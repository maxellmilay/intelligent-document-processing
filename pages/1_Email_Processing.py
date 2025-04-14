import streamlit as st
import os
import sys
import json
import tempfile
import logging
import base64
import shutil
import pandas as pd

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.tools.extractors.ocr import OCRExtractor
from lib.tools.extractors.ner import NERExtractor
from lib.tools.email.parser import EmailParser
from lib.tools.llm.wrapper import LLM

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def process_email(email_data, selected_attachments=None, attachment_files=None):
    """
    Process email data following the flow:
    1. Extract text data (email body, metadata, text documents) and process with NER
    2. Extract image/PDF data and process with OCR
    3. Feed OCR results into NER
    4. Build prompt from NER results
    5. Send prompt to LLM
    """
    
    # Configure extraction engines
    ocr_config = {
        'lang': 'eng',
        'preprocess': True
    }
    
    ner_config = {
        'spacy_model': 'en_core_web_sm',
        'confidence_threshold': 0.7,
        'custom_entities': {
            'POLICY_NUMBER': [
                {'PATTERN': [{'LOWER': 'policy'}, {'LOWER': 'number'}, {'TEXT': ':'}, {'IS_DIGIT': True}]},
                {'PATTERN': [{'LOWER': 'policy'}, {'LOWER': '#'}, {'IS_DIGIT': True}]},
                {'PATTERN': [{'LOWER': 'policy'}, {'TEXT': '#'}, {'IS_DIGIT': True}]}
            ],
            'CLAIM_NUMBER': [
                {'PATTERN': [{'LOWER': 'claim'}, {'LOWER': 'number'}, {'TEXT': ':'}, {'IS_DIGIT': True}]},
                {'PATTERN': [{'LOWER': 'claim'}, {'LOWER': '#'}, {'IS_DIGIT': True}]},
                {'PATTERN': [{'LOWER': 'claim'}, {'TEXT': '#'}, {'IS_DIGIT': True}]}
            ]
        }
    }
    
    # Initialize extractors with configs
    ocr_extractor = OCRExtractor(config=ocr_config)
    ner_extractor = NERExtractor(config=ner_config)
    
    # Initialize LLM
    llm_wrapper = LLM(config={
        'model': 'gpt-3.5-turbo',
        'temperature': 0.3
    })
    
    # Filter attachments if selected
    attachments_to_process = email_data['attachments']
    if selected_attachments is not None:
        attachments_to_process = [
            attachment for i, attachment in enumerate(email_data['attachments'])
            if i in selected_attachments
        ]
    
    # Initialize results holders
    text_content_list = []  # All text content for NER
    image_content_list = []  # Image attachments for OCR
    ocr_results = {}  # Results from OCR
    
    # --- STEP 1: Process all text data with NER ---
    
    # Add email body to text content
    text_content_list.append({
        'content': email_data['body'],
        'source': 'Email Body',
        'type': 'text'
    })
    
    # Add metadata as text
    metadata_text = "\n".join([f"{k}: {v}" for k, v in email_data['metadata'].items()])
    text_content_list.append({
        'content': metadata_text,
        'source': 'Email Metadata',
        'type': 'text'
    })
    
    # Add text attachments
    for i, attachment in enumerate(attachments_to_process):
        attachment_index = email_data['attachments'].index(attachment)
        
        # Skip corrupted attachments
        if attachment.get('is_corrupted', False):
            continue
            
        content_type = attachment['type']
        
        # Process text attachments with NER
        if content_type.startswith('text/'):
            try:
                # Get file path if available
                file_info = attachment_files.get(attachment_index) if attachment_files else None
                file_path = file_info['path'] if file_info else None
                
                # Read text content
                if file_path and os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()
                else:
                    text_content = attachment['content'].decode('utf-8', errors='ignore')
                
                # Add to text content list
                source = f"Attachment: {attachment['filename']}"
                text_content_list.append({
                    'content': text_content,
                    'source': source,
                    'type': 'text'
                })
            except Exception as e:
                st.warning(f"Could not read text from {attachment['filename']}: {str(e)}")
        elif content_type.startswith('image/') or content_type == 'application/pdf':
            # Collect images for OCR processing
            source = f"Attachment: {attachment['filename']}"
            
            # Get file path if available
            file_info = attachment_files.get(attachment_index) if attachment_files else None
            file_path = file_info['path'] if file_info else None
            
            image_content_list.append({
                'content': attachment['content'],
                'path': file_path,
                'source': source,
                'type': content_type,
                'filename': attachment['filename']
            })
    
    # Add attachment references from email body
    if 'attachment_references' in email_data and email_data['attachment_references']:
        for ref in email_data['attachment_references']:
            text_content_list.append({
                'content': ref['context'],
                'source': f"Attachment Reference: {ref['text']}",
                'type': 'text'
            })
    
    # --- STEP 2: Process all text with NER ---
    with st.spinner("Processing text content with NER..."):
        all_ner_results = {}
        for text_item in text_content_list:
            try:
                result = ner_extractor.extract(text_item['content'], 'text')
                
                # Add source information to each entity
                if 'entities' in result:
                    for entity_type, entities in result['entities'].items():
                        for entity in entities:
                            entity['source'] = text_item['source']
                
                # Store results by source
                all_ner_results[text_item['source']] = result
            except Exception as e:
                st.warning(f"Could not process {text_item['source']} with NER: {str(e)}")
    
    # --- STEP 3: Process images with OCR ---
    ocr_text_results = []
    with st.spinner("Processing images with OCR..."):
        for image_item in image_content_list:
            try:
                # Use file path if available, otherwise use content
                if image_item['path'] and os.path.exists(image_item['path']):
                    ocr_result = ocr_extractor.extract(image_item['path'], 'image')
                else:
                    ocr_result = ocr_extractor.extract(image_item['content'], 'image')
                
                # Extract text from OCR result
                ocr_text = ""
                if 'text' in ocr_result:
                    ocr_text = ocr_result['text']
                elif 'raw_text' in ocr_result:
                    ocr_text = ocr_result['raw_text']
                
                # Store OCR results
                ocr_results[image_item['source']] = {
                    'text': ocr_text,
                    'filename': image_item['filename'],
                    'type': image_item['type']
                }
                
                # Add OCR text for NER processing in step 4
                if ocr_text:
                    ocr_text_results.append({
                        'content': ocr_text,
                        'source': f"OCR from {image_item['source']}",
                        'type': 'text'
                    })
            except Exception as e:
                st.warning(f"Could not process {image_item['source']} with OCR: {str(e)}")
    
    # --- STEP 4: Process OCR results with NER ---
    with st.spinner("Processing OCR results with NER..."):
        for ocr_item in ocr_text_results:
            try:
                ocr_ner_result = ner_extractor.extract(ocr_item['content'], 'text')
                
                # Add source information to each entity
                if 'entities' in ocr_ner_result:
                    for entity_type, entities in ocr_ner_result['entities'].items():
                        for entity in entities:
                            entity['source'] = ocr_item['source']
                
                # Store OCR-NER results
                all_ner_results[ocr_item['source']] = ocr_ner_result
            except Exception as e:
                st.warning(f"Could not process {ocr_item['source']} with NER: {str(e)}")
    
    # --- STEP 5: Build prompt for LLM ---
    prompt = None
    llm_results = None
    
    with st.spinner("Building prompt and querying LLM..."):
        # Extract all entities and fields
        all_entities = {}
        all_fields = {}
        
        for source, result in all_ner_results.items():
            if 'entities' in result:
                for entity_type, entities in result['entities'].items():
                    if entity_type not in all_entities:
                        all_entities[entity_type] = []
                    all_entities[entity_type].extend(entities)
            
            if 'fields' in result:
                for field_name, field_value in result['fields'].items():
                    # Add source if it's not already in the field name
                    if source not in field_name:
                        field_name = f"{field_name} ({source})"
                    all_fields[field_name] = field_value
        
        # Build prompt
        prompt = build_llm_prompt(
            email_data=email_data,
            entities=all_entities,
            fields=all_fields,
            ocr_results=ocr_results
        )
        
        # Query LLM
        try:
            # Use complete() method instead of query() which doesn't exist
            llm_results = llm_wrapper.complete(prompt)
            
            # Try to parse the result as JSON if possible
            try:
                import json
                llm_results = json.loads(llm_results)
            except json.JSONDecodeError:
                # If not valid JSON, keep as string
                pass
                
        except Exception as e:
            st.error(f"Error querying LLM: {str(e)}")
            llm_results = {"error": str(e)}
    
    # --- STEP 6: Combine all results ---
    combined_results = {
        'email_metadata': email_data['metadata'],
        'processed_attachments': [
            item['filename'] for source, item in ocr_results.items()
        ] + [
            text_item['source'].split(': ')[1] 
            for text_item in text_content_list 
            if ': ' in text_item['source'] and not text_item['source'].startswith('OCR')
        ],
        'ocr_processed': list(ocr_results.keys()),
        'attachment_files': [
            item['path'] for item in image_content_list if item['path']
        ],
        'attachment_references': email_data.get('attachment_references', []),
        'extracted_fields': {},
        'entities': {},
        'ocr_text': {source: data['text'] for source, data in ocr_results.items()},
    }
    
    # Add all extracted fields
    for source, result in all_ner_results.items():
        if 'fields' in result:
            for field, value in result['fields'].items():
                field_key = f"{field} ({source})"
                combined_results['extracted_fields'][field_key] = value
    
    # Add all entities
    for source, result in all_ner_results.items():
        if 'entities' in result:
            for entity_type, entities in result['entities'].items():
                if entity_type not in combined_results['entities']:
                    combined_results['entities'][entity_type] = []
                combined_results['entities'][entity_type].extend(entities)
    
    # Add LLM results if available
    if llm_results:
        combined_results['llm_results'] = llm_results
        combined_results['llm_prompt'] = prompt
    
    return combined_results

def build_llm_prompt(email_data, entities, fields, ocr_results):
    """
    Build a prompt for the LLM based on extracted information
    
    Args:
        email_data: Dictionary containing email data
        entities: Dictionary of entities extracted by NER
        fields: Dictionary of fields extracted
        ocr_results: Dictionary of OCR results
        
    Returns:
        Prompt string for LLM
    """
    prompt = f"""
You are an AI assistant specialized in extracting information from emails and their attachments.
I have processed an email with NER and OCR and need your help to extract structured information.

Email Subject: {email_data['metadata'].get('subject', 'N/A')}
From: {email_data['metadata'].get('from', 'N/A')}
To: {email_data['metadata'].get('to', 'N/A')}
Date: {email_data['metadata'].get('date', 'N/A')}

Email Body Summary:
{email_data['body'][:500]}{"..." if len(email_data['body']) > 500 else ""}

The following entities and fields have been extracted:

EXTRACTED FIELDS:
{json.dumps(fields, indent=2) if fields else "No fields extracted"}

EXTRACTED ENTITIES:
{json.dumps(entities, indent=2) if entities else "No entities extracted"}

OCR RESULTS:
{json.dumps({src: data['text'][:100] + "..." for src, data in ocr_results.items()}) if ocr_results else "No OCR results"}

Based on this information, please:
1. Identify the key information in this email (topic, purpose, actions required)
2. Extract any specific fields like policy numbers, claim numbers, dates, amounts, etc.
3. Summarize the content in a structured way
4. Note any inconsistencies or missing information

Format your response in JSON with these sections:
- summary: A brief summary of the email
- extracted_fields: Key information fields you've identified
- missing_info: Any critical information that seems to be missing
- suggested_action: What action should be taken with this email
"""
    return prompt

def get_file_download_link(file_content, file_name, mime_type):
    """Generate a download link for a file"""
    b64 = base64.b64encode(file_content).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{file_name}">Download {file_name}</a>'

def get_image_html(file_content, file_name, content_type):
    """Generate HTML to display an image"""
    b64 = base64.b64encode(file_content).decode()
    return f'<img src="data:{content_type};base64,{b64}" alt="{file_name}" style="max-width: 100%; max-height: 300px;">'

# Page title and description
st.title("Email Processing")
st.write("Upload and process email (.eml) files to extract information using OCR and NER.")

# Create email parser instance
email_parser = EmailParser()

# File uploader
uploaded_file = st.file_uploader("Upload an email file", type=["eml"])

# Set a session state for temporary directory
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
    st.session_state.attachment_files = None

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    
    # Process the email using the EmailParser class
    with st.spinner("Parsing email..."):
        email_data = email_parser.parse(uploaded_file)
    
    # Check for attachment references
    if 'attachment_references' in email_data and email_data['attachment_references']:
        st.subheader("Attachment References Found in Email Text")
        for ref in email_data['attachment_references']:
            st.markdown(f"**Referenced File:** {ref['text']}")
            st.markdown(f"*Context:* \"{ref['context']}\"")
        
        st.info("These references were extracted from the email body and may provide context about attachments.")
    
    # Check for corrupted attachments
    corrupted_count = sum(1 for a in email_data['attachments'] if a.get('is_corrupted', False))
    if corrupted_count > 0:
        st.warning(f"⚠️ {corrupted_count} out of {len(email_data['attachments'])} attachments appear to be corrupted or contain placeholder data.")
        st.info("The application will skip processing corrupted attachments but will extract information from the email body and any context that references these attachments.")
    
    # Create temporary directory for attachments if needed
    if st.session_state.temp_dir is None or not os.path.exists(st.session_state.temp_dir):
        st.session_state.temp_dir = tempfile.mkdtemp()
        st.session_state.attachment_files = {}
    
    # Save attachments to disk using EmailParser
    if email_data['attachments']:
        with st.spinner("Saving attachments to disk..."):
            temp_dir, attachment_files = email_parser.save_attachments_to_disk(
                email_data['attachments'], 
                st.session_state.temp_dir
            )
            st.session_state.temp_dir = temp_dir
            st.session_state.attachment_files = attachment_files
    
    # Display email metadata
    st.subheader("Email Metadata")
    metadata_df = pd.DataFrame({
        "Field": list(email_data['metadata'].keys()),
        "Value": list(email_data['metadata'].values())
    })
    st.dataframe(metadata_df)
    
    # Display email body preview
    st.subheader("Email Body Preview")
    st.text_area("Content", email_data['body'][:500] + ("..." if len(email_data['body']) > 500 else ""), height=150)
    
    # Display and handle attachments
    if email_data['attachments']:
        st.subheader("Attachments")
        
        # Create tabs for attachments
        attachment_names = [
            f"{i+1}. {a['filename']} {'⚠️' if a.get('is_corrupted', False) else ''}" 
            for i, a in enumerate(email_data['attachments'])
        ]
        tabs = st.tabs(attachment_names)
        
        for i, (tab, attachment) in enumerate(zip(tabs, email_data['attachments'])):
            with tab:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Type:** {attachment['type']}")
                    st.write(f"**Size:** {len(attachment['content'])} bytes")
                    
                    # Show warning for corrupted attachments
                    if attachment.get('is_corrupted', False):
                        st.warning("⚠️ This attachment appears to be corrupted or contains placeholder data.")
                    
                    # Get file path if available
                    file_info = st.session_state.attachment_files.get(i)
                    file_path = file_info['path'] if file_info else None
                    
                    if file_path:
                        st.write(f"**File Path:** {file_path}")
                    
                    # Show preview of the attachment
                    st.write("**Preview:**")
                    email_parser.display_attachment_preview(attachment, file_path)
                
                with col2:
                    # Download button for the attachment
                    st.download_button(
                        label=f"Download",
                        data=attachment['content'],
                        file_name=attachment['filename'],
                        mime=attachment['type']
                    )
        
        # Select attachments to process
        st.subheader("Select Attachments to Process")
        attachment_options = [f"{i+1}. {a['filename']} {'⚠️' if a.get('is_corrupted', False) else ''}" for i, a in enumerate(email_data['attachments'])]
        default_selections = list(range(len(email_data['attachments'])))
        selected_attachment_indices = []
        
        for i, option in enumerate(attachment_options):
            is_selected = st.checkbox(option, value=True, key=f"attachment_{i}")
            if is_selected:
                selected_attachment_indices.append(i)
    
    # Process button
    if st.button("Process Email"):
        # If there are attachments, use only the selected ones
        selected_attachments = selected_attachment_indices if email_data['attachments'] else None
        
        with st.spinner("Processing email and attachments..."):
            try:
                results = process_email(
                    email_data, 
                    selected_attachments, 
                    st.session_state.attachment_files
                )
                
                st.subheader("Extraction Results")
                
                # Show which attachments were processed
                if 'processed_attachments' in results and results['processed_attachments']:
                    st.write("**Processed Attachments:**")
                    for attachment_name in results['processed_attachments']:
                        st.write(f"- {attachment_name}")
                
                # Display attachment references if available
                if 'attachment_references' in results and results['attachment_references']:
                    st.write("**Attachment References from Email Body:**")
                    for ref in results['attachment_references']:
                        st.markdown(f"- **{ref['text']}**: \"{ref['context']}\"")
                
                # Display file paths
                if 'attachment_files' in results and results['attachment_files']:
                    st.write("**Attachment Files:**")
                    for file_path in results['attachment_files']:
                        if file_path and os.path.exists(file_path):
                            st.write(f"- {file_path}")
                
                # Display extracted fields
                if results['extracted_fields']:
                    st.write("**Extracted Fields:**")
                    fields_df = pd.DataFrame({
                        "Field": list(results['extracted_fields'].keys()),
                        "Value": list(results['extracted_fields'].values())
                    })
                    st.dataframe(fields_df)
                else:
                    st.info("No fields were extracted from the email.")
                
                # Display entities if available
                if 'entities' in results and results['entities']:
                    st.write("**Named Entities:**")
                    
                    # Format entities data for display
                    all_entities_data = []
                    for entity_type, entities in results['entities'].items():
                        for entity in entities:
                            all_entities_data.append({
                                "Type": entity_type,
                                "Text": entity['text'],
                                "Source": entity.get('source', 'Email Body'),
                                "Confidence": entity.get('confidence', 'N/A')
                            })
                    
                    if all_entities_data:
                        entities_df = pd.DataFrame(all_entities_data)
                        st.dataframe(entities_df)
                    else:
                        st.info("No entities were detected in the email.")
                
                # Download results as JSON
                if st.download_button(
                    label="Download Results as JSON",
                    data=json.dumps(results, indent=2),
                    file_name=f"{uploaded_file.name.split('.')[0]}_results.json",
                    mime="application/json"
                ):
                    st.success("Results downloaded successfully!")
            except Exception as e:
                st.error(f"Error processing email: {str(e)}")
                st.error("Please check the log for details and try a different email.")

# Clean up temporary directory when the app is closed
def cleanup():
    if 'temp_dir' in st.session_state and st.session_state.temp_dir:
        try:
            if os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir)
                logger.info(f"Cleaned up temporary directory: {st.session_state.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")

# Register cleanup function
import atexit
atexit.register(cleanup) 