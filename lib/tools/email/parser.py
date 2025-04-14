import os
import re
import tempfile
import shutil
import streamlit as st
from email.parser import BytesParser
from email.policy import default
from io import BytesIO

class EmailParser:
    """
    Class for parsing EML files and extracting relevant information
    """
    
    def __init__(self):
        """Initialize the EmailParser"""
        self.parser = BytesParser(policy=default)
    
    def parse(self, uploaded_file):
        """
        Parse an uploaded .eml file into its components.
        
        Args:
            uploaded_file: Streamlit uploaded file object or file-like object
            
        Returns:
            Dictionary with email components
        """
        # Handle both Streamlit UploadedFile and regular file paths
        if hasattr(uploaded_file, 'getvalue'):
            # It's a Streamlit UploadedFile
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
                
            with open(temp_path, 'rb') as fp:
                msg = self.parser.parse(fp)
                
            # Clean up the temp file
            os.unlink(temp_path)
        else:
            # Assume it's a file path
            with open(uploaded_file, 'rb') as fp:
                msg = self.parser.parse(fp)
        
        email_data = {
            'body': '',
            'metadata': {
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'cc': msg.get('Cc', ''),
                'bcc': msg.get('Bcc', ''),
                'subject': msg.get('Subject', ''),
                'date': msg.get('Date', '')
            },
            'attachments': []
        }
        
        # Get body content
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                # If it's plain text and not an attachment, it's probably the body
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    email_data['body'] = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
        else:
            email_data['body'] = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        
        # Get attachments
        for part in msg.walk():
            if part.get_content_disposition() == 'attachment':
                filename = part.get_filename()
                content = part.get_payload(decode=True)
                content_type = part.get_content_type()
                
                # Check if content is likely corrupted/placeholder
                is_corrupted = self.is_attachment_corrupted(content, content_type)
                
                email_data['attachments'].append({
                    'filename': filename,
                    'content': content,
                    'type': content_type,
                    'is_corrupted': is_corrupted,
                    'metadata': {
                        'content_type': content_type,
                        'filename': filename,
                        'is_corrupted': is_corrupted
                    }
                })
        
        # Extract attachment references from the email body
        attachment_references = self.extract_attachment_references(email_data['body'])
        email_data['attachment_references'] = attachment_references
        
        return email_data
    
    def is_attachment_corrupted(self, content, content_type):
        """
        Check if attachment content is likely corrupted or a placeholder
        
        Args:
            content: Attachment binary content
            content_type: Content type of the attachment
            
        Returns:
            Boolean indicating if attachment appears corrupted
        """
        # Check for very small size
        if len(content) < 100:
            return True
        
        # For images, try to open with PIL
        if content_type.startswith('image/'):
            try:
                from PIL import Image
                image = Image.open(BytesIO(content))
                image.verify()  # Verify it's a valid image
                return False
            except Exception:
                return True
        
        # For PDFs, check for PDF header
        if content_type == 'application/pdf':
            return not content.startswith(b'%PDF')
        
        # For text files, try to decode
        if content_type.startswith('text/'):
            try:
                text = content.decode('utf-8', errors='strict')
                # If very short or all the same character, it's likely a placeholder
                if len(text) < 20 or len(set(text)) < 5:
                    return True
                return False
            except UnicodeDecodeError:
                return True
        
        # Default to assuming it's not corrupted
        return False
    
    def extract_attachment_references(self, email_body):
        """
        Extract references to attachments from the email body
        
        Args:
            email_body: The email body text
            
        Returns:
            List of dictionaries containing attachment references
        """
        references = []
        
        # Look for specific file types mentioned
        file_extensions = [
            r"(?i)(\w+\.(pdf|doc|docx|jpg|jpeg|png|xls|xlsx|csv|txt))",
        ]
        
        for pattern in file_extensions:
            matches = re.finditer(pattern, email_body)
            for match in matches:
                filename = match.group(1).strip()
                ref_context = email_body[max(0, match.start()-30):min(len(email_body), match.end()+30)]
                references.append({
                    'text': filename,
                    'context': ref_context,
                    'position': match.start()
                })
        
        return references
        
    def save_attachments_to_disk(self, attachments, temp_dir=None):
        """
        Save email attachments to disk
        
        Args:
            attachments: List of attachment objects
            temp_dir: Optional temporary directory path
            
        Returns:
            Tuple of (directory_path, dictionary mapping attachment indices to file paths)
        """
        # Create a temporary directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        else:
            os.makedirs(temp_dir, exist_ok=True)
        
        attachment_files = {}
        
        for i, attachment in enumerate(attachments):
            # Make sure the filename is safe for the filesystem
            safe_filename = "".join(c for c in attachment['filename'] if c.isalnum() or c in "._- ")
            
            # Create a unique filename in case of duplicates
            unique_filename = f"{i+1}_{safe_filename}"
            file_path = os.path.join(temp_dir, unique_filename)
            
            # Save the attachment to disk
            with open(file_path, 'wb') as f:
                f.write(attachment['content'])
            
            # Store the file path in the attachment mapping
            attachment_files[i] = {
                'path': file_path,
                'filename': attachment['filename'],
                'type': attachment['type'],
                'is_corrupted': attachment.get('is_corrupted', False)
            }
        
        return temp_dir, attachment_files
    
    def export_attachments(self, attachment_files, output_dir):
        """
        Export attachments to a specified directory
        
        Args:
            attachment_files: Dictionary of attachment files
            output_dir: Directory to export to
            
        Returns:
            List of exported file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        exported_files = []
        
        for attachment_id, file_info in attachment_files.items():
            src_path = file_info['path']
            filename = file_info['filename']
            
            # Create a safe filename
            safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
            dest_path = os.path.join(output_dir, safe_filename)
            
            # Handle duplicate filenames
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(safe_filename)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(output_dir, f"{base}_{counter}{ext}")
                    counter += 1
            
            # Copy the file
            shutil.copy2(src_path, dest_path)
            exported_files.append(dest_path)
        
        return exported_files
    
    def prepare_content_for_extraction(self, content, content_type):
        """
        Prepare content for extraction by ensuring it's in a format the extractors can handle
        
        Args:
            content: The content to prepare
            content_type: The type of the content
            
        Returns:
            Prepared content and updated content type
        """
        # If it's already text, return as is
        if isinstance(content, str):
            return content, 'text'
        
        # For images and PDFs, we can't process them directly in the Streamlit app
        # But we'll return them with the correct type so the extractors can handle them
        if content_type.startswith('image/'):
            return content, 'image'
        
        # For other binary data, try to convert to text if possible
        try:
            text_content = content.decode('utf-8', errors='ignore')
            return text_content, 'text'
        except:
            # If we can't decode, return as is
            return content, content_type
            
    def prepare_attachment_preview(self, attachment, file_path=None):
        """
        Prepare attachment preview data without UI-specific components
        
        Args:
            attachment: Attachment data dictionary
            file_path: Optional file path to the attachment on disk
            
        Returns:
            Dictionary with preview information
        """
        content_type = attachment['type']
        is_corrupted = attachment.get('is_corrupted', False)
        preview_data = {
            'type': content_type,
            'filename': attachment['filename'],
            'is_corrupted': is_corrupted,
            'file_path': file_path,
            'preview_type': None,
            'content': None,
            'error': None
        }
        
        # For images
        if content_type.startswith('image/'):
            preview_data['preview_type'] = 'image'
            
            try:
                if file_path and os.path.exists(file_path) and not is_corrupted:
                    preview_data['content'] = file_path  # Return the file path for the UI to load
                elif not is_corrupted:
                    preview_data['content'] = attachment['content']  # Return the binary content
                else:
                    preview_data['error'] = "Cannot display corrupted image"
            except Exception as e:
                preview_data['error'] = f"Could not prepare image preview: {str(e)}"
                preview_data['content'] = None
        
        # For text files
        elif content_type in ['text/plain', 'text/html', 'text/csv', 'application/json']:
            preview_data['preview_type'] = 'text'
            
            try:
                # Try to read from file if path is provided
                if file_path and os.path.exists(file_path) and not is_corrupted:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                else:
                    text = attachment['content'].decode('utf-8', errors='ignore')
                
                preview_data['content'] = text
            except Exception as e:
                preview_data['error'] = f"Unable to decode text content: {str(e)}"
        
        # For PDFs
        elif content_type == 'application/pdf':
            preview_data['preview_type'] = 'pdf'
            
            if is_corrupted:
                preview_data['error'] = "PDF data appears to be corrupted or incomplete"
            else:
                preview_data['content'] = attachment['content']
        
        # Other file types
        else:
            preview_data['preview_type'] = 'other'
            preview_data['content'] = attachment['content']
        
        return preview_data
        
    def display_attachment_preview(self, attachment, file_path=None):
        """
        Display attachment preview in Streamlit UI
        
        Args:
            attachment: Attachment data dictionary
            file_path: Optional file path to the attachment on disk
        """
        # Get the preview data
        preview_data = self.prepare_attachment_preview(attachment, file_path)
        
        # Display appropriate warnings for corrupted files
        if preview_data['is_corrupted']:
            st.warning("⚠️ This attachment appears to be corrupted or contains placeholder data.")
        
        # Display error message if there was an error preparing the preview
        if preview_data['error']:
            st.error(preview_data['error'])
            return
        
        # Display based on preview type
        if preview_data['preview_type'] == 'image':
            try:
                # If we have a file path or content, display it
                if isinstance(preview_data['content'], str) and os.path.exists(preview_data['content']):
                    st.image(preview_data['content'], caption=preview_data['filename'])
                elif preview_data['content']:
                    st.image(preview_data['content'], caption=preview_data['filename'])
            except Exception as e:
                st.error(f"Could not display image: {str(e)}")
                st.write("The image attachment may be truncated or corrupted.")
        
        # Text files
        elif preview_data['preview_type'] == 'text':
            if preview_data['content']:
                text = preview_data['content']
                if len(text) > 1000:
                    st.text_area("Content Preview", text[:1000] + "...", height=200)
                else:
                    st.text_area("Content", text, height=200)
        
        # PDFs
        elif preview_data['preview_type'] == 'pdf':
            st.write("PDF preview is not available, but you can download the file below.")
            
            # If we have a file path, offer a link to open it
            if preview_data['file_path'] and os.path.exists(preview_data['file_path']):
                st.write(f"PDF saved at: {preview_data['file_path']}")
        
        # Other file types
        else:
            st.write(f"Preview not available for {preview_data['type']} files")
            
            # If we have a file path, show it
            if preview_data['file_path'] and os.path.exists(preview_data['file_path']):
                st.write(f"File saved at: {preview_data['file_path']}")
