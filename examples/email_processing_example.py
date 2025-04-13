#!/usr/bin/env python3

import os
import sys
import json
import logging
from email import message_from_file
from email.parser import BytesParser
from email.policy import default
import argparse

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.tools.extractors.base import Extractor
from lib.tools.extractors.ocr_extractor import OCRExtractor
from lib.tools.extractors.ner_extractor import NERExtractor

# Optional: Import a language model wrapper if available
try:
    from lib.tools.llm.wrapper import LLMWrapper
    has_llm = True
except ImportError:
    has_llm = False


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def parse_eml_file(eml_path):
    """
    Parse an .eml file into its components.
    
    Args:
        eml_path: Path to the .eml file
        
    Returns:
        Dictionary with email components
    """
    with open(eml_path, 'rb') as fp:
        msg = BytesParser(policy=default).parse(fp)
    
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
            
            email_data['attachments'].append({
                'filename': filename,
                'content': content,
                'type': content_type,
                'metadata': {
                    'content_type': content_type,
                    'filename': filename
                }
            })
    
    return email_data


def main():
    parser = argparse.ArgumentParser(description='Process an insurance claim email.')
    parser.add_argument('eml_file', help='Path to the .eml file to process')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for extraction refinement')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info(f"Processing email: {args.eml_file}")
    
    # Parse the email
    email_data = parse_eml_file(args.eml_file)
    logger.info(f"Parsed email with {len(email_data['attachments'])} attachments")
    
    # Configure extraction engines if requested
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
    
    # Add LLM configuration if requested
    if args.use_llm and has_llm:
        logger.info("Initializing LLM extraction engine...")
        # Example LLM configuration
        llm_engine_config = {
            'extraction_engine_type': 'llm',
            'engine_config': {
                'model': 'gpt-3.5-turbo',
                'temperature': 0.3
            }
        }
        ocr_config.update(llm_engine_config)
        ner_config.update(llm_engine_config)
    
    # Initialize extractors with configs
    ocr_extractor = OCRExtractor(config=ocr_config)
    ner_extractor = NERExtractor(config=ner_config)
    
    # Process the email with both extractors
    ocr_results = ocr_extractor.process_email(email_data)
    ner_results = ner_extractor.process_email(email_data)
    
    # Combine results, giving preference to NER results for overlapping fields
    combined_results = {
        'email_metadata': email_data['metadata'],
        'extracted_fields': {}
    }
    
    # Add OCR results first
    if 'fields' in ocr_results:
        combined_results['extracted_fields'].update(ocr_results['fields'])
    elif 'extracted_text' in ocr_results:
        combined_results['extracted_fields']['ocr_text'] = ocr_results['extracted_text']
    
    # Add/override with NER results
    if 'fields' in ner_results:
        combined_results['extracted_fields'].update(ner_results['fields'])
    
    # Add all entities from NER
    if 'entities' in ner_results:
        combined_results['entities'] = ner_results['entities']
    
    # Print a summary
    logger.info("=== Extraction Results ===")
    for field, value in combined_results['extracted_fields'].items():
        logger.info(f"{field}: {value}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(combined_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main() 