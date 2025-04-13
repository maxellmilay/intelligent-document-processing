# Intelligent Document Processing

A framework for extracting structured information from documents, with a focus on insurance claim emails.

## Overview

This project provides a set of tools for extracting structured information from emails, including:

- Email body text
- Email metadata (sender, recipients, subject, etc.)
- Attachments (images, PDFs, and other documents)

The framework is built around a flexible Extractor base class that can be extended to implement different extraction methods.

## Installation

Clone the repository and install required dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/intelligent-document-processing.git
cd intelligent-document-processing

# Install basic dependencies
pip install -r requirements.txt

# Install OCR dependencies (optional)
pip install pytesseract pillow

# Install NER dependencies (optional)
pip install spacy
python -m spacy download en_core_web_sm
```

## Extractor Framework

The core of this project is the `Extractor` base class, which provides a common interface for various information extraction techniques:

```python
from lib.tools.extractors.base import Extractor
```

The base class provides:
- A common interface for information extraction
- Processing methods for emails and their components
- Support for different extraction engines (Tesseract, spaCy, LLMs, etc.)
- Customizable extraction logic

### Available Extractors

- **OCRExtractor**: Uses Optical Character Recognition to extract text from images and then extracts structured information from that text.
- **NERExtractor**: Uses Named Entity Recognition to extract structured information from text.

## Example Usage

Here's a simple example of how to use the extractors:

```python
from lib.tools.extractors.ocr_extractor import OCRExtractor
from lib.tools.extractors.ner_extractor import NERExtractor

# Initialize extractors with configuration
ocr_extractor = OCRExtractor(config={
    'lang': 'eng',
    'preprocess': True,
    # Optional extraction engine configuration
    'extraction_engine_type': 'llm',
    'engine_config': {
        'model': 'gpt-3.5-turbo'
    }
})

ner_extractor = NERExtractor(config={
    'spacy_model': 'en_core_web_sm',
    'confidence_threshold': 0.7
})

# Process an email (dictionary with body, metadata, and attachments)
ocr_results = ocr_extractor.process_email(email_data)
ner_results = ner_extractor.process_email(email_data)

# Extract from specific content
text_content = "Policy #: ABC123456 - Claim filed by John Doe"
ner_results = ner_extractor.extract(text_content, "text")
```

### Processing an Email File

The project includes an example script for processing `.eml` files:

```bash
python examples/email_processing_example.py path/to/email.eml --output results.json --use-llm
```

## Creating Custom Extractors

You can create custom extractors by inheriting from the `Extractor` base class and implementing the `extraction_logic` method:

```python
from lib.tools.extractors.base import Extractor

class MyCustomExtractor(Extractor):
    def __init__(self, config=None):
        super().__init__(config)
        # Initialize your custom extractor
        
    def _initialize_extraction_engine(self):
        # Initialize any extraction engines based on config
        if 'custom_engine' in self.config:
            # Load and set up your extraction engine
            self.extraction_engine = CustomEngine(**self.config.get('engine_config', {}))
    
    def extraction_logic(self, content, content_type="text", metadata=None):
        # Implement your extraction logic
        # This is where the actual extraction happens
        extracted_data = {}
        
        # Example: simple regex extraction
        if content_type == "text" and isinstance(content, str):
            import re
            policy_match = re.search(r'Policy\s*#?:?\s*(\w+)', content)
            if policy_match:
                extracted_data['policy_number'] = policy_match.group(1)
        
        # Use extraction engine if available
        if self.extraction_engine and hasattr(self.extraction_engine, 'process'):
            additional_data = self.extraction_engine.process(content)
            extracted_data.update(additional_data)
            
        return extracted_data
```

## Extraction Engines

The framework is designed to work with various extraction engines:

1. **OCR Engines**: Like Tesseract for converting images to text
2. **NLP Models**: Such as spaCy for entity recognition
3. **LLMs**: For advanced extraction and refinement
4. **Custom Models**: Any model or tool that extracts structured information

You can configure the extraction engine in the extractor's configuration:

```python
extractor = OCRExtractor(config={
    # Base configuration
    'lang': 'eng',
    
    # Extraction engine configuration
    'extraction_engine_type': 'llm',
    'engine_config': {
        'model': 'gpt-4',
        'temperature': 0.3
    }
})
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.