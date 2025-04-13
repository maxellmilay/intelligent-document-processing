from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import logging


class Extractor(ABC):
    """
    Base class for all extractors used in Intelligent Document Processing (IDP).
    
    Extractors are responsible for extracting structured information from various 
    sources within emails, including the email body, metadata, and attachments.
    
    This class is meant to be subclassed by specific extraction implementations
    such as OCR, NER, regex-based extractors, etc.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extractor.
        
        Args:
            config: Configuration options for the extractor, including:
                   - extraction_engine: The engine to use for extraction (optional)
                   - engine_config: Configuration for the extraction engine
                   - Any other extractor-specific settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize extraction engine if specified in config
        self.extraction_engine = None
        if 'extraction_engine' in self.config:
            self._initialize_extraction_engine()
    
    def _initialize_extraction_engine(self):
        """
        Initialize the extraction engine based on configuration.
        This method should be overridden by subclasses.
        """
        pass
    
    @abstractmethod
    def extraction_logic(self, content: Union[str, bytes], content_type: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Custom extraction logic to be implemented by each extractor.
        This is where the specific extraction algorithm/logic should be implemented.
        
        Args:
            content: The content to extract information from
            content_type: Type of the content
            metadata: Additional metadata that might be useful for extraction
            
        Returns:
            Dictionary containing the extracted fields and their values
        """
        pass
    
    def extract(self, content: Union[str, bytes], content_type: str = "text", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract information from the provided content.
        This is the main public API method for extraction.
        
        Args:
            content: The content to extract information from (text, image bytes, etc.)
            content_type: Type of the content ('text', 'image', 'pdf', 'email', etc.)
            metadata: Additional metadata that might be useful for extraction
            
        Returns:
            Dictionary containing the extracted fields and their values
        """
        # Perform extraction using custom logic
        results = self.extraction_logic(content, content_type, metadata)
        
        # Validate results
        validated_results = self.validate_results(results)
        
        return validated_results
    
    def process_email(self, email_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an entire email including body, metadata, and attachments.
        
        Args:
            email_content: Dictionary containing parsed email data with keys like:
                - 'body': The email body text
                - 'metadata': Dictionary with sender, recipients, subject, etc.
                - 'attachments': List of attachment objects with content and type
                
        Returns:
            Combined dictionary of all extracted fields from all sources
        """
        results = {}
        
        # Extract from email body
        if 'body' in email_content:
            body_results = self.extract(email_content['body'], 'text', email_content.get('metadata'))
            results.update(body_results)
        
        # Extract from email metadata
        if 'metadata' in email_content:
            metadata_results = self.extract_from_metadata(email_content['metadata'])
            results.update(metadata_results)
        
        # Extract from attachments
        if 'attachments' in email_content:
            attachment_results = self.extract_from_attachments(email_content['attachments'])
            results.update(attachment_results)
            
        return results
    
    def extract_from_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information from email metadata.
        
        Args:
            metadata: Dictionary containing email metadata (sender, recipients, etc.)
            
        Returns:
            Dictionary of extracted fields from metadata
        """
        # Default implementation - can be overridden by subclasses
        return self.extract(str(metadata), 'metadata', metadata)
    
    def extract_from_attachments(self, attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract information from email attachments.
        
        Args:
            attachments: List of attachment objects with content and type
            
        Returns:
            Dictionary of extracted fields from all attachments
        """
        results = {}
        
        for attachment in attachments:
            content = attachment.get('content')
            content_type = attachment.get('type', 'unknown')
            
            if content:
                attachment_result = self.extract(content, content_type, attachment.get('metadata'))
                results.update(attachment_result)
                
        return results
    
    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the extracted results.
        
        Args:
            results: The extracted results to validate
            
        Returns:
            Validated (and possibly corrected) results
        """
        # Default implementation - can be overridden by subclasses
        return results
