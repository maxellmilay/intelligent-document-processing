from typing import Dict, Any, Optional, Union
import logging

try:
    import pytesseract
    from PIL import Image
    import io
except ImportError:
    logging.warning("OCR dependencies not installed. Run 'pip install pytesseract pillow'")

from lib.tools.extractors.base import Extractor


class OCRExtractor(Extractor):
    """
    Extractor that uses OCR to extract text from images and then extract structured 
    information from the OCR results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OCR extractor.
        
        Args:
            config: Configuration including:
                - tesseract_cmd: Path to tesseract executable
                - lang: Language for OCR (default: 'eng')
                - preprocess: Whether to preprocess images (default: True)
                - extraction_engine: Optional LLM or other text extraction engine for structured extraction
                - engine_config: Configuration for the extraction engine
        """
        super().__init__(config)
        
        # Set up OCR-specific configuration
        self.lang = self.config.get('lang', 'eng')
        self.preprocess = self.config.get('preprocess', True)
        
    def _initialize_extraction_engine(self):
        """
        Initialize the extraction engine based on configuration.
        For OCR, this could be an LLM or other text processing engine.
        """
        # Set up Tesseract configuration if available
        if 'tesseract_cmd' in self.config:
            try:
                pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
            except NameError:
                self.logger.error("pytesseract not installed")
        
        # Initialize the extraction engine (e.g., an LLM) if provided
        engine_type = self.config.get('extraction_engine_type')
        engine_config = self.config.get('engine_config', {})
        
        if engine_type == 'llm':
            try:
                # Attempt to load the specified LLM class
                llm_module = self.config.get('llm_module', 'lib.utils.llm_wrapper')
                llm_class = self.config.get('llm_class', 'LLMWrapper')
                
                module = __import__(llm_module, fromlist=[llm_class])
                LLMClass = getattr(module, llm_class)
                
                self.extraction_engine = LLMClass(**engine_config)
                self.logger.info(f"Initialized LLM extraction engine: {llm_class}")
            except (ImportError, AttributeError) as e:
                self.logger.error(f"Failed to initialize LLM extraction engine: {str(e)}")
    
    def extraction_logic(self, content: Union[str, bytes], content_type: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract information using OCR technology.
        
        Args:
            content: The content to extract from (image bytes for OCR)
            content_type: Type of the content ('image', 'pdf', etc.)
            metadata: Additional metadata about the content
            
        Returns:
            Dictionary containing extracted fields and values
        """
        extracted_text = ""
        
        # Use OCR for image content types
        if content_type.startswith('image/') or content_type == 'image':
            try:
                extracted_text = self._perform_ocr(content)
            except Exception as e:
                self.logger.error(f"OCR failed: {str(e)}")
                return {}
        # Pass through text content
        elif content_type == 'text':
            extracted_text = content if isinstance(content, str) else content.decode('utf-8', errors='ignore')
        else:
            self.logger.warning(f"Unsupported content type for OCR: {content_type}")
            return {}
        
        # If we have extracted text and an extraction engine, use it to extract structured information
        if extracted_text and self.extraction_engine:
            return self._extract_fields_with_engine(extracted_text, metadata)
        
        # If we just have text but no extraction engine, return the text as-is
        return {"extracted_text": extracted_text}
    
    def _perform_ocr(self, image_data: Union[str, bytes]) -> str:
        """
        Perform OCR on the given image data.
        
        Args:
            image_data: Either bytes of an image or a path to an image file
            
        Returns:
            Extracted text from the image
        """
        try:
            # Convert bytes to PIL Image if necessary
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = Image.open(image_data)
            
            # Preprocess image if configured
            if self.preprocess:
                image = self._preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text
        except Exception as e:
            self.logger.error(f"OCR processing error: {str(e)}")
            raise
    
    def _preprocess_image(self, image):
        """
        Preprocess the image to improve OCR results.
        Simple implementation - can be extended with more sophisticated techniques.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
            
        # Additional preprocessing can be added here
        # Example: thresholding, noise removal, deskewing, etc.
        
        return image
    
    def _extract_fields_with_engine(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use the extraction engine to extract structured fields from OCR text.
        
        Args:
            text: The OCR-extracted text
            metadata: Additional context about the document
            
        Returns:
            Dictionary of extracted fields and values
        """
        if not self.extraction_engine:
            return {"extracted_text": text}
        
        # Example prompt construction for extraction engine
        prompt = self._construct_extraction_prompt(text, metadata)
        
        try:
            # This can work with different types of extraction engines
            if hasattr(self.extraction_engine, 'generate'):
                # LLM-style interface
                engine_response = self.extraction_engine.generate(prompt)
            elif hasattr(self.extraction_engine, 'extract'):
                # Generic extractor interface
                engine_response = self.extraction_engine.extract(prompt)
            elif hasattr(self.extraction_engine, 'process'):
                # NLP processor interface
                engine_response = self.extraction_engine.process(prompt)
            else:
                self.logger.error("Extraction engine has no compatible interface")
                return {"extracted_text": text}
            
            # Parse engine response into structured data
            parsed_fields = self._parse_engine_response(engine_response)
            return parsed_fields
        except Exception as e:
            self.logger.error(f"Extraction engine failed: {str(e)}")
            return {"extracted_text": text, "error": str(e)}
    
    def _construct_extraction_prompt(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Construct a prompt for the extraction engine to extract structured information.
        
        Args:
            text: The OCR-extracted text
            metadata: Additional context about the document
            
        Returns:
            Prompt string for the extraction engine
        """
        # This is a simplified prompt - actual implementation would be more sophisticated
        prompt = (
            "Extract the following fields from this insurance claim email text:\n"
            "- Claim Number\n"
            "- Policy Number\n"
            "- Claimant Name\n"
            "- Date of Loss\n"
            "- Description of Loss\n"
            "- Estimated Amount\n\n"
            f"Email text:\n{text}\n\n"
            "Return the extracted fields in JSON format."
        )
        
        # Add metadata context if available
        if metadata:
            context = f"\n\nAdditional context: {metadata}"
            prompt += context
            
        return prompt
    
    def _parse_engine_response(self, engine_response: str) -> Dict[str, Any]:
        """
        Parse the extraction engine response into a structured dictionary.
        
        Args:
            engine_response: The response from the extraction engine
            
        Returns:
            Dictionary of extracted fields
        """
        # This is a simplified parser - actual implementation would be more robust
        # and would depend on the expected response format from the engine
        
        try:
            # If the engine returns JSON, we could parse it directly
            import json
            return json.loads(engine_response)
        except:
            # Fallback to a simple key-value extraction
            extracted = {}
            lines = engine_response.strip().split('\n')
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    extracted[key.strip()] = value.strip()
                    
            return extracted 