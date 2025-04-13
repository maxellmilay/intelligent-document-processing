from typing import Dict, Any, Optional, Union, List
import logging

try:
    import spacy
except ImportError:
    logging.warning("NER dependencies not installed. Run 'pip install spacy'")
    logging.warning("You may also need to download a language model: python -m spacy download en_core_web_sm")

from lib.tools.extractors.base import Extractor


class NERExtractor(Extractor):
    """
    Extractor that uses Named Entity Recognition to extract structured information
    from text content.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the NER extractor.
        
        Args:
            config: Configuration including:
                - spacy_model: Name of spaCy model to use (default: 'en_core_web_sm')
                - custom_entities: Dictionary mapping custom entity types to patterns
                - confidence_threshold: Minimum confidence score for extracted entities
                - extraction_engine: Optional refinement engine (like an LLM)
                - engine_config: Configuration for the extraction engine
        """
        super().__init__(config)
        
        # Set up NER-specific configuration
        self.spacy_model = self.config.get('spacy_model', 'en_core_web_sm')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.nlp = None
        
        # Initialize spaCy NLP pipeline
        try:
            self.nlp = spacy.load(self.spacy_model)
            
            # Add custom entity rules if provided
            if 'custom_entities' in self.config:
                self._add_custom_entity_rules(self.config['custom_entities'])
                
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {str(e)}")
            
    def _initialize_extraction_engine(self):
        """
        Initialize the extraction engine based on configuration.
        For NER, this would typically be a refinement engine like an LLM.
        """
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
        elif engine_type == 'transformers':
            try:
                # Example of initializing a Hugging Face transformers model
                from transformers import pipeline
                model_name = engine_config.get('model_name', 'distilbert-base-uncased')
                task = engine_config.get('task', 'ner')
                
                self.extraction_engine = pipeline(task, model=model_name)
                self.logger.info(f"Initialized transformers model: {model_name}")
            except ImportError:
                self.logger.error("Failed to initialize transformers model: transformers package not installed")
            
    def extraction_logic(self, content: Union[str, bytes], content_type: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract named entities from the provided content.
        
        Args:
            content: The text content to extract entities from
            content_type: Type of the content (should be 'text')
            metadata: Additional metadata about the content
            
        Returns:
            Dictionary containing extracted entities grouped by type
        """
        if not self.nlp:
            self.logger.error("NER model not initialized")
            return {}
            
        # Ensure we have text to process
        if not isinstance(content, str):
            try:
                content = content.decode('utf-8')
            except (AttributeError, UnicodeDecodeError):
                self.logger.error("Content must be text for NER extraction")
                return {}
                
        # Process text with spaCy NER
        doc = self.nlp(content)
        
        # Extract entities and group by type
        entity_results = self._extract_entities(doc)
        
        # Use extraction engine to refine results if available
        if self.extraction_engine:
            refined_results = self._refine_with_engine(entity_results, content, metadata)
            return refined_results
            
        return entity_results
    
    def _extract_entities(self, doc) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities from a processed spaCy document.
        
        Args:
            doc: Processed spaCy document
            
        Returns:
            Dictionary of entities grouped by type
        """
        entities = {}
        
        for ent in doc.ents:
            # Skip low-confidence entities
            if hasattr(ent, 'score') and ent.score < self.confidence_threshold:
                continue
                
            entity_type = ent.label_
            entity_info = {
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'score', 1.0)
            }
            
            # Group by entity type
            if entity_type not in entities:
                entities[entity_type] = []
                
            entities[entity_type].append(entity_info)
            
        # Extract specific insurance claim fields based on entity types and patterns
        claim_fields = self._extract_claim_fields(doc, entities)
        
        # Merge specific fields with general entities
        results = {
            'entities': entities,
            'fields': claim_fields
        }
        
        return results
    
    def _extract_claim_fields(self, doc, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Extract specific insurance claim fields from entities and document.
        
        Args:
            doc: Processed spaCy document
            entities: Already extracted entities
            
        Returns:
            Dictionary of specific claim fields
        """
        claim_fields = {}
        
        # Extract policy number (looking for patterns like "Policy #: 12345678")
        policy_matches = [sent for sent in doc.sents if "policy" in sent.text.lower()]
        for sent in policy_matches:
            for token in sent:
                if token.like_num and len(token.text) >= 5:
                    claim_fields['policy_number'] = token.text
                    break
        
        # Extract claim number
        claim_matches = [sent for sent in doc.sents if "claim" in sent.text.lower()]
        for sent in claim_matches:
            for token in sent:
                if token.like_num and len(token.text) >= 5:
                    claim_fields['claim_number'] = token.text
                    break
        
        # Extract dates (use DATE entities)
        if 'DATE' in entities:
            date_entities = entities['DATE']
            # Look for date of loss specifically
            for date in date_entities:
                date_context = doc.text[max(0, date['start']-20):min(len(doc.text), date['end']+20)]
                if "loss" in date_context.lower() or "incident" in date_context.lower():
                    claim_fields['date_of_loss'] = date['text']
                    break
            
            # If we didn't find a specific date of loss, use the first date
            if 'date_of_loss' not in claim_fields and date_entities:
                claim_fields['date_of_loss'] = date_entities[0]['text']
        
        # Extract money amounts (use MONEY entities)
        if 'MONEY' in entities:
            money_entities = entities['MONEY']
            for money in money_entities:
                money_context = doc.text[max(0, money['start']-30):min(len(doc.text), money['end']+30)]
                if "damage" in money_context.lower() or "estimated" in money_context.lower() or "amount" in money_context.lower():
                    claim_fields['estimated_amount'] = money['text']
                    break
        
        # Extract person names (use PERSON entities for claimant)
        if 'PERSON' in entities:
            person_entities = entities['PERSON']
            # Look for claimant specifically
            for person in person_entities:
                person_context = doc.text[max(0, person['start']-20):min(len(doc.text), person['end']+20)]
                if "claimant" in person_context.lower() or "insured" in person_context.lower():
                    claim_fields['claimant_name'] = person['text']
                    break
            
            # If we didn't find a specific claimant, use the first person
            if 'claimant_name' not in claim_fields and person_entities:
                claim_fields['claimant_name'] = person_entities[0]['text']
        
        return claim_fields
    
    def _add_custom_entity_rules(self, custom_entities: Dict[str, List[Dict[str, Any]]]):
        """
        Add custom entity rules to the spaCy pipeline.
        
        Args:
            custom_entities: Dictionary mapping entity types to pattern lists
        """
        try:
            # Check if the EntityRuler already exists in the pipeline
            if 'entity_ruler' not in self.nlp.pipe_names:
                # Create the ruler and add it to the pipeline
                ruler = self.nlp.add_pipe('entity_ruler', name='entity_ruler')
            else:
                # Get the existing ruler
                ruler = self.nlp.get_pipe('entity_ruler')
            
            # Add the custom patterns
            patterns = []
            for entity_type, entity_patterns in custom_entities.items():
                for pattern in entity_patterns:
                    patterns.append({'label': entity_type, 'pattern': pattern})
            
            ruler.add_patterns(patterns)
            
        except Exception as e:
            self.logger.error(f"Failed to add custom entity rules: {str(e)}")
    
    def _refine_with_engine(self, entity_results: Dict[str, Any], original_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use the extraction engine to refine and extract additional fields from NER results.
        
        Args:
            entity_results: The results from NER extraction
            original_text: The original text content
            metadata: Additional context about the document
            
        Returns:
            Refined dictionary of extracted fields
        """
        if not self.extraction_engine:
            return entity_results
        
        # Construct prompt for engine
        prompt = self._construct_refinement_prompt(entity_results, original_text, metadata)
        
        try:
            # Different interfaces for different engine types
            if hasattr(self.extraction_engine, 'generate'):
                # LLM-style interface
                engine_response = self.extraction_engine.generate(prompt)
            elif hasattr(self.extraction_engine, 'extract'):
                # Generic extractor interface
                engine_response = self.extraction_engine.extract(prompt)
            elif hasattr(self.extraction_engine, 'process'):
                # NLP processor interface
                engine_response = self.extraction_engine.process(prompt)
            elif callable(self.extraction_engine):
                # Function-like interface (e.g., HuggingFace pipeline)
                engine_response = self.extraction_engine(prompt)
            else:
                self.logger.error("Extraction engine has no compatible interface")
                return entity_results
            
            # Parse the engine response
            if isinstance(engine_response, dict):
                # Some engines might return structured data directly
                refined_fields = engine_response
            else:
                # Otherwise parse the text response
                refined_fields = self._parse_engine_response(engine_response)
            
            # Merge with original NER results
            merged_results = entity_results.copy()
            merged_results['fields'].update(refined_fields)
            
            return merged_results
            
        except Exception as e:
            self.logger.error(f"Extraction engine refinement failed: {str(e)}")
            return entity_results
    
    def _construct_refinement_prompt(self, entity_results: Dict[str, Any], original_text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Construct a prompt for the extraction engine to refine NER results.
        
        Args:
            entity_results: The results from NER extraction
            original_text: The original text content
            metadata: Additional context about the document
            
        Returns:
            Prompt string for the extraction engine
        """
        # Extract current fields for context
        current_fields = entity_results.get('fields', {})
        fields_str = "\n".join([f"- {k}: {v}" for k, v in current_fields.items()])
        
        prompt = (
            "I've extracted the following fields from an insurance claim email using NER:\n"
            f"{fields_str}\n\n"
            "The original text is:\n"
            f"{original_text}\n\n"
            "Please extract or correct the following fields if they are present in the text:\n"
            "- Claim Number\n"
            "- Policy Number\n"
            "- Claimant Name\n"
            "- Date of Loss\n"
            "- Description of Loss\n"
            "- Estimated Amount\n\n"
            "Return only the extracted fields in JSON format."
        )
        
        # Add metadata context if available
        if metadata:
            metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            prompt += f"\n\nAdditional email metadata:\n{metadata_str}"
            
        return prompt
    
    def _parse_engine_response(self, engine_response: str) -> Dict[str, Any]:
        """
        Parse the extraction engine response into a structured dictionary.
        
        Args:
            engine_response: The response from the extraction engine
            
        Returns:
            Dictionary of extracted fields
        """
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