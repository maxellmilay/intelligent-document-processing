import json

class PromptProcessor:
    def __init__(self, llm_instance, prompts, interface=None):
        self.llm = llm_instance
        self.prompts = prompts
        self.interface = interface
        self.responses = {}
        
        # Process prompts to populate responses
        self._process_prompts()
        
        # Filter responses based on interface if provided
        if self.interface:
            self._filter_responses()

    def _process_prompts(self):
        for prompt_obj in self.prompts:
            field_name = prompt_obj["field_name"]
            prompt = prompt_obj["prompt"]

            # Get response using the LLM class
            response = self.llm.chat([{"role": "user", "content": prompt}])
            self.responses[field_name] = response

    def _filter_responses(self):
        """
        Filters responses to only include fields specified in the interface.
        """
        if not self.interface:
            return
            
        filtered_responses = {}
        for field in self.interface:
            if field in self.responses:
                filtered_responses[field] = self.responses[field]
                
        self.responses = filtered_responses

    def print_json(self):
        """
        Formats and prints the responses as properly indented JSON.
        """
        formatted_json = json.dumps(self.responses, indent=2)
        print(formatted_json)
