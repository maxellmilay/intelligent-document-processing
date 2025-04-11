import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class LLM:
    """
    A client for interacting with language models via the OpenAI API.
    
    This class provides a simplified interface for making requests to OpenAI's
    various language models, supporting both chat completions and text completions.
    
    Attributes:
        api_key (str): The OpenAI API key from environment variables
        client (OpenAI): The OpenAI client instance
        provider (str): The API provider (currently only supports "openai")
        model (str): The default model to use for completions
        default_config (dict): Default configuration parameters for API calls
    """
    
    def __init__(self, provider="openai", model="gpt-3.5-turbo", **config_options):
        """
        Initialize the LLM client.
        
        Args:
            provider (str, optional): The API provider. Defaults to "openai".
            model (str, optional): The default model to use. Defaults to "gpt-3.5-turbo".
            **config_options: Additional configuration options for API calls:
                - temperature: Controls randomness (0-2). Higher is more random.
                - max_tokens: Maximum number of tokens to generate.
                - top_p: Controls diversity via nucleus sampling.
                - frequency_penalty: Reduces repetition of token sequences.
                - presence_penalty: Reduces repetition of topics.
                - stop: Sequences where the API will stop generating tokens.
                - user: A unique identifier for the end user.
        
        Note:
            Requires OPENAI_API_KEY to be set in environment variables or .env file.
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.provider = provider
        self.model = model
        
        # Default configuration with option to override through constructor
        self.default_config = {
            "temperature": config_options.get("temperature", 0.7),
            "max_tokens": config_options.get("max_tokens", 256),
            "top_p": config_options.get("top_p", 1.0),
            "frequency_penalty": config_options.get("frequency_penalty", 0.0),
            "presence_penalty": config_options.get("presence_penalty", 0.0),
            "stop": config_options.get("stop", None),
            "user": config_options.get("user", None)
        }

    def generate(self, payload):
        """
        Generate completions based on provided payload.
        
        This is the core method used by both chat() and complete() methods.
        
        Args:
            payload (dict): Configuration for the API request, including:
                - messages: List of message objects for chat completions
                - prompt: String prompt for text completions
                - model: Model to use (overrides instance default)
                - Any other configuration parameters
                
        Returns:
            str: The generated text response
            
        Raises:
            ValueError: If neither messages nor prompt is provided
        """
        # Merge defaults with any overrides
        config = {**self.default_config, **{k: v for k, v in payload.items() if k in self.default_config and k not in ['model', 'messages', 'prompt']}}

        messages = payload.get("messages")
        prompt = payload.get("prompt")
        model = payload.get("model", self.model)

        if messages is None and prompt is None:
            raise ValueError("You must provide either 'messages' (for chat models) or 'prompt' (for completion models).")

        if model.startswith("gpt-"):
            if not messages and prompt:
                messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **config
            )
            return response.choices[0].message.content

        else:
            if not prompt:
                raise ValueError("Prompt is required for non-chat models.")
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                **config
            )
            return response.choices[0].text
            
    def chat(self, messages, model=None, **kwargs):
        """
        Generate a response using chat completion models.
        
        Args:
            messages (list): A list of message objects with 'role' and 'content'.
                Example: [{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": "Hello!"}]
            model (str, optional): The model to use. Defaults to the instance's model.
            **kwargs: Additional parameters to override default configuration.
                
        Returns:
            str: The generated response text
            
        Example:
            llm = LLM()
            response = llm.chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather like today?"}
            ])
        """
        payload = {
            "messages": messages,
            "model": model or self.model,
            **kwargs
        }
        return self.generate(payload)
        
    def complete(self, prompt, model=None, **kwargs):
        """
        Generate a response using text completion models.
        
        Args:
            prompt (str): The text prompt to complete.
            model (str, optional): The model to use. Defaults to the instance's model.
            **kwargs: Additional parameters to override default configuration.
                
        Returns:
            str: The generated completion text
            
        Example:
            llm = LLM()
            response = llm.complete("Once upon a time")
        """
        payload = {
            "prompt": prompt,
            "model": model or self.model,
            **kwargs
        }
        return self.generate(payload)
