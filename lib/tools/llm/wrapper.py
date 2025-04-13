import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

class LLM:
    """
    A client for interacting with language models via LangChain.
    
    This class provides a simplified interface for making requests to language models,
    supporting both chat completions and text completions.
    
    Attributes:
        api_key (str): The OpenAI API key from environment variables
        provider (str): The API provider (currently only supports "openai")
        model (str): The default model to use for completions
        default_config (dict): Default configuration parameters for API calls
        llm (ChatOpenAI): The LangChain model instance
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
            "model_kwargs": {"user": config_options.get("user", None)} if config_options.get("user") else {}
        }
        
        # Initialize the LangChain model
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            **{k: v for k, v in self.default_config.items() if k != "user"}
        )

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
        # Extract parameters
        messages = payload.get("messages")
        prompt = payload.get("prompt")
        model = payload.get("model", self.model)
        
        # Create config dict for LangChain model
        config = {
            k: v for k, v in self.default_config.items() 
            if k in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'stop']
        }
        
        # Update config with any overrides from payload
        for k in config.keys():
            if k in payload:
                config[k] = payload[k]
        
        # Check if we need to create a new model instance with different parameters
        if model != self.model or any(payload.get(k) != self.default_config.get(k) for k in config.keys()):
            llm = ChatOpenAI(
                model=model,
                openai_api_key=self.api_key,
                **config
            )
        else:
            llm = self.llm

        if messages is None and prompt is None:
            raise ValueError("You must provide either 'messages' (for chat models) or 'prompt' (for completion models).")

        # Handle chat completions
        if messages is not None:
            # Convert the message format to LangChain's format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Invoke the model and return the content
            response = llm.invoke(langchain_messages)
            return response.content
        
        # Handle text completions (using chat model in the background)
        elif prompt is not None:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
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
