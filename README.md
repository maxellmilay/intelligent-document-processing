# Prompt Dynamism

A simple yet powerful wrapper for interacting with OpenAI's language models, providing an easy-to-use interface for both chat completions and text completions.

## Features

- Simple API for OpenAI language models
- Support for both chat completions and text completions
- Configurable parameters (temperature, max_tokens, etc.)
- Environment-based API key management

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prompt-dynamism.git
cd prompt-dynamism

# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key in .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Requirements

- Python 3.7+
- OpenAI Python client
- python-dotenv

## Usage

### Basic Usage

```python
from llm import LLM

# Initialize with default settings
llm = LLM()

# Chat completion
response = llm.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
])
print(response)

# Text completion
response = llm.complete("The capital of France is")
print(response)
```

### Advanced Configuration

```python
# Configure client with custom parameters
llm = LLM(
    model="gpt-4",         # Use a more powerful model
    temperature=0.9,       # Higher creativity
    max_tokens=500         # Longer responses
)

# Override parameters for a specific request
response = llm.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about AI."}
    ],
    temperature=0.2,       # More focused response
    max_tokens=100         # Shorter response
)
```

## API Reference

### LLM Class

The main class for interacting with language models:

```python
LLM(provider="openai", model="gpt-3.5-turbo", **config_options)
```

### Methods

- `chat(messages, model=None, **kwargs)`: Generate responses using chat completion models
- `complete(prompt, model=None, **kwargs)`: Generate responses using text completion models

## License

[Your License Here]