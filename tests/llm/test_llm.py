from lib.llm.llm import LLM
from lib.llm.processor import PromptProcessor

from lib.prompts.insurance import insurance_prompts
from lib.interfaces.insurance import insurance_types
 
def main():
    # Initialize LLM instance with LangChain
    llm = LLM(temperature=0.7)

    # Process prompts and get responses using LangChain
    processor = PromptProcessor(llm, prompts=insurance_prompts, interface=insurance_types)
    
    # Print the responses as properly formatted JSON
    processor.print_json()

if __name__ == "__main__":
    main()
