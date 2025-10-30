# src/components/summarizer.py

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def summarize_text(text: str, model_name: str = "t5-small") -> str:
    """
    Generates a summary for a given text using a T5 model.

    Args:
        text (str): The input text to summarize.
        model_name (str): The name of the T5 model to use from Hugging Face Hub. 
                          Defaults to "t5-small".

    Returns:
        str: The generated summary.
    """
    try:
        # 1. Load the tokenizer and model from Hugging Face
        print(f"Loading model: {model_name}...")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        print("Model loaded successfully.")

        # 2. Pre-process the text: T5 requires a prefix for summarization
        # We also need to add a check for text length, as T5 has a token limit.
        # We'll truncate to a safe number of tokens if necessary.
        input_text = "summarize: " + text
        
        # Tokenize the input text
        inputs = tokenizer.encode(
            input_text, 
            return_tensors="pt",       # Return PyTorch tensors
            max_length=512,            # Truncate to a max of 512 tokens
            truncation=True
        )

        # 3. Generate the summary
        print("Generating summary...")
        summary_ids = model.generate(
            inputs, 
            max_length=150,      # Maximum length of the summary
            min_length=40,       # Minimum length of the summary
            length_penalty=2.0,  # Penalizes longer summaries to keep them concise
            num_beams=4,         # Use beam search for better quality results
            early_stopping=True
        )

        # 4. Decode the generated summary IDs back to a string
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print("Summary generated.")
        
        return summary

    except Exception as e:
        print(f"An error occurred during summarization: {e}")
        return ""

# --- Example of how to test this file directly ---
if __name__ == '__main__':
    # This block runs only when you execute `python summarizer.py`
    
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to the natural intelligence displayed by animals and humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of successfully achieving its goals. The term 
    "artificial intelligence" had previously been used to describe machines 
    that mimic and display "human" cognitive skills that are associated with 
    the human mind, such as "learning" and "problem-solving". This definition 
    has since been rejected by major AI researchers who now describe AI in 
    terms of rationality and acting rationally, which does not limit AI to 
    human-like intelligence.
    """
    
    print("--- Testing T5 Summarizer ---")
    print(f"\nOriginal Text:\n{sample_text}")
    
    generated_summary = summarize_text(sample_text)
    
    if generated_summary:
        print(f"\n✅ Generated Summary:\n{generated_summary}")
    else:
        print("\n❌ Summary generation failed.")