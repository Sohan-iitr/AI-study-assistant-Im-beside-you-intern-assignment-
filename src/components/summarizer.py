# src/components/summarizer.py

from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .pdf_processor import extract_text_from_pdf  # Import our PDF extractor
import torch

# --- This is the local T5 summarization function from before ---
def _summarize_chunk(text: str, model, tokenizer) -> str:
    """Helper function to summarize a single chunk of text."""
    input_text = "summarize: " + text
    inputs = tokenizer.encode(
        input_text, 
        return_tensors="pt",
        max_length=512,  # T5's max input
        truncation=True
    )
    
    summary_ids = model.generate(
        inputs, 
        max_length=150,
        min_length=30,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- This is our new "smart" function that does the MapReduce ---
def summarize_pdf_locally(pdf_path: str, model_name: str = "t5-small") -> str:
    """
    Extracts text from a PDF, chunks it, summarizes each chunk,
    and then summarizes the combined summaries. (MapReduce)
    """
    try:
        # 1. Load the T5 model and tokenizer (locally)
        print(f"Loading local model: {model_name}...")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        print("Model loaded.")

        # 2. Extract text from the PDF
        print(f"Extracting text from {pdf_path}...")
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text:
            return "Error: Could not extract text from PDF."
        print(f"Extracted {len(full_text)} characters.")

        # 3. Create a text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Split the text into 1000-char chunks
            chunk_overlap=100 # Add 100-char overlap
        )
        chunks = text_splitter.split_text(full_text)
        print(f"Split text into {len(chunks)} chunks.")

        # 4. MAP step: Summarize each chunk
        chunk_summaries = []
        print("Summarizing chunks (Map step)...")
        for i, chunk in enumerate(chunks):
            summary = _summarize_chunk(chunk, model, tokenizer)
            chunk_summaries.append(summary)
            print(f"  Summarized chunk {i+1}/{len(chunks)}")
        
        # 5. REDUCE step: Combine and do a final summary
        print("Combining summaries (Reduce step)...")
        combined_summaries = "\n".join(chunk_summaries)
        
        final_summary = _summarize_chunk(combined_summaries, model, tokenizer)
        
        print("Local summarization complete.")
        return final_summary

    except Exception as e:
        print(f"An error occurred during local summarization: {e}")
        return f"Error processing PDF: {e}"