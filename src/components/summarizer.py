import os
from transformers import T5ForConditionalGeneration, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .pdf_processor import extract_text_from_pdf
import torch

# ------------------------------
# ✅ Correct model path setup
# ------------------------------
# Use model folder under "models", not "model"
MODEL_DIR = os.path.join("model")
# Get absolute path safely (no spaces, normalized)
MODEL_PATH = os.path.abspath(MODEL_DIR)
print("✅ Local model path:", MODEL_PATH)
# ------------------------------

def _summarize_chunk(text: str, model, tokenizer) -> str:
    """Helper function to summarize a single chunk of text."""
    input_text = "summarize: " + text

    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
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


def summarize_pdf_locally(pdf_path: str) -> str:
    """
    Extracts text from a PDF, chunks it, and uses your LOCAL
    fine-tuned T5 model to summarize it chunk by chunk (MapReduce style).
    """
    try:
        print(f"📂 Loading local fine-tuned model from: {MODEL_PATH}")

        # ✅ Check if model path exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model path not found: {MODEL_PATH}")

        # ✅ Load tokenizer and model correctly from local directory
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)
        print("✅ Model loaded successfully.")

        # Extract text from the PDF
        print(f"📘 Extracting text from: {pdf_path}")
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text:
            return "⚠️ Error: Could not extract text from PDF."
        print(f"📄 Extracted {len(full_text)} characters from PDF.")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(full_text)
        print(f"✂️ Split text into {len(chunks)} chunks.")

        # Summarize chunks
        chunk_summaries = []
        print("🧠 Summarizing chunks (Map step)...")
        for i, chunk in enumerate(chunks):
            summary = _summarize_chunk(chunk, model, tokenizer)
            chunk_summaries.append(summary)
            print(f"   ✅ Summarized chunk {i+1}/{len(chunks)}")

        # Combine summaries and generate final summary
        print("🧩 Combining summaries (Reduce step)...")
        combined_summaries = "\n".join(chunk_summaries)
        final_summary = _summarize_chunk(combined_summaries, model, tokenizer)

        print("🏁 Local summarization complete.")
        return final_summary

    except Exception as e:
        print(f"❌ An error occurred during local summarization: {e}")
        return f"Error: {e}"


if __name__ == '__main__':
    print("Run this via Streamlit, not directly. Example:")
    print("   streamlit run src/app.py")


# # src/components/summarizer.py
# import os
# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from .pdf_processor import extract_text_from_pdf

# load_dotenv()

# def summarize_pdf_locally(pdf_path: str) -> str:
#     """
#     Summarize a PDF file using Groq's LLM (Map-Reduce style manually).
#     """
#     try:
#         llm = ChatGroq(
#             model="llama-3.1-8b-instant",
#             temperature=0.3,
#             api_key=os.getenv("GROQ_API_KEY")
#         )

#         print(f"Extracting text from {pdf_path}...")
#         full_text = extract_text_from_pdf(pdf_path)
#         if not full_text:
#             return "Error: Could not extract text from PDF."

#         print(f"Extracted {len(full_text)} characters.")

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=4000,
#             chunk_overlap=300
#         )
#         docs = text_splitter.create_documents([full_text])
#         print(f"Split text into {len(docs)} chunks.")

#         # --- Map step ---
#         map_prompt = PromptTemplate.from_template(
#             "Summarize the following text concisely:\n\n{text}\n\nSummary:"
#         )

#         summaries = []
#         for i, doc in enumerate(docs):
#             print(f"Summarizing chunk {i+1}/{len(docs)}...")
#             response = llm.invoke(map_prompt.format(text=doc.page_content))
#             summaries.append(response.content.strip())

#         # --- Reduce step ---
#         combined_text = "\n".join(summaries)
#         reduce_prompt = PromptTemplate.from_template(
#             "Combine and refine these summaries into a cohesive final summary:\n\n{chunks}\n\nFinal Summary:"
#         )
#         final_response = llm.invoke(reduce_prompt.format(chunks=combined_text))

#         print("Summarization complete.")
#         return final_response.content.strip()

#     except Exception as e:
#         print(f"Error during summarization: {e}")
#         return f"Error processing PDF: {e}"
