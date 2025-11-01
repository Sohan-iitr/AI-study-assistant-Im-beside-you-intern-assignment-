# # src/components/summarizer.py

# import os
# from dotenv import load_dotenv

# # ✅ Correct import for LangChain >= 1.0
# from langchain_classic.chains.summarize import load_summarize_chain

# from langchain_core.documents import Document
# from langchain_groq import ChatGroq
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from .pdf_processor import extract_text_from_pdf

# # Load the API key
# load_dotenv()

# def summarize_pdf_locally(pdf_path: str) -> str:
#     """
#     This function uses an API-based MapReduce strategy for speed.
#     """
#     try:
#         # 1. Initialize the LLM
#         llm = ChatGroq(
#             model="llama-3.1-8b-instant",
#             temperature=0.3,
#             api_key=os.getenv("GROQ_API_KEY")
#         )

#         # 2. Extract text from PDF
#         print(f"Extracting text from {pdf_path}...")
#         full_text = extract_text_from_pdf(pdf_path)
#         if not full_text:
#             return "Error: Could not extract text from PDF."
#         print(f"Extracted {len(full_text)} characters.")

#         # 3. Split the text
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=3000,
#             chunk_overlap=200
#         )
#         docs = text_splitter.create_documents([full_text])
#         print(f"Split text into {len(docs)} chunks.")

#         # 4. Load and run the MapReduce summarization chain
#         print("Running API-based MapReduce summarization chain...")
#         chain = load_summarize_chain(llm, chain_type="map_reduce")

#         # ✅ In LangChain v1.x, use `chain.invoke()` for structured results
#         result = chain.invoke({"input_documents": docs})

#         final_summary = result["output_text"]
#         print("Summarization complete.")
#         return final_summary

#     except Exception as e:
#         print(f"An error occurred during summarization: {e}")
#         return f"Error processing PDF: {e}"

# if __name__ == "__main__":
#     print("This file cannot be run directly anymore.")
#     print("Please run 'streamlit run src/app.py' to test.")



# src/components/summarizer.py
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .pdf_processor import extract_text_from_pdf

load_dotenv()

def summarize_pdf_locally(pdf_path: str) -> str:
    """
    Summarize a PDF file using Groq's LLM (Map-Reduce style manually).
    """
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY")
        )

        print(f"Extracting text from {pdf_path}...")
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text:
            return "Error: Could not extract text from PDF."

        print(f"Extracted {len(full_text)} characters.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=300
        )
        docs = text_splitter.create_documents([full_text])
        print(f"Split text into {len(docs)} chunks.")

        # --- Map step ---
        map_prompt = PromptTemplate.from_template(
            "Summarize the following text concisely:\n\n{text}\n\nSummary:"
        )

        summaries = []
        for i, doc in enumerate(docs):
            print(f"Summarizing chunk {i+1}/{len(docs)}...")
            response = llm.invoke(map_prompt.format(text=doc.page_content))
            summaries.append(response.content.strip())

        # --- Reduce step ---
        combined_text = "\n".join(summaries)
        reduce_prompt = PromptTemplate.from_template(
            "Combine and refine these summaries into a cohesive final summary:\n\n{chunks}\n\nFinal Summary:"
        )
        final_response = llm.invoke(reduce_prompt.format(chunks=combined_text))

        print("Summarization complete.")
        return final_response.content.strip()

    except Exception as e:
        print(f"Error during summarization: {e}")
        return f"Error processing PDF: {e}"
