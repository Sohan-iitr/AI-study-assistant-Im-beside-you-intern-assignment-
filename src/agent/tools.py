# src/agent/tools.py

import os
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv

# Import your custom functions from the components
from src.components.pdf_processor import extract_text_from_pdf
from src.components.summarizer import summarize_text
from src.components.question_generator import generate_questions

# Load environment variables for the Tavily API
load_dotenv()

# --- Tool 1: PDF Extractor ---
@tool
def pdf_text_extractor_tool(pdf_path: str) -> str:
    """
    Use this tool to extract all text from a PDF document.
    The input must be a string representing the file path to the PDF.
    """
    return extract_text_from_pdf(pdf_path)

# --- Tool 2: Tavily Web Search ---
@tool
def tavily_web_search_tool(query: str) -> str:
    """
    Use this tool to search the web for recent information, updates,
    or to find context on a specific topic.
    The input must be a string containing the search query.
    """
    try:
        tavily_api_key = os.getenv("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")
        if not tavily_api_key:
            return "Error: TAVILY_API_KEY not found in environment variables."
        
        client = TavilyClient(api_key=tavily_api_key)
        # We ask for 3 search results
        response = client.search(query=query, max_results=3)
        
        # Format the results into a single string
        results_string = "\n".join([
            f"Source: {res['url']}\nContent: {res['content']}" 
            for res in response.results
        ])
        return results_string
    
    except Exception as e:
        return f"An error occurred during the web search: {e}"

# --- Tool 3: T5 Summarizer ---
@tool
def t5_summarizer_tool(text: str) -> str:
    """
    Use this tool to generate a summary of a long piece of text.
    The input must be the text you want to summarize.
    """
    # Note: We can specify the model to use, or just use the default "t5-small"
    return summarize_text(text, model_name="t5-small")

# --- Tool 4: Question Generator ---
@tool
def question_generator_tool(summary_text: str) -> str:
    """
    Use this tool to generate 5 study questions based on a summary.
    The input must be the summary text.
    The output will be a list of questions as a single string.
    """
    questions_list = generate_questions(summary_text)
    # Convert the list of questions into a single, newline-separated string
    return "\n".join(questions_list)
