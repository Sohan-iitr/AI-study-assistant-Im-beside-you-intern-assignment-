import os
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv

# Import your NEW smart function and the other tools
from src.components.summarizer import summarize_pdf_locally
from src.components.question_generator import generate_questions

load_dotenv()

# --- NEW TOOL 1: Smart PDF Summarizer ---
@tool
def summarize_pdf_tool(pdf_path: str) -> str:
    """
    Use this tool to generate a summary of a PDF document.
    It handles extracting text, chunking, and summarizing all at once.
    The input MUST be a string representing the file path to the PDF.
    """
    # This function now does all the heavy lifting locally
    return summarize_pdf_locally(pdf_path)

# --- TOOL 2: Tavily Web Search (Unchanged) ---
@tool
def tavily_web_search_tool(query: str) -> str:
    """
    Use this tool to search the web for recent information, updates,
    or to find context on a specific topic.
    The input must be a string containing the search query.
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return "Error: TAVILY_API_KEY not found in environment variables."
        
        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(query=query, max_results=3)
        
        results_string = "\n".join([
            f"Source: {res['url']}\nContent: {res['content']}" 
            for res in response.results
        ])
        return results_string
    
    except Exception as e:
        return f"An error occurred during the web search: {e}"

# --- TOOL 3: Question Generator (Unchanged) ---
@tool
def question_generator_tool(summary_text: str) -> str:
    """
    Use this tool to generate 5 study questions based on a summary.
    The input must be the summary text.
    The output will be a list of questions as a single string.
    """
    questions_list = generate_questions(summary_text)
    return "\n".join(questions_list)