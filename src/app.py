# src/app.py

import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# --- Add Project Root to sys.path ---
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
# -----------------------------------

# --- We import the tools as regular Python functions ---
from src.agent.tools import (
    summarize_pdf_tool,
    tavily_web_search_tool,
    question_generator_tool,
)

# --- Load Environment Variables ---
load_dotenv(project_root / ".env")


def main():
    """
    The main function that runs the Streamlit UI.
    """
    st.set_page_config(page_title="AI Study Assistant", page_icon="🤖")
    st.title("AI Study Assistant 🤖")
    st.markdown("Upload a PDF and I'll help you summarize it, find recent updates, and generate study questions.")

    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    
    use_tavily_toggle = st.toggle(
        "Cross-check with web search (uses Tavily)?", 
        value=True
    )
    
    st.text_input(
        "Your goal:", 
        "Summarize the document and generate study questions.",
        disabled=True
    )

    if st.button("Run Agent"):
        if uploaded_file is not None:
            temp_dir = Path("./temp_files")
            temp_dir.mkdir(exist_ok=True)
            temp_file_path = temp_dir / uploaded_file.name
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"PDF saved temporarily to: {temp_file_path}")

            try:
                # --- STEP 1: DIRECTLY CALL THE SUMMARY TOOL ---
                with st.spinner("Summarizing... (This may take a moment)"):
                    # --- CHANGE 1: Call the .func attribute ---
                    summary_text = summarize_pdf_tool.func(str(temp_file_path))
                
                if not summary_text or summary_text.startswith("Error:"):
                    st.error(f"Failed to generate summary: {summary_text}")
                    raise Exception("Summary generation failed.")
                
                st.success("Summary Complete!")
                st.markdown("### Summary:")
                st.markdown(summary_text)

                # --- STEP 2: (OPTIONAL) DIRECTLY CALL TAVILY ---
                search_results = ""
                if use_tavily_toggle:
                    with st.spinner("Searching for recent updates..."):
                        # --- CHANGE 2: Call the .func attribute ---
                        search_results = tavily_web_search_tool.func(summary_text)
                    st.success("Web search complete!")
                    st.markdown("### Related Web Info:")
                    st.markdown(search_results)

                # --- STEP 3: DIRECTLY CALL THE QUESTION TOOL ---
                with st.spinner("Generating questions..."):
                    # --- CHANGE 3: Call the .func attribute ---
                    question_string = question_generator_tool.func(summary_text)
                
                st.success("Questions Complete!")
                st.markdown("### Study Questions:")
                st.markdown(question_string)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # This cleanup step remains the same
                os.remove(temp_file_path)
                st.info(f"Temporary file {temp_file_path.name} removed.")

        elif not uploaded_file:
            st.warning("Please upload a PDF file first.")

if __name__ == "__main__":
    main()