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

# --- Correct Imports ---
from langgraph.prebuilt import create_react_agent   # ✅ Works with current LangGraph version
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# --- Import Tools ---
from src.agent.tools import (
    summarize_pdf_tool,
    tavily_web_search_tool,
    question_generator_tool,
)
from src.components.summarizer import summarize_pdf_locally  # ✅ Added import for direct summarization

# --- Load Environment Variables ---
load_dotenv(project_root / ".env")


# --- Create Agent Executor Function ---
def create_agent_executor(use_tavily: bool = True):
    """Creates and returns a ReAct agent with the given tools."""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    tools = [summarize_pdf_tool]
    if use_tavily:
        tools.append(tavily_web_search_tool)
        print("--- Web Search Tool IS ENABLED ---")
    else:
        print("--- Web Search Tool IS DISABLED ---")

    agent_executor = create_react_agent(llm, tools)
    print("--- LangGraph ReAct Agent Created (Summary-Focused) ---")

    return agent_executor


# --- Streamlit App ---
def main():
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

            st.info(f"📂 PDF saved temporarily to: {temp_file_path}")

            try:
                # --- STEP 1: SUMMARIZE LOCALLY ---
                with st.spinner("Summarizing document... This may take a moment ⏳"):
                    summary_text = summarize_pdf_locally(str(temp_file_path))

                if not summary_text or summary_text.startswith("Error"):
                    st.error("❌ Failed to generate a proper summary.")
                    return

                st.success("✅ Summary Complete!")
                st.markdown("### 📘 Summary:")
                st.markdown(summary_text)

                # --- STEP 2: GENERATE QUESTIONS ---
                with st.spinner("Generating study questions..."):
                    question_string = question_generator_tool.func(summary_text)

                st.success("✅ Questions Ready!")
                st.markdown("### 🎯 Study Questions:")
                st.markdown(question_string)

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")
            finally:
                try:
                    os.remove(temp_file_path)
                    st.info(f"🧹 Temporary file {temp_file_path.name} removed.")
                except Exception:
                    pass

        else:
            st.warning("⚠️ Please upload a PDF file first.")


if __name__ == "__main__":
    main()
