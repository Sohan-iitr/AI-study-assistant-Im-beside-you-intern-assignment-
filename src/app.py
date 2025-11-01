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

# --- NEW, CORRECTED IMPORTS ---
# The agent code now lives in 'langgraph'
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# --- Import Your Custom Tools ---
# This part was always correct
from src.agent.tools import (
    pdf_text_extractor_tool,
    tavily_web_search_tool,
    t5_summarizer_tool,
    question_generator_tool,
)

# --- Load Environment Variables ---
load_dotenv(project_root / ".env")

def create_agent_executor(use_tavily: bool = True):
    """
    Creates and returns the LangChain Agent.
    """
    
    # 1. Initialize the LLM (the "brain")
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # 2. Dynamically build the tools list
    tools = [
        pdf_text_extractor_tool,
        t5_summarizer_tool,
        question_generator_tool
    ]
    
    if use_tavily:
        tools.append(tavily_web_search_tool)
        print("--- Web Search Tool IS ENABLED ---")
    else:
        print("--- Web Search Tool IS DISABLED ---")

    # 3. Create the ReAct Agent using the modern 'langgraph' function
    # This 'agent_executor' is the runnable object we need.
    agent_executor = create_react_agent(llm, tools)
    print("--- LangGraph ReAct Agent Created ---")
    
    return agent_executor

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
    
    user_prompt = st.text_input(
        "What should I do with this PDF?", 
        "Summarize the document and generate study questions."
    )

    if st.button("Run Agent"):
        if uploaded_file is not None and user_prompt:
            temp_dir = Path("./temp_files")
            temp_dir.mkdir(exist_ok=True)
            temp_file_path = temp_dir / uploaded_file.name
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"PDF saved temporarily to: {temp_file_path}")

            with st.spinner("Agent is thinking..."):
                try:
                    combined_input = (
                        f"User Task: '{user_prompt}'\n"
                        f"File Path: '{temp_file_path}'"
                    )
                    
                    # Create the agent executor
                    agent_executor = create_agent_executor(use_tavily=use_tavily_toggle)
                    
                    # --- UPDATED AGENT INVOCATION ---
                    # We pass the input as a list of messages
                    messages = [HumanMessage(content=combined_input)]
                    
                    # We stream the agent's response to get the final output
                    response_chunks = []
                    for chunk in agent_executor.stream({"messages": messages}):
                        if "messages" in chunk:
                            # This is the final answer
                            final_message = chunk["messages"][-1]
                            if final_message.role == "assistant":
                                response_chunks.append(final_message.content)
                    
                    final_response = "".join(response_chunks)
                    
                    st.success("Task Complete!")
                    st.markdown("### Agent's Final Answer:")
                    st.markdown(final_response)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    os.remove(temp_file_path)
                    st.info(f"Temporary file {temp_file_path.name} removed.")

        elif not uploaded_file:
            st.warning("Please upload a PDF file first.")
        else:
            st.warning("Please enter a prompt for the agent.")

if __name__ == "__main__":
    main()








# # src/app.py

# import os
# import sys
# import streamlit as st
# from pathlib import Path
# from dotenv import load_dotenv

# # --- Add Project Root to sys.path ---
# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))
# # -----------------------------------

# # --- LangChain Core Imports ---
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_core.prompts import PromptTemplate
# from langchain_hub import pull
# from langchain_groq import ChatGroq

# # --- Import Your Custom Tools ---
# from src.agent.tools import (
#     pdf_text_extractor_tool,
#     tavily_web_search_tool,
#     t5_summarizer_tool,
#     question_generator_tool,
# )

# # --- Load Environment Variables ---
# load_dotenv(project_root / ".env")

# # --- CHANGE 1: Modify the function to accept the toggle's value ---
# def create_agent_executor(use_tavily: bool = True):
#     """
#     Creates and returns the LangChain AgentExecutor.
    
#     Args:
#         use_tavily (bool): Whether to include the Tavily web search tool.
#     """
    
#     # 1. Initialize the LLM (the "brain")
#     llm = ChatGroq(
#         model="llama-3.1-8b-instant", 
#         temperature=0,
#         api_key=os.getenv("GROQ_API_KEY")
#     )

#     # 2. --- CHANGE 2: Dynamically build the tools list ---
#     # Start with the core tools that are always available
#     tools = [
#         pdf_text_extractor_tool,
#         t5_summarizer_tool,
#         question_generator_tool
#     ]
    
#     # Conditionally add the Tavily tool
#     if use_tavily:
#         tools.append(tavily_web_search_tool)
#         print("--- Web Search Tool IS ENABLED ---")
#     else:
#         print("--- Web Search Tool IS DISABLED ---")
#     # ----------------------------------------------------

#     # 3. Get the ReAct prompt template
#     prompt = pull("hwchase17/react")

#     # 4. Create the ReAct Agent
#     agent = create_react_agent(llm, tools, prompt)

#     # 5. Create the Agent Executor
#     agent_executor = AgentExecutor(
#         agent=agent, 
#         tools=tools, 
#         verbose=True,
#         handle_parsing_errors=True
#     )
    
#     return agent_executor

# def main():
#     """
#     The main function that runs the Streamlit UI.
#     """
#     st.set_page_config(page_title="AI Study Assistant", page_icon="🤖")
#     st.title("AI Study Assistant 🤖")
#     st.markdown("Upload a PDF and I'll help you summarize it, find recent updates, and generate study questions.")

#     # 1. File Uploader
#     uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    
#     # --- CHANGE 3: Add the Streamlit toggle switch ---
#     use_tavily_toggle = st.toggle(
#         "Cross-check with web search (uses Tavily)?", 
#         value=True  # Default to 'on'
#     )
#     # ------------------------------------------------
    
#     # 2. Text Input for the task
#     user_prompt = st.text_input(
#         "What should I do with this PDF?", 
#         "Summarize the document and generate study questions."
#     )

#     # 3. Run Button
#     if st.button("Run Agent"):
#         if uploaded_file is not None and user_prompt:
#             # --- File Handling ---
#             temp_dir = Path("./temp_files")
#             temp_dir.mkdir(exist_ok=True)
#             temp_file_path = temp_dir / uploaded_file.name
            
#             with open(temp_file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             st.info(f"PDF saved temporarily to: {temp_file_path}")

#             # --- Run the Agent ---
#             with st.spinner("Agent is thinking... (Check your terminal for detailed thoughts)"):
#                 try:
#                     combined_input = (
#                         f"User Task: '{user_prompt}'\n"
#                         f"File Path: '{temp_file_path}'"
#                     )
                    
#                     # --- CHANGE 4: Pass the toggle's value to the function ---
#                     agent_executor = create_agent_executor(use_tavily=use_tavily_toggle)
                    
#                     # Invoke the agent
#                     response = agent_executor.invoke({
#                         "input": combined_input
#                     })
                    
#                     st.success("Task Complete!")
#                     st.markdown("### Agent's Final Answer:")
#                     st.markdown(response["output"])
                    
#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")
#                 finally:
#                     # Clean up the temporary file
#                     os.remove(temp_file_path)
#                     st.info(f"Temporary file {temp_file_path.name} removed.")

#         elif not uploaded_file:
#             st.warning("Please upload a PDF file first.")
#         else:
#             st.warning("Please enter a prompt for the agent.")

# if __name__ == "__main__":
#     main()