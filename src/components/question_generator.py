# src/components/question_generator.py

import os
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

def generate_questions(summary_text: str) -> List[str]:
    """
    Generates study questions based on a given summary text using the Groq API.
    """
    load_dotenv() 

    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in .env file.")
        return []

    try:
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful study assistant. Based on the following text summary, "
            "generate 5 distinct and insightful questions that would be useful for a student "
            "to test their understanding. Each question should be on a new line, starting with a number.\n\n"
            "Summary:\n{summary}\n\n"
            "Questions:"
        )

        # CORRECTED LINE: Use the new, supported model name
        model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)

        chain = prompt | model | StrOutputParser()

        print("Generating questions with Groq...")
        question_string = chain.invoke({"summary": summary_text})
        
        questions = [q.strip() for q in question_string.split("\n") if q.strip()]
        print("Questions generated.")
        
        return questions

    except Exception as e:
        print(f"An error occurred during question generation: {e}")
        return []

# --- The test block remains the same ---
if __name__ == '__main__':
    sample_summary = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "contrasting with the natural intelligence of humans and animals. The field "
        "studies intelligent agents, which are systems that perceive their environment "
        "and act to maximize success. Originally, AI was about mimicking human cognitive "
        "skills like learning and problem-solving, but major researchers now define it "
        "more broadly in terms of rationality and acting rationally."
    )

    print("--- Testing Question Generator with Groq ---")
    print(f"\nInput Summary:\n{sample_summary}\n")

    generated_questions = generate_questions(sample_summary)

    if generated_questions:
        print("✅ Generated Questions:")
        for i, question in enumerate(generated_questions, 1):
            print(f"{question}")
    else:
        print("❌ Question generation failed.")