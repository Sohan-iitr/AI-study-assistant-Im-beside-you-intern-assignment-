# 🧠 AI Study Assistant — Automated Note Summarizer & Q&A Generator

## 📘 Overview
This project is an **AI-powered Study Assistant** designed to help students and educators summarize study material and generate test/viva-style questions automatically.  

Users can upload lecture notes or textbook PDFs, and the system will:
1. Extract and clean text from the file.  
2. Use a **Retrieval-Augmented Generation (RAG)** tool to check for updated information (e.g., new laws, data, or tech developments).  
3. Pass the enriched text to a **fine-tuned T5-small model** for summarization.  
4. Generate intelligent **viva/test questions** based on the content.  
5. Display the output through an easy-to-use **Streamlit interface**.

---

## ⚙️ Tech Stack

| Component | Technology Used |
|------------|-----------------|
| Frontend/UI | Streamlit |
| Agent Framework | LangChain |
| Retrieval / RAG Tool | DuckDuckGo / Tavily API (via LangChain) |
| Model | Fine-tuned **T5-small** (≈60M parameters) |
| Libraries | Transformers, Datasets, PEFT, PyMuPDF, Rouge-score |
| Environment | Python 3.10+, VS Code / Google Colab |
| Version Control | Git + GitHub |

---
link for finetuned model download https://drive.google.com/drive/folders/16eeG99XW7YczgfSDXJTOSVWlcdco_DFB?usp=sharing


## 🧩 System Architecture

User Upload (PDF)
↓
PDF Text Extractor (PyMuPDF)
↓
RAG Tool (Tavily search)
↓
Fine-tuned T5-small Model
↓
Q&A Generator
↓
Streamlit Display (Summary + Viva Questions)

youtube video link: 