### Name: Sohan Awate
### College: IIT Roorkee
### Branch: Production and Industrial Engineering


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
| Retrieval / RAG Tool | Tavily API (via LangChain) |
| Model | Fine-tuned **T5-small** (≈60M parameters) |
| Libraries | Transformers, Datasets, PEFT, PyMuPDF, Rouge-score |
| Environment | Python 3.10+, VS Code / Google Colab |
| Version Control | Git + GitHub |

---

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


📊 **Architecture & Training Resources:**  
All model files, training notebooks, and reports are available here:  
🔗 [Google Drive Folder — Models, Training Code & Documents](https://drive.google.com/drive/folders/16eeG99XW7YczgfSDXJTOSVWlcdco_DFB?usp=sharing)

---

## 🎯 Key Features
- 📝 **Automated Summarization:** Extracts and condenses lengthy academic PDFs.  
- 🔍 **Web Verification (Optional):** Uses Tavily API for real-time fact updates.  
- ❓ **Question Generation:** Creates study/viva-style questions via Groq API.  
- ⚙️ **Agentic Architecture:** Modular LangChain workflow with defined tools.  
- 💻 **Interactive UI:** Streamlit-based interface for smooth user experience.  

---

## 🚀 How to Run the Project

```bash
# 1️⃣ Clone the repository
git clone https://github.com/Sohan-iitr/AI-study-assistant-Im-beside-you-intern-assignment-.git
cd AI-study-assistant

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the Streamlit application
streamlit run src/app.py

