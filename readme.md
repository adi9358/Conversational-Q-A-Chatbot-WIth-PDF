# 🧠 Conversational Q&A Chatbot With PDF

A **Streamlit-based conversational AI assistant** that answers your questions from uploaded PDF files using **LangChain**, **FAISS**, **Groq’s Llama 3.1 model**, and **HuggingFace embeddings**.\
It retains chat history for context-aware responses.

---

## 🚀 Features

- 📄 Upload one or more PDFs
- 🧠 Ask context-aware questions about the PDF content
- 🗞 Chat history-aware reformulation using LangChain's history-aware retriever
- 💬 LLM-powered responses using [Groq's](https://groq.com/) `llama3-8b-instant`
- 📚 Document embeddings via HuggingFace (`all-MiniLM-L6-v2`)
- 🔍 Fast semantic retrieval via FAISS

---

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq API (LLaMA 3.1)](https://console.groq.com/)
- [Hugging Face Transformers](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)

---

## 📦 Installation

### 1. Clone the Repo

```bash
git clone https://github.com/adi9358/Conversational-Q-A-Chatbot-WIth-PDF.git
cd Conversational-Q-A-Chatbot-WIth-PDF
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root folder with the following:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Get your:

- Groq key from [https://console.groq.com/keys](https://console.groq.com/keys)
- HF token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## ▶️ Usage

```bash
streamlit run app.py
```

- Upload one or more PDF files
- Ask natural-language questions about the content
- View contextual and concise AI-generated answers
- Review the session’s chat history

---

## 📂 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── .env                    # API keys (not included in repo)
```

---

## ❗ Notes

- Make sure your PDFs contain selectable text (not scanned images).
- If you see SQLite-related errors, don't worry — this project uses **FAISS**, not SQLite.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## ✨ Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Groq LLaMA 3](https://groq.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)

