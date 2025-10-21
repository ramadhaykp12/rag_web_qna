# RAG QnA Web App

This repository contains a **Retrieval-Augmented Generation (RAG)** QnA web application built with **Python**, **LangChain**, **Streamlit**, and **Gemini Flash 2.5 API**.

The app allows you to **upload a PDF document** and **ask questions** about its content. It retrieves relevant document sections and generates accurate answers using a **Gemini LLM**.

The **embedding model** used in this project is an open-source model from **Hugging Face**.

---

## Features

* Upload PDF documents directly through the web interface
* Retrieve relevant content using embeddings
* Generate answers using Gemini Flash 2.5
* Built with LangChain and Streamlit for simplicity and performance

---

## Installation

Clone this repository:

```bash
git clone https://github.com/ramadhaykp12/rag-qna-webapp.git
cd rag-qna-webapp
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Before running the app, create a `.env` file in the project root directory and add your API keys:

```bash
GOOGLE_API_KEY="your_gemini_api_key"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"
```

You can obtain:

* **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
* **Hugging Face Token** from [Hugging Face Access Tokens](https://huggingface.co/settings/tokens)

---

## Running the App

Once everything is configured, start the Streamlit app:

```bash
streamlit run app.py
```

Then open the provided local URL in your browser (usually `http://localhost:8501`).

---

## Tech Stack

* **Python**
* **LangChain**
* **Gemini Flash 2.5 API**
* **Hugging Face Embeddings**
* **Streamlit**

---

## Future Improvements

* Support for multiple document uploads
* Add chat history memory
* Integrate vector database (FAISS or Chroma)
* Improved UI/UX design
