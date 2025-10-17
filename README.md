# RAG QNA Web App 
This repository contains RAG QNA Web App using Python. Gemini Flash 2.5 API used as a LLM model with help of Langchain Library and Streamlit for building web interface. Th embedding model used in this project was an open source embedding model from HuggingFace. The web app used to answer your question based on the PDF document that you upload to the web app. 
## Library Installation
All required library used in this project written in requirements.txt. Run the command below to install all the library needed for this project
```
pip install -r requirements.txt
```
## Secret Key Configuration
To use Gemini API and the embedding model from HuggingFace, you need to configue the API key token in .env file. You can get the Gemini API key from goolge AI studio and for HuggingFace you can get it from Access token menu in HuggingFace Official website
```
GOOGLE_API_KEY="your token"
HUGGINGFACEHUB_API_TOKEN="your token"
```

