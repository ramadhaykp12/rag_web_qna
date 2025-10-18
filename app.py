import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key dari .env
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Inisialisasi halaman Streamlit
st.set_page_config(page_title="RAG PDF Q&A with Gemini", layout="wide")
st.title("Chat with your Document")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing the document"):
        # Simpan PDF ke sementara
        temp_pdf = "temp.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.read())

        # Load dan pecah PDF
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = splitter.split_documents(docs)

        # Buat vectorstore FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Inisialisasi Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.2,
        )

        # Buat QA Chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("Document Procssing success")

        # Kolom untuk menanyakan sesuatu
        st.subheader("Ask anything about your uploaded document")
        user_query = st.text_input("Insert the question:")

        if st.button("Submit"):
            if user_query.strip() == "":
                st.warning("Please insert the question.")
            else:
                with st.spinner("Processing the answer..."):
                    response = qa_chain.invoke({"query": user_query})
                    st.markdown("Answer:")
                    st.write(response["result"])
