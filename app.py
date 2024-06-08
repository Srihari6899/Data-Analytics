import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Print API keys to ensure they are loaded correctly
print("GROQ_API_KEY:", groq_api_key)
print("GOOGLE_API_KEY:", google_api_key)

# Check if the API keys are retrieved correctly
if not groq_api_key:
    raise ValueError("GROQ API key not found. Please set the GROQ_API_KEY environment variable.")
if not google_api_key:
    raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("PDF Document Question And Answering")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
""")

# Initialize session state variables
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "loader" not in st.session_state:
    st.session_state.loader = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "text_splitter" not in st.session_state:
    st.session_state.text_splitter = None
if "final_documents" not in st.session_state:
    st.session_state.final_documents = None

def vector_embedding():
    if st.session_state.vectors is None:
        try:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            
            if not st.session_state.docs:
                st.error("No documents loaded. Please check the directory and files.")
                return
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
            
            if not st.session_state.final_documents:
                st.error("No final documents after splitting. Please check the splitting parameters.")
                return

            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings
        
        except Exception as e:
            st.error(f"An error occurred during vector embedding: {e}")
            logging.exception("Exception during vector embedding")

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    if st.session_state.vectors is None:
        st.warning("Please initialize the vector store by clicking 'Documents Embedding' button first.")
    else:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            logging.info(f"Response time: {time.process_time() - start}")
            st.write(response['answer'])

            # With a streamlit expander
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        except Exception as e:
            st.error(f"An error occurred during retrieval: {e}")
            logging.exception("Exception during retrieval")
