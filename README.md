# Chat-with-pdf
This project is a PDF Chatbot built using Python, Streamlit, LangChain, FAISS, HuggingFace Embeddings, and the Groq Llama 3.3 LLM. It uses the Retrieval-Augmented Generation (RAG) technique to answer questions based on uploaded PDF documents.

The application extracts text from PDFs, splits it into smaller chunks, converts the chunks into vector embeddings, and stores them in a FAISS vector database. When a user asks a question, the system retrieves the most relevant information from the documents and provides it as context to the LLM, which generates an accurate, context-based response. This approach improves answer accuracy and minimizes hallucinations.

Key Features:

Upload multiple PDF files
Ask questions in natural language
Fast document retrieval using FAISS
Context-aware answers using RAG
Interactive chat interface with Streamlit
