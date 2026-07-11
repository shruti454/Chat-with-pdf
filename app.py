import streamlit as st
from dotenv import load_dotenv
from rag import build_rag_pipeline

load_dotenv()

st.set_page_config(
    page_title="PDF ChatBot",
    page_icon="📚"
)

st.title("📚 Chat with your PDFs")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    if st.button("Process PDFs"):

        with st.spinner("Processing..."):

            chain = build_rag_pipeline(uploaded_files)

            st.session_state.chain = chain

        st.success("PDFs processed successfully!")



if "messages" not in st.session_state:
    st.session_state.messages = []
    
question = st.chat_input("Ask your question",key = "Chat_input")

if question:

    if "chain" not in st.session_state:
        st.warning("Please upload PDFs first.")

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        response = st.session_state.chain.invoke(
            {
                "input": question
            }
        )

        answer = response["answer"]

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])      
        