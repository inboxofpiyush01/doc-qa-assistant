import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Page & Session State Initialization ---
st.set_page_config(page_title="Smart Doc Q&A 2026", layout="wide")
st.title("📄 Smart Document Assistant")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. Sidebar Setup ---
with st.sidebar:
    st.header("Settings")
    hf_token = st.text_input("HuggingFace API Token", type="password", help="Requires 'Inference' permissions.")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 3. Core RAG Logic ---
if uploaded_file and hf_token:
    # A. PDF Processing (Cached for speed)
    @st.cache_resource
    def process_pdf(file):
        reader = PdfReader(file)
        raw_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        # Split text into manageable chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(raw_text)
        
        # Create Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    try:
        retriever = process_pdf(uploaded_file)

        # B. LLM & Chat Wrapper (Fixes the 'Conversational Task' Provider Error)
        llm_engine = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            huggingfacehub_api_token=hf_token,
            temperature=0.1,
            max_new_tokens=512,
        )
        
        # ChatHuggingFace formats the prompt into the 'chat' JSON the provider expects
        llm = ChatHuggingFace(llm=llm_engine)

        # C. RAG Chain Construction
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the question using ONLY the provided context: {context}"),
            ("human", "{question}")
        ])

        # Modern LCEL Pipeline
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # --- 4. Chat Interface ---
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User Input
        if user_query := st.chat_input("Ask about your document..."):
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document..."):
                    try:
                        # Invoke the chain
                        response = rag_chain.invoke(user_query)
                        st.markdown(response)
                        # Store in history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Generation Error: {e}")

    except Exception as e:
        st.error(f"Setup Error: {e}")

elif not hf_token:
    st.info("Please enter your HuggingFace Token in the sidebar to begin.")
else:
    st.info("Upload a PDF to start chatting.")