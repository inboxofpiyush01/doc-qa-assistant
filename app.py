# import streamlit as st
# import tempfile
# import os
# from document_loader import load_document, chunk_document
# from embeddings import build_vectorstore, load_vectorstore
# from rag_pipeline import build_qa_chain, ask

# # ── Page config ──────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Doc Q&A Assistant",
#     page_icon="📄",
#     layout="wide",
# )

# # ── Custom CSS ───────────────────────────────────────────────────
# st.markdown("""
# <style>
# .answer-box {
#     background: #f0f7ff;
#     border-left: 4px solid #2563eb;
#     padding: 16px 20px;
#     border-radius: 6px;
#     font-size: 16px;
#     line-height: 1.7;
# }
# .source-box {
#     background: #f8fafc;
#     border: 1px solid #e2e8f0;
#     border-radius: 6px;
#     padding: 10px 14px;
#     font-size: 13px;
#     color: #64748b;
#     margin-top: 6px;
# }
# </style>
# """, unsafe_allow_html=True)

# # ── Session state ─────────────────────────────────────────────────
# if "qa_chain" not in st.session_state:
#     st.session_state.qa_chain = None
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "doc_name" not in st.session_state:
#     st.session_state.doc_name = None

# # ── Sidebar ──────────────────────────────────────────────────────
# with st.sidebar:
#     st.title("📄 Doc Q&A Assistant")
#     st.markdown("**Upload a document, then ask anything about it.**")
#     st.divider()

#     uploaded_file = st.file_uploader(
#         "Upload PDF or TXT",
#         type=["pdf", "txt"],
#         help="Supports PDF and plain text files",
#     )

#     if uploaded_file:
#         if st.button("📥 Process Document", use_container_width=True):
#             with st.spinner("Reading and indexing document..."):
#                 # Save to temp file
#                 suffix = os.path.splitext(uploaded_file.name)[-1]
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                     tmp.write(uploaded_file.read())
#                     tmp_path = tmp.name

#                 try:
#                     text = load_document(tmp_path)
#                     chunks = chunk_document(text, source_name=uploaded_file.name)
#                     vs = build_vectorstore(chunks)
#                     st.session_state.qa_chain = build_qa_chain(vs)
#                     st.session_state.doc_name = uploaded_file.name
#                     st.session_state.chat_history = []
#                     st.success(f"✅ Processed {len(chunks)} chunks!")
#                 except Exception as e:
#                     st.error(f"Error: {e}")
#                 finally:
#                     os.unlink(tmp_path)

#     st.divider()

#     if st.button("🗑️ Clear Chat History", use_container_width=True):
#         st.session_state.chat_history = []

#     st.markdown("---")
#     st.markdown("""
# **How it works:**
# 1. Upload your document
# 2. It gets split into chunks
# 3. Chunks are embedded with a free model
# 4. Your question retrieves the top matches
# 5. An LLM reads those chunks and answers

# **Model:** `all-MiniLM-L6-v2` (embeddings)  
# **LLM:** `flan-t5-large` (free) or GPT-3.5  
# """)

# # ── Main area ─────────────────────────────────────────────────────
# st.title("📄 Smart Document Q&A Assistant")

# if st.session_state.doc_name:
#     st.info(f"Active document: **{st.session_state.doc_name}**")
# else:
#     st.warning("👈 Please upload and process a document to get started.")
#     st.stop()

# # ── Chat history ──────────────────────────────────────────────────
# for item in st.session_state.chat_history:
#     with st.chat_message("user"):
#         st.write(item["question"])
#     with st.chat_message("assistant"):
#         st.markdown(
#             f'<div class="answer-box">{item["answer"]}</div>',
#             unsafe_allow_html=True,
#         )
#         if item["sources"]:
#             with st.expander(f"📎 {len(item['sources'])} source chunks used"):
#                 for src in item["sources"]:
#                     st.markdown(
#                         f'<div class="source-box">'
#                         f'<b>Chunk #{src["chunk_id"]}</b> from <i>{src["source"]}</i><br>'
#                         f'{src["preview"]}'
#                         f'</div>',
#                         unsafe_allow_html=True,
#                     )

# # ── Question input ────────────────────────────────────────────────
# question = st.chat_input("Ask a question about your document...")

# if question:
#     with st.chat_message("user"):
#         st.write(question)

#     with st.chat_message("assistant"):
#         with st.spinner("Searching document and generating answer..."):
#             try:
#                 result = ask(st.session_state.qa_chain, question)
#                 answer = result["answer"]
#                 sources = result["sources"]
#             except Exception as e:
#                 answer = f"Error: {e}"
#                 sources = []

#         st.markdown(
#             f'<div class="answer-box">{answer}</div>',
#             unsafe_allow_html=True,
#         )
#         if sources:
#             with st.expander(f"📎 {len(sources)} source chunks used"):
#                 for src in sources:
#                     st.markdown(
#                         f'<div class="source-box">'
#                         f'<b>Chunk #{src["chunk_id"]}</b> from <i>{src["source"]}</i><br>'
#                         f'{src["preview"]}'
#                         f'</div>',
#                         unsafe_allow_html=True,
#                     )

#     st.session_state.chat_history.append({
#         "question": question,
#         "answer": answer,
#         "sources": sources,
#     })

###==============================================================================
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace

# --- Page Setup ---
st.set_page_config(page_title="2026 Smart Doc Q&A", layout="wide")
st.title("📄 Smart Document Assistant")

with st.sidebar:
    hf_token = st.text_input("HuggingFace Token", type="password")
    pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf and hf_token:
    # 1. Extract and Split
    reader = PdfReader(pdf)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    
    # 2. Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 3. The LLM (Modern Router)
    llm_base = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_token
    )
    llm = ChatHuggingFace(llm=llm_base)
    
    # 4. The LCEL Chain (No 'langchain.chains' needed!)
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # This pipeline replaces the old 'chains' modules
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 5. UI
    user_q = st.text_input("What do you want to know?")
    if user_q:
        with st.spinner("Analyzing..."):
            response = rag_chain.invoke(user_q)
            st.write("### Answer")
            st.info(response)

elif not hf_token:
    st.warning("Enter your HF Token to start.")