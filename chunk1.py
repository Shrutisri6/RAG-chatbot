import streamlit as st
import pandas as pd
import json
import sqlite3
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Conversational RAG Chatbot", layout="wide")

st.title("🤖 Conversational RAG Chatbot")

# ==============================
# SESSION STATE FOR CHAT
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader(
    "Upload any document (PDF, DOCX, PPTX, TXT, MD, JSON, XML, CSV, SQLite)",
    type=["pdf", "docx", "pptx", "txt", "md", "json", "xml", "csv", "db", "sqlite"]
)

# ==============================
# FILE LOADER
# ==============================
def load_file(file_path, file_extension):

    if file_extension == "pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()

    elif file_extension == "docx":
        loader = Docx2txtLoader(file_path)
        return loader.load()

    elif file_extension == "pptx":
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()

    elif file_extension in ["txt", "md"]:
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    elif file_extension == "json":
        with open(file_path, "r") as f:
            data = json.load(f)
        return [Document(page_content=json.dumps(data, indent=2))]

    elif file_extension == "xml":
        with open(file_path, "r") as f:
            data = f.read()
        return [Document(page_content=data)]

    elif file_extension == "csv":
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)
        return [Document(page_content=text)]

    elif file_extension in ["db", "sqlite"]:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()

        text_data = ""
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()

        for table in tables:
            table_name = table[0]
            rows = cursor.execute(f"SELECT * FROM {table_name}").fetchall()
            text_data += f"\nTable: {table_name}\n"
            for row in rows:
                text_data += str(row) + "\n"

        conn.close()
        return [Document(page_content=text_data)]

    return []

# ==============================
# VECTORSTORE
# ==============================
@st.cache_resource
def setup_vectorstore(uploaded_file):

    file_extension = uploaded_file.name.split(".")[-1].lower()
    temp_path = f"temp.{file_extension}"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    documents = load_file(temp_path, file_extension)

    full_text = "\n".join([doc.page_content for doc in documents])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(full_text)

    docs = [
        Document(page_content=chunk)
        for chunk in chunks
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    return FAISS.from_documents(docs, embeddings)

# ==============================
# MAIN CHAT LOGIC
# ==============================
if uploaded_file:

    vectorstore = setup_vectorstore(uploaded_file)

    # Display previous chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (ChatGPT style)
    user_input = st.chat_input("Ask something about the document...")

    if user_input:

        # Save user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # Retrieve context
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 6, "fetch_k": 15}
                )

                retrieved_docs = retriever.invoke(user_input)

                context = "\n\n".join(
                    [doc.page_content for doc in retrieved_docs]
                )

                # Build chat history text
                chat_history = "\n".join(
                    [f"{m['role']}: {m['content']}"
                     for m in st.session_state.messages]
                )

                llm = ChatOllama(
                    model="llama3.2:1b",
                    temperature=0
                )

                prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant having a conversation with the user.

Use the conversation history and the provided context to answer.

If the answer is not in the context, say:
"I could not find this in the document."

Conversation History:
{history}

Context:
{context}

User Question:
{question}
""")

                chain = prompt | llm

                response = chain.invoke({
                    "history": chat_history,
                    "context": context,
                    "question": user_input
                })

                st.markdown(response.content)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )
