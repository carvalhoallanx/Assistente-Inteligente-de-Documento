import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def add_documents_to_store(pdf_list):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 🔁 Verifica se já existe banco
    if os.path.exists("vector_store"):
        db = FAISS.load_local(
            "vector_store",
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        db = None

    all_docs = []

    for pdf in pdf_list:
        loader = PyPDFLoader(pdf)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf
            doc.metadata["page"] = doc.metadata.get("page", 0)

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(all_docs)

    # 🧠 Criar ou adicionar
    if db:
        db.add_documents(split_docs)
    else:
        db = FAISS.from_documents(split_docs, embeddings)

    db.save_local("vector_store")

    print("✅ Documentos adicionados ao banco!")