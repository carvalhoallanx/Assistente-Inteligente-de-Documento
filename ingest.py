from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

    # 1. Carregar PDF
def create_vector_store(pdf_path='doctest/documentotest.pdf'):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Quebrar em chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # 3. Criar embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Criar banco vetorial
    db = FAISS.from_documents(docs, embeddings)

    # 5. Salvar no disco
    db.save_local("vector_store")

    print("✅ Vector store criado!")
    
create_vector_store()    