import os
import warnings
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    # Adicione o parâmetro abaixo para autorizar o carregamento do arquivo
    return FAISS.load_local(
        "vector_store", 
        embeddings, 
        allow_dangerous_deserialization=True
    )

def answer_question(vector_store, query):
    # 1. Busca os trechos de texto mais relevantes no banco de dados
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

   # 2. Prepara o prompt para o Gemini
    client = genai.Client(api_key='')
    
    prompt = f""" Use o contexto abaixo para responder à pergunta. 
    Se não souber, diga que não encontrou no documento.
    
    Contexto: {context}
    Pergunta: {query}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text