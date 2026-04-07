import os
import warnings
import requests
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

def answer_question(vector_store, query, k=3):
    # 1. Busca os trechos de texto mais relevantes no banco de dados
    docs = vector_store.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])

    # 2. Prepara o prompt para o Ollama
    prompt = f""" Use o contexto abaixo para responder à pergunta. 
    Se não souber, diga que não encontrou no documento.
    
    Contexto: {context}
    Pergunta: {query}
    """

    model = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

    try:
        response = requests.post(
            ollama_url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        if not answer:
            answer = "Nao consegui gerar uma resposta com o Ollama."
    except requests.RequestException as exc:
        raise RuntimeError(
            "Falha ao conectar no Ollama. Verifique se o servidor esta ativo em "
            "http://localhost:11434 e se o modelo foi baixado."
        ) from exc

    return answer, docs