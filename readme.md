📄🤖 Chat com Documentos (RAG em Python)

Um assistente inteligente que permite conversar com seus próprios arquivos usando IA.

🧠 Sobre o projeto

Este projeto implementa um sistema de RAG (Retrieval-Augmented Generation) que permite:

📄 Fazer upload de documentos (PDF, TXT, etc)<p>
❓ Fazer perguntas sobre o conteúdo<p>
🤖 Receber respostas baseadas nos dados enviados<p>

👉 Basicamente: seu próprio ChatGPT com arquivos personalizados

⚙️ Tecnologias utilizadas<p>
🐍 Python<p>
🔗 LangChain<p>
💬 Ollama<p>
🤖 StreamLit<p>
🧠 OllamaEmbeddings (Sentence Transformers)<p>
📊 FAISS (Vector Store)<p>
🚀 FastAPI<p>

## Como Executar

Executar API:

in terminal - pip install requirements.txt or pip install 

In powershell - irm https://ollama.com/install.ps1 | iex - install ollama

in cmd - ollama pull nomic-embed-text  
ollama pull qwen2.5-coder:7b

run in terminal - 
cd ChatDOC
uvicorn api.main:app --reload
or
uvicorn api.main:app --reload

streamlit run ChatDOC\app.py
or 
streamlit run app.py
