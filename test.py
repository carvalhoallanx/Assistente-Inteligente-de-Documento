from rag_pipeline import load_db, answer_question

#test 
if __name__ == "__main__":
    # 1. Carrega o banco de dados criado anteriormente
        try:
            db = load_db()
            print("✅ Banco de dados carregado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao carregar o banco: {e}")
            exit()

# 2. Loop de interação
print("\n--- Assistente RAG Pronto (digite 'sair' para encerrar) ---")
while True:
        pergunta = input("\nVocê: ")
        
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
        
        print("Analisando documentos...")
        resposta = answer_question(db, pergunta)
        print(f"Gemini: {resposta}")