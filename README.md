#  ABC Grocery RAG Chatbot
<img width="947" height="418" alt="ABC" src="https://github.com/user-attachments/assets/9a6072d1-566c-4638-b059-f49413d82e5d" />

##  Live Demo

**▶️ Try the app here:**  
👉 https://grocery-rag-chatbot-with-memory.streamlit.app/

A Retrieval-Augmented Generation (RAG) AI assistant built for **ABC Grocery** to answer customer and staff questions using verified internal documentation.  
This Streamlit application integrates **OpenAI GPT-5**, **LangChain**, and **ChromaDB** to provide fast, accurate, and context-aware responses.

---

##  Features

- **RAG-powered question answering**  
  Retrieves relevant context from the internal help guide before generating responses.

- **Conversational memory**  
  Remembers user information during the session and personalizes the conversation.

- **Hallucination-safe responses**  
  The model is restricted to answer **only** from provided documentation.

- **Modern Streamlit UI**  
  Clean chat interface with message history and session-based memory.

- **ChromaDB vector store**  
  Efficient embedding storage and fast similarity search.

---

##  Tech Stack

| Component | Technology |
|----------|------------|
| Language Model | OpenAI GPT-5 |
| Embeddings | `text-embedding-3-small` |
| Framework | LangChain |
| Vector Database | ChromaDB |
| UI | Streamlit |
| Language | Python |

---

##  How It Works

1. The help-desk document is loaded and split into semantic chunks.  
2. Each chunk is embedded using `text-embedding-3-small`.  
3. Embeddings are stored in **ChromaDB**.  
4. User queries → Retriever performs similarity search → relevant chunks returned.  
5. GPT-5 receives:
   - the retrieved context  
   - the conversation history  
   - the user question  
6. The assistant responds following strict grounding rules.

---

##  Grounding & Safety Rules

To ensure reliability and avoid hallucination, the assistant:

- Answers **only** from `<context> … </context>`
- Never invents facts
- Does not use external or world knowledge
- Uses memory only for personalization, not factual answers
- Falls back to a safe message if information is missing

---

##  Running Locally



```bash
1. Clone the repo
git clone https://github.com/LShahmiri/Grocery-RAG-Chatbot.git
cd Grocery-RAG-Chatbot
2.Install dependencies
pip install -r requirements.txt
3. Add your OpenAI API key
OPENAI_API_KEY="your-key-here"
4. Run the app
streamlit run streamlit_app.py

