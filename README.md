#  ABC Grocery RAG Chatbot

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

##  Live Demo

**▶️ Try the app here:**  
👉 https://grocery-rag-chatbot-with-memory.streamlit.app/

---

##  Project Structure


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

### 1. Clone the repo

```bash
git clone https://github.com/LShahmiri/Grocery-RAG-Chatbot.git
cd Grocery-RAG-Chatbot
pip install -r requirements.txt
OPENAI_API_KEY="your-key-here"
streamlit run streamlit_app.py

