# 📚 AI-Powered RAG PDF Question Answering System

A Retrieval-Augmented Generation (RAG) application built with Python, LangChain, ChromaDB, Mistral AI, and Streamlit that allows users to upload PDF documents and ask natural-language questions based on their content.

The system processes the uploaded PDF, splits it into smaller chunks, generates semantic embeddings, stores them in a Chroma vector store, retrieves relevant document context using Maximum Marginal Relevance (MMR), and uses a Mistral language model to generate context-grounded answers.

---

## 🚀 Overview

Large Language Models may not have access to information contained inside a user's private documents.

This project addresses this problem using a **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of asking the LLM to answer directly, the application first retrieves relevant information from the uploaded PDF and then provides that information to the LLM as context.

### Core Workflow

```text
              PDF Upload
                   │
                   ▼
             PyPDFLoader
                   │
                   ▼
             Text Extraction
                   │
                   ▼
       Recursive Text Splitting
                   │
                   ▼
          Mistral Embeddings
                   │
                   ▼
            ChromaDB Vector Store
                   │
                   │
             User Question
                   │
                   ▼
              MMR Retrieval
                   │
                   ▼
          Relevant PDF Chunks
                   │
                   ▼
           Prompt Construction
                   │
                   ▼
             Mistral Small
                   │
                   ▼
             Final Answer

