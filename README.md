# 📚 AI-Powered RAG PDF Question Answering System

An AI-powered **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions about their content.

The system uses **LangChain, ChromaDB, Mistral AI embeddings, Mistral Small, and Streamlit** to retrieve relevant document context and generate grounded answers based only on the uploaded PDF.

## 🔗 Links

- 🌐 **Live Demo:** [Try the RAG PDF Chatbot](https://pdf-rag-project-pna66jay958az92eyqpyak.streamlit.app/)
- 💻 **GitHub Repository:** [View Source Code](https://github.com/bhavya782292/PDF-RAG-PROJECT)

---

## 🎯 Project Overview

Large Language Models can sometimes generate answers that are not supported by the source document.

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that first retrieves relevant information from an uploaded PDF and then provides that context to an LLM for answer generation.

The application is designed to answer questions using **only the retrieved document context**.

If the required information is not present in the retrieved context, the system responds:

> I could not find the answer in the document.

This helps keep responses grounded in the uploaded document.

---

## ✨ Key Features

### 📄 PDF Upload

Users can upload a PDF document directly through the Streamlit interface.

The application temporarily stores the uploaded file and processes it using **PyPDFLoader**.

### ✂️ Document Chunking

The extracted document content is divided into smaller chunks using:

**RecursiveCharacterTextSplitter**

Configuration:

```text
Chunk Size: 1000 characters
Chunk Overlap: 200 characters 
