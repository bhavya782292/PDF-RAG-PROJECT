from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

# Below are the documents

docs = [
    Document(page_content="Python is widely used in artificial intelligence.", metadata={"source": "AI_BOOK"}),
    Document(page_content="Pandas is widely used  for data analysis.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural network is widely used in deep learning.", metadata={"source": "DL_book"}),
]
# Create embeddings

embeddings_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    persist_directory="chroma-db"
)


result = vectorstore.similarity_search("What is used for data analysis? ", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)
    
retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)
    
    