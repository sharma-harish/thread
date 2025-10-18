from logging import getLogger
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph_project.settings import settings
from pathlib import Path
import chromadb
from chromadb import Client
from chromadb.config import Settings

from langgraph_project.vector_store.loader import load_documents_from_folder

index_logger = getLogger("index")
PERSIST_DIR = "chroma_db"
client = chromadb.PersistentClient(path=PERSIST_DIR)

def create_index(doc_path: str, collection: str) -> Chroma:
    """
    Create a simple index using the in memory Chroma
    """
    if settings.use_google:
        embeddings_model = GoogleGenerativeAIEmbeddings(model=settings.google_embeddings_model_name)
    else:
        embeddings_model = OpenAIEmbeddings(
            model=settings.embeddings_model_name,
            api_key=settings.openai_api_key.get_secret_value()
        )
    collections = [collection.name for collection in client.list_collections()]
    if collection in collections:
        index = Chroma(
            collection_name=collection,
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings_model
        )
        return index
    else:
        documents = load_documents_from_folder(doc_path)
        index_logger.info(f"Processing Index for {len(documents)} docs")
        index = Chroma.from_documents(
            documents,
            embeddings_model,
            collection_name=collection,
            persist_directory=PERSIST_DIR
        )
    return index
