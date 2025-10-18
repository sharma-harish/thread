from langchain_core.tools import tool

from langgraph_project.settings import settings
from langgraph_project.vector_store.index import create_index

# Initialize the VectorStore and set the index

index = create_index(settings.knowledge_base_path + "/users", "users_collection", )

@tool
def retrieve_user_information_vectorbase(query: str):
    """Retrieve information from a MongoDB Atlas Vector Search index.
    
    Args:
        query (str): The search query.

    Returns:
        str: Retrieved documents concatenated as a string.
    """
    documents = index.similarity_search(query=query, k=1)
    return "\n\n".join([doc.page_content for doc in documents]) 
