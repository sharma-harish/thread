from logging import getLogger
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

loader_logger = getLogger("loader")

def load_documents_from_folder(knowledge_base_path: str):
    """
    Load all `.md` files from knowledge_base_path folder and return them as a list of Document.
    Args:
        knowledge_base_path (str): Path to the knowledge base containing `.md` files.
    Returns:
        List[Document]: A list of Document objects.
    """
    print(f"Loading documents from {knowledge_base_path}")
    loader = DirectoryLoader(knowledge_base_path, glob="*.md", loader_cls=TextLoader, show_progress=True)
    docs = loader.load()
    headers_to_split_on = [
        ("#", "Header 1")
    ]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True
    )

    split_docs = []
    for doc in docs:
        chunks = header_splitter.split_text(doc.page_content)
        # Convert each chunk to a LangChain Document, preserving metadata
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
            split_docs.append(chunk)
    loader_logger.info(f"Read {len(docs)} files")

    return docs