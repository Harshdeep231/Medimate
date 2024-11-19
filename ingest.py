from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Constants
DATA_PATH = 'DataUsed/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def load_documents(data_path):
    """
    Load PDF documents from a specified directory.

    Args:
        data_path (str): Path to the directory containing PDF files.

    Returns:
        list: A list of documents loaded from the PDF files.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory '{data_path}' does not exist.")

    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        raise ValueError("No documents found in the specified directory.")

    return documents


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents (list): List of documents to be split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap size between chunks.

    Returns:
        list: List of split documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def create_embeddings(model_name=EMBEDDING_MODEL):
    """
    Create HuggingFace embeddings for vectorization.

    Args:
        model_name (str): Name of the embedding model.

    Returns:
        HuggingFaceEmbeddings: Instance of the embedding model.
    """
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})


def save_vector_db(texts, embeddings, db_path):
    """
    Create and save a FAISS vector database.

    Args:
        texts (list): List of text chunks to be stored.
        embeddings (HuggingFaceEmbeddings): Embedding model.
        db_path (str): Path to save the FAISS vector database.
    """
    db = FAISS.from_documents(texts, embeddings)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_local(db_path)


def create_vector_db():
    """
    Complete pipeline to load documents, process them into embeddings,
    and save the FAISS vector database.
    """
    try:
        print("Loading documents...")
        documents = load_documents(DATA_PATH)

        print("Splitting documents into chunks...")
        texts = split_documents(documents)

        print(f"Creating embeddings using model '{EMBEDDING_MODEL}'...")
        embeddings = create_embeddings()

        print(f"Saving FAISS vector database to '{DB_FAISS_PATH}'...")
        save_vector_db(texts, embeddings, DB_FAISS_PATH)

        print("Vector database created and saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    create_vector_db()
