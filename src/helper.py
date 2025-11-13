# from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from pathlib import Path
from langchain.schema import Document
# from langchain.embeddings import HuggingFaceBgeEmbeddings

# ✅ NEW
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings



# Build the PDF path relative to this file so it works regardless of the
# current working directory when the module is imported or the script run.
pdf_path = str(
    Path(__file__).resolve().parent.parent.joinpath("data", "3-Lecture-Common-Skin-Disease-2019.pdf")
)
def load_and_split_data(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # Fixed method name to lowercase
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)  # Split the documents
    return texts  # Return the split texts

extracted_data=load_and_split_data(pdf_path)
extracted_data # Display the first chunk of extracted data

def filter_to_minimal_chunk(extracted) -> List[Document]:
    """Convert full Document objects into minimal Documents carrying
    only page_content and a source metadata field.

    Accepts the list of documents to process (does not rely on globals).
    """
    minimal_docs: List[Document] = []
    for doc in extracted:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": doc.metadata.get("source")},
            )
        )
    return minimal_docs

def split(minimal_doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunk = text_splitter.split_documents(minimal_doc)
    return text_chunk



def download_embeddings():
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embeddings=HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


