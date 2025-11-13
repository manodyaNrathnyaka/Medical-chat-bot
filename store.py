from dotenv import load_dotenv
import os
from src.helper import (
    load_and_split_data,
    split,
    filter_to_minimal_chunk,
    download_embeddings,
    pdf_path,
)
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-bot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",  # cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # specification
    )

# Use clear names and prepare documents/embeddings exactly as in the notebook
# Load and split the PDF into Document chunks
extracted = load_and_split_data(pdf_path)
minimal_docs = filter_to_minimal_chunk(extracted)
text_chunk = split(minimal_docs)

# Create/prepare embeddings
embedding = download_embeddings()

# Create the Pinecone vector store using the index name (string)
doc_store = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name,
)

print("Pinecone vector store created:", hasattr(doc_store, "client"))