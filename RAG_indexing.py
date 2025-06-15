import os
from openai import images
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from typing import List
import pytesseract
from pdf2image import convert_from_path


# Load the environment variables from the .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

SOURCE_DOCUMENT = 'https://rupertstudies.weebly.com/uploads/9/5/8/4/9584887/basic.mathematics.for.economists_-_rosser.rootledge_2003_second.edition.pdf'
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
postgres_connection = os.getenv("POSTGRES_CONNECTION")

def main():
    print("Extracting...")
    pdf_text = extract_pdf_text(SOURCE_DOCUMENT)

    print("Chunking...")
    chunks = pdf_chunk(pdf_text)
    print(f"Number of chunks = {len(chunks)}")

    print("Creating Vector Store...")
    create_vector_store(chunks)
    print("Done.")


def extract_pdf_text(pdf_url: str) -> List[Document]:
    response = requests.get(pdf_url)
    pdf_path = 'toancaocap.pdf'
    with open(pdf_path, 'wb') as file:
        file.write(response.content)
    
    print("PDF file text is extracted...")
    loader = PyPDFLoader(pdf_path)
    pdf_text = loader.load()
    print(f"Number of documents = {len(pdf_text)}")

    return pdf_text

def pdf_chunk(pdf_text: List[Document]) -> List[Document]:
    print("PDF file text is chunked....")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        # Existing args
    )

    chunks = text_splitter.split_documents(pdf_text)
    return chunks


def create_vector_store(documents: List[Document]):
    print("Postgres vector store is created...\n")
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=documents,
        collection_name=COLLECTION_NAME,
        connection_string=postgres_connection,
        use_jsonb=True,
    )
    return db

if __name__ == "__main__":
    main()