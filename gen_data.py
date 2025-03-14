import getpass
import pprint
import os

from dotenv import load_dotenv

load_dotenv(override=True)

if not os.environ.get("OPENAI_API_KEY"): 
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

from openai import OpenAI

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, Docx2txtLoader, DirectoryLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_iris import IRISVector

import os
import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document  # For reading .docx files

import pandas as pd

client = OpenAI()
embeddings = OpenAIEmbeddings()

username = 'demo'
password = 'demo' 
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972' 
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"
# Under the hood, this becomes a SQL table. CANNOT have '.' in the name

loader = DirectoryLoader('data', glob='*.docx', loader_cls=Docx2txtLoader)
docs = loader.load()
len(docs)

text_splitter = SpacyTextSplitter(chunk_size=400, chunk_overlap=20)
docs = text_splitter.split_documents(docs)

COLLECTION_NAME = "cancer_db"
# This creates a persistent vector store (a SQL table). You should run this ONCE only
db = IRISVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

print(f"Number of docs in vector store: {len(db.get()['ids'])}")

f = open("data/s_test.txt", "r", encoding='ISO-8859-1')
# query = "new technology"
scenario = f.read()

f = open("data/knowledge.docx", "r", encoding='ISO-8859-1')
# query = "new technology"
knowledge = f.read()

loader = CSVLoader('data/treatment_selection.csv')#, csv_args={'fieldnames':['']})
docs = loader.load()
len(docs)

COLLECTION_NAME = "pictures_db"
# This creates a persistent vector store (a SQL table). You should run this ONCE only
db1 = IRISVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)