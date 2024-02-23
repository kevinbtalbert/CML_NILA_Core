
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader,PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import pinecone

# Initialize Pinecone and OpenAI API
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
openai_api_key = os.getenv("OPENAI_API_KEY")
index_name = os.getenv("PINECONE_INDEX")

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
    # openai_api_base=OPENAI_API_BASE, 
    # openai_api_type=OPENAI_API_TYPE, 
    # openai_api_version=OPENAI_API_VERSION,
    # chunk_size=1
)

# Check if the Pinecone index exists, if not, create it
existing_indexes = pinecone.list_indexes()
if index_name not in existing_indexes:
    pinecone.create_index(index_name, dimension=1536, metric="cosine")
    
# Initialize Pinecone index object
index = pinecone.Index(index_name)

# Define the path to your documents folder
folder_path = '/home/cdsw/data/TXT'

# List all PDF files in the folder
def list_txt_files(folder_path):
    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

txt_files = list_txt_files(folder_path)

for txt_file in txt_files:
    print(txt_file)
    loader = TextLoader(txt_file)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    print("Loaded TXT document " + txt_file + " successfully to Pinecone index " + index_name)
    