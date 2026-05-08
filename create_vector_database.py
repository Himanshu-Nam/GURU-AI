#load pdf
#split into text
#create embedding
#store embedding in db(chroma)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

#load document
loader = PyPDFLoader('document_loader/deeplearning.pdf')
docs = loader.load()



#text-splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

#create chunks
chunks = splitter.split_documents(docs)

#embedding model
embedding_model = OpenAIEmbeddings()

vector_db = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory="chroma_db"
)
