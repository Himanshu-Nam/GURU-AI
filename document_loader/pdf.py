from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
loader = PyPDFLoader('document_loader/GRU.pdf')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,
    chunk_overlap=10
)
chuks = splitter.split_documents(docs)
print(chuks[0].page_content)
