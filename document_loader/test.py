from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
data = TextLoader('document_loader/text.txt')
splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=5
)
docs = data.load()
chuks = splitter.split_documents(docs)
for i in chuks:
    print(i.page_content)
   