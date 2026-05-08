from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
url='https://www.apple.com/in/shop/buy-mac/macbook-air'
data = WebBaseLoader(url)
docs = data.load()
print(docs[0].page_content)