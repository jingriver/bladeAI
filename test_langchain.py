from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

baseUrl = 'http://localhost:11434'

ollama = Ollama(
    base_url=baseUrl,
    model="llama3"
)
#print(ollama.invoke("why is the sky blue"))

# load the document
loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
data = loader.load()
print(f'loader: {len(data)}')

# split it into chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
all_splits = text_splitter.split_documents(data)
print(f'all_splits: {len(all_splits)}')

# create embedding function
oembed = OllamaEmbeddings(base_url=baseUrl, model="nomic-embed-text")
print(f'embeddings: {oembed}')

# load docs into Chroma DB and save to disk
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed, persist_directory="./chroma_db")
print(f'vectorstore: {vectorstore._collection.count()}')

# query the DB
question="Who is Neleus and who is in Neleus' family?"
docs = vectorstore.similarity_search(question)
print(f'similarity_search: {docs[0].metadata, docs[0].page_content}')

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Retrieval
qachain = RetrievalQA.from_chain_type(
    llm=ollama,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
print(qachain.invoke({"query": question}))
