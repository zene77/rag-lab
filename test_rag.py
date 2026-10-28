from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains import RetrievalQA

PDF_PATH = "documento_prueba.pdf"

print("1) Cargando PDF...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"   Paginas cargadas: {len(docs)}")

print("2) Dividiendo en chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)
print(f"   Chunks generados: {len(splits)}")

print("3) Creando base vectorial con Chroma...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory="./chroma_local_db"
)
print("   Chroma listo.")

print("4) Construyendo cadena RAG...")
qa = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="llama3.1"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)
print("   Cadena lista.")

print("\n5) Pregunta de prueba:")
pregunta = "¿Cuándo se fundó la empresa y quién es su director ejecutivo?"
print(f"   > {pregunta}\n")

respuesta = qa.invoke({"query": pregunta})["result"]
print("Respuesta del modelo:\n")
print(respuesta)
