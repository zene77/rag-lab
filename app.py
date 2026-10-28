import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


def get_llm():
    return OllamaLLM(model="llama3.1", temperature=0.1)


def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


# ---------- NUEVA FUNCIÓN: RESUMEN ----------
def resumir_pdf(file):
    if not file:
        return "Por favor, sube un archivo PDF primero."

    loader = PyPDFLoader(file.name)
    docs = loader.load()

    # Unimos todo el texto del PDF
    texto_completo = "\n\n".join(doc.page_content for doc in docs)

    if not texto_completo.strip():
        return "El PDF no contiene texto extraíble."

    # Si el PDF es muy largo, lo truncamos para no desbordar el contexto
    # llama3.1 tiene ~8k tokens de contexto; dejamos margen.
    MAX_CHARS = 12000
    if len(texto_completo) > MAX_CHARS:
        texto_completo = texto_completo[:MAX_CHARS] + "\n\n[...texto truncado...]"

    prompt = PromptTemplate(
        template="""Eres un asistente que resume documentos.
Resume el siguiente texto en español, en 5-8 frases claras.
Usa solo la información del texto. No inventes nada.

Texto:
{texto}

Resumen:""",
        input_variables=["texto"],
    )

    llm = get_llm()
    cadena = prompt | llm
    respuesta = cadena.invoke({"texto": texto_completo})
    return respuesta


# ---------- FUNCIÓN EXISTENTE: PREGUNTAS ----------
def responder_pregunta(file, query):
    if not file:
        return "Por favor, sube un archivo PDF primero."
    if not query or not query.strip():
        return "Escribe una pregunta sobre el documento."

    loader = PyPDFLoader(file.name)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    if not splits:
        return "El PDF no contiene texto extraíble. Prueba con otro archivo."

    try:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=get_embeddings(),
        )
    except Exception as e:
        return f"Error al crear la base vectorial: {str(e)}"

    prompt = PromptTemplate(
        template="""Responde solo usando el siguiente contexto.
Si la respuesta no está en el contexto, di exactamente:
"No encuentro esa información en el documento."

Contexto: {context}

Pregunta: {question}

Respuesta:""",
        input_variables=["context", "question"],
    )

    qa = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

    respuesta = qa.invoke({"query": query})
    return respuesta["result"]


# ---------- INTERFAZ CON DOS PESTAÑAS ----------
with gr.Blocks(title="Asistente de Documentos Local") as demo:
    gr.Markdown("# Asistente de Documentos Privado (Local)")
    gr.Markdown("RAG con Ollama y ChromaDB. Tus datos nunca salen de tu computadora.")

    with gr.Tab("Preguntar"):
        file_preg = gr.File(label="Sube tu PDF", file_types=[".pdf"], type="filepath")
        query = gr.Textbox(label="Pregunta sobre el documento",
                           placeholder="¿De qué trata este documento?")
        btn_preg = gr.Button("Preguntar", variant="primary")
        out_preg = gr.Textbox(label="Respuesta", lines=8)
        btn_preg.click(responder_pregunta, inputs=[file_preg, query], outputs=out_preg)

    with gr.Tab("Resumen"):
        file_res = gr.File(label="Sube tu PDF", file_types=[".pdf"], type="filepath")
        btn_res = gr.Button("Generar resumen", variant="primary")
        out_res = gr.Textbox(label="Resumen", lines=10)
        btn_res.click(resumir_pdf, inputs=[file_res], outputs=out_res)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
