# RAG Local con Ollama + ChromaDB + Gradio

Asistente que responde preguntas y genera resúmenes sobre tus PDFs **sin enviar datos a la nube**.
Todo el procesamiento (embeddings y generación) ocurre en tu máquina con [Ollama](https://ollama.com).

## Características

- 🔒 **100% local y privado**: ni el PDF ni las preguntas salen de tu ordenador.
- **Preguntas sobre el documento**: recuperación semántica + respuesta con contexto.
- **Resumen automático**: síntesis del PDF en 5–8 frases.
- **Sin APIs de pago ni claves**: solo modelos open source.
- **Respuestas honestas**: si la respuesta no está en el documento, lo dice explícitamente.

## Stack

| Componente | Tecnología |
|---|---|
| LLM | `llama3.1` (vía Ollama) |
| Embeddings | `nomic-embed-text` (vía Ollama) |
| Base vectorial | ChromaDB (`langchain-chroma`) |
| Orquestación | LangChain |
| Interfaz | Gradio |

## Requisitos

- Linux (probado en Ubuntu 22.04)
- [Ollama](https://ollama.com) instalado
- Python 3.10+

### Instalar Ollama y los modelos

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Instalación del proyecto

```bash
git clone https://github.com/zene77/rag-lab.git
cd rag-lab

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python app.py
```

Abre [http://127.0.0.1:7860](http://127.0.0.1:7860). Verás dos pestañas:

### Pestaña "Preguntar"

1. Sube un PDF.
2. Escribe una pregunta sobre su contenido.
3. Pulsa **Preguntar**.

La primera pregunta tarda 20–60 s (Ollama carga los modelos en memoria). Las siguientes son mucho más rápidas.

### Pestaña "Resumen"

1. Sube un PDF.
2. Pulsa **Generar resumen**.
3. Obtén un resumen en 5–8 frases basado solo en el contenido del documento.

## Ejemplos reales

Probado con el PDF *"Bases de Datos Vectoriales y Recuperadores Avanzados"*.

**Pregunta:**
> ¿Qué es Chroma DB?

**Respuesta:**
> Chroma DB es una base de datos vectorial de código abierto diseñada específicamente para almacenar y recuperar incrustaciones y metadatos. Se utiliza para implementar motores de búsqueda semántica sobre texto y se integra fácilmente con LangChain.

**Resumen (extracto):**
> Las bases de datos vectoriales, como Chroma DB, permiten almacenar y recuperar incrustaciones y metadatos de manera eficiente. Los retrievers de LangChain son interfaces genéricas que devuelven documentos a partir de una consulta no estructurada. LangChain también ofrece mecanismos avanzados de recuperación, como consultas múltiples, autoconsulta y documentos principales... Las bases de datos vectoriales y los retrievers de LangChain forman la columna vertebral de los sistemas RAG.

## Estructura del proyecto

```
rag-lab/
├── app.py              # Interfaz Gradio + pipeline RAG (preguntas y resumen)
├── test_rag.py         # Test del pipeline sin interfaz
├── crear_pdf.py        # Genera un PDF de prueba (documento_prueba.pdf)
├── requirements.txt    # Dependencias con versiones fijas
├── .gitignore          # Excluye .venv/, __pycache__/ y chroma_local_db/
└── README.md
```

## Cómo funciona

1. **Carga**: `PyPDFLoader` extrae el texto del PDF.
2. **División**: `RecursiveCharacterTextSplitter` parte el texto en chunks de 1000 caracteres con 200 de solapamiento.
3. **Embeddings**: `nomic-embed-text` convierte cada chunk en un vector.
4. **Almacenamiento**: ChromaDB indexa los vectores en memoria.
5. **Recuperación**: dada una pregunta, se recuperan los `k=3` chunks más similares.
6. **Generación**: `llama3.1` redacta la respuesta usando solo esos chunks como contexto.
7. **Resumen**: en la pestaña de resumen se pasa el texto completo (hasta 12 000 caracteres) al LLM con un prompt específico.

## Limitaciones conocidas

- **Contexto limitado**: `llama3.1` tiene ~8k tokens de ventana. Para PDFs muy largos, el resumen solo cubre los primeros 12 000 caracteres (el código trunca el resto).
- **Respuestas conservadoras**: el prompt estricto hace que el modelo diga *"No encuentro esa información en el documento"* si el retriever no recupera el fragmento adecuado. Aumentar `k` o reformular la pregunta suele solucionarlo.
- **Sin memoria conversacional**: cada pregunta es independiente; no hay historial de chat.

## Posibles mejoras

- [ ] Resumen jerárquico (map-reduce) para PDFs largos.
- [ ] Multi-Query Retriever para preguntas ambiguas.
- [ ] Historial de conversación con `ConversationBufferMemory`.
- [ ] Despliegue en Hugging Face Spaces con `share=True`.

## Licencia

MIT
EOF
```

### Añadirlo al repositorio

```bash
cd ~/rag-lab
git add README.md
git commit -m "Añadir README con instrucciones, ejemplos y limitaciones"
git push
```
