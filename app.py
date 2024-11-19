from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import pyttsx3

# Constants
DB_FAISS_PATH = 'vectorstore/db_faiss'
MODEL_PATH = "./llama-2-7b-chat.ggmlv3.q8_0.bin"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GREETING_KEYWORDS = {"hi", "hello", "hey", "greetings"}
ENDING_KEYWORDS = {"thank you", "thanks", "thank you very much"}
CUSTOM_PROMPT_TEMPLATE = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know. Do not make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:

(Note: I am not a doctor. Please consult a healthcare professional before taking any medication.)
"""

# Text-to-Speech Engine Initialization
tts_engine = pyttsx3.init()

def set_custom_prompt():
    """Creates a prompt template for QA retrieval."""
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=['context', 'question'])

def load_llm():
    """Loads the Llama 2 model."""
    return CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.5
    )

def initialize_faiss_db():
    """Initializes and loads the FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        raise RuntimeError(f"Error loading FAISS database: {e}")

def create_qa_chain():
    """Creates the QA retrieval chain."""
    db = initialize_faiss_db()
    llm = load_llm()
    prompt = set_custom_prompt()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def speak(text):
    """Converts text to speech using pyttsx3."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def format_response(result):
    """Formats the response with sources if available."""
    answer = result.get("result", "No response generated.")
    sources = result.get("source_documents", [])

    if sources:
        relevant_sources = []
        for source in sources:
            source_info = source.metadata.get("source", "Unknown Source")
            page = source.metadata.get("page", "N/A")
            relevant_sources.append(f"{source_info} (Page {page})")
        answer += "\nSources:\n" + "\n".join(relevant_sources)
    else:
        answer += "\nNo sources found."

    return answer

@cl.on_chat_start
async def start():
    """Handles the start of the chat."""
    try:
        chain = create_qa_chain()
        cl.user_session.set("chain", chain)

        msg = cl.Message(content="Starting the bot...")
        await msg.send()
        msg.content = "Hi, Welcome to MediMate AI. What is your query?"
        await msg.update()
    except Exception as e:
        await cl.Message(content=f"Error initializing chatbot: {e}").send()

@cl.on_message
async def main(message: cl.Message):
    """Handles user messages."""
    content = message.content.strip().lower()
    chain = cl.user_session.get("chain")

    # Greeting response
    if content in GREETING_KEYWORDS:
        await cl.Message(content="Hello! How can I assist you today?").send()
        return

    # Thank you response
    if content in ENDING_KEYWORDS:
        await cl.Message(content="You're very welcome! 😊 Let me know if there's anything else I can help with.").send()
        return

    # Process user query
    try:
        cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
        response = await chain.ainvoke(content, callbacks=[cb])
        formatted_response = format_response(response)
        await cl.Message(content=formatted_response).send()
        speak(formatted_response)
    except Exception as e:
        await cl.Message(content=f"Error processing your query: {e}").send()
