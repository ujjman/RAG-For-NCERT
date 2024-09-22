import os
import dotenv
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.utils.math import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.tools import YouTubeSearchTool
import uvicorn
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from fastapi.responses import JSONResponse
import requests

class TextToSpeechRequest(BaseModel):
    text: str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SARVAM_API_KEY = "eab52b74-5405-45f7-af26-410d99bf849a"

if not OPENAI_API_KEY:
    logger.error("OpenAI API key not found in environment variables.")
    raise ValueError("OpenAI API key not found.")

if not SARVAM_API_KEY:
    logger.error("Sarvam AI API key not found in environment variables.")
    raise ValueError("Sarvam AI API key not found.")

client = OpenAI(api_key=OPENAI_API_KEY)

youtube_tool = YouTubeSearchTool()

general_template = """You are ChatGPT, a large language model trained by OpenAI.
Provide concise and accurate answers to general queries. If you don't know the answer, say "I don't know."

Question: {query}"""

document_template = """You are an assistant specialized in the content of the provided document. Answer the question based on the document context. If the answer is not in the document, say "I don't know."

Context: {context}
Question: {question}"""

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

prompt_templates = [general_template, document_template]

prompt_embeddings = embeddings.embed_documents(prompt_templates)




class TranslationRequest(BaseModel):
    text: str
    target_language_code: str

@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    url = "https://api.sarvam.ai/translate"
    payload = {
        "input": request.text,
        "source_language_code": "en-IN",
        "target_language_code": request.target_language_code,
        "speaker_gender": "Female",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True
    }
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": SARVAM_API_KEY
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        translated_text = response.json().get("translated_text")
        return {"translated_text": translated_text}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in translation request: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in translation service")












def prompt_router(input):
    logger.info(f"Input received in prompt_router: {input}")
    if isinstance(input, dict) and "query" in input:
        query = input["query"]
    else:
        query = str(input)

    if not isinstance(query, str):
        raise ValueError(f"Query must be a string, received: {type(query)}")

    query_embedding = embeddings.embed_query(query)
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar_index = similarity.argmax()
    most_similar = prompt_templates[most_similar_index]

    if most_similar == general_template:
        logger.info("Routing to General LLM")
        return "general"
    else:
        logger.info("Routing to RAG Chain")
        return "rag"

general_prompt_template = PromptTemplate(template=general_template, input_variables=["query"])
document_prompt_template = PromptTemplate(template=document_template, input_variables=["context", "question"])

general_llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)


try:
    loader = PyPDFLoader("Sound.pdf")
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(document)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    rag_llm = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Error setting up RAG components: {str(e)}")
    raise

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.post("/text-to-speech/")
async def text_to_speech(request: TextToSpeechRequest):
    url = "https://api.sarvam.ai/text-to-speech"
    payload = {
        "inputs": [request.text],
        "target_language_code": "bn-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1,
        "loudness": 2.8,
        "speech_sample_rate": 22050,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": SARVAM_API_KEY
    }

    try:
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()
        audio_string = response.text[12:-3]  
        audio_data = base64.b64decode(audio_string) 
        temp_audio_file = "temp_audio.wav"
        print("dodd")
        with open(temp_audio_file, "wb") as audio_file:
            audio_file.write(audio_data)
        return {"audio": "temp_audio.wav"} 
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in text-to-speech request: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in text-to-speech service")

def rag_chain(input):
    question = input["query"] if isinstance(input, dict) else str(input)
    docs = retriever.get_relevant_documents(question)
    context = format_docs(docs)
    prompt = document_prompt_template.format(context=context, question=question)
    return rag_llm.predict(prompt)

def general_chain(input):
    query = input["query"] if isinstance(input, dict) else str(input)
    prompt = general_prompt_template.format(query=query)
    return general_llm.predict(prompt)

def routing_chain(input):
    route = prompt_router(input)
    if route == "general":
        return general_chain(input), "general"
    else:
        return rag_chain(input), "rag"

class Query(BaseModel):
    question: str

class SoundWaveRequest(BaseModel):
    frequency: float
    amplitude: float
    time: float = 1.0

class SearchQuery(BaseModel):
    query: str

@app.post("/ask/")
async def ask(query: Query):
    try:
        logger.info(f"Received query: {query.question}")
        response, source = routing_chain(query.question)
        logger.info(f"Generated response: {response}")
        return {"question": query.question, "response": response, "source": source}
    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please check the server logs for more information.")

@app.post("/simulate-sound-wave/")
async def simulate_sound_wave(request: SoundWaveRequest):
    frequency = request.frequency
    amplitude = request.amplitude
    time = request.time

    sampling_rate = 44100
    t = np.linspace(0, time, int(sampling_rate * time), endpoint=False)
    sound_wave = amplitude * np.sin(2 * np.pi * frequency * t)

    plt.figure(figsize=(10, 4))
    plt.plot(t, sound_wave)
    plt.title(f'Sine Wave with Frequency {frequency}Hz and Amplitude {amplitude} (Duration: {time} sec)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.xlim(0, time)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return {"image": img_str}

@app.post("/search_youtube/")
async def search_youtube(query: SearchQuery):
    try:
        video_urls = youtube_tool.run(query.query)
        return {"videos": eval(video_urls)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/extract_parameters/")
async def extract_parameters(query: Query):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Extract the values of amplitude, frequency, and time from the following command: '{query.question}'."},
                {"role": "system", "content": "Format the response as: 'Amplitude: [value], Frequency: [value], Time: [value]'. If any value is not present then make it 0 by default."}
            ],
            max_tokens=50,
            temperature=0.3
        )
        return {"result": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate_title/")
async def generate_title(query: Query):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Generate a concise YouTube search title of maximum 5 words based on the question: '{query.question}'"}
            ],
            max_tokens=15,
            temperature=0.5
        )
        return {"title": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)