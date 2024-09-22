
# Retrieval-Augmented Generation (RAG) System with Agent and Extensions

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system designed to complete the technical assignment. The goal of this project is to demonstrate familiarity with key concepts related to RAG systems, agent-based actions, and advanced features such as voice and translation functionalities. The system works with NCERT PDF text data and serves the generated results using FastAPI.

  

## Project Overview

### Assignment Breakdown:

#### Part 1: Building a RAG System

A RAG (Retrieval-Augmented Generation) system has been implemented, where external NCERT PDF data is used along with a vector database and OpenAI's API for natural language queries. The system is served via a FastAPI endpoint and accepts user queries to retrieve relevant responses from the dataset.

  

#### Part 2: Building an Agent with Smart Actions

The RAG system has been extended to include an intelligent agent capable of performing additional actions. The agent is designed to decide when to call the Vector Database based on user input and can trigger two additional tools depending on the user's query. These tools include:

  

1. Sound Wave Simulator: Generates a graphical sound wave based on user-defined parameters such as amplitude, time, and frequency. The prompt is parsed using OpenAI's API to extract these parameters from the user's input. This feature is controlled by enabling the "Wave Generator" checkbox in the frontend.

  

2. YouTube Video Search: Fetches two YouTube videos related to the user's query when the "YouTube Search" extension is enabled in the frontend.

  

**Note:** The Sound Wave Simulator and YouTube Search tools cannot be enabled simultaneously.

  

#### Part 3: Adding Voice to the Agent (Bonus)

Voice functionality has been added to the agent, allowing the user to listen to the generated answers. In addition, users can translate the generated answers into their preferred language. This feature uses Sarvam's AI API for both translation and voice output.

  

## Tools and Libraries Used:

  

### Python Libraries: 
FastAPI, Langchain, PyPDFLoader, Chroma, OpenAI, Matplotlib, and Sarvam AI API.

  

### Frontend: 
A simple frontend interface is provided to allow users to interact with the system by enabling/disabling extensions and submitting queries.

  

### Backend: 
All queries are handled using a FastAPI backend, and responses are generated based on user input and the selected features.

  

## Features

#### RAG System:

Retrieves and processes data from NCERT PDF texts.
Leverages OpenAI’s language model for query understanding and response generation.

  

### Agent with Smart Tools:

#### Sound Wave Simulator: 
Generates graphical plots of sound waves based on user prompts.

  

#### YouTube Video Search:
 Fetches relevant YouTube videos based on the user's query.

  

#### Translation and Voice:

Users can translate the generated text into multiple languages.
Provides a voice-enabled response using Sarvam’s AI API for text-to-speech.

  

## How to Run the Project

  
### Prerequisites:
a. Python 3.x
b. Install required Dependencies

  

```

pip install -r requirements.txt

```

  

#### OPEN AI API KEY

1. Put your open AI API in the environment variables because the back.py code fetches API key from environment variables

  

```

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

```

  

2. Or if the above does not work, you can directly put the API key as string in the back.py code

  

```

OPENAI_API_KEY = "YOUR_API_KEY"

```

  

### To run the code

  

1. Make sure your `localhost` is free and port `8000` is not being used by any other program.
2. Then open a cmd in the same directory where the code files are extracted.
3. Then enter below command in the cmd to start the backend

```

python back.py

```

4. Then to start the fronend, open another window of cmd and enter below command

```

streamlit run front.py

```

5. Then you will be automatically redirected to the front end. If not then check the cmd terminal for the frontend url and put that in your browser.

  

**If the 1st part when running the code does not work then try to change the port from `8000` to some other port number that is not being used by any other program.**
1. You have to change it in both the files, `front.py` and `back.py` at - 

**back.py**
```
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
```
**front.py**
```
BACKEND_URL = "http://localhost:8000"
```
