
# 🎯 Retrieval-Augmented Generation (RAG) System with Intelligent Agent & Extensions 🚀

Welcome to the implementation of a **Retrieval-Augmented Generation (RAG)** system enhanced with an intelligent agent and advanced features like voice and translation! 🎉 This project showcases key concepts in RAG systems, agent-based actions, and more. It uses NCERT PDF text data and serves results via FastAPI.

## 📖 Table of Contents

-   [Project Overview](#project-overview)
    -   [Assignment Breakdown](#assignment-breakdown)
        -   [Part 1: Building a RAG System](#part-1-building-a-rag-system)
        -   [Part 2: Building an Agent with Smart Actions](#part-2-building-an-agent-with-smart-actions)
        -   [Part 3: Adding Voice to the Agent (Bonus)](#part-3-adding-voice-to-the-agent-bonus)
-   [Tools and Libraries Used](#tools-and-libraries-used)
-   [Features](#features)
    -   [RAG System](#rag-system)
    -   [Agent with Smart Tools](#agent-with-smart-tools)
        -   [Sound Wave Simulator 🎵](#sound-wave-simulator-)
        -   [YouTube Video Search 🎥](#youtube-video-search-)
    -   [Translation and Voice 🗣️](#translation-and-voice-)
-   [How to Run the Project](#how-to-run-the-project)
    -   [Prerequisites](#prerequisites)
    -   [Setting up API Keys](#setting-up-api-keys)
    -   [Running the Code](#running-the-code)
-   [Using the Features](#using-the-features)
    -   [Enabling Extensions](#enabling-extensions)
    -   [Using YouTube Search](#using-youtube-search)
    -   [Using Wave Generator](#using-wave-generator)
-   [Troubleshooting](#troubleshooting)

## Project Overview

### Assignment Breakdown

#### Part 1: Building a RAG System

A Retrieval-Augmented Generation (RAG) system is implemented, leveraging external NCERT PDF data along with a vector database and OpenAI's API for natural language queries. The system is served via a FastAPI endpoint and accepts user queries to retrieve relevant responses from the dataset.

#### Part 2: Building an Agent with Smart Actions

The RAG system is extended to include an intelligent agent capable of performing additional actions. The agent decides when to call the vector database based on user input and can trigger two additional tools depending on the user's query:

1.  **Sound Wave Simulator 🎵**: Generates a graphical sound wave based on user-defined parameters such as amplitude, time, and frequency. The prompt is parsed using OpenAI's API to extract these parameters from the user's input. This feature is controlled by enabling the **"Wave Generator"** checkbox in the frontend.
    
2.  **YouTube Video Search 🎥**: Fetches two YouTube videos related to the user's query when the **"YouTube Search"** extension is enabled in the frontend.
    

> **Note:** The Sound Wave Simulator and YouTube Search tools cannot be enabled simultaneously.

#### Part 3: Adding Voice to the Agent (Bonus)

Voice functionality has been added to the agent, allowing users to listen to the generated answers. Additionally, users can translate the generated answers into their preferred language. This feature uses Sarvam's AI API for both translation and voice output.

## Tools and Libraries Used

-   **Python Libraries**: FastAPI, LangChain, PyPDFLoader, Chroma, OpenAI, Matplotlib, and Sarvam AI API.
-   **Frontend**: A simple interface provided to allow users to interact with the system by enabling/disabling extensions and submitting queries.
-   **Backend**: All queries are handled using a FastAPI backend, and responses are generated based on user input and the selected features.

## Features

### RAG System

-   Retrieves and processes data from NCERT PDF texts.
-   Leverages OpenAI’s language model for query understanding and response generation.

### Agent with Smart Tools

#### Sound Wave Simulator 🎵

-   Generates graphical plots of sound waves based on user prompts.

#### YouTube Video Search 🎥

-   Fetches relevant YouTube videos based on the user's query.

### Translation and Voice 🗣️

-   Users can translate the generated text into multiple languages.
-   Provides a voice-enabled response using Sarvam’s AI API for text-to-speech.

## How to Run the Project

### Prerequisites

-   **Python 3.x**
    
-   Install required dependencies:
    
    bash
    
    Copy code
    
    `pip install -r requirements.txt` 
    

### Setting up API Keys

#### OpenAI and Sarvam AI API Keys

1.  **Environment Variables**: It's recommended to set your OpenAI and Sarvam API keys in environment variables, as the `back.py` code fetches API keys from environment variables.
    
    python
    
    Copy code
    
    `OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")` 
    
2.  **Direct Assignment**: If setting environment variables is not feasible, you can directly assign the API keys in the `back.py` code.
    
    python
    
    Copy code
    
    `OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
    SARVAM_API_KEY = "YOUR_SARVAM_API_KEY"` 
    

### Running the Code

1.  **Check Port Availability**: Ensure your `localhost` is free and port `8000` is not being used by any other program.
    
2.  **Start Backend Server**:
    
    -   Open a command prompt in the directory where the code files are located.
        
    -   Run the following command:
        
        bash
        
        Copy code
        
        `python back.py` 
        
3.  **Start Frontend Interface**:
    
    -   Open another command prompt window.
        
    -   Run the following command:
        
        bash
        
        Copy code
        
        `streamlit run front.py` 
        
4.  **Access the Frontend**:
    
    -   The frontend should automatically open in your browser.
    -   If not, check the command prompt for the frontend URL and open it manually.
5.  **Place NCERT PDF**:
    
    -   **Important**: Place the downloaded NCERT PDF in the same folder as `back.py` and `front.py`.
    -   Rename the PDF to `Sound.pdf`.

## Using the Features

### Enabling Extensions

-   **Note**: Only one extension can be enabled at a time.
    
-   In the frontend, select either:
    
    -   **Enable YouTube Search**
    -   **Enable Wave Generator**

### Using YouTube Search

1.  **Enable YouTube Search** in the frontend.
2.  Enter your prompt.
3.  The model will generate a meaningful title from your prompt and search YouTube for related videos.
4.  The videos will be embedded below your generated answer.

### Using Wave Generator

1.  **Enable Wave Generator** in the frontend.
    
2.  Enter prompts specifying wave parameters, such as:
    
    -   `generate sound wave using amplitude as 2, frequency as 23Hz, and time as 2 seconds`
    -   `create a sound wave having freq=23, amp=3, t=0.5`
    -   `simulate wave using t=1 sec, a=3, f=12`
3.  A graph of the sound wave will be generated using the parameters you've defined.
    

## Troubleshooting

-   **Port Issues**: If the code doesn't work initially, the default port `8000` might be in use. Change the port to a different number in both `back.py` and `front.py`.
    
    -   **In `back.py`**:
        
        python
        
        Copy code
        
        `if __name__ == "__main__":
            uvicorn.run(app, host="localhost", port=YOUR_PORT_NUMBER)` 
        
    -   **In `front.py`**:
        
        python
        
        Copy code
        
        `BACKEND_URL = "http://localhost:YOUR_PORT_NUMBER"` 
        
    -   Replace `YOUR_PORT_NUMBER` with your chosen port number in both files.
        
-   **API Key Errors**: Ensure that your API keys are correctly set either in environment variables or directly in the code.
    
-   **Missing PDF File**: Confirm that `Sound.pdf` is placed in the correct directory and is properly named.