import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO
import os
from IPython.display import Audio

BACKEND_URL = "http://localhost:8000"

def extract_video_id(video_url):
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
        return video_id
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[1].split("?")[0]
        return video_id
    elif "shorts//" in video_url:
        video_id = video_url.split("shorts/")[1].split("?")[0]
        return video_id
    else:
        return None

def display_youtube_video(video_id):
    if video_id:
        video_url = f"https://www.youtube.com/embed/{video_id}"
        st.video(video_url)
    else:
        st.error("Could not extract video ID.")

def is_sound_wave_query(query):
    sound_wave_keywords = ["sound wave", "generate sound", "create wave", "simulate wave", "wave generator"]
    return any(keyword in query.lower() for keyword in sound_wave_keywords)

def main():
    st.title("Interactive Learning Platform")

    st.sidebar.title("Settings")
    youtube_enabled = st.sidebar.checkbox("Enable YouTube Search", value=False)
    wave_enabled = st.sidebar.checkbox("Enable Wave Generator", value=False)

    ask_mode = st.sidebar.radio(
        "Select ask mode:",
        ("Ask Normally", "Ask only RAG")
    )
    
    endpoint = "/ask/" if ask_mode == "Ask Normally" else "/ask-only-rag"
    
    if youtube_enabled and wave_enabled:
        st.sidebar.warning("Please enable only one feature at a time.")
        youtube_enabled = False
        wave_enabled = False

    st.header("Ask a Question or Enter a Command")
    query = st.text_input("Enter your query:")

    if 'answer' not in st.session_state:
        st.session_state['answer'] = None
    if 'audio' not in st.session_state:
        st.session_state['audio'] = None
    if 'translated_text' not in st.session_state:
        st.session_state['translated_text'] = None

    if st.button("Submit"):
        if not query:
            st.warning("Please enter a query.")
        else:
            st.session_state['answer'] = None
            st.session_state['audio'] = None
            st.session_state['translated_text'] = None

            if is_sound_wave_query(query):
                if wave_enabled:
                    with st.spinner("Extracting parameters..."):
                        params_response = requests.post(f"{BACKEND_URL}/extract_parameters/", json={"question": query})
                    if params_response.status_code == 200:
                        params_data = params_response.json()
                        st.success("Extracted Parameters:")
                        st.write(params_data['result'])

                        try:
                            params = params_data['result'].split(', ')
                            amplitude = float(params[0].split(': ')[1])
                            frequency = float(params[1].split(': ')[1])
                            time = float(params[2].split(': ')[1])
                        except (IndexError, ValueError) as e:
                            st.error(f"Error parsing parameters: {str(e)}")
                            return

                        if(amplitude == 0 and frequency == 0 and time == 0):
                            st.error("Failed to generate sound wave as no parameters were provided or all parameters are 0.")
                            return

                        with st.spinner("Generating sound wave..."):
                            wave_response = requests.post(f"{BACKEND_URL}/simulate-sound-wave/", 
                                                          json={"frequency": frequency, "amplitude": amplitude, "time": time})

                        if wave_response.status_code == 200:
                            wave_data = wave_response.json()
                            img = base64.b64decode(wave_data["image"])
                            img = Image.open(BytesIO(img))
                            st.image(img, caption="Sound Wave Visualization")
                        else:
                            st.error(f"Failed to generate sound wave: {wave_response.status_code} - {wave_response.text}")
                    else:
                        st.error(f"Failed to extract parameters: {params_response.status_code} - {params_response.text}")
                else:
                    st.warning("The Wave Generator is currently disabled. Please enable it in the sidebar to generate sound waves.")
            else:
                with st.spinner("Fetching answer..."):
                    response = requests.post(f"{BACKEND_URL}{endpoint}", json={"question": query})
                if response.status_code == 200:
                    data = response.json()
                    st.session_state['answer'] = data['response']
                    st.session_state['source'] = data.get('source', 'rag')  # Default to 'rag' if source is not provided

                    st.subheader("Answer:")
                    st.write(f"Question: {data['question']}")
                    st.write(f"Answer: {data['response']}")

                    if youtube_enabled and st.session_state['source'] == 'rag' and data['response'].lower() != "i don't know":
                        with st.spinner("Generating YouTube search title..."):
                            title_response = requests.post(f"{BACKEND_URL}/generate_title/", json={"question": query})
                        if title_response.status_code == 200:
                            title_data = title_response.json()
                            generated_title = title_data['title']
                            st.success(f"Generated YouTube Search Title: {generated_title}")

                            with st.spinner("Searching YouTube..."):
                                search_response = requests.post(f"{BACKEND_URL}/search_youtube/", json={"query": generated_title})
                            if search_response.status_code == 200:
                                search_data = search_response.json()
                                video_urls = search_data.get("videos", [])
                                
                                if video_urls:
                                    st.subheader("Related YouTube Videos:")
                                    for video_url in video_urls:
                                        video_id = extract_video_id(video_url)
                                        if video_id:
                                            display_youtube_video(video_id)
                                        else:
                                            st.write(f"Could not extract video ID from URL: {video_url}")
                                else:
                                    st.write("No videos found for this query.")
                            else:
                                st.error(f"Failed to fetch videos: {search_response.status_code} - {search_response.text}")
                        else:
                            st.error(f"Failed to generate title: {title_response.status_code} - {title_response.text}")
                    elif youtube_enabled:
                        st.info("No YouTube videos displayed for this query as it was not answered using the document knowledge base.")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

    if st.session_state['answer']:
        st.subheader("Answer:")
        st.write(st.session_state['answer'])

        target_languages = {
            "Hindi": "hi-IN",
            "Bengali": "bn-IN",
            "Kannada": "kn-IN",
            "Malayalam": "ml-IN",
            "Marathi": "mr-IN",
            "Odia": "od-IN",
            "Punjabi": "pa-IN",
            "Tamil": "ta-IN",
            "Telugu": "te-IN",
            "Gujarati": "gu-IN"
        }
        selected_language = st.selectbox("Select target language for translation:", list(target_languages.keys()))

        if st.button("Translate"):
            with st.spinner("Translating..."):
                translation_response = requests.post(
                    f"{BACKEND_URL}/translate/",
                    json={"text": st.session_state['answer'], "target_language_code": target_languages[selected_language]}
                )
            if translation_response.status_code == 200:
                translation_data = translation_response.json()
                st.session_state['translated_text'] = translation_data['translated_text']
                st.subheader(f"Translated Answer ({selected_language}):")
                st.write(st.session_state['translated_text'])
            else:
                st.error(f"Failed to translate: {translation_response.status_code} - {translation_response.text}")

    if st.session_state['answer']:
        if st.button("Listen to Answer"):
            with st.spinner("Generating audio..."):
                tts_response = requests.post(f"{BACKEND_URL}/text-to-speech/", json={"text": st.session_state['answer']})
            if tts_response.status_code == 200:
                tts_data = tts_response.json()
                audio_base64 = tts_data.get("audio", "")
                if audio_base64:
                    try:
                        temp_audio_file = audio_base64
                        st.audio(temp_audio_file, format="audio/wav")
                        
                        os.remove(temp_audio_file)
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                else:
                    st.error("No audio data received from TTS service.")
            else:
                st.error(f"Failed to generate audio: {tts_response.status_code} - {tts_response.text}")

if __name__ == "__main__":
    main()