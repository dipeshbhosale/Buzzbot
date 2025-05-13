import os
import json
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import urllib.parse
import threading # Use threading for non-blocking I/O/tasks
import concurrent.futures # For managing threads
import PyPDF2 # Explicitly import PyPDF2 now that async wrapper is removed
import time  # Add this import near the top with other imports
import streamlit.components.v1 as components
import numpy as np # For OpenCV frame manipulation
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_ENABLED = True
except ImportError:
    CLIP_ENABLED = False
try:
    from ultralytics import YOLO
    YOLO_ENABLED = True
except ImportError:
    YOLO_ENABLED = False
try:
    import openai
    from PIL import Image
    import base64
    from io import BytesIO
    OPENAI_ENABLED = True
    PILLOW_ENABLED = True
except ImportError:
    OPENAI_ENABLED = False
    PILLOW_ENABLED = False

try:
    import cv2
except ImportError:
    cv2 = None
# Try importing docx and pandas, but handle ImportError gracefully
try:
    import docx
except ImportError:
    docx = None
try:
    import pandas as pd
except ImportError:
    pd = None

# Import RAG and Chat History Managers (assuming they are well-behaved)
from rag_utils import RAGManager
from chat_history_manager import get_chat_history, create_session_id

# Try importing speech recognition modules and pyttsx3
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False
    # Warning will be displayed in the sidebar if disabled

# Ensure session state is initialized before any access
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = create_session_id()
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'voice_input_active' not in st.session_state:
    st.session_state['voice_input_active'] = False
if 'selected_theme' not in st.session_state:
    st.session_state['selected_theme'] = "Light"
if 'session_state_initialized' not in st.session_state:
    st.session_state['session_state_initialized'] = True

# Must be the first Streamlit command after session state initialization
st.set_page_config(
    page_title="Buzzbot - Your AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure required dependencies are installed
required_packages = {
    'python-docx': docx,
    'pandas': pd,
    'PyPDF2': 'pdf_support',
    'nest_asyncio': 'async_support', # This might be a placeholder, nest_asyncio is not directly used for core features
    'opencv-python': cv2, # Still useful for image processing even if not for direct capture
    'streamlit-webrtc': WEBRTC_ENABLED, # Use the boolean flag
    'av': WEBRTC_ENABLED, # Use the boolean flag, av is a dependency of streamlit-webrtc
    'ultralytics': YOLO_ENABLED,
    'openai': OPENAI_ENABLED, # Pillow is a dependency for OpenAI vision part too
    'Pillow': PILLOW_ENABLED,
}

missing_packages = []
for package, module in required_packages.items():
    if module is None:
        missing_packages.append(f"pip install {package}")



if missing_packages:
    st.sidebar.warning("Missing packages. Install with:\n" + "\n".join(missing_packages))

# --- Configuration and Setup ---

# Load environment variables from .env file - this is fast
load_dotenv()

def get_env_variable(key: str, default=None) -> str | None:
    """Retrieves environment variable, displays error if missing."""
    value = os.environ.get(key, default)
    if not value:
        st.sidebar.error(f"Environment variable '{key}' is missing. Please add it to your .env file.")
    return value

# USE st.cache_resource TO CACHE EXPENSIVE OBJECT INITIALIZATION
# These objects are created once per session and reused across reruns
@st.cache_resource
def get_groq_client(api_key_val: str | None) -> Groq | None:
    """Initializes and caches the Groq client."""
    if not api_key_val:
        return None
    try:
        return Groq(api_key=api_key_val)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Groq client: {e}")
        return None

@st.cache_resource
def get_rag_manager() -> RAGManager | None:
    """Initializes and caches the RAG Manager."""
    try:
        return RAGManager()
    except Exception as e:
        st.sidebar.error(f"Failed to initialize RAGManager: {e}")
        return None

@st.cache_resource
def get_pyttsx3_engine_cached() -> pyttsx3.Engine | None:
    """Initializes and returns a pyttsx3 engine, cached."""
    if not VOICE_ENABLED:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 1.0)
        return engine
    except Exception as e:
        print(f"Failed to initialize cached text-to-speech engine: {e}")
        return None

@st.cache_resource
def get_yolo_model() -> YOLO | None:
    """Loads and caches the YOLO model."""
    if not YOLO_ENABLED:
        return None
    try:
        model = YOLO("yolov8n.pt")  # Or yolov8s.pt, yolov8m.pt for different sizes/accuracies
        return model
    except Exception as e:
        st.sidebar.error(f"Failed to load YOLO model: {e}")
        return None

# Move session state initialization to the very beginning, right after imports
def initialize_session_state():
    """Initializes necessary session state variables if not already present."""
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = create_session_id()
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'voice_input_active' not in st.session_state:
        st.session_state['voice_input_active'] = False
    if 'selected_theme' not in st.session_state:
        st.session_state['selected_theme'] = "Light"
    if 'session_state_initialized' not in st.session_state:
        st.session_state['session_state_initialized'] = True

# Load environment variables and initialize API clients
api_keys = {
    "GROQ_API_KEY": None,
    "PIXABAY_API_KEY": None,
    "OPENAI_API_KEY": None,
}
api_keys["GROQ_API_KEY"] = get_env_variable("GROQ_API_KEY") # Recommended: Load from .env
api_keys["PIXABAY_API_KEY"] = get_env_variable("PIXABAY_API_KEY") # Recommended: Load from .env

# Load OpenAI API key from environment variable for better security
api_keys["OPENAI_API_KEY"] = get_env_variable("OPENAI_API_KEY")

groq_client = get_groq_client(api_keys["GROQ_API_KEY"])
rag_manager = get_rag_manager()
yolo_model_instance = get_yolo_model()

# Initialize TTS engine if voice is enabled
cached_tts_engine = None
if VOICE_ENABLED:
    cached_tts_engine = get_pyttsx3_engine_cached()
    if cached_tts_engine is None:
        # If engine failed to init, effectively disable voice output
        VOICE_ENABLED = False

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes necessary session state variables if not already present."""
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = create_session_id()
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'voice_input_active' not in st.session_state:
        st.session_state['voice_input_active'] = False
    if 'selected_theme' not in st.session_state:
        st.session_state['selected_theme'] = "Light"
    if 'session_state_initialized' not in st.session_state:
        st.session_state['session_state_initialized'] = True


# --- Theme Application ---
# This function applies CSS, which is fast.
def apply_theme(theme: str):
    """Applies a custom theme using markdown CSS."""
    themes = {
        "Dark": {
            "background": "#0E1117", "text": "#FAFAFA", "sidebar_bg": "#1E1E1E",
            "button_bg": "#007BFF", "button_text": "white", "chat_user_bg": "#2E2E2E",
            "chat_bot_bg": "#1A3A50"
        },
        "Light": {
            "background": "#FFFFFF", "text": "#000000", "sidebar_bg": "#F0F2F6",
            "button_bg": "#4CAF50", "button_text": "white", "chat_user_bg": "#E8E8E8",
            "chat_bot_bg": "#D1E7DD"
        }
    }
    selected_theme = themes.get(theme, themes["Light"])

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {selected_theme['background']}; color: {selected_theme['text']}; }}
        .stSidebar {{ background-color: {selected_theme['sidebar_bg']}; color: {selected_theme['text']}; }}
        .stChatMessage {{ border-radius: 10px; padding: 10px; margin-bottom: 10px; }}
        .stChatMessage.stChatUser {{ background-color: {selected_theme['chat_user_bg']}; }}
        .stChatMessage.stChatAssistant {{ background-color: {selected_theme['chat_bot_bg']}; }}
        .stChatMessage p, .stChatMessage a, .stChatMessage li, .stChatMessage ul {{ color: {selected_theme['text']} !important; }}
        .stChatMessage a {{ color: #007BFF !important; }}
        div.stChatInput {{ background-color: {selected_theme['sidebar_bg']}; padding: 10px; border-top: 1px solid #ccc; }}
        div.stChatInput > label > div > div > textarea {{ background-color: {selected_theme['background']}; color: {selected_theme['text']}; }}
        .stButton button {{ background-color: {selected_theme['button_bg']}; color: {selected_theme['button_text']}; border-radius: 5px; padding: 8px 15px; border: none; cursor: pointer; }}
        .stButton button:hover {{ opacity: 0.9; }}
        @keyframes colorChange {{ 0% {{ color: #4CAF50; }} 25% {{ color: #2196F3; }} 50% {{ color: #FF9800; }} 75% {{ color: #E91E63; }} 100% {{ color: #4CAF50; }} }}
        @keyframes rotate {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
        .rotating-robot {{ display: inline-block; animation: rotate 2s linear infinite; }}
        .color-changing-title {{ font-size: 3em; font-weight: bold; text-align: center; animation: colorChange 5s infinite; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- API Interaction Functions ---

# Use a thread pool executor for non-CPU intensive blocking tasks like network calls
# Max workers can be tuned based on expected concurrent users and task nature
thread_pool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# Function to submit a task to the thread pool
def run_in_thread(func, *args, **kwargs):
    """Runs a function in a thread and returns a Future."""
    return thread_pool_executor.submit(func, *args, **kwargs)

def call_groq_api(user_message: str) -> str:
    """Calls the Groq API for chatbot response with RAG enhancement."""
    if groq_client is None:
        return "Error: API client not initialized. Check your API key."
    if rag_manager is None:
        return "Error: RAG manager not initialized."

    # Get chat history for context
    chat_history = get_chat_history(st.session_state.session_id)
    
    # Use messages attribute instead of get_recent_messages
    recent_messages = chat_history.messages[-20:] if chat_history.messages else []  # Get last 5 messages
    
    # Create augmented prompt with both RAG and chat history context
    conversation_context = "\n".join([
        f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
        for msg in recent_messages
    ])
    
    augmented_message = f"""
Previous conversation:
{conversation_context}

Knowledge base context:
{rag_manager.get_augmented_prompt(user_message)}

Current message:
{user_message}
"""

    messages = [{"role": "system", "content": "You are BuzzBot. Keep responses very brief and direct."}]
    # Prepare messages for the API, including a brief history for context
    history_for_api = [
        {"role": "user" if role == "User" else "assistant", "content": content}
        for role, content in st.session_state.chat_history if role in ["User", "BuzzBot"]
    ]
    # Limit history to avoid hitting context window limits and potentially speed up processing
    messages.extend(history_for_api[-6:]) # Add last 6 messages for context

    messages.append({"role": "user", "content": augmented_message})

    try:
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            timeout=60 # Add a timeout to the API call itself
        )
        if response and response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            return "I'm sorry, I couldn't process your request. Please try again."
    except Exception as e:
        print(f"Error during Groq API call: {e}") # Log error server-side
        return f"Error: An API error occurred: {e}"

def call_image_search_api(query: str) -> str:
    """Generates a Google Image Search URL."""
    search_url = f"https://www.google.com/search?tbm=isch&q={urllib.parse.quote_plus(query)}"
    return search_url

@st.cache_data(ttl=3600) # Cache news results for 1 hour - GOOD practice
def call_news_api(location: str) -> str:
    """Fetches recent news headlines for a location using Google News RSS."""
    try:
        rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(location)}&hl=en&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(rss_url, headers=headers, timeout=15) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")
        summaries = []
        now = datetime.utcnow()

        for item in items:
            title = item.title.text if item.title else "No Title"
            link = item.link.text if item.link else "#"
            description_soup = BeautifulSoup(item.description.text, "html.parser") if item.description else None
            desc = description_soup.get_text().strip() if description_soup else "No Description"
            pub_date_str = item.pubDate.text if item.pubDate else None

            pub_dt = None
            if pub_date_str:
                try:
                    pub_dt = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
                except (ValueError, TypeError):
                     print(f"Could not parse date '{pub_date_str}' for news item.")
                     pub_dt = None
                except Exception as e:
                     print(f"Unexpected error parsing date '{pub_date_str}': {e}")
                     pub_dt = None

            if pub_dt and (now - pub_dt > timedelta(days=1)):
                continue

            summaries.append(f"- **{title}**\n  {desc}\n  [Read more]({link})")

            if len(summaries) >= 3:
                break

        return "\n\n".join(summaries) if summaries else f"No recent news found for '{location}'."
    except requests.exceptions.Timeout:
         print(f"News API request timed out for location '{location}'.")
         return f"Error fetching news: Request timed out."
    except requests.exceptions.RequestException as e:
        print(f"Network or HTTP error fetching news for location '{location}': {e}")
        return f"Error fetching news: Could not connect or retrieve news feed."
    except Exception as e:
        print(f"An error occurred while processing news feed for location '{location}': {e}")
        return f"Error fetching news: An unexpected error occurred."


# --- Voice Input/Output Functions ---

def listen_for_voice() -> str | None:
    """Listens for voice input using SpeechRecognition."""
    if not VOICE_ENABLED:
        return None

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_adjustment_damping = 0.15
    recognizer.dynamic_energy_adjustment_ratio = 1.5
    recognizer.pause_threshold = 0.8
    recognizer.non_speaking_duration = 0.5

    status_placeholder = st.empty()

    try:
        with sr.Microphone() as source:
            status_placeholder.info("Adjusting for ambient noise... Please be quiet.")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            status_placeholder.info("🎤 Listening... Speak now!")
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=5)
            status_placeholder.info("Processing audio...")
            with st.spinner("Transcribing speech..."):
                text = recognizer.recognize_google(audio, language='en-US', pfilter=0, show_all=False)
            status_placeholder.empty()
            return text

    except sr.WaitTimeoutError:
        status_placeholder.warning("Listening timed out. No speech detected.")
    except sr.UnknownValueError:
        status_placeholder.warning("Could not understand the audio. Please speak clearly.")
    except sr.RequestError as e:
        status_placeholder.error(f"Could not request results from speech recognition service; {e}")
    except Exception as e:
        status_placeholder.error(f"An unexpected error occurred during voice input: {e}")

    status_placeholder.empty()
    return None

def speak_response_threaded(text: str):
    """Speaks the given text using pyttsx3 in a separate thread."""
    if not VOICE_ENABLED or cached_tts_engine is None:
        return

    engine = cached_tts_engine

    def run_speak():
        """Helper function to run the speaking task in a thread."""
        try:
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Threaded speech output error: {str(e)}")

    thread = threading.Thread(target=run_speak, daemon=True)
    thread.start()


# --- File Processing Functions ---

def process_pdf_file(uploaded_file) -> str | None:
    """Process PDF file and extract text safely."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        extracted_text = []

        progress_text = "Reading PDF pages..."
        my_bar = st.progress(0, text=progress_text)

        for page_num in range(len(reader.pages)):
            try:
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    extracted_text.append(text)
            except Exception as e:
                print(f"Error reading PDF page {page_num}: {e}")
                pass
            progress = (page_num + 1) / len(reader.pages)
            my_bar.progress(progress, text=progress_text)

        my_bar.empty()

        if not extracted_text:
            print("Could not extract any text from the PDF.")
            return "Could not extract any text from the PDF."

        return "\n".join(extracted_text)
    except ImportError:
        return "Error: PyPDF2 not installed. Cannot read PDF."
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return f"Error: An error occurred while reading the PDF: {e}"


def process_docx_file(uploaded_file) -> str | None:
    """Process DOCX file and extract text."""
    if docx is None:
        return "Error: python-docx not installed. Cannot read DOCX."
    try:
        doc = docx.Document(uploaded_file)
        extracted_text = "\n".join([para.text for para in doc.paragraphs])
        if not extracted_text:
             print("Could not extract any text from the DOCX.")
             return "Could not extract any text from the DOCX."
        return extracted_text
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return f"Error: An error occurred while reading the DOCX: {e}"

def process_csv_file(uploaded_file) -> str | None:
    """Process CSV file and extract text representation."""
    if pd is None:
        return "Error: pandas not installed. Cannot read CSV."
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            print("Could not read CSV or it was empty.")
            return "Could not read CSV or it was empty."

        return df.to_string()

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return f"Error: An error occurred while reading the CSV: {e}"


def process_uploaded_file_content(uploaded_file) -> str | None:
    """Handles reading content based on file type, displaying errors."""
    file_contents = None
    file_type = uploaded_file.type
    file_name = uploaded_file.name

    try:
        if file_type == "text/plain":
            file_contents = uploaded_file.read().decode("utf-8")
        elif file_type == "application/json":
            data = json.load(uploaded_file)
            file_contents = json.dumps(data, indent=2)
        elif file_type == "application/pdf":
            file_contents = process_pdf_file(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
             file_contents = process_docx_file(uploaded_file)
        elif file_type == "text/csv":
             file_contents = process_csv_file(uploaded_file)
        else:
            file_contents = f"Error: Unsupported file type: {file_type}"

    except Exception as e:
        print(f"An unexpected error occurred during initial file read for {file_name}: {e}")
        file_contents = f"Error: An unexpected error occurred during file reading: {e}"

    if file_contents is not None and isinstance(file_contents, str) and file_contents.startswith("Error:"):
        st.error(f"Failed to process file '{file_name}': {file_contents[7:]}")
        return None

    if file_contents is None:
        st.warning(f"No content extracted from file '{file_name}'.")

    return file_contents


# --- Core Chat Logic ---
def chatbot_response(user_input: str) -> str:
    """Generates a chatbot response based on input, handling special commands."""
    user_input_lower = user_input.lower().strip()

    if "session_id" not in st.session_state:
        initialize_session_state()
        if "session_id" not in st.session_state:
            print("Error: Critical session state 'session_id' is missing after re-initialization.")
            st.error("A critical application error occurred. Please refresh the page.")
            return "Error: Unable to process request due to application state error."

    chat_history = get_chat_history(st.session_state.session_id)
    if "my name is" in user_input_lower:
        name = user_input.split("my name is")[-1].strip().title()
        if name:
            chat_history.add_user_message(f"My name is {name}")
            chat_history.add_ai_message(f"Nice to meet you, {name}!")
            return f"Nice to meet you, {name}!"
        return "Okay, I noted your name!"
    elif "what is my name" in user_input_lower:
        for message in chat_history.messages:
            if message.role == "User" and "my name is" in message.content.lower():
                name = message.content.split("my name is")[-1].strip().title()
                return f"Your name is {name}."
        return "I don't have your name stored yet. You can tell me by saying 'My name is [Your Name]'."

    return call_groq_api(user_input)


# --- Input Processing and History Management ---

response_placeholder = st.empty()

def process_input(user_input: str, action_type: str, is_voice_input: bool):
    """Processes user input based on selected action type."""
    if not user_input:
        return

    if "session_id" not in st.session_state or "chat_history" not in st.session_state:
        initialize_session_state()

    # Get chat history manager instance
    chat_history = get_chat_history(st.session_state.session_id)
    
    # Add user message to both display history and chat history manager
    st.session_state.chat_history.append(("User", user_input))
    chat_history.add_user_message(user_input)

    response_content = ""

    with st.spinner(f"Processing '{action_type}' request..."):
        try:
            if action_type == "Chat":
                future = run_in_thread(chatbot_response, user_input)
                response_content = future.result()

            elif action_type == "Image Search":
                response_content = call_image_search_api(user_input)
                st.session_state.chat_history.append(("Image Search Results", response_content))

            elif action_type == "News":
                future = run_in_thread(call_news_api, user_input)
                response_content = future.result()

            elif action_type == "Movie Search":
                movie_prompt = (
                    f"Please list exactly 10 movie recommendations based on or related to '{user_input}'. "
                    "Provide only the movie titles, one per line, numbered 1 through 10."
                )
                future = run_in_thread(call_groq_api, movie_prompt)
                movie_titles_response = future.result()

                if movie_titles_response and not movie_titles_response.startswith("Error:"):
                    titles = []
                    for line in movie_titles_response.splitlines():
                        cleaned_line = line.strip()
                        if cleaned_line:
                            if cleaned_line[0].isdigit() and '.' in cleaned_line:
                                parts = cleaned_line.split('.', 1)
                                title = parts[1].strip() if len(parts) > 1 else ''
                            elif cleaned_line.startswith('-'):
                                title = cleaned_line[1:].strip()
                            else:
                                title = cleaned_line
                            if title:
                                titles.append(title)
                    formatted_recommendations = []
                    for i, title in enumerate(titles[:10]):
                        encoded_title = urllib.parse.quote_plus(title)
                        google_link = f"https://www.google.com/search?q={encoded_title}+movie"
                        formatted_recommendations.append(f"{i+1}. {title} - [Google Search Link]({google_link})")
                    if formatted_recommendations:
                        response_content = "Here are some movie recommendations:\n\n" + "\n".join(formatted_recommendations)
                    else:
                        response_content = f"Could not find clear movie recommendations for '{user_input}'."
                else:
                    response_content = movie_titles_response

            else:
                response_content = f"Error: Unsupported action type: {action_type}"

        except Exception as e:
            print(f"An unexpected error occurred during process_input execution: {e}")
            response_content = f"Error: An unexpected error occurred: {e}"

    # After getting response_content, add to both histories
    if not (action_type == "Image Search") and \
       not response_content.startswith("Error: Unable to process request due to application state error."):
        st.session_state.chat_history.append(("BuzzBot", response_content))
        chat_history.add_ai_message(response_content)

    if is_voice_input and response_content and VOICE_ENABLED and \
       st.session_state.chat_history[-1][0] == "BuzzBot" and \
       not response_content.startswith("Error:"):
        speak_response_threaded(response_content)

    st.rerun()


# --- UI Components ---
def display_chat_history():
    """Displays the chat history using st.chat_message."""
    for speaker, message in st.session_state.chat_history:
        if speaker == "User":
            with st.chat_message("user"):
                st.markdown(message)
        elif speaker == "BuzzBot":
            with st.chat_message("assistant"):
                st.markdown(message)
        elif speaker == "Image Search Results":
            with st.chat_message("assistant"):
                 st.markdown(message)


# Placeholder for a function that would use an image captioning model
def get_image_description_from_image_data(image_data_or_frame, source_type: str) -> str:
    """
    Placeholder: Simulates getting a description from image data (OpenCV frame).
    In a real implementation, this would involve sending the image_file_object's
    content to an image captioning model.
    'source_type' can be "live view" or "captured picture".
    """
    if image_data_or_frame is None:
        return "Error: No image data received."

    # This function now always expects an OpenCV frame (NumPy array)
    is_cv_frame = isinstance(image_data_or_frame, np.ndarray)
    
    # Simulate dynamic content or detected movement
    # In a real scenario, you'd analyze frame_data_url for actual movement.
    current_time_str = datetime.now().strftime("%H:%M:%S")
    
    # Cycle through a few mock descriptions to simulate change
    if 'mock_scene_state' not in st.session_state:
        st.session_state.mock_scene_state = 0
    
    descriptions = [
        f"At {current_time_str}, the {source_type} shows a scene with a desk and a chair. A coffee mug is on the desk.",
        f"At {current_time_str}, the {source_type} indicates a person might have just left the room. The chair is slightly pushed out.",
        f"At {current_time_str}, the {source_type} captured a window with daylight outside. The coffee mug remains.",
        f"At {current_time_str}, this {source_type} shows a laptop on the desk, which appears to be closed.",
        f"At {current_time_str}, the {source_type} caught a glimpse of a bookshelf in the background with several books."
    ]
    
    description = descriptions[st.session_state.mock_scene_state % len(descriptions)]
    st.session_state.mock_scene_state += 1
    
    return description

# Custom video processor for streamlit-webrtc
class CameraVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def create_camera_ui():
    """Creates the camera UI component using streamlit-webrtc for live feed."""
    st.header("📹 Live Camera & Snapshot Analysis")
    
    # Initialize states
    if 'camera_chat' not in st.session_state:
        st.session_state.camera_chat = []
    if 'captured_frame' not in st.session_state: # For the snapshot from webrtc
        st.session_state.captured_frame = None
    if 'analyzed_image_with_boxes' not in st.session_state: # For YOLO output image (RGB)
        st.session_state.analyzed_image_with_boxes = None
    if 'detected_labels' not in st.session_state: # For YOLO labels
        st.session_state.detected_labels = None
    if 'openai_vision_response' not in st.session_state: # For OpenAI Vision text response
        st.session_state.openai_vision_response = None
    if 'webrtc_ctx' not in st.session_state:
        st.session_state.webrtc_ctx = None # To store webrtc_streamer context

    # Create two columns: camera and chat
    cam_col, chat_col = st.columns([0.7, 0.3])
    
    with cam_col:
        st.subheader("Live Camera Feed")
        if not WEBRTC_ENABLED:
            st.error("WebRTC components are not available. Please install `streamlit-webrtc` and `av`. Camera tab will not function.")
        else:
            ctx = webrtc_streamer(
                key="live_camera_streamer",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=CameraVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            # st.session_state.webrtc_ctx = ctx # Storing full context might not be necessary if only processor is used

            if ctx.video_processor:
                if st.button("📸 Capture Frame", key="capture_frame_button", use_container_width=True):
                    if ctx.video_processor.latest_frame is not None:
                        st.session_state.captured_frame = ctx.video_processor.latest_frame.copy()
                        st.session_state.analyzed_image_with_boxes = None # Clear previous analysis
                        st.session_state.detected_labels = None
                        st.session_state.openai_vision_response = None
                        
                        current_time = datetime.now()
                        primary_analysis_done = False # Flag to track if primary AI observation is set

                        # --- OpenAI GPT-4 Vision Analysis ---
                        if OPENAI_ENABLED and PILLOW_ENABLED and cv2 and api_keys.get("OPENAI_API_KEY"):
                            try:
                                with st.spinner("Analyzing with OpenAI GPT-4 Vision..."):
                                    # Convert captured OpenCV BGR frame to RGB PIL Image
                                    img_pil_rgb = Image.fromarray(cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
                                    
                                    buffered = BytesIO()
                                    img_pil_rgb.save(buffered, format="PNG") # PNG is good for vision
                                    img_bytes = buffered.getvalue()
                                    img_base64 = base64.b64encode(img_bytes).decode()

                                    # Initialize OpenAI client (New SDK style)
                                    client = openai.OpenAI(api_key=api_keys["OPENAI_API_KEY"])
                                    response = client.chat.completions.create( # Use the client instance
                                        model="gpt-4o", # Updated to the latest vision-capable model
                                        messages=[
                                            {"role": "user", "content": [
                                                {"type": "text", "text": "Describe this image concisely. What are the main elements or actions?"},
                                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                                            ]}
                                        ], # Corrected: messages is a list of dicts
                                        max_tokens=500 # As per your new snippet
                                    )
                                    vision_text_response = response.choices[0].message.content
                                    st.session_state.openai_vision_response = vision_text_response
                                    ai_observation_message = f"OpenAI Vision analysis: {vision_text_response}"
                                    st.session_state.camera_chat.append(("AI", ai_observation_message, current_time))
                                    primary_analysis_done = True
                            except Exception as e:
                                # More specific error handling for OpenAI API
                                if isinstance(e, openai.APIError):
                                    if 'insufficient_quota' in str(e).lower(): # Check for quota issue
                                        st.error("OpenAI API Error: You've run out of OpenAI API credits. Please check your billing or try later.")
                                        st.session_state.openai_vision_response = "OpenAI Vision Error: Insufficient quota."
                                    else:
                                        st.error(f"OpenAI API Error: {e}")
                                        st.session_state.openai_vision_response = f"OpenAI Vision Error: {e}"
                                else: # Catch other potential exceptions during the process
                                    st.error(f"An unexpected error occurred during OpenAI Vision analysis: {e}")
                                    st.session_state.openai_vision_response = f"OpenAI Vision Error: Unexpected error - {e}"
                        
                        # --- YOLO Analysis (runs for visual output, and for chat if OpenAI failed) ---
                        if yolo_model_instance and cv2 :
                            with st.spinner("Running YOLO object detection..."):
                                img_rgb_for_yolo = cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB)
                                results_yolo = yolo_model_instance(img_rgb_for_yolo)[0]
                                st.session_state.analyzed_image_with_boxes = results_yolo.plot() # This is RGB
                                st.session_state.detected_labels = [yolo_model_instance.names[int(cls)] for cls in results_yolo.boxes.cls]
                                
                            if not primary_analysis_done: # If OpenAI didn't set the primary message
                                h, w, _ = st.session_state.captured_frame.shape
                                labels_str = ", ".join(st.session_state.detected_labels) if st.session_state.detected_labels else "No objects detected"
                                ai_observation_message_yolo = f"YOLO detected in picture ({w}x{h}): {labels_str}."
                                st.session_state.camera_chat.append(("AI", ai_observation_message_yolo, current_time))
                                primary_analysis_done = True
                        
                        # --- Fallback to Mock Analysis ---
                        if not primary_analysis_done:
                            with st.spinner("Analyzing captured picture (mock)..."):
                                visual_description = get_image_description_from_image_data(st.session_state.captured_frame, source_type="captured picture")
                                if visual_description and not visual_description.startswith("Error:"):
                                    h, w, _ = st.session_state.captured_frame.shape
                                    ai_observation_message_mock = f"Mock analysis of picture ({w}x{h}): {visual_description}"
                                    st.session_state.camera_chat.append(("AI", ai_observation_message_mock, current_time))
                                else:
                                    st.session_state.camera_chat.append(("System", f"Mock analysis failed: {visual_description}", current_time))

                        st.rerun() # Rerun to update chat and display image
                    else:
                        st.warning("No frame available from camera to capture.")
            elif ctx.state.playing:
                 st.info("Live camera feed is active. Click 'Capture Frame' to take a snapshot.")
            else:
                st.info("Camera is not active. The component should start it automatically if permissions are granted.")

        # Display captured picture if available
        if st.session_state.captured_frame is not None and cv2:
            st.image(
                cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB), 
                caption="Last Captured Picture (Original)", 
                use_container_width=True
            )
        
        # Display YOLO analyzed image if available
        if st.session_state.analyzed_image_with_boxes is not None:
            st.image(st.session_state.analyzed_image_with_boxes, caption="Analyzed Picture (with Detections)", use_container_width=True)

        # Display OpenAI Vision response if available
        if st.session_state.openai_vision_response is not None:
            st.markdown("---")
            st.subheader("OpenAI Vision Analysis:")
            st.info(st.session_state.openai_vision_response)

        # Display YOLO detected labels if available
        if st.session_state.detected_labels is not None:
            if st.session_state.detected_labels:
                st.write("Detected Objects:", ", ".join(st.session_state.detected_labels))
            else:
                st.write("Detected Objects: None")

    with chat_col:
        st.subheader("Picture Discussion") 
        
        # Display camera chat history
        camera_chat_display_container = st.container(height=400) # Match main chat height
        with camera_chat_display_container:
            if 'camera_chat' in st.session_state and st.session_state.camera_chat:
                for speaker, content, timestamp in st.session_state.camera_chat:
                    if speaker == "User":
                        with st.chat_message("user"):
                            st.markdown(content)
                    elif speaker == "AI":
                        with st.chat_message("assistant"):
                            st.markdown(content)
                    elif speaker == "System":
                        # Using a different avatar for system messages for clarity
                        with st.chat_message("assistant", avatar="⚙️"): 
                            st.markdown(f"*{content}*")
            else:
                st.caption("No discussion about captured pictures yet.")
        
        # Input for camera-specific chat
        camera_user_query = st.chat_input("Ask about the captured picture...", key="cv_camera_chat_input_widget")
        
        if camera_user_query:
            query_time = datetime.now()
            st.session_state.camera_chat.append(("User", camera_user_query, query_time))
            
            # Get the latest AI observation (which will be about a captured picture)
            latest_observation = "the last analyzed frame or picture" # Default
            if 'camera_chat' in st.session_state and st.session_state.camera_chat:
                for speaker, content, _ in reversed(st.session_state.camera_chat):
                    if speaker == "AI" and \
                       (content.startswith("I see in the current view:") or content.startswith("I analyzed the captured picture:")):
                        latest_observation = content
                        break
            
            # Construct a prompt for the AI based on its last "visual" observation and the user's query
            prompt_for_ai = f"Context: You previously observed '{latest_observation}'. Now, the user asks: {camera_user_query}"
                        
            with st.spinner("Thinking about the picture..."):
                ai_response_to_camera = call_groq_api(prompt_for_ai)
            
            st.session_state.camera_chat.append(("AI", ai_response_to_camera, datetime.now()))
            st.rerun() # Rerun to display the new user message and AI response
            

# --- Main Application Flow ---
def main():
    """Main function to run the Streamlit application."""
    if not st.session_state.get('session_state_initialized', False):
        initialize_session_state()
    
    current_theme = st.session_state.get('selected_theme', 'Light')
    apply_theme(current_theme)

    st.markdown("<h1>Buzzbot <span class='rotating-robot'>🤖</span></h1>", unsafe_allow_html=True)

    st.sidebar.header("Theme")
    theme_options = ["Light", "Dark"]
    theme = st.sidebar.selectbox("🎨 Choose Theme", options=theme_options, key="theme_selectbox", index=theme_options.index(current_theme))
    if theme != current_theme:
        st.session_state.selected_theme = theme
        st.rerun()

    st.sidebar.header("Settings")
    action_options = ["Chat", "Image Search", "News", "Movie Search"]
    action_type = st.sidebar.radio("⚡ Select Action", options=action_options, key="action_radio")

    if not VOICE_ENABLED:
        st.sidebar.warning(
            "🎤 Voice features disabled. Install required packages: `pip install SpeechRecognition pyaudio pyttsx3` (and ensure a microphone is configured)"
        )
    elif cached_tts_engine is None:
         st.sidebar.warning("Failed to initialize Text-to-Speech engine. Voice output disabled.")

    if rag_manager is None:
         st.sidebar.warning("RAG Manager initialization failed. File upload features may not work.")
    if docx is None:
         st.sidebar.warning("`python-docx` not installed. DOCX uploads not supported. Install with `pip install python-docx`.")
    if pd is None:
         st.sidebar.warning("`pandas` not installed. CSV uploads not supported. Install with `pip install pandas`.")
    if cv2 is None:
        st.sidebar.warning("`opencv-python` not installed. Live camera features in the 'Camera' tab will not work. Install with `pip install opencv-python`.")
    if not WEBRTC_ENABLED:
        st.sidebar.error("`streamlit-webrtc` or `av` not installed. Camera tab will not function correctly. Install with `pip install streamlit-webrtc av`")
    if not YOLO_ENABLED:
        st.sidebar.warning("`ultralytics` not installed. YOLO object detection will not be available. Install with `pip install ultralytics`")
    if not OPENAI_ENABLED:
        st.sidebar.warning("`openai` library not installed. OpenAI Vision features will be disabled. Install with `pip install openai`")
    if not PILLOW_ENABLED:
        st.sidebar.warning("`Pillow` library not installed. Image processing for OpenAI Vision will be disabled. Install with `pip install Pillow`")


    chat_tab, camera_tab = st.tabs(["💬 Chat", "📹 Camera"])
    
    with chat_tab:
        chat_history_container = st.container(height=400)
        with chat_history_container:
            display_chat_history()

        user_query = st.chat_input("What can I help you with?", key="chat_input_main")

        col_voice, col_spacer = st.columns([1, 5])
        with col_voice:
            if VOICE_ENABLED:
                voice_button_label = (
                    "🎤 Start Listening" if not st.session_state.voice_input_active else "⏹️ Stop Listening"
                )
                voice_button_clicked = st.button(voice_button_label, key="voice_button")

                if voice_button_clicked:
                     st.session_state.voice_input_active = not st.session_state.voice_input_active
                     if st.session_state.voice_input_active:
                          voice_input = listen_for_voice()
                          st.session_state.voice_input_active = False
                          if voice_input:
                               st.info(f"You said: {voice_input}")
                               process_input(voice_input, action_type, is_voice_input=True)
                          else:
                               st.rerun()
                     else:
                          st.rerun()


        if user_query:
            st.session_state.voice_input_active = False
            process_input(user_query, action_type, is_voice_input=False)


        st.markdown("---")
        with st.expander("📂 Upload a File"):
            st.write("Upload a text-based file (TXT, JSON, CSV, PDF, DOCX) to process its content.")
            allowed_file_types = ["txt", "json"]
            if docx is not None: allowed_file_types.append("docx")
            if pd is not None: allowed_file_types.append("csv")
            allowed_file_types.append("pdf")

            uploaded_file = st.file_uploader(
                "Choose a file",
                type=allowed_file_types,
                key="file_uploader"
            )

            if uploaded_file is not None:
                file_details = {"name": uploaded_file.name, "type": uploaded_file.type, "size": uploaded_file.size}
                st.write("File Details:")
                st.json(file_details)

                with st.spinner(f"Reading and processing {uploaded_file.name}..."):
                     file_contents = process_uploaded_file_content(uploaded_file)

                if file_contents is not None and not isinstance(file_contents, str) or (isinstance(file_contents, str) and not file_contents.startswith("Error:")):

                    st.success("File content extracted successfully!")
                    st.subheader("Extracted Content Preview:")
                    if uploaded_file.type == "application/json":
                        try:
                            st.json(json.loads(file_contents))
                        except json.JSONDecodeError:
                            st.text_area(
                                "Content Preview",
                                str(file_contents)[:4000] + ("..." if len(str(file_contents)) > 4000 else ""),
                                height=300,
                            )
                    else:
                        st.text_area(
                            "Content Preview",
                            str(file_contents)[:4000] + ("..." if len(str(file_contents)) > 4000 else ""),
                            height=300,
                        )

                    if rag_manager is not None:
                        add_to_knowledge_base = st.button("Add to Knowledge Base", key="add_to_kb")
                        if add_to_knowledge_base:
                            with st.spinner("Adding content to knowledge base..."):
                                process_uploaded_file(file_contents)


                    process_file_content_button = st.button(
                        f"Process file content with Chat", key="process_file_button"
                    )
                    if process_file_content_button:
                        prompt_prefix = f"Analyze the content from {uploaded_file.name}:\n\n"
                        process_input(
                            prompt_prefix + str(file_contents), action_type="Chat", is_voice_input=False
                        )

                elif file_contents is not None and isinstance(file_contents, str) and file_contents.startswith("Error:"):
                     st.subheader("Extracted Content Preview:")
                     st.error(f"Could not display preview:\n{file_contents}")

    with camera_tab:
        create_camera_ui()


def process_uploaded_file(file_contents: str):
    """Process uploaded file contents and add to RAG knowledge base."""
    if rag_manager is None:
        st.error("RAG Manager is not initialized. Cannot add documents.")
        return

    chunks = [s.strip() for s in file_contents.split('.') if s.strip()]
    if chunks:
        try:
            rag_manager.add_documents(chunks)
            st.success("File content added to knowledge base successfully!")
        except Exception as e:
            st.error(f"Error adding documents to knowledge base: {e}")
    else:
        st.warning("No processable chunks found in the file content.")


if __name__ == "__main__":
    main()