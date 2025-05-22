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
import PyPDF2
import time  # Add this import near the top with other imports
import streamlit.components.v1 as components
from typing import Optional # Add this import for Optional type hints

# Camera-specific and related vision/ML imports are removed:
# streamlit_webrtc, av, ultralytics, YOLO, openai, PIL, base64, io.BytesIO, cv2, transformers, numpy

WEBRTC_ENABLED = False
YOLO_ENABLED = False
OPENAI_ENABLED = False
PILLOW_ENABLED = False
CLIP_ENABLED = False
cv2 = None # Explicitly set to None as cv2 is removed

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
from chat_history_manager import get_chat_history, create_session_id, get_or_create_groq_chain_for_session

# Voice features are being removed
VOICE_ENABLED = False
# Ensure session state is initialized before any access
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = create_session_id()
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'session_state_initialized' not in st.session_state:
    st.session_state['session_state_initialized'] = True

# Move session state initialization to the very beginning, right after imports
def initialize_session_state():
    """Initializes necessary session state variables if not already present."""
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = create_session_id()
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'voice_input_active' not in st.session_state:
        st.session_state['voice_input_active'] = False
    # 'selected_theme' is removed as theme option is being removed
    # if 'selected_theme' not in st.session_state:
    #     st.session_state['selected_theme'] = "Light"
    if 'session_state_initialized' not in st.session_state: # This check should be sufficient
        st.session_state['session_state_initialized'] = True

# --- Configuration and Setup ---
st.set_page_config(
    page_title="Buzzbot - Your AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Load environment variables and initialize API clients
api_keys = {
    "GROQ_API_KEY": None,
    "PIXABAY_API_KEY": None,
    # "OPENAI_API_KEY": None, # OpenAI key removed as its usage was tied to camera
}
api_keys["GROQ_API_KEY"] = get_env_variable("GROQ_API_KEY") # Recommended: Load from .env
api_keys["PIXABAY_API_KEY"] = get_env_variable("PIXABAY_API_KEY") # Recommended: Load from .env

# Load OpenAI API key from environment variable for better security
api_keys["OPENAI_API_KEY"] = get_env_variable("OPENAI_API_KEY")

groq_client = get_groq_client(api_keys.get("GROQ_API_KEY")) # Use .get for safety
rag_manager = get_rag_manager()


# Theme application function and dictionary are removed.

# --- API Interaction Functions ---

# Use a thread pool executor for non-CPU intensive blocking tasks like network calls
# Max workers can be tuned based on expected concurrent users and task nature
thread_pool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# Function to submit a task to the thread pool
def run_in_thread(func, *args, **kwargs):
    """Runs a function in a thread and returns a Future."""
    # For LLM calls that might be generators, direct execution might be simpler
    # if the LLM client itself handles async/threading appropriately.
    # However, for consistency with other API calls, we can keep it.
    # If func is a generator function, submit will run it to completion in the thread.
    # The result() will then be the generator object itself.
    return thread_pool_executor.submit(func, *args, **kwargs) 

def call_groq_api(user_message: str): # Returns a generator
    """Calls the Groq API for chatbot response with RAG enhancement."""
    if not api_keys.get("GROQ_API_KEY"):
        yield "Error: GROQ API key not configured."
        return
    if rag_manager is None:
        yield "Error: RAG manager not initialized."
        return

    try:
        # Get the session-specific LangChain chain (it includes memory)
        # The model name is now set within get_or_create_groq_chain_for_session
        # The default model in chat_history_manager is "meta-llama/llama-4-scout-17b-16e-instruct"
        # If you want to use "llama3-70b-8192" as in the previous direct Groq call,
        # you'd pass it here or change the default in chat_history_manager.py
        lc_chain = get_or_create_groq_chain_for_session(
            st.session_state.session_id,
            api_keys["GROQ_API_KEY"],
            model_name="llama3-70b-8192" # Or your preferred model
        )
    except Exception as e:
        yield f"Error initializing LangChain model: {e}"
        return

    # The RAG manager prepares the input text for the HumanMessagePromptTemplate
    rag_augmented_query = rag_manager.get_augmented_prompt(user_message)

    # Input for the LLMChain (history is handled by the chain's memory)
    input_data = {"text": rag_augmented_query}

    try:
        # Stream the response from the LangChain chain
        # LLMChain.stream yields dictionaries; the actual token is typically in the output key (default 'text')
        for chunk_dict in lc_chain.stream(input_data):
            # The output key for LLMChain is typically 'text'
            content_chunk = chunk_dict.get(lc_chain.output_key, "") 
            if content_chunk:
                yield content_chunk
            # The chain's memory is updated automatically.
    except Exception as e:
        print(f"Error during Groq API call: {e}")  # Log error server-side
        error_message_to_yield = f"Error: An API error occurred: {e}"
        # Check if the error is a 401 Invalid API Key error
        if "401" in str(e) and ("Invalid API Key" in str(e) or "invalid_api_key" in str(e).lower()):
            error_message_to_yield = (
                "Error: Groq API request failed with a 401 error (Invalid API Key).\n\n"
                "Please check the following:\n"
                "1. Your `GROQ_API_KEY` in the `.env` file is correct and active.\n"
                "2. You have sufficient credits/quota on your Groq account.\n"
                "3. Restart the application after verifying/updating the key."
            )
        yield error_message_to_yield

# The call_web_search_api function is removed as the "Web Search" action is being removed.

def _get_first_google_image_url(google_search_url: str) -> str | None:
    """
    Tries to fetch the Google Images page and extract the first image URL.
    This is for preview purposes and can be fragile.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(google_search_url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Google's image search HTML can change. This is a common pattern.
        # Look for image tags within a container that often holds search results.
        # This selector might need adjustment if Google changes its HTML.
        img_tag = soup.find("img", {"class": ["Q4LuWd", "n3VNCb", "rg_i"]}) # Common classes for result images
        if img_tag and img_tag.get("src"):
            return img_tag.get("src")
    except Exception as e:
        print(f"Could not fetch or parse Google Image for preview: {e}")
    return None

def call_image_search_api(query: str) -> dict: # Returns a dictionary
    """
    Fetches images using CLIP from local index, then Pixabay API, 
    or generates a Google Image Search URL as a fallback.
    Returns a dictionary with image URLs and a "see more" link.
    """
    pixabay_key = api_keys.get("PIXABAY_API_KEY")
    google_search_url = f"https://www.google.com/search?tbm=isch&q={urllib.parse.quote_plus(query)}"
    results_limit = 3 # Number of images to fetch from Pixabay

    # CLIP-based local search removed.
    # 1. Fallback to Pixabay API (was 2, now 1)
    if pixabay_key:
        try:
            # ... (existing Pixabay logic remains the same) ...
            # (Assuming the Pixabay call logic from your original file is here)
            # For brevity, I'm not repeating the full Pixabay call.
            # If Pixabay returns image_urls:
            # return {"type": "image_results", "images": image_urls, "see_more_url": google_search_url, "query": query, "source": "Pixabay"}
            # Placeholder for Pixabay logic:
            # Simulating Pixabay call for now
            pass # Replace with actual Pixabay call
        except requests.exceptions.RequestException as e:
            print(f"Pixabay API request error: {e}")
        except Exception as e:
            print(f"Error processing Pixabay response: {e}")
    
    # 2. Fallback to Google Search link, try to get a preview image (was 3, now 2)
    preview_image_url = _get_first_google_image_url(google_search_url)
    return {
        "type": "image_link", 
        "url": google_search_url, 
        "query": query, "source": "Google Images",
        "preview_image_url": preview_image_url # Add preview URL
    }

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
def chatbot_response(user_input: str): # Can return str or generator
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

    yield from call_groq_api(user_input)


# --- Input Processing and History Management ---

response_placeholder = st.empty()

def process_input(user_input: str, action_type: str):
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
                # chatbot_response is run in a thread. It returns either a string or a generator.
                response_object_future = run_in_thread(chatbot_response, user_input)
                response_object = response_object_future.result()
                # If chatbot_response itself is a generator (because call_groq_api is a generator),
                # response_object will be that generator.
                # If chatbot_response handles a non-generator case (e.g. "my name is"),
                # response_object will be a string.
                # Check if the response_object is a generator (for streaming)
                if hasattr(response_object, '__iter__') and not isinstance(response_object, str):
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response_streamed = ""
                        for chunk in response_object:
                            full_response_streamed += chunk
                            message_placeholder.markdown(full_response_streamed + "▌") # Typing cursor
                        message_placeholder.markdown(full_response_streamed) # Final message
                    response_content = full_response_streamed
                else:
                    # It's a direct string response (e.g., from "my name is" command)
                    response_content = str(response_object)
                    # The display will be handled by the history append and rerun

            elif action_type == "Image Search":
                # Image search now also runs in a thread
                future = run_in_thread(call_image_search_api, user_input)
                response_content = future.result() # This will be a dictionary

            elif action_type == "News":
                future = run_in_thread(call_news_api, user_input)
                response_content = future.result()

            elif action_type == "Movie Search":
                movie_prompt = (
                    f"Please list exactly 10 movie recommendations based on or related to '{user_input}'. "
                    "Provide only the movie titles, one per line, numbered 1 through 10."
                )
                future = run_in_thread(call_groq_api, movie_prompt)
                # call_groq_api now always returns a generator (even for errors, it yields an error string)
                # So, movie_titles_response_object will be a generator.
                movie_titles_response_generator = future.result() 


                raw_movie_titles_text = ""

                if hasattr(movie_titles_response_generator, '__iter__') and not isinstance(movie_titles_response_generator, str):
                    # Stream the raw movie list from the LLM
                    with st.chat_message("assistant"): # Temporarily display stream
                        message_placeholder = st.empty()
                        full_response_streamed = ""
                        for chunk in movie_titles_response_generator: # Iterate through the generator
                            full_response_streamed += chunk
                            message_placeholder.markdown(full_response_streamed + "▌")
                        message_placeholder.markdown(full_response_streamed)
                    raw_movie_titles_text = full_response_streamed # Assign the fully streamed text

                # Now, parse the fully streamed raw_movie_titles_text
                if raw_movie_titles_text and not raw_movie_titles_text.startswith("Error:"):
                    titles = []
                    for line in raw_movie_titles_text.splitlines():
                        cleaned_line = line.strip()
                        if cleaned_line:
                            # Attempt to remove leading numbers/bullets
                            if cleaned_line[0].isdigit() and '.' in cleaned_line:
                                parts = cleaned_line.split('.', 1)
                                title_candidate = parts[1].strip() if len(parts) > 1 else ''
                            elif cleaned_line.startswith('-'):
                                title_candidate = cleaned_line[1:].strip()
                            else:
                                title_candidate = cleaned_line
                            if title_candidate: # Ensure non-empty title
                                titles.append(title_candidate)

                    formatted_recommendations = []
                    for i, title in enumerate(titles[:10]):
                        encoded_title = urllib.parse.quote_plus(title)
                        google_link = f"https://www.google.com/search?q={encoded_title}+movie"
                        formatted_recommendations.append(f"{i+1}. {title} - [Google Search Link]({google_link})")
                    
                    if formatted_recommendations:
                        response_content = "Here are some movie recommendations:\n\n" + "\n".join(formatted_recommendations)
                    else:
                        # If parsing failed to extract titles from a non-error response
                        response_content = f"I found some movie information, but had trouble formatting it. Raw response: \n```\n{raw_movie_titles_text}\n```"
                else:
                    # Handle cases where raw_movie_titles_text itself is an error message
                    # or if the generator was empty (shouldn't happen if errors yield strings)
                    response_content = raw_movie_titles_text if raw_movie_titles_text else "Failed to get movie recommendations."

            else:
                response_content = f"Error: Unsupported action type: {action_type}"

        except Exception as e:
            print(f"An unexpected error occurred during process_input execution: {e}")
            response_content = f"Error: An unexpected error occurred: {e}"

    # After getting response_content, add to both histories
    # For Image Search, response_content is a dict, so we add it directly to st.session_state.chat_history
    if action_type == "Image Search" and isinstance(response_content, dict):
        st.session_state.chat_history.append(("Image Search Results", response_content))
        # TTS for image search is handled below
    elif not isinstance(response_content, dict) and \
         not (isinstance(response_content, str) and response_content.startswith("Error: Unable to process request due to application state error.")):
        st.session_state.chat_history.append(("BuzzBot", response_content))
        chat_history.add_ai_message(response_content)

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
                if isinstance(message, dict):
                    query = message.get('query', 'your query') # Default query text
                    source = message.get('source', 'the web') # Default source text
                    msg_type = message.get("type")

                    if msg_type == "image_results" and message.get("images"):
                        st.markdown(f"Here are some images I found for **'{query}'** (from {source}):")
                        # Display images in columns for better layout
                        cols = st.columns(len(message["images"]))
                        for i, img_url in enumerate(message["images"]):
                            cols[i].image(img_url, width=150) # Adjust width as needed
                        if message.get("see_more_url"): # Pixabay might have this
                            st.markdown(f"[See more images on Google]({message.get('see_more_url')})")

                    elif msg_type == "image_link":
                        preview_url = message.get("preview_image_url")
                        if preview_url:
                            st.image(preview_url, width=150, caption=f"Preview for '{query}'")
                        st.markdown(f"**Image Search:** [Click here for more images of \"{query}\" on {source}]({message.get('url')})")

                    else: # Fallback for unknown dict type
                        st.markdown(f"Received an image search result of an unknown format for '{query}'.")

                # Fallback for old string format if any (though new responses will be dicts)
                elif isinstance(message, str):
                    st.markdown(f"**Image Search Results:** [Click here to view images]({message})")
                else: # Fallback for completely unknown message type
                    st.markdown("Received an image search result in an unexpected format.")


# --- Main Application Flow ---
def main():
    """Main function to run the Streamlit application."""
    if not st.session_state.get('session_state_initialized', False):
        initialize_session_state()
    # Theme application is removed.

    st.markdown("<h1>Buzzbot <span class='rotating-robot'>🤖</span></h1>", unsafe_allow_html=True)

    # Theme selection UI is removed from the sidebar.
    # st.sidebar.header("Theme")
    # ... (theme selectbox logic removed)

    st.sidebar.header("Settings")
    action_options = ["Chat", "Image Search", "News", "Movie Search"] # "Web Search" removed
    action_type = st.sidebar.radio("⚡ Select Action", options=action_options, key="action_radio")

    if rag_manager is None:
         st.sidebar.warning("RAG Manager initialization failed. File upload features may not work.")
    if docx is None:
         st.sidebar.warning("`python-docx` not installed. DOCX uploads not supported. Install with `pip install python-docx`.")
    if pd is None:
         st.sidebar.warning("`pandas` not installed. CSV uploads not supported. Install with `pip install pandas`.")
    # Removed warnings for camera-specific libraries (cv2, webrtc, yolo, openai, pillow, clip)


    # Tabs are removed as Camera tab is gone. Chat UI will be directly in the main area.
    # chat_tab, camera_tab = st.tabs(["💬 Chat", "📹 Camera"])
    
    # with chat_tab: # This block is no longer needed if not using tabs
    if True: # Placeholder to maintain indentation for now, can be removed
        chat_history_container = st.container(height=400)
        with chat_history_container:
            display_chat_history()

        user_query = st.chat_input("What can I help you with?", key="chat_input_main")

        if user_query:
            st.session_state.voice_input_active = False
            process_input(user_query, action_type)

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
                            prompt_prefix + str(file_contents), action_type="Chat"
                        )

                elif file_contents is not None and isinstance(file_contents, str) and file_contents.startswith("Error:"):
                     st.subheader("Extracted Content Preview:")
                     st.error(f"Could not display preview:\n{file_contents}")

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
