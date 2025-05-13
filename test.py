import os
import json
# from charset_normalizer import detect # Not used in the final plan
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
# from langdetect import DetectorFactory # Not used in the final plan
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import urllib.parse  # Import for URL encoding

# Try importing speech recognition modules and pyttsx3
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False
    # Warning will be displayed in the sidebar if disabled

# DetectorFactory.seed = 0 # Not needed if langdetect is not used

# Load environment variables from .env file
load_dotenv()

def get_env_variable(key, default=None):
    """Retrieves environment variable, displays error if missing."""
    value = os.environ.get(key, default)
    if not value:
        st.error(f"Environment variable '{key}' is missing. Please add it to your .env file.")
    return value

# --- API Clients and Setup ---
api_key = get_env_variable("GROQ_API_KEY")
# Initialize Groq client only if API key is available
groq_client = Groq(api_key=api_key) if api_key else None

# --- Memory Management ---
MEMORY_FILE = "memory.json"

def load_memory():
    """Loads persistent memory from a JSON file."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            st.warning("Error loading memory file. Starting with empty memory.")
            return {}
        except Exception as e:
            st.error(f"An unexpected error occurred loading memory: {e}")
            return {}
    return {}

def save_memory(data):
    """Saves persistent memory to a JSON file."""
    try:
        with open(MEMORY_FILE, "w") as file:
            json.dump(data, file, indent=4)  # Use indent for readability
    except Exception as e:
        st.error(f"An error occurred saving memory: {e}")

# Load memory on startup
memory = load_memory()

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes necessary session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "voice_input_active" not in st.session_state:
        st.session_state.voice_input_active = False
    # Add other session state variables here if needed

# --- Theme Application ---
def apply_theme(theme):
    """Applies a custom theme using markdown CSS."""
    # Define themes
    themes = {
        "Dark": {
            "background": "#0E1117",
            "text": "#FAFAFA",
            "sidebar_bg": "#1E1E1E",
            "button_bg": "#007BFF",
            "button_text": "white",
            "chat_user_bg": "#2E2E2E",
            "chat_bot_bg": "#1A3A50"
        },
        "Light": {
            "background": "#FFFFFF",
            "text": "#000000",
            "sidebar_bg": "#F0F2F6",
            "button_bg": "#4CAF50",
            "button_text": "white",
            "chat_user_bg": "#E8E8E8",
            "chat_bot_bg": "#D1E7DD"
        }
    }

    selected_theme = themes.get(theme, themes["Light"])

    st.markdown(
        f"""
        <style>
        /* General app styling */
        .stApp {{
            background-color: {selected_theme['background']};
            color: {selected_theme['text']};
        }}

        /* Sidebar styling */
        .stSidebar {{
            background-color: {selected_theme['sidebar_bg']};
            color: {selected_theme['text']};
        }}

        /* Chat message styling */
        .stChatMessage {{
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }}

        .stChatMessage.stChatUser {{
            background-color: {selected_theme['chat_user_bg']};
        }}

        .stChatMessage.stChatAssistant {{
            background-color: {selected_theme['chat_bot_bg']};
        }}

        .stChatMessage p, .stChatMessage a, .stChatMessage li, .stChatMessage ul {{
            color: {selected_theme['text']} !important;
        }}

        .stChatMessage a {{
            color: #007BFF !important;
        }}

        /* Chat input styling */
        div.stChatInput {{
            background-color: {selected_theme['sidebar_bg']};
            padding: 10px;
            border-top: 1px solid #ccc;
        }}

        div.stChatInput > label > div > div > textarea {{
            background-color: {selected_theme['background']};
            color: {selected_theme['text']};
        }}

        /* Button styling */
        .stButton button {{
            background-color: {selected_theme['button_bg']};
            color: {selected_theme['button_text']};
            border-radius: 5px;
            padding: 8px 15px;
            border: none;
            cursor: pointer;
        }}

        .stButton button:hover {{
            opacity: 0.9;
        }}

        /* Title animation */
        @keyframes colorChange {{
            0% {{ color: #4CAF50; }}
            25% {{ color: #2196F3; }}
            50% {{ color: #FF9800; }}
            75% {{ color: #E91E63; }}
            100% {{ color: #4CAF50; }}
        }}

        /* Robot animation */
        @keyframes rotate {{
            from {{
                transform: rotate(0deg);
            }}
            to {{
                transform: rotate(360deg);
            }}
        }}

        .rotating-robot {{
            display: inline-block;
            animation: rotate 2s linear infinite;
        }}

        .color-changing-title {{
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            animation: colorChange 5s infinite;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        </style>
    """,
        unsafe_allow_html=True,
    )

# --- API Interaction Functions ---
def get_groq_client():
    """Returns the Groq client instance."""
    return groq_client

def call_groq_api(user_message):
    """Calls the Groq API for chatbot response."""
    client = get_groq_client()
    if not client:
        return "API client not initialized. Check your API key."

    # Prepare messages for the API, including a brief history for context
    messages = [{"role": "system", "content": "You are BuzzBot. Keep responses very brief and direct."}]
    # Append previous turns, limiting to last few for brevity and token control
    # Only include 'User' and 'BuzzBot' messages in the history sent to the API
    history_for_api = [
        (role, content) for role, content in st.session_state.chat_history if role in ["User", "BuzzBot"]
    ]
    for role, content in history_for_api[-6:]:  # Limit to last 6 turns
        api_role = "user" if role == "User" else "assistant"
        messages.append({"role": api_role, "content": content})

    messages.append({"role": "user", "content": user_message})

    try:
        # Using st.spinner for visual feedback during API call
        with st.spinner("Buzzbot is thinking..."):
            response = client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192",  # Using a suitable model
                # Increased tokens slightly for potentially better responses
            )
            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                return "I'm sorry, I couldn't process your request. Please try again."
    except Exception as e:
        st.error(f"An error occurred while calling the API: {e}")
        return f"Error: {e}"

def call_image_search_api(query):
    """Generates a Google Image Search URL."""
    # Note: This is a simple URL redirect, not an actual image search API call returning images.
    search_url = f"https://www.google.com/search?tbm=isch&q={urllib.parse.quote_plus(query)}"  # Use urllib.parse.quote_plus
    return search_url

@st.cache_data(ttl=3600)  # Cache news results for 1 hour to avoid repeated calls
def call_news_api(location):
    """Fetches recent news headlines for a location using Google News RSS."""
    try:
        # Construct RSS feed URL
        rss_url = (
            f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(location)}&hl=en&gl=US&ceid=US:en"
        )  # Use urllib.parse.quote_plus
        headers = {"User-Agent": "Mozilla/5.0"}  # Use a common User-Agent
        response = requests.get(rss_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")
        summaries = []
        now = datetime.utcnow()

        for item in items:
            title = item.title.text if item.title else "No Title"
            link = item.link.text if item.link else "#"
            # Extract text content from description, which might contain HTML
            description_soup = (
                BeautifulSoup(item.description.text, "html.parser") if item.description else None
            )
            desc = description_soup.get_text().strip() if description_soup else "No Description"
            pub_date_str = item.pubDate.text if item.pubDate else None

            # Parse publication date and filter by recency (last 24 hours)
            pub_dt = None
            if pub_date_str:
                try:
                    # Attempt parsing common RSS date format
                    pub_dt = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
                except ValueError:
                    # Fallback for other potential formats if necessary
                    pass  # Add more format handling if required
                except Exception as e:
                    st.warning(f"Could not parse date '{pub_date_str}': {e}")  # Log parsing errors

            if pub_dt and (now - pub_dt > timedelta(days=1)):
                continue  # Skip news older than 24 hours

            # Format the news item for display
            summaries.append(f"- **{title}**\n  {desc}\n  [Read more]({link})")

            if len(summaries) >= 3:  # Limit to top 3 recent articles
                break

        return "\n\n".join(summaries) if summaries else f"No recent news found for '{location}'."
    except requests.exceptions.RequestException as e:
        st.error(f"Network or HTTP error fetching news: {e}")
        return f"Error fetching news: Could not connect to news source."
    except Exception as e:
        st.error(f"An error occurred while parsing news feed: {e}")
        return f"Error fetching news: {e}"

# --- Voice Input/Output Functions ---
def listen_for_voice():
    """Listens for voice input using SpeechRecognition."""
    if not VOICE_ENABLED:
        # Warning is shown in sidebar
        return None

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_adjustment_damping = 0.15
    recognizer.dynamic_energy_adjustment_ratio = 1.5
    recognizer.pause_threshold = 0.8

    status_placeholder = st.empty()  # Placeholder for voice status messages

    try:
        with sr.Microphone() as source:
            status_placeholder.info("Adjusting for ambient noise... Please be quiet.")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            status_placeholder.info("🎤 Listening... Speak now!")
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=5)
            status_placeholder.info("Processing audio...")
            # Use a spinner while processing
            with st.spinner("Transcribing speech..."):
                text = recognizer.recognize_google(audio, language='en-US')  # Specify language
            status_placeholder.empty()  # Clear status on success
            return text

    except sr.WaitTimeoutError:
        status_placeholder.warning("Listening timed out. No speech detected.")
    except sr.UnknownValueError:
        status_placeholder.warning("Could not understand the audio. Please speak clearly.")
    except sr.RequestError as e:
        status_placeholder.error(f"Could not request results from speech recognition service; {e}")
    except Exception as e:
        status_placeholder.error(f"An unexpected error occurred during voice input: {e}")

    status_placeholder.empty()  # Ensure placeholder is cleared on error too
    return None

def get_pyttsx3_engine():
    """Initializes and returns a pyttsx3 engine."""
    try:
        engine = pyttsx3.init()
        return engine
    except Exception as e:
        st.error(f"Failed to initialize text-to-speech engine: {e}")
        return None

def speak_response(text):
    """Speaks the given text using pyttsx3."""
    if not VOICE_ENABLED:
        return
    engine = get_pyttsx3_engine()
    if engine:
        try:
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            st.error(f"Speech output error: {str(e)}")

# --- Core Chat Logic ---
def chatbot_response(user_input):
    """Generates a chatbot response based on input, handling special commands."""
    user_input_lower = user_input.lower().strip()

    # Handle name memory
    if "my name is" in user_input_lower:
        name = user_input.split("my name is")[-1].strip().title()
        if name:
            memory['name'] = name
            save_memory(memory)
            return f"Nice to meet you, {name}!"
        else:
            return "Okay, I'll remember you mentioned your name, but I didn't catch it."
    elif "what is my name" in user_input_lower:
        name = memory.get('name')
        if name:
            return f"Your name is {name}."
        else:
            return "I don't have your name stored yet. You can tell me by saying 'My name is [Your Name]'."
    elif "forget my name" in user_input_lower or "clear my name" in user_input_lower:
        if 'name' in memory:
            del memory['name']
            save_memory(memory)
            return "Okay, I've forgotten your name."
        else:
            return "I don't have your name stored anyway."

    # Default to calling Groq API
    return call_groq_api(user_input)

# --- Input Processing and History Management ---
def process_input(user_input, action_type, is_voice_input):
    """Processes user input based on selected action type."""
    if not user_input:
        return  # Do nothing if input is empty

    # Add user message to history
    st.session_state.chat_history.append(("User", user_input))

    response_content = ""
    # No need for spinner here, as specific functions like call_groq_api have their own spinners

    if action_type == "Chat":
        response_content = chatbot_response(user_input)
        st.session_state.chat_history.append(("BuzzBot", response_content))
        if is_voice_input and response_content:
            speak_response(response_content)  # Speak chatbot response
    elif action_type == "Image Search":
        # The image search API call is fast (just generates a URL), no spinner needed here
        search_url = call_image_search_api(user_input)
        # Store as link for display
        st.session_state.chat_history.append(("Image Search Results", search_url))
        # response_content = f"Here are the image search results for '{user_input}': {search_url}" # No need to set response_content here
        if is_voice_input:
            speak_response("Here are the image search results.")  # Speak a confirmation
    elif action_type == "News":
        # call_news_api has its own spinner
        news_info = call_news_api(user_input)
        st.session_state.chat_history.append(("BuzzBot", news_info))
        # response_content = news_info # No need to set response_content here
        if is_voice_input:
            # Speak a summary or just a confirmation
            speak_response("Here is the latest news.")  # Keep spoken response brief
    elif action_type == "Movie Search":
        # Use the existing Groq API to get movie recommendations
        # Craft a specific prompt to ask for exactly 10 movie titles
        movie_prompt = (
            f"Please list exactly 10 movie recommendations based on or related to '{user_input}'. "
            "Provide only the movie titles, one per line, numbered 1 through 10."
        )
        movie_titles_response = call_groq_api(movie_prompt)

        if movie_titles_response and "Error:" not in movie_titles_response:
            # Parse the response to extract titles
            titles = []
            # Split by lines and try to clean up numbered lists
            for line in movie_titles_response.splitlines():
                cleaned_line = line.strip()
                # Remove leading numbers (e.g., "1.", "2)") and extra space
                if cleaned_line and (cleaned_line[0].isdigit() or cleaned_line.startswith("-")):
                    parts = cleaned_line.split('.', 1)  # Split on first dot
                    if len(parts) > 1:
                        title = parts[1].strip()
                    else:  # Handle cases like "- Movie Title"
                        title = cleaned_line[1:].strip() if cleaned_line.startswith("-") else cleaned_line.strip()

                    if title:
                        titles.append(title)
                else:
                    # If it doesn't look like a numbered/bulleted list, just take the line
                    if cleaned_line:
                        titles.append(cleaned_line)

            # Format the output as requested
            formatted_recommendations = []
            for i, title in enumerate(titles[:10]):  # Take up to 10 titles found
                # Encode the title for the URL
                encoded_title = urllib.parse.quote_plus(title)
                google_link = f"https://www.google.com/search?q={encoded_title}"
                formatted_recommendations.append(f"{i+1}. {title} - [Google Search Link]({google_link})")

            if formatted_recommendations:
                response_content = "Here are some movie recommendations:\n\n" + "\n".join(
                    formatted_recommendations
                )
            else:
                response_content = (
                    f"Could not find movie recommendations for '{user_input}'. Please try a different query."
                )

        else:
            response_content = f"Error fetching movie recommendations: {movie_titles_response}"  # Display the error from API call

        st.session_state.chat_history.append(("BuzzBot", response_content))
        if is_voice_input and response_content:
            speak_response("Here are some movie recommendations.")  # Keep spoken response brief

    # st.rerun() is called by the caller after process_input finishes

# --- UI Components ---
def display_chat_history():
    """Displays the chat history using st.chat_message."""
    # Using a container allows placing chat history above input, and st.chat_input handles scroll
    # chat_container = st.container() # Not strictly needed, can just iterate directly

    for speaker, message in st.session_state.chat_history:
        if speaker == "User":
            with st.chat_message("user"):
                st.markdown(message)
        elif speaker == "BuzzBot":
            with st.chat_message("assistant"):  # Use assistant role for bot
                st.markdown(message)
        elif speaker == "Image Search Results":
            with st.chat_message("assistant"):  # Display search results as assistant message
                st.markdown(f"**Image Search Results:** [Click here to view images]({message})")
        # Add other message types if needed

# --- Main Application Flow ---
def main():
    """Main function to run the Streamlit application."""

    # --- Page Configuration ---
    st.set_page_config(
        page_title="Buzzbot - Your AI Assistant",
        page_icon="🤖",  # You can use an emoji or a path to an image file
        layout="wide",  # Use wide layout
        initial_sidebar_state="expanded",  # Sidebar open by default
    )

    # Initialize session state variables
    initialize_session_state()

    # Apply theme
    apply_theme("Light")  # Default theme

    # --- Header ---
    st.markdown("<h1>Buzzbot <span class='rotating-robot'>🤖</span></h1>", unsafe_allow_html=True)
    # Optional: Add a subheader or caption
    # st.subheader("Your friendly neighborhood AI assistant")

    # --- Sidebar Controls ---
    st.sidebar.header("Theme")
    theme = st.sidebar.selectbox("🎨 Choose Theme", ["Light", "Dark"], index=0)
    apply_theme(theme)

    st.sidebar.header("Settings")
    # REDEFINE the options list to include the new action
    action_options = ["Chat", "Image Search", "News", "Movie Search"]
    action_type = st.sidebar.radio("⚡ Select Action", options=action_options)

    # Display voice feature warning in sidebar if disabled
    if not VOICE_ENABLED:
        st.sidebar.warning(
            "🎤 Voice features disabled. Install required packages: `pip install SpeechRecognition pyaudio pyttsx3`"
        )

    # --- Chat History Display Area ---
    # Use a container to hold the chat messages. st.chat_input handles scrolling to the bottom.
    chat_history_container = st.container(height=400)  # Optional: Set a fixed height with scrollbar
    with chat_history_container:
        display_chat_history()

    # --- Input Area (at the bottom) ---
    # st.chat_input automatically places itself at the bottom and handles submission on Enter
    user_query = st.chat_input("What can I help you with?", key="chat_input_main")

    # Voice Input Button (placed above the chat input)
    # Use columns to control button width
    col_voice, col_spacer = st.columns([1, 5])  # Give voice button less width
    with col_voice:
        if VOICE_ENABLED:
            # Change button label based on voice input active state
            voice_button_label = (
                "🎤 Start Listening" if not st.session_state.voice_input_active else "⏹️ Stop Listening"
            )
            voice_button_clicked = st.button(voice_button_label, key="voice_button")

            if voice_button_clicked:
                # Toggle voice input active state
                st.session_state.voice_input_active = not st.session_state.voice_input_active
                if st.session_state.voice_input_active:
                    # Start listening
                    voice_input = listen_for_voice()
                    # Listening is done, set active state back to False regardless of outcome
                    st.session_state.voice_input_active = False  # Reset the flag
                    if voice_input:
                        st.info(f"You said: {voice_input}")  # Show transcribed text
                        # Process the voice input as if it were text input
                        process_input(voice_input, action_type, is_voice_input=True)
                        st.rerun()  # Rerun to update chat history

    # Process text input from st.chat_input
    if user_query:
        # Ensure voice listening is stopped if user types
        st.session_state.voice_input_active = False  # Reset the flag
        process_input(user_query, action_type, is_voice_input=False)
        st.rerun()  # Rerun to update chat history

    # --- File Upload Section ---
    st.markdown("---")  # Separator using markdown
    # Use an expander to keep the UI clean
    with st.expander("📂 Upload a File"):
        st.write("Upload a text-based file (TXT, JSON, CSV, PDF, DOCX) to process its content.")
        uploaded_file = st.file_uploader(
            "Choose a file", type=["txt", "json", "csv", "pdf", "docx"], key="file_uploader"
        )

        if uploaded_file is not None:
            file_contents = None
            file_details = {"name": uploaded_file.name, "type": uploaded_file.type, "size": uploaded_file.size}
            st.write("File Details:")
            st.json(file_details)  # Display file details nicely

            # Add a spinner for file processing
            with st.spinner(f"Reading {uploaded_file.type} file..."):
                try:
                    if uploaded_file.type == "text/plain":
                        file_contents = uploaded_file.read().decode("utf-8")
                    elif uploaded_file.type == "application/json":
                        data = json.load(uploaded_file)
                        file_contents = json.dumps(data, indent=2)  # Keep as string for processing
                    elif uploaded_file.type == "application/pdf":
                        try:
                            import PyPDF2

                            reader = PyPDF2.PdfReader(uploaded_file)
                            # Extract text page by page
                            extracted_text = []
                            for page_num in range(len(reader.pages)):
                                page = reader.pages[page_num]
                                text = page.extract_text()
                                if text:
                                    extracted_text.append(text)
                            file_contents = "\n".join(extracted_text)
                            if not file_contents:
                                st.warning("Could not extract any text from the PDF.")
                                file_contents = "Could not extract text from PDF."  # Provide a message
                        except ImportError:
                            st.error("PyPDF2 not installed. Cannot read PDF. Please install: `pip install PyPDF2`")
                            file_contents = None  # Indicate failure
                        except Exception as e:
                            st.error(f"Error reading PDF: {e}")
                            file_contents = f"Error reading PDF: {e}"  # Provide error message

                    elif (
                        uploaded_file.type
                        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    ):
                        try:
                            import docx

                            doc = docx.Document(uploaded_file)
                            file_contents = "\n".join([para.text for para in doc.paragraphs])
                            if not file_contents:
                                st.warning("Could not extract any text from the DOCX.")
                                file_contents = "Could not extract text from DOCX."  # Provide a message
                        except ImportError:
                            st.error(
                                "python-docx not installed. Cannot read DOCX. Please install: `pip install python-docx`"
                            )
                            file_contents = None  # Indicate failure
                        except Exception as e:
                            st.error(f"Error reading DOCX: {e}")
                            file_contents = f"Error reading DOCX: {e}"  # Provide error message

                    elif uploaded_file.type == "text/csv":
                        try:
                            import pandas as pd

                            df = pd.read_csv(uploaded_file)
                            # Convert DataFrame to string representation. Limit output for large files.
                            if len(df) > 50:  # Display head and tail for larger CSVs
                                file_contents = df.head().to_string() + "\n...\n" + df.tail().to_string()
                            else:
                                file_contents = df.to_string()
                            if not file_contents:
                                st.warning("Could not read CSV or it was empty.")
                                file_contents = "Could not read CSV or it was empty."  # Provide a message
                        except ImportError:
                            st.error("pandas not installed. Cannot read CSV. Please install: `pip install pandas`")
                            file_contents = None  # Indicate failure
                        except Exception as e:
                            st.error(f"Error reading CSV: {e}")
                            file_contents = f"Error reading CSV: {e}"  # Provide error message
                    else:
                        st.warning(f"Unsupported file type: {uploaded_file.type}")
                        file_contents = None  # Indicate unsupported type

                except Exception as e:
                    st.error(f"An unexpected error occurred during file reading: {e}")
                    file_contents = f"Error reading file: {e}"  # Provide error message

            if file_contents is not None:  # Check if content was successfully extracted
                st.success("File content extracted successfully!")
                # Display extracted content preview
                st.subheader("Extracted Content Preview:")
                # Use st.text_area for general text, st.json for JSON
                if uploaded_file.type == "application/json":
                    try:
                        st.json(json.loads(file_contents))  # Display JSON nicely
                    except json.JSONDecodeError:
                        st.text_area(
                            "Content Preview",
                            file_contents[:2000] + ("..." if len(file_contents) > 2000 else ""),
                            height=300,
                        )
                # Check if it's a text-like format that might benefit from text_area
                elif uploaded_file.type in [
                    "text/plain",
                    "text/csv",
                    "application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ]:
                    st.text_area(
                        "Content Preview",
                        file_contents[:4000] + ("..." if len(file_contents) > 4000 else ""),
                        height=300,
                    )  # Limit preview length
                else:  # Fallback for other types if needed, though handled above
                    st.text_area(
                        "Content Preview",
                        file_contents[:4000] + ("..." if len(file_contents) > 4000 else ""),
                        height=300,
                    )

                # Offer to process the content with the selected action (defaulting to Chat as per plan)
                process_file_content_button = st.button(
                    f"Process file content with Chat", key="process_file_button"
                )
                if process_file_content_button:
                    # Call process_input with the extracted file content
                    # Prepend a prompt to guide the model
                    prompt_prefix = "Analyze the following document content:\n\n"
                    process_input(
                        prompt_prefix + file_contents, action_type="Chat", is_voice_input=False
                    )  # Always process files as Chat
                    st.rerun()  # Rerun to show chat response

# --- Run the App ---
if __name__ == "__main__":
    main()