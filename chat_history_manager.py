import uuid
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate, # Added for system prompt
)
from langchain_groq import ChatGroq

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ChatHistory:
    def __init__(self):
        self.messages: List[ChatMessage] = []
    
    def add_user_message(self, content: str):
        self.messages.append(ChatMessage(role="user", content=content))
    
    def add_ai_message(self, content: str):
        self.messages.append(ChatMessage(role="assistant", content=content))
    
    def clear(self):
        self.messages = []
    
    @property
    def message_history(self) -> List[ChatMessage]:
        return self.messages

chats_by_session_id: Dict[str, ChatHistory] = {}

# Store LLMChains per session to maintain memory context
langchain_chains_by_session_id: Dict[str, LLMChain] = {}

def get_chat_history(session_id: str) -> ChatHistory:
    chat_history = chats_by_session_id.get(session_id)
    if chat_history is None:
        chat_history = ChatHistory()
        chats_by_session_id[session_id] = chat_history
    return chat_history
    
def create_session_id() -> str:
    return str(uuid.uuid4())

# Function to create a Groq LLMChain with conversation memory
def get_or_create_groq_chain_for_session(session_id: str, groq_api_key: str, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> LLMChain:
    if session_id in langchain_chains_by_session_id:
        return langchain_chains_by_session_id[session_id]

    # Define the prompt structure, including a system message
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are BuzzBot, a friendly and very helpful AI assistant. Your main purpose is to assist the user with their requests in a gentle and understanding manner. "
            "Always try your best to complete any task the user asks. If a request seems impossible or unclear, "
            "politely ask for more details or gently suggest an alternative way you might be able to help, rather than directly refusing. "
            "When formulating your response, please integrate insights from the 'Knowledge base context' (if provided within the user's current message/text input) "
            "with the ongoing 'chat_history' to ensure your answer is both contextually relevant and maintains a natural conversational flow. "
            "Your primary aim is to be helpful and clear, in a soft, encouraging, and positive tone. Keep responses very brief and direct unless asked for detail."
        ),
        MessagesPlaceholder(variable_name="chat_history"), # Populated by ConversationBufferMemory
        HumanMessagePromptTemplate.from_template("{text}"), # User's RAG-augmented query
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, streaming=True)
        groq_lc_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory, # This memory is now tied to this chain instance for this session
        )
        langchain_chains_by_session_id[session_id] = groq_lc_chain
        return groq_lc_chain
    except Exception as e:
        print(f"Error creating Groq LLMChain for session {session_id}: {e}")
        raise # Re-raise to allow Streamlit to catch and display it