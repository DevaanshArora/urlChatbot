import streamlit as st
#from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
import uuid
import chromadb

# Load the .env file
load_dotenv()

# Add this near the top of the file after imports
CHROMA_DB_DIR = "chroma_db"
if not os.path.exists(CHROMA_DB_DIR):
    os.makedirs(CHROMA_DB_DIR)

# Store chat chains for different sessions
if "chat_chains" not in st.session_state:
    st.session_state.chat_chains = {}
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "loaded_urls" not in st.session_state:
    st.session_state.loaded_urls = {}

def create_chat_chain():
    model = ChatGroq(model="llama3-8b-8192", temperature=0.7)
    memory = ConversationBufferMemory()
    return ConversationChain(
        llm=model,
        memory=memory,
        verbose=True
    )

def load_and_process_url(url, session_id):
    # Skip if URL already loaded for this session
    if session_id in st.session_state.loaded_urls and url in st.session_state.loaded_urls[session_id]:
        st.sidebar.warning(f"URL already loaded: {url}")
        return None
    
    try:
        # Load the webpage
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        
        # Initialize session's loaded URLs if not exists
        if session_id not in st.session_state.loaded_urls:
            st.session_state.loaded_urls[session_id] = set()
        
        # Add URL to loaded set
        st.session_state.loaded_urls[session_id].add(url)
        
        return splits
    except Exception as e:
        st.sidebar.error(f"Error loading URL {url}: {str(e)}")
        return None

# Page configuration
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")
st.title("AI Sensy Chat Assistant 🤖")

# Enhanced explanatory text at the top
st.markdown("""
### Welcome to AI Sensy's Intelligent Chat Assistant! 

#### 🌟 Key Features:
1. **Intelligent Conversations**: Powered by Groq's LLaMA 3 for natural and informative responses
2. **Session Management**:
   - Save and reload conversations using Session IDs
   - Share sessions with teammates
   - Create multiple conversation threads
3. **Context-Aware Responses**:
   - Load web pages for domain-specific knowledge
   - AI considers loaded sources when answering
4. **Persistent Memory**:
   - Remembers conversation history
   - Maintains context throughout the chat
   
#### 📚 How to Use:

**Session Management:**
1. Find your Session ID in the sidebar
2. Save it to continue conversations later
3. Share it with others for collaborative discussions
4. Click "Create New Session" for a fresh start

**Adding Knowledge Sources:**
1. Locate the "Add Knowledge Sources" section in sidebar
2. Enter URLs (one per line)
3. Click "Load URLs" to process
4. View loaded sources below the input

**Chat Interface:**
1. Type your questions in the chat box
2. AI will consider both loaded sources and its general knowledge
3. Use "Clear Chat" to reset the conversation
4. Previous messages remain visible for context

**Pro Tips:**
- Load relevant URLs before asking domain-specific questions
- Create different sessions for different topics
- Clear chat while keeping sources when starting a new subtopic
- Check the "Currently Loaded Sources" to track your knowledge base

Need help? Just ask in the chat and I'll guide you through any feature!
""")

# Session ID handling
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar organization
st.sidebar.title("Settings & Data Sources")

# Session management section with explanations
st.sidebar.markdown("### Session Management")
st.sidebar.markdown("""
Keep track of your conversations by using session IDs. You can:
- Create a new session
- Load an existing session
- Share your session ID with others
""")

# Display current session ID
st.sidebar.text(f"Current Session ID: {st.session_state.session_id}")

# Option to enter a specific session ID
new_session_id = st.sidebar.text_input("Enter an existing Session ID:", 
                                      help="Paste a session ID here to reload a previous conversation")

if new_session_id and new_session_id != st.session_state.session_id:
    st.session_state.session_id = new_session_id
    st.rerun()

# Create new session button
if st.sidebar.button("Create New Session", 
                     help="Start a fresh conversation with a new session ID"):
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

# URL input section with better explanation
st.sidebar.markdown("---")
st.sidebar.markdown("### Add Knowledge Sources")
st.sidebar.markdown("""
Enhance the AI's responses by providing relevant web pages. The assistant will use 
this content to provide more accurate and contextual answers.
""")

url_input = st.sidebar.text_area(
    "Enter URLs (one per line)",
    height=100,
    help="Add web pages that contain relevant information for your conversation"
)

if st.sidebar.button("Load URLs", 
                     help="Click to process and load the entered URLs as knowledge sources"):
    if url_input:
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        
        # Initialize vector store if not exists for this session
        if st.session_state.session_id not in st.session_state.vector_stores:
            st.session_state.vector_stores[st.session_state.session_id] = Chroma(
                collection_name=f"session_{st.session_state.session_id}",
                embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
                persist_directory=CHROMA_DB_DIR
            )
        
        vector_store = st.session_state.vector_stores[st.session_state.session_id]
        
        for url in urls:
            splits = load_and_process_url(url, st.session_state.session_id)
            if splits:
                vector_store.add_documents(splits)
                st.sidebar.success(f"Successfully loaded: {url}")

# Display loaded URLs with better formatting
if st.session_state.session_id in st.session_state.loaded_urls:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Currently Loaded Sources")
    st.sidebar.markdown("The following sources are being used for context:")
    for url in st.session_state.loaded_urls[st.session_state.session_id]:
        st.sidebar.markdown(f"• [{url}]({url})")

# Initialize chat chain for current session if it doesn't exist
if st.session_state.session_id not in st.session_state.chat_chains:
    st.session_state.chat_chains[st.session_state.session_id] = {
        "chain": create_chat_chain(),
        "messages": []
    }

current_session = st.session_state.chat_chains[st.session_state.session_id]

# Clear chat button with explanation
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Clear Chat", 
                 help="Reset the current conversation while keeping loaded sources"):
        current_session["chain"] = create_chat_chain()
        current_session["messages"] = []
        st.rerun()

# Add a divider before the chat
st.markdown("---")
st.markdown("### Chat Interface")

# Display chat messages
for message in current_session["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    current_session["messages"].append({"role": "user", "content": prompt})

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check if we have any loaded URLs for this session
            has_sources = (st.session_state.session_id in st.session_state.vector_stores and 
                         st.session_state.loaded_urls.get(st.session_state.session_id))
            
            if has_sources:
                # Get relevant context from loaded URLs
                vector_store = st.session_state.vector_stores[st.session_state.session_id]
                relevant_docs = vector_store.similarity_search(prompt, k=3)
                context = "\n".join([doc.page_content for doc in relevant_docs])
                
                enhanced_prompt = f"""You are a helpful AI assistant. Use the following context from loaded sources to answer the question. If the context doesn't contain relevant information, say so and provide a general response.

                Context from loaded sources:
                {context}
                
                Question: {prompt}"""
                
                response = current_session["chain"].predict(input=enhanced_prompt)
            else:
                # No sources loaded - use general knowledge
                response = current_session["chain"].predict(input=prompt)
            
            st.write(response)
    current_session["messages"].append({"role": "assistant", "content": response})

# Update styling
st.markdown("""
    <style>
    .stChat {
        padding: 20px;
        border-radius: 10px;
        background-color: #f5f5f5;
    }
    .stChatMessage {
        margin: 10px 0;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton button {
        border-radius: 20px;
        padding: 10px 24px;
    }
    .sidebar .sidebar-content {
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
