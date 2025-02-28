# urlChatbot
# AI Chat Assistant 🤖

A powerful, context-aware chat assistant built with Streamlit and powered by Groq's LLaMA 3 model. This application allows users to have intelligent conversations while incorporating knowledge from web sources.

## Features 🌟

- **Intelligent Conversations**: Powered by Groq's LLaMA 3 model
- **Multi-Source Knowledge**: Incorporate multiple web sources for enhanced responses
- **Session Management**: Create, save, and share conversation sessions
- **Context-Aware Responses**: Load and reference web pages for domain-specific knowledge
- **Persistent Memory**: Maintains conversation history throughout the chat
- **Google AI Embeddings**: Uses Google's text embedding model for efficient document search

## Prerequisites 📋

- Python 3.8+
- A Groq API key
- A Google AI API key

## Installation 🛠️

1. Clone the repository:
git clone https://github.com/DevaanshArora/urlChatbot.git
2. Create and activate a virtual environment:
bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install required packages:
bash
pip install -r requirements.txt


4. Create a `.env` file in the project root with your API keys:
fyr, my personal api keys:
GROQ_API_KEY="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"
OPENAI_API_KEY="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"
GOOGLE_API_KEY="AIzaSyCqNcbRFMIO7pg95mvQoH4lcXYQFDX_z2E"
copy them as it is in the ".env" file.


## Usage 🚀

1. Start the application:
streamlit run chatBot.py


2. Access the application in your web browser (typically at `http://localhost:8501`)

### Key Features Guide 📚

1. **Session Management**:
   - Find your Session ID in the sidebar
   - Save it to continue conversations later
   - Share it with others for collaborative discussions
   - Create new sessions as needed

2. **Adding Knowledge Sources**:
   - Use the sidebar to add URLs
   - Enter URLs one per line
   - Click "Load URLs" to process
   - View loaded sources below the input

3. **Chat Interface**:
   - Type questions in the chat box
   - View AI responses that consider both loaded sources and general knowledge
   - Use "Clear Chat" to reset conversations
   - Previous messages remain visible for context

## Dependencies 📦

- `streamlit`: Web application framework
- `langchain`: LLM framework for chaining operations
- `langchain_groq`: Groq integration
- `langchain_google_genai`: Google AI integration
- `chromadb`: Vector store for document embeddings
- `python-dotenv`: Environment variable management

## Project Structure 📁
