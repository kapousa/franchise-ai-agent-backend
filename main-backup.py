from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import uuid # Import uuid for generating unique session IDs

# Initialize FastAPI app
app = FastAPI()

# Configure CORS to allow requests from your React frontend.
# IMPORTANT: In a production environment, restrict origins to your specific frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your React app to connect
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# In-memory storage for chat sessions.
# In a real-world application, you would use a persistent storage solution
# like a database (e.g., Firestore, Redis) for scalability and persistence.
# Key: session_id (str), Value: google.generativeai.ChatSession object
chat_sessions = {}

# Pydantic model for incoming chat messages
class ChatMessage(BaseModel):
    message: str
    session_id: str = None # Add an optional session_id field

# Configure the Gemini API client
genai.configure(api_key="AIzaSyBbFpqTeTKwpD1RPjSETrroELVwSX4Xu7M")

# Initialize the Generative Model
# Using 'gemini-2.0-flash' for faster responses, 'gemini-pro' is another option.
model = genai.GenerativeModel('gemini-2.0-flash')

# Define the system instruction for the AI Franchise Agent
# This prompt guides the model to act as a specialized franchise consultant.
SYSTEM_INSTRUCTION = """
You are an AI Franchise Agent, designed to modernize franchise consultancy services. Your role is to act as an intelligent, always-available digital assistant for prospective franchisees, franchisors, and human consultants.

Your core objectives are:
- To automate repetitive consultancy tasks.
- To deliver consistent, standardized guidance regarding franchises.
- To provide data-driven insights (when applicable and if you have the data).
- To personalize recommendations to each user (based on information provided in the conversation).
- To improve scalability and reduce costs in franchise consulting.
- To ensure 24/7 availability for franchise-related queries.

You can assist with:
- Answering FAQs about franchises.
- Guiding users through automated questionnaires (conceptually, you'd ask questions).
- Providing persona-based franchise matching (by asking about user preferences/criteria).
- Summarizing FDDs (Franchise Disclosure Documents) if provided with text or asked about general FDD aspects.
- Discussing document generation (NDAs, forms) in a general sense.
- Offering personalized recommendations and financial projections (conceptually, by asking for data).
- Discussing operations analytics, compliance monitoring, market/site selection analysis, localized marketing content generation, and predictive risk analysis in the context of franchises.
- Providing insights for consultants and discussing knowledge base management.

Important Guidelines:
- Stay strictly within the domain of franchise consultancy.
- Do not provide legal advice, financial advice, or specific investment recommendations. Advise users to consult human experts for such matters.
- If a question is outside your scope or requires specific data you don't have, politely state your limitation and suggest consulting a human consultant or providing the necessary information.
- Maintain a helpful, professional, and informative tone.
- Focus on general information and guidance based on common franchise practices, as you do not have access to real-time market data or specific company documents unless explicitly provided in the conversation.
"""

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """
    Handles incoming chat messages, manages chat sessions,
    sends messages to the Gemini API within a session,
    and returns the generated response along with the session ID.
    """
    session_id = chat_message.session_id

    # If no session ID is provided or if the session doesn't exist, create a new one
    if not session_id or session_id not in chat_sessions:
        session_id = str(uuid.uuid4()) # Generate a new unique session ID
        # Start a new chat session with the model, including the system instruction
        chat_sessions[session_id] = model.start_chat(history=[
            {"role": "user", "parts": [SYSTEM_INSTRUCTION]},
            {"role": "model", "parts": ["Understood. I am ready to assist as an AI Franchise Agent. How can I help you with your franchise queries today?"]}
        ])
        print(f"New chat session created: {session_id}")
    else:
        print(f"Using existing chat session: {session_id}")

    # Get the chat session object
    chat = chat_sessions[session_id]

    try:
        # Send the message within the context of the chat session
        response = chat.send_message(chat_message.message)

        # Extract the text from the Gemini response.
        if response.candidates and response.candidates[0].content.parts:
            bot_response = response.candidates[0].content.parts[0].text
        else:
            bot_response = "No response from Gemini API."
            print(f"Gemini API response structure unexpected: {response}")

        # Return the response along with the session ID
        return {"response": bot_response, "session_id": session_id}
    except Exception as e:
        # Log the error and return an HTTP 500 internal server error
        print(f"Error communicating with Gemini API: {e}")
        # Optionally, remove the session if an error occurs to allow a fresh start
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        raise HTTPException(status_code=500, detail="Error communicating with the chatbot service.")

# To run this FastAPI application:
# 1. Save the code as `main.py`.
# 2. Install dependencies: `pip install fastapi uvicorn google-generativeai pydantic`
# 3. Run the server: `uvicorn main:app --reload`
#    The `--reload` flag is useful for development as it restarts the server on code changes.
