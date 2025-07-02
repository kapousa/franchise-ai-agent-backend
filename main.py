from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional # Import Optional
import google.generativeai as genai
import os
import uuid # Import uuid for generating unique session IDs
import base64 # For decoding base64 image data

# Initialize FastAPI app
app = FastAPI()

# Configure CORS to allow requests from your React frontend.
# IMPORTANT: In a production environment, restrict origins to your specific frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow your React app to connect
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
    message: Optional[str] = None # Make message explicitly Optional
    session_id: Optional[str] = None # Add an optional session_id field
    file_content: Optional[str] = None # Base64 encoded content for images, or raw text for text files
    file_mime_type: Optional[str] = None # MIME type of the uploaded file

# Configure the Gemini API client
# It's highly recommended to load your API key from environment variables
# For demonstration, a placeholder is used. Replace with your actual API key.
# Example: export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key="AIzaSyBbFpqTeTKwpD1RPjSETrroELVwSX4Xu7M") # Replace with your actual Gemini API Key or load from env

# --- Define Tools for "Live Data" Integration ---
# In a real application, these functions would make API calls to external services
# or query databases to fetch actual live data.
def get_franchise_market_data(industry: str = None) -> str:
    """
    Retrieves simulated market data for a given franchise industry.
    If no industry is specified, provides general market trends.

    Args:
        industry (str, optional): The specific industry (e.g., "fast food", "fitness").
                                  Defaults to None for general trends.

    Returns:
        str: A string containing the simulated market data.
    """
    if industry:
        # This is mock data. Replace with actual API calls or database lookups.
        mock_data = {
            "fast food": "The fast food industry is seeing a 5% growth in Q2 2024, with a strong emphasis on drive-thru efficiency and plant-based options. Average initial investment: $200,000 - $1,000,000.",
            "fitness": "The fitness franchise sector is experiencing a 7% annual growth, driven by boutique studios and personalized training. Average initial investment: $150,000 - $500,000.",
            "education": "Education franchises are stable, with a 3% growth, focusing on STEM and tutoring services. Average initial investment: $100,000 - $300,000.",
            "retail": "Retail franchises are adapting to e-commerce integration, with varied growth depending on niche. Average initial investment: $50,000 - $250,000."
        }
        return mock_data.get(industry.lower(), f"No specific data found for {industry} franchise industry. General trends indicate a resilient market.")
    else:
        return "General franchise market trends show steady growth across service-based and food sectors, with increasing demand for sustainable and tech-integrated models."

# Initialize the Generative Model and register the tools
# The 'tools' argument makes the model aware of the functions it can "call".
model = genai.GenerativeModel('gemini-2.0-flash', tools=[get_franchise_market_data])

# Define the system instruction for the AI Franchise Agent
# This prompt guides the model to act as a specialized franchise consultant.
SYSTEM_INSTRUCTION = """
You are an AI Franchise Agent, specifically designed to provide information and guidance solely within the domain of franchise consultancy services. Your role is to act as an intelligent, always-available digital assistant for prospective franchisees, franchisors, and human consultants.

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
- Discussing operations analytics, compliance monitoring, market and site selection analysis, localized marketing content generation, and predictive risk analysis in the context of franchises.
- You have access to specialized tools to retrieve relevant information to assist with your responses.

Important Guidelines:
- Your responses must strictly adhere to the domain of franchise consultancy.
- You cannot answer questions or engage in discussions outside of franchise-related topics. If a user asks a question unrelated to franchises, politely inform them that you are an AI Franchise Agent and can only assist with franchise-specific inquiries.
- You can, however, engage in normal greetings and brief opening discussions before transitioning to franchise-related topics.
- Do not provide legal advice, financial advice, or specific investment recommendations. Always advise users to consult human experts for such matters.
- If a question requires specific data you don't have (even with tools), politely state your limitation and suggest consulting a human consultant or providing the necessary information.
- Maintain a helpful, professional, and informative tone.
- Focus on general information and guidance based on common franchise practices, utilizing your tools for enhanced data where appropriate.
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
        # Construct the parts for the Gemini API call
        contents_parts = []

        # Add the user's text message if available and not empty
        if chat_message.message is not None and chat_message.message.strip() != "":
            contents_parts.append({"text": chat_message.message})

        # Add the file content if available
        if chat_message.file_content and chat_message.file_mime_type:
            if chat_message.file_mime_type.startswith('image/'):
                contents_parts.append({
                    "inlineData": {
                        "mimeType": chat_message.file_mime_type,
                        "data": chat_message.file_content # Base64 encoded image data
                    }
                })
            elif chat_message.file_mime_type == 'text/plain':
                contents_parts.append({"text": f"User uploaded a text document:\n{chat_message.file_content}"})
            else:
                # Handle other file types or provide a message
                contents_parts.append({"text": f"User uploaded a file of unsupported type: {chat_message.file_mime_type}. Content not processed."})

        # Ensure there's at least one part to send to the model
        if not contents_parts:
            raise HTTPException(status_code=400, detail="No message or file content provided.")

        # Send the message (and file content) within the context of the chat session
        response_stream = chat.send_message(contents_parts, stream=True)

        bot_response_parts = []
        # Loop through the response stream to handle potential tool calls
        for chunk in response_stream:
            # Ensure candidates and content exist before accessing parts
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                print(f"Warning: Unexpected chunk structure or empty content parts: {chunk}")
                continue # Skip to the next chunk

            # Iterate over each part within the chunk
            for part in chunk.candidates[0].content.parts:
                if part.text:
                    # If the part contains text, append it to the response
                    bot_response_parts.append(part.text)
                elif part.function_call:
                    # If the part contains a function call, execute the tool
                    function_call = part.function_call
                    function_name = function_call.name
                    function_args = {k: v for k, v in function_call.args.items()}

                    print(f"Model requested tool: {function_name} with args: {function_args}")

                    # Execute the corresponding Python function (tool)
                    if function_name == "get_franchise_market_data":
                        tool_output = get_franchise_market_data(**function_args)
                        print(f"Tool output: {tool_output}")

                        # Send the tool's output back to the model for it to generate the final response
                        tool_response_stream = chat.send_message(
                            genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=function_name,
                                    response={
                                        "content": tool_output
                                    }
                                )
                            ),
                            stream=True # Continue streaming for the model's response after tool output
                        )
                        # Append the model's response after processing the tool output
                        for tool_chunk in tool_response_stream:
                            if not tool_chunk.candidates or not tool_chunk.candidates[0].content or not tool_chunk.candidates[0].content.parts:
                                print(f"Warning: Unexpected tool response chunk structure or empty content parts: {tool_chunk}")
                                continue
                            for tool_part in tool_chunk.candidates[0].content.parts:
                                if tool_part.text:
                                    bot_response_parts.append(tool_part.text)
                    else:
                        bot_response_parts.append(f"Sorry, I don't have a tool for '{function_name}'.")
                # You can add more elif conditions here for other tool types or content types
                # if your application needs to handle them.

        bot_response = "".join(bot_response_parts)

        if not bot_response:
            bot_response = "No response from Gemini API."
            print(f"Gemini API response structure unexpected or empty: {response_stream}")

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
