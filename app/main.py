"""
It is a discussion between human 'user' and a chatbot 'assistant'
The discussion uses speech-to-text (STT), then an ollama LLM as 'assistant', then text-to-speech (TTS)
The client, web browser, stores history of chat between 'user' and 'assistant'

The process:
The user records his question using the button from the UI
The client send the user's question to the API, using sound format, for voice recognition.
The solution used for speech-to-text is openai-whisper
The endpoint for posting audio question is /speech-to-text, the server do not store text, only respond text of the question
The client store the text of the question in the chat history and send the chat history to /start-chat
The server launch the background task that prepare the response
The client request /stream_chat to get the ChatResponseChunk chunks (status,text and audio) one-by-one
Once the 'FINISHED' status received in the last chunk, the response is finished
The client store the full response of the 'assistant' in the chat history
The client is ready for another question from 'user'
The API server store chat history only during the preparation of a response
"""
from http.client import HTTPResponse
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, status, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pathlib import Path
from chatbot_server import ChatbotServer, logger, DEFAULT_LOCAL_OLLAMA_SERVER
from schemas import ChatResponseChunk, Message, NewChat
from typing import List
LOCAL_WEB_SERVER = True
LOCAL_OLLAMA_SERVER = True

if LOCAL_WEB_SERVER:
    web_server_host_address = "127.0.0.1"
    web_server_port = 8080
    website_address = f"http://{web_server_host_address}:{web_server_port}"
else:
    web_server_host_address = "192.168.1.150"
    web_server_port = 8000
    website_address = 'https://anyapp.julienmasclef2.synology.me:54321'

if LOCAL_OLLAMA_SERVER:
    ollama_server = DEFAULT_LOCAL_OLLAMA_SERVER
else:
    ollama_server = "https://ollama-api.julienmasclef2.synology.me:60001"

logger.info(f"Ollama server: {ollama_server}")
logger.info(f"Website address: {website_address}")
# Directory setup
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Template setup
templates = Jinja2Templates(directory=BASE_DIR / "templates")

chatbot_server = ChatbotServer(lang='en', ollama_server=ollama_server)


@app.get("/")
async def serve_frontend(request: Request):
    """
    Render the chat room HTML.
    """
    return templates.TemplateResponse("chat_room.html", {"request": request})


# Serve the dynamic host.js
@app.get("/host.js")
async def get_host_js(request: Request):
    return templates.TemplateResponse(request=request,
                                      name="host.js",
                                      context={"website_address": website_address},
                                      media_type="application/javascript")


@app.post("/speech-to-text")
async def post_sound(sound_file: UploadFile = File(...), messages: str = Form(...)):
    """
    Process uploaded sound and return transcribed text.
    Recommended Process:
    1. In browser, capture audio and convert to WAV 16kHz, 16bits, Mono
    2. Normalize audio in browser
    3. Send to Whisper API
    4. Return transcription
    """
    # Parse the messages JSON string into a list of Message objects

    try:
        parsed_messages = json.loads(messages)  # Convert JSON string into a Python object (list of Message)
        messages = [Message(**message) for message in parsed_messages]
    except json.JSONDecodeError as e:
        return {"error": "Invalid JSON format for messages", "details": str(e)}

    # Read the uploaded audio file
    file_content = await sound_file.read()
    # Call the simplified transcription method
    transcription = chatbot_server.catch_user_question_from_audio_browser(file_content, messages)

    return {"text": transcription}


@app.post("/start_chat", response_model=NewChat)
async def start_chat(messages: List[Message], background_tasks: BackgroundTasks):
    """
    Process to assistant response based on a chat history, launch the job in background tasks

    Args:
        messages (List[Message]): A list of user and assistant messages history
    Returns:
        NewChat: The result of the chat session, the background task will populate response in an async way.
    """
    # Initialize a new chat session with the provided messages
    new_chat = chatbot_server.start_chat(messages=messages)

    # Add a task to prepare the response for the newly created chat session
    logger.info(f"Background process for chat {new_chat.session_id} is starting...")
    background_tasks.add_task(chatbot_server.build_chat_response, session_id=new_chat.session_id)
    # await chatbot_server.build_chat_response(session_id=new_chat.session_id)

    return new_chat


@app.get("/stream_chat", response_model=ChatResponseChunk | None)
async def stream_chat(session_id: str):
    """
    Get one chunk of the response: text and sound
    audio chunk is bytes, produced by coqui-tts tts() method and normalized:
    """
    if session_id in chatbot_server.chat_sessions.keys():
        return await chatbot_server.stream_chat(session_id=session_id)
    else:
        return JSONResponse(content='Session ID not found', status_code=status.HTTP_404_NOT_FOUND)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app="main:app",  # "main" is the filename, "app" is the FastAPI instance
        host=web_server_host_address,  # Host address
        port=web_server_port,  # Port number
        reload=True  # Enable auto-reload for development
    )
