"""
QuantAI Restaurant API Server (Auckland, New Zealand)
This module provides a FastAPI server that integrates with `agent.py`
and uses Coqui XTTS-v2 for text-to-speech synthesis.

Endpoints:
1. Process text queries via `agent.py` (Kai).
2. Convert responses to speech using Coqui XTTS-v2.
3. Handle audio file playback.
4. Stream real-time TTS via WebSockets.
5. Integrate with Twilio for telephone calls.
6. Manage outbound calls for reservation confirmations.
"""

import os
# Fix OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import time
import signal
import sys
from pathlib import Path
import uuid
import io
import wave
import tempfile
from typing import Optional, Dict, Any, Tuple, List, Union, AsyncGenerator
from datetime import datetime
import speech_recognition as sr
import numpy as np
from pydub import AudioSegment
import voice  # Unified TTS functionality (Coqui XTTS v2 + ElevenLabs fallback)
from agent import QuantAIRestaurantAgent
import asyncio
import uvloop  # High-performance event loop replacement
import orjson  # Faster JSON processing
from concurrent.futures import ThreadPoolExecutor

# Twilio integration
from twilio_voice_app import TwilioVoiceServer
from twilio_outbound_calls import TwilioOutboundCaller
from flask import Flask, request, Response

# Set uvloop as the event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create temp directory for audio files
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

# Default speaker voice path
DEFAULT_SPEAKER_WAV = "/Users/harjas/Desktop/voice1.wav"

# Thread pool for parallel processing
THREAD_POOL = ThreadPoolExecutor(max_workers=8)

# Custom JSONResponse using orjson for better performance
class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS)

# Pydantic models for request/response
class TextRequest(BaseModel):
    """Model for text query requests."""
    text: str
    language: str = "en"

class TextResponse(BaseModel):
    """Model for text query responses."""
    success: bool
    response: str
    language: str = "en"
    error: Optional[str] = None

class TTSRequest(BaseModel):
    """Model for text-to-speech requests."""
    text: str
    language: str = "en"
    speaker: Optional[str] = None  # Optional voice name for cloning (ElevenLabs or Coqui)
    speaker_wav: Optional[str] = None  # Path/URL to reference WAV file for voice cloning

class TTSResponse(BaseModel):
    """Model for text-to-speech responses."""
    success: bool
    audio_url: Optional[str] = None
    text: str
    error: Optional[str] = None

class VoiceResponse(BaseModel):
    """Model for voice query responses."""
    user_text: str
    response_text: str
    audio_url: Optional[str] = None
    error: Optional[str] = None

class WebSocketTTSRequest(BaseModel):
    """Model for WebSocket TTS requests."""
    text: str
    language: str = "en"
    speaker: Optional[str] = None
    speaker_wav: Optional[str] = None
    chunk_size: int = 1024

# Models for phone call functionality
class OutboundCallRequest(BaseModel):
    """Model for outbound call requests."""
    phone_number: str
    purpose: str = "general_inquiry"
    context: Optional[Dict[str, Any]] = None

class OutboundCallResponse(BaseModel):
    """Model for outbound call responses."""
    success: bool
    call_id: Optional[str] = None
    status: str
    error: Optional[str] = None

class BulkCallRequest(BaseModel):
    """Model for bulk call requests."""
    phone_numbers: List[str]
    purpose: str = "general_inquiry"
    delay_seconds: int = 5

class TwilioStatusResponse(BaseModel):
    """Model for Twilio status response."""
    status: str
    twilio_server_running: bool
    phone_numbers: List[Dict[str, Any]] = []
    error: Optional[str] = None

class QuantAIRestaurantServer:
    """Server for QuantAI Restaurant Assistant."""
    
    def __init__(self, speaker_wav: str = None):
        """Initialize the server.
        
        Args:
            speaker_wav: Path to the speaker voice WAV file for XTTS-v2
        """
        self.speaker_wav = speaker_wav or DEFAULT_SPEAKER_WAV
        # TTS is now fully managed by voice.py; no local initialisation needed
        self._initialize_agent()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def _initialize_agent(self):
        """Initialize the QuantAI Restaurant Agent."""
        logger.info("Initializing QuantAI Restaurant Agent...")
        try:
            self.agent = QuantAIRestaurantAgent()
            logger.info("✓ QuantAI Restaurant Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise RuntimeError(f"Agent initialization failed: {e}")
    
    def transcribe_audio(self, audio_data: bytes) -> Tuple[str, bool]:
        """Transcribe audio data to text using Google Speech Recognition.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Tuple of (transcribed_text, success)
        """
        if not audio_data:
            return "", False
            
        try:
            # Convert bytes to an AudioData object that recognizer can use
            with io.BytesIO(audio_data) as f:
                with wave.open(f, 'rb') as wav:
                    audio = sr.AudioData(
                        wav.readframes(wav.getnframes()),
                        wav.getframerate(),
                        wav.getsampwidth()
                    )
            
            # Use Google Speech Recognition
            recognizer = sr.Recognizer()
            text = recognizer.recognize_google(audio)
            logger.info(f"Transcribed text: {text}")
            return text, True
        except sr.UnknownValueError:
            logger.warning("Speech could not be understood")
            return "", False
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition service error: {e}")
            return "", False
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", False
    
    async def get_agent_response(self, text: str) -> str:
        """Get a response from the QuantAI Restaurant Agent.
        
        Args:
            text: User query text
            
        Returns:
            Agent response text
        """
        logger.info(f"Getting agent response for: {text}")
        try:
            response = await self.agent.generate_response(text)
            logger.info(f"Agent response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error getting agent response: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}. Please try again."
    
    async def get_agent_response_streaming(self, text: str) -> AsyncGenerator[str, None]:
        """Get a streaming response from the QuantAI Restaurant Agent.
        
        Args:
            text: User query text
            
        Returns:
            AsyncGenerator yielding response text chunks
        """
        logger.info(f"Getting streaming agent response for: {text}")
        try:
            async for chunk in self.agent.answer_streaming(text):
                yield chunk
        except Exception as e:
            logger.error(f"Error getting streaming agent response: {e}")
            yield f"I'm sorry, I encountered an error: {str(e)}. Please try again."

def cleanup_old_files(directory: Path, max_age_hours: int = 1):
    """Clean up old temporary files."""
    current_time = datetime.now().timestamp()
    for file in directory.glob("*"):
        if current_time - file.stat().st_mtime > (max_age_hours * 3600):
            try:
                file.unlink()
                logger.info(f"Cleaned up old file: {file}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file}: {e}")

def validate_and_convert_audio(audio_data: bytes) -> Optional[bytes]:
    """Try to open the audio file as WAV, or convert to WAV if needed."""
    try:
        # Try to open as WAV
        with io.BytesIO(audio_data) as f:
            with wave.open(f, 'rb') as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                framerate = wav.getframerate()
                logger.info(f"WAV file validated: channels={channels}, sample_width={sample_width}, framerate={framerate}")
                return audio_data
    except Exception as e:
        logger.warning(f"Audio file is not a valid WAV: {e}. Attempting conversion with pydub.")
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            logger.info("Audio file converted to WAV using pydub.")
            return wav_io.getvalue()
        except Exception as e2:
            logger.error(f"Audio file could not be processed: {e2}")
            return None

# Server startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    """
    # Create server instance for global use
    global server
    global twilio_flask_app
    
    logger.info("Initializing QuantAI Restaurant server...")
    try:
        # Initialize server
        server = QuantAIRestaurantServer(speaker_wav=DEFAULT_SPEAKER_WAV)
        logger.info("Server initialized.")
        
        # Initialize Twilio Flask app in a separate thread
        from threading import Thread
        from twilio_voice_app import app as flask_app
        twilio_flask_app = flask_app
        
        # Start Twilio Flask app in a separate thread
        twilio_thread = Thread(target=lambda: twilio_flask_app.run(host='0.0.0.0', port=5000))
        twilio_thread.daemon = True
        twilio_thread.start()
        logger.info("Twilio voice server started on port 5000")
        
        yield
        
    except Exception as e:
        logger.error(f"Error initializing server: {e}")
        raise
    finally:
        # Clean up on shutdown
        logger.info("Shutting down server...")
        # Any cleanup code for the server

# Initialize FastAPI app
app = FastAPI(
    title="QuantAI Restaurant API",
    description="API server for QuantAI Restaurant's text and voice processing capabilities",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse  # Use orjson for all responses
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "agent": "ok",
            "tts": voice.engine() if voice.engine() else "unavailable"
        }
    }
    return status

@app.post("/query", response_model=TextResponse)
async def process_text_query(request: TextRequest):
    """Process a text query and return the response.
    
    This endpoint takes a text query, processes it through agent_groq.py,
    and returns the response.
    """
    try:
        logger.info(f"Processing text query: {request.text[:100]}...")
        
        # Get response from agent
        response = await app.state.server.get_agent_response(request.text)
        
        return TextResponse(
            success=True,
            response=response,
            language=request.language
        )
    except Exception as e:
        logger.error(f"Error processing text query: {e}")
        return TextResponse(
            success=False,
            response="",
            language=request.language,
            error=str(e)
        )

@app.post("/query-stream")
async def process_text_query_streaming(request: TextRequest):
    """Process a text query and stream the response.
    
    This endpoint takes a text query, processes it through agent_groq.py,
    and streams the response as it's generated.
    """
    try:
        logger.info(f"Processing streaming text query: {request.text[:100]}...")
        
        async def generate_response():
            async for chunk in app.state.server.get_agent_response_streaming(request.text):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error processing streaming text query: {e}")
        return StreamingResponse(
            iter([f"data: Error: {str(e)}\n\n"]),
            media_type="text/event-stream"
        )

@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """Convert text to speech using Coqui XTTS-v2.
    
    This endpoint takes text and converts it to speech using Coqui XTTS-v2.
    """
    try:
        logger.info(f"Processing TTS request: {request.text[:100]}...")
        
        # Synthesize speech using unified voice module
        audio_data = voice.speak(
            request.text,
            language=request.language,
            speaker=request.speaker,
            speaker_wav=request.speaker_wav or DEFAULT_SPEAKER_WAV,
        )
        
        if audio_data is None:
            return TTSResponse(
                success=False,
                text=request.text,
                error="Speech synthesis failed"
            )
        
        # Save audio to file
        filename = f"tts_{uuid.uuid4()}.wav"
        file_path = TEMP_DIR / filename
        
        with open(file_path, "wb") as f:
            f.write(audio_data)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_old_files, TEMP_DIR)
        
        return TTSResponse(
            success=True,
            audio_url=f"/audio/{filename}",
            text=request.text
        )
    except Exception as e:
        logger.error(f"Error processing TTS request: {e}")
        return TTSResponse(
            success=False,
            text=request.text,
            error=str(e)
        )

@app.post("/voice-query", response_model=VoiceResponse)
async def process_voice_query(
    audio_file: UploadFile = File(...),
    language: str = Form("en"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Process a voice query and return both text and synthesized speech response.
    
    This endpoint takes an audio file, transcribes it using Google Speech Recognition,
    processes the text through agent_groq.py, and returns both the text response
    and synthesized speech.
    """
    try:
        logger.info(f"Processing voice query from file: {audio_file.filename}")
        
        # Read audio file
        audio_data = await audio_file.read()
        logger.info(f"Received audio data: {len(audio_data)} bytes")
        
        # Validate and convert audio
        wav_data = validate_and_convert_audio(audio_data)
        if not wav_data:
            return VoiceResponse(
                user_text="",
                response_text="Invalid audio format",
                error="Invalid audio format"
            )
        
        # Transcribe audio
        text, success = app.state.server.transcribe_audio(wav_data)
        if not success:
            return VoiceResponse(
                user_text="",
                response_text="Could not transcribe audio",
                error="Speech recognition failed"
            )
        
        logger.info(f"Transcribed text: {text}")
        
        # Get response from agent
        response = await app.state.server.get_agent_response(text)
        
        # Synthesize speech
        audio_data = voice.speak(response, language=language, speaker_wav=DEFAULT_SPEAKER_WAV)
        
        if audio_data is None:
            return VoiceResponse(
                user_text=text,
                response_text=response,
                error="Speech synthesis failed"
            )
        
        # Save audio to file
        filename = f"response_{uuid.uuid4()}.wav"
        file_path = TEMP_DIR / filename
        
        with open(file_path, "wb") as f:
            f.write(audio_data)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_old_files, TEMP_DIR)
        
        return VoiceResponse(
            user_text=text,
            response_text=response,
            audio_url=f"/audio/{filename}"
        )
    except Exception as e:
        logger.error(f"Error processing voice query: {e}")
        return VoiceResponse(
            user_text="",
            response_text="",
            error=str(e)
        )

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio files."""
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )

@app.post("/tts-stream")
async def tts_stream(request: TTSRequest):
    """Stream TTS audio directly."""
    try:
        audio_data = voice.speak(
            request.text,
            language=request.language,
            speaker=request.speaker,
            speaker_wav=request.speaker_wav or DEFAULT_SPEAKER_WAV,
        )
        if not audio_data:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
    except Exception as e:
        logger.error(f"TTS stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """WebSocket endpoint for real-time TTS streaming.
    
    This endpoint allows clients to connect via WebSocket and receive
    audio chunks as they are generated.
    """
    await websocket.accept()
    try:
        while True:
            # Receive and parse the request
            data = await websocket.receive_text()
            try:
                # Use orjson for faster parsing
                request_dict = orjson.loads(data)
                request_data = WebSocketTTSRequest.parse_obj(request_dict)
            except Exception as e:
                await websocket.send_json({"error": f"Invalid request format: {str(e)}"}, 
                                         dumps=lambda v: orjson.dumps(v).decode())
                continue
            
            logger.info(f"WebSocket TTS request: {request_data.text[:50]}...")
            
            # Stream audio chunks
            try:
                # First, send a start message
                await websocket.send_json({"status": "start", "text": request_data.text},
                                         dumps=lambda v: orjson.dumps(v).decode())
                
                # Stream the audio chunks
                chunk_count = 0
                async for chunk in voice.speak_streaming(
                    request_data.text,
                    language=request_data.language,
                    speaker=request_data.speaker,
                    speaker_wav=request_data.speaker_wav or DEFAULT_SPEAKER_WAV,
                    chunk_size=request_data.chunk_size
                ):
                    await websocket.send_bytes(chunk)
                    chunk_count += 1
                
                # Send an end message
                await websocket.send_json({"status": "end", "chunks": chunk_count},
                                         dumps=lambda v: orjson.dumps(v).decode())
                
            except Exception as e:
                logger.error(f"WebSocket TTS streaming error: {e}")
                await websocket.send_json({"error": f"TTS streaming failed: {str(e)}"},
                                         dumps=lambda v: orjson.dumps(v).decode())
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": f"WebSocket error: {str(e)}"},
                                     dumps=lambda v: orjson.dumps(v).decode())
        except:
            pass

@app.websocket("/ws/conversation")
async def websocket_conversation(websocket: WebSocket):
    """WebSocket endpoint for real-time conversation.
    
    This endpoint allows clients to connect via WebSocket, send text queries,
    and receive both text responses and audio chunks as they are generated.
    """
    await websocket.accept()
    try:
        while True:
            # Receive the query
            data = await websocket.receive_json(mode="text")
            # Parse with orjson for better performance
            data_dict = orjson.loads(data) if isinstance(data, str) else data
            query = data_dict.get("text", "")
            language = data_dict.get("language", "en")
            
            if not query:
                await websocket.send_json({"error": "No query provided"},
                                         dumps=lambda v: orjson.dumps(v).decode())
                continue
            
            logger.info(f"WebSocket conversation query: {query[:50]}...")
            
            try:
                # Send a start message
                await websocket.send_json({"status": "processing", "type": "text"},
                                         dumps=lambda v: orjson.dumps(v).decode())
                
                # Process in parallel: generate response and prepare for TTS
                response_chunks = []
                
                # Stream text response
                async for chunk in app.state.server.get_agent_response_streaming(query):
                    response_chunks.append(chunk)
                    await websocket.send_json({
                        "type": "text",
                        "chunk": chunk
                    }, dumps=lambda v: orjson.dumps(v).decode())
                
                # Combine chunks for full response
                full_response = "".join(response_chunks)
                
                # Send audio start message
                await websocket.send_json({"status": "processing", "type": "audio"},
                                         dumps=lambda v: orjson.dumps(v).decode())
                
                # Stream audio chunks
                async for audio_chunk in voice.speak_streaming(
                    full_response,
                    language=language,
                    speaker_wav=DEFAULT_SPEAKER_WAV
                ):
                    await websocket.send_bytes(audio_chunk)
                
                # Send completion message
                await websocket.send_json({
                    "status": "complete",
                    "response": full_response
                }, dumps=lambda v: orjson.dumps(v).decode())
                
            except Exception as e:
                logger.error(f"WebSocket conversation error: {e}")
                await websocket.send_json({"error": f"Conversation failed: {str(e)}"},
                                         dumps=lambda v: orjson.dumps(v).decode())
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": f"WebSocket error: {str(e)}"},
                                     dumps=lambda v: orjson.dumps(v).decode())
        except:
            pass

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "QuantAI Restaurant API",
        "version": "1.0.0",
        "description": "API server for QuantAI Restaurant's text and voice processing capabilities",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/query", "method": "POST", "description": "Process text queries"},
            {"path": "/query-stream", "method": "POST", "description": "Stream text query responses"},
            {"path": "/tts", "method": "POST", "description": "Convert text to speech"},
            {"path": "/voice-query", "method": "POST", "description": "Process voice queries"},
            {"path": "/audio/{filename}", "method": "GET", "description": "Serve audio files"},
            {"path": "/tts-stream", "method": "POST", "description": "Stream TTS audio"},
            {"path": "/ws/tts", "method": "WebSocket", "description": "WebSocket for real-time TTS streaming"},
            {"path": "/ws/conversation", "method": "WebSocket", "description": "WebSocket for real-time conversation"}
        ]
    }

# -----------------------------------------------------------------------------
# Compatibility endpoints for frontend (TextMode & VoiceMode)
# -----------------------------------------------------------------------------

# Static language list (could be expanded or generated dynamically)
LANGUAGE_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
}

@app.get("/languages-text")
async def get_text_languages():
    """Return supported languages for text chat."""
    return {"success": True, "languages": LANGUAGE_MAP}

@app.get("/languages-voice")
async def get_voice_languages():
    """Return supported languages for voice chat (same set for now)."""
    return {"success": True, "languages": LANGUAGE_MAP}

# Alias /text-query to /query for backward compatibility with frontend
@app.post("/text-query", response_model=TextResponse)
async def process_text_query_alias(request: TextRequest):
    return await process_text_query(request)

# -----------------------------------------------------------------------------
# Phone call endpoints with Twilio
# -----------------------------------------------------------------------------

@app.post("/calls/outbound", response_model=OutboundCallResponse)
async def make_outbound_call(request: OutboundCallRequest):
    """Make an outbound call to a customer."""
    try:
        # Create caller
        caller = TwilioOutboundCaller()
        
        # Make the call
        result = await caller.make_call(
            request.phone_number,
            request.purpose,
            request.context
        )
        
        if result.get("status") == "initiated":
            return OutboundCallResponse(
                success=True,
                call_id=result.get("call_id"),
                status="initiated"
            )
        else:
            return OutboundCallResponse(
                success=False,
                status="failed",
                error=result.get("error", "Unknown error")
            )
    except Exception as e:
        logger.error(f"Error making outbound call: {e}")
        return OutboundCallResponse(
            success=False,
            status="error",
            error=str(e)
        )

@app.post("/calls/bulk", response_model=List[OutboundCallResponse])
async def make_bulk_calls(request: BulkCallRequest):
    """Make multiple outbound calls."""
    try:
        # Create caller
        caller = TwilioOutboundCaller()
        
        # Make bulk calls
        results = await caller.make_bulk_calls(
            request.phone_numbers,
            request.purpose,
            request.delay_seconds
        )
        
        # Convert to response model
        responses = []
        for result in results:
            success = result.get("status") == "initiated"
            response = OutboundCallResponse(
                success=success,
                call_id=result.get("call_id"),
                status=result.get("status"),
                error=result.get("error") if not success else None
            )
            responses.append(response)
        
        return responses
    except Exception as e:
        logger.error(f"Error making bulk calls: {e}")
        return [
            OutboundCallResponse(
                success=False,
                status="error",
                error=str(e)
            )
        ]

@app.get("/twilio/status")
async def get_twilio_status():
    """Get the status of the Twilio connection."""
    try:
        # Check if Twilio credentials are set
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not all([account_sid, auth_token, twilio_number]):
            return {
                "status": "not_configured",
                "error": "Missing Twilio credentials"
            }
        
        # Check if the Flask app is running
        twilio_server_running = twilio_flask_app is not None
        
        # Get Twilio phone numbers
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        phone_numbers = list(client.incoming_phone_numbers.list(limit=10))
        
        return {
            "status": "ok",
            "twilio_server_running": twilio_server_running,
            "phone_numbers": [
                {
                    "phone_number": number.phone_number,
                    "friendly_name": number.friendly_name,
                    "sid": number.sid
                }
                for number in phone_numbers
            ]
        }
    except Exception as e:
        logger.error(f"Error checking Twilio status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/twilio/setup")
async def setup_twilio():
    """Set up Twilio webhooks for the server."""
    try:
        # Get Twilio credentials
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not all([account_sid, auth_token, twilio_number]):
            return {
                "status": "error", 
                "message": "Missing Twilio credentials. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER environment variables."
            }
        
        # Get server public URL
        server_url = os.getenv("TWILIO_WEBHOOK_BASE_URL")
        if not server_url:
            return {
                "status": "error",
                "message": "Missing TWILIO_WEBHOOK_BASE_URL environment variable."
            }
        
        # Set up webhooks
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        
        # Update the phone number with webhooks
        try:
            # Find the phone number in the account
            incoming_phone_numbers = client.incoming_phone_numbers.list(phone_number=twilio_number)
            if not incoming_phone_numbers:
                return {
                    "status": "error",
                    "message": f"Phone number {twilio_number} not found in your Twilio account."
                }
            
            # Update the first matching phone number
            phone_number = incoming_phone_numbers[0]
            phone_number.update(
                voice_url=f"{server_url}/voice",
                voice_method="POST"
            )
            
            return {
                "status": "ok",
                "message": f"Twilio webhooks set up successfully for {twilio_number}",
                "voice_url": f"{server_url}/voice"
            }
        except Exception as e:
            logger.error(f"Error updating phone number: {e}")
            return {
                "status": "error",
                "message": f"Error updating phone number: {str(e)}"
            }
    except Exception as e:
        logger.error(f"Error setting up Twilio: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8006,
        reload=False,  # Disable reload in production for better performance
        log_level="info",
        loop="uvloop",  # Use uvloop for better performance
        workers=2  # Use multiple workers for better concurrency
    )