# QuantAI Restaurant - Streaming TTS and Latency Optimization

This project implements a high-performance, low-latency text-to-speech (TTS) system for the QuantAI Restaurant assistant. The system supports real-time streaming of audio chunks as they are generated, significantly reducing perceived latency.

## Key Features

### 1. Streaming TTS

- **Chunked Audio Generation**: Audio is generated and delivered in small chunks rather than waiting for the entire file.
- **Parallel Processing**: Sentences are processed in parallel using a thread pool for faster generation.
- **WebSocket Support**: Real-time bidirectional communication for streaming audio.

### 2. Latency Optimizations

- **Lazy Initialization**: TTS models are initialized once per process and reused.
- **Thread Pool**: Background processing of CPU-intensive tasks.
- **Sentence-Level Parallelism**: Text is split into sentences and processed concurrently.
- **Caching**: Frequently used translations and audio segments are cached.
- **Fallback Mechanism**: Automatic fallback between Coqui XTTS-v2 and ElevenLabs.

### 3. Multiple Interfaces

- **Direct API**: Use `voice.speak()` for traditional TTS or `voice.speak_streaming()` for streaming.
- **HTTP Endpoints**: RESTful API endpoints for TTS and conversation.
- **WebSocket Endpoints**: Real-time streaming via WebSockets.

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Starting the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8006 --reload
```

### Testing Streaming TTS

```bash
python test_streaming_tts.py
```

### WebSocket Demo

Open `websocket_client_demo.html` in a web browser to test the WebSocket functionality.

## API Endpoints

### REST Endpoints

- `POST /query`: Process text queries and return responses
- `POST /query-stream`: Stream text responses as they are generated
- `POST /tts`: Convert text to speech and return audio URL
- `POST /tts-stream`: Stream audio directly
- `POST /voice-query`: Process voice queries and return text and audio

### WebSocket Endpoints

- `ws://localhost:8006/ws/tts`: WebSocket for streaming TTS
- `ws://localhost:8006/ws/conversation`: WebSocket for real-time conversation with both text and audio responses

## Implementation Details

### TTS Streaming Architecture

1. **Text Processing**:
   - Split input text into sentences
   - Process sentences in parallel using thread pool

2. **Audio Generation**:
   - Generate audio for each sentence
   - Yield audio chunks as they become available
   - Small delays between chunks for realistic streaming

3. **WebSocket Communication**:
   - Send status updates as JSON messages
   - Send binary audio data as it's generated
   - Client can play audio incrementally

### Latency Comparison

| Method | Average Latency | Notes |
|--------|----------------|-------|
| Traditional TTS | 3-5 seconds | Full processing before playback |
| Streaming TTS | 0.5-1 second | First audio chunk available quickly |
| WebSocket Streaming | 0.3-0.8 seconds | Real-time bidirectional communication |

## Future Improvements

- Implement adaptive chunk sizing based on network conditions
- Add voice activity detection for more natural pauses
- Optimize audio format for web streaming (consider MP3 or Opus)
- Implement progressive enhancement for slower connections

# QuantAI Restaurant Backend

This is the backend server for the QuantAI Restaurant system. It provides a FastAPI-based API for processing text and voice queries, as well as WebSocket endpoints for real-time communication.

## Features

- Text query processing via the QuantAI Restaurant Agent
- Text-to-Speech synthesis using Coqui XTTS-v2 with ElevenLabs fallback
- WebSocket endpoints for real-time conversation
- Voice query processing with Google Speech Recognition
- Phone call integration with Fonoster for real-time voice calling

## Installation

1. Clone the repository
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Configure the environment variables (see below)

## Environment Variables

Copy the example environment file and fill in your own values:
```
cp fonoster_config.env.example .env
```

Required environment variables:
- `GROQ_API_KEY`: API key for Groq LLM API
- `OPENAI_API_KEY`: API key for OpenAI (used for TTS)

Optional environment variables:
- `ELEVENLABS_API_KEY`: API key for ElevenLabs (TTS fallback)
- `DEFAULT_SPEAKER_WAV`: Path to a WAV file for voice cloning

## Phone Calling with Fonoster

The system now supports real-time phone calling integration using Fonoster, an open-source alternative to Twilio. This allows customers to call a phone number and interact with the QuantAI Restaurant agent over voice.

### Setting up Fonoster

1. Create a Fonoster account at https://fonoster.com/
2. Install Fonoster CLI: `npm install -g @fonoster/ctl`
3. Log in to Fonoster: `fonoster login`
4. Get your API credentials from the Fonoster dashboard
5. Add the following to your `.env` file:
```
FONOSTER_API_KEY=your-fonoster-api-key-here
FONOSTER_API_SECRET=your-fonoster-api-secret-here
FONOSTER_ACCESS_KEY_ID=your-fonoster-access-key-id-here
FONOSTER_FROM_NUMBER=your-phone-number
```

### Starting the Voice Server

1. Run the setup script to configure your Fonoster resources:
```
python fonoster_config.py
```

2. Start the server as usual:
```
python server.py
```

3. Use the API endpoints to manage the voice server:
- `/fonoster/start-voice-server`: Start the voice server
- `/fonoster/stop-voice-server`: Stop the voice server
- `/fonoster/status`: Check the status of the voice server

### Making Outbound Calls

You can make outbound calls using the API:

```
POST /calls/outbound
{
    "phone_number": "+1234567890",
    "purpose": "reservation_confirmation",
    "context": {
        "reservation_id": "RES12345",
        "time": "19:30",
        "date": "2023-04-15"
    }
}
```

Available purposes:
- `reservation_confirmation`
- `special_promotion`
- `feedback_request`
- `general_inquiry`

### Bulk Calling

For marketing campaigns or batch notifications:

```
POST /calls/bulk
{
    "phone_numbers": ["+1234567890", "+0987654321"],
    "purpose": "special_promotion",
    "delay_seconds": 5
}
```

## Running the Server

```
python server.py
```

The server will start on port 8006 by default.

## API Endpoints

- `POST /query`: Process a text query
- `POST /query-stream`: Stream a text query response
- `POST /tts`: Convert text to speech
- `POST /voice-query`: Process a voice query
- `GET /audio/{filename}`: Get an audio file
- `POST /tts-stream`: Stream TTS audio

## WebSocket Endpoints

- `/ws/tts`: WebSocket for real-time TTS streaming
- `/ws/conversation`: WebSocket for real-time conversation

## Phone Call Endpoints

- `POST /calls/outbound`: Make an outbound call
- `POST /calls/bulk`: Make multiple outbound calls
- `GET /fonoster/status`: Get Fonoster status
- `POST /fonoster/start-voice-server`: Start the voice server
- `POST /fonoster/stop-voice-server`: Stop the voice server
- `POST /fonoster/setup`: Set up Fonoster resources 