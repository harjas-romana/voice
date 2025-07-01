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
- **Phone Calling**: Integrate with Twilio to handle real telephone calls.

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

### WebSocket Endpoints

- `/ws/tts`: WebSocket endpoint for streaming TTS
- `/ws/conversation`: WebSocket endpoint for two-way conversation

## Environment Variables

Copy the example environment file and fill in your own values:
```
cp twilio_env.example .env
```

Required environment variables:
- `GROQ_API_KEY`: API key for Groq LLM API
- `OPENAI_API_KEY`: API key for OpenAI (used for TTS)

Optional environment variables:
- `ELEVENLABS_API_KEY`: API key for ElevenLabs (TTS fallback)
- `DEFAULT_SPEAKER_WAV`: Path to a WAV file for voice cloning

## Phone Calling with Twilio

The system now supports real-time phone calling integration using Twilio. This allows customers to call a phone number and interact with the QuantAI Restaurant agent over voice.

### Setting up Twilio

1. Create a Twilio account at https://www.twilio.com/
2. Buy a phone number from the Twilio console
3. Get your Twilio credentials (Account SID and Auth Token)
4. Add the following to your `.env` file:
```
TWILIO_ACCOUNT_SID=your-twilio-account-sid
TWILIO_AUTH_TOKEN=your-twilio-auth-token
TWILIO_PHONE_NUMBER=your-twilio-phone-number
TWILIO_WEBHOOK_BASE_URL=https://your-server-url.com
```

### Starting the Voice Server

1. Start the server as usual:
```
python server.py
```
The Twilio webhook server will start automatically alongside the FastAPI server.

2. Set up your Twilio webhooks by visiting:
```
http://your-server-url.com:8006/twilio/setup
```

3. Use the API endpoint to check Twilio status:
```
GET /twilio/status
```

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

## AWS Deployment

For deploying to AWS EC2:

1. Launch an EC2 instance (t2.micro is sufficient for testing)
2. Install Docker and Docker Compose
3. Clone this repository
4. Configure your `.env` file with your Twilio credentials
5. Run the deployment script:

```bash
./aws_deploy.sh
```

This will:
- Set up your environment
- Configure the server with your public IP
- Build and start Docker containers
- Set up Twilio webhooks

## Architecture

The system uses a layered architecture:
1. **FastAPI Server**: Handles HTTP requests and WebSockets
2. **Twilio Webhook Server**: Handles telephone calls
3. **QuantAI Agent**: Processes user queries
4. **Voice Layer**: Manages TTS with ElevenLabs and Coqui XTTS-v2

## Contributions

Contributions are welcome! Please feel free to submit a Pull Request. 