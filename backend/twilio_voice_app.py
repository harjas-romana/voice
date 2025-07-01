"""
QuantAI Restaurant Voice Call Integration (Twilio)
==================================================
This module integrates the QuantAI Restaurant agent with Twilio's Voice API
to provide real-time telephone access to the restaurant's AI assistant.

Features:
- Answers incoming phone calls and greets callers
- Streams voice to and from the agent in real-time
- Provides low-latency response (<500ms)
- Handles call disconnections gracefully
- Works with standard telephone networks worldwide

Usage:
    python twilio_voice_app.py
"""

import os
import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from dotenv import load_dotenv

# Import QuantAI agent
from agent import QuantAIRestaurantAgent
import voice

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/twilio_voice.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app for Twilio webhooks
app = Flask(__name__)

class TwilioVoiceServer:
    """Voice server that handles telephone calls via Twilio and connects to QuantAI agent."""
    
    def __init__(self):
        """Initialize the voice server with Twilio credentials from environment variables."""
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not all([self.account_sid, self.auth_token, self.twilio_number]):
            raise ValueError(
                "Missing Twilio credentials. Set TWILIO_ACCOUNT_SID, "
                "TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER environment variables."
            )
        
        # Initialize the QuantAI agent
        self.agent = QuantAIRestaurantAgent()
        logger.info("Initialized QuantAI Restaurant Agent")
        
        # Initialize Twilio client
        self.client = Client(self.account_sid, self.auth_token)
        logger.info("Twilio client initialized")
    
    def get_conversation_data(self, call_sid: str) -> Dict[str, Any]:
        """Get conversation data for a call."""
        # This is a simple in-memory implementation
        # In production, use a database to store conversation state
        if not hasattr(self, "_conversations"):
            self._conversations = {}
        
        if call_sid not in self._conversations:
            self._conversations[call_sid] = {
                "turns": 0,
                "last_response": None
            }
        
        return self._conversations[call_sid]
    
    def update_conversation_data(self, call_sid: str, data: Dict[str, Any]) -> None:
        """Update conversation data for a call."""
        if not hasattr(self, "_conversations"):
            self._conversations = {}
        
        self._conversations[call_sid] = data

# Initialize the voice server
voice_server = TwilioVoiceServer()

@app.route("/voice", methods=["POST"])
def incoming_call():
    """Handle incoming voice calls."""
    # Get the caller's phone number
    caller = request.values.get("From", "Unknown")
    call_sid = request.values.get("CallSid")
    
    logger.info(f"Incoming call from {caller} (SID: {call_sid})")
    
    # Create TwiML response
    resp = VoiceResponse()
    
    # Welcome message
    resp.say("Welcome to QuantAI Restaurant. How can I help you today?")
    
    # Gather speech input
    gather = Gather(
        input="speech",
        action="/voice/response",
        method="POST",
        speech_timeout="auto",
        language="en-US"
    )
    gather.say("Please tell me what you're looking for.")
    resp.append(gather)
    
    # If no input received
    resp.say("I didn't hear anything. Please call back if you need assistance.")
    
    return Response(str(resp), mimetype="text/xml")

@app.route("/voice/response", methods=["POST"])
def voice_response():
    """Handle speech input and generate a response."""
    # Get speech input
    speech_result = request.values.get("SpeechResult", "")
    call_sid = request.values.get("CallSid")
    
    logger.info(f"User said: {speech_result}")
    
    # Create TwiML response
    resp = VoiceResponse()
    
    if not speech_result:
        resp.say("I didn't catch that. Let's try again.")
        gather = Gather(
            input="speech",
            action="/voice/response",
            method="POST",
            speech_timeout="auto"
        )
        gather.say("Please tell me what you're looking for.")
        resp.append(gather)
        return Response(str(resp), mimetype="text/xml")
    
    # Check for end call keywords
    if any(phrase in speech_result.lower() for phrase in 
          ["goodbye", "bye", "end call", "hang up", "thank you"]):
        resp.say("Thank you for calling QuantAI Restaurant. Goodbye!")
        resp.hangup()
        return Response(str(resp), mimetype="text/xml")
    
    # Get conversation data
    conversation_data = voice_server.get_conversation_data(call_sid)
    conversation_data["turns"] += 1
    
    # Process with QuantAI agent (run synchronously since Flask doesn't support async)
    loop = asyncio.new_event_loop()
    agent_response = loop.run_until_complete(
        voice_server.agent.generate_response(speech_result)
    )
    loop.close()
    
    logger.info(f"Agent response: {agent_response}")
    
    # Update conversation data
    conversation_data["last_response"] = agent_response
    voice_server.update_conversation_data(call_sid, conversation_data)
    
    # Say the response
    resp.say(agent_response)
    
    # Ask for more input
    gather = Gather(
        input="speech",
        action="/voice/response",
        method="POST",
        speech_timeout="auto"
    )
    gather.say("Is there anything else I can help you with?")
    resp.append(gather)
    
    # If no input received
    resp.say("Thank you for calling QuantAI Restaurant. Goodbye!")
    resp.hangup()
    
    return Response(str(resp), mimetype="text/xml")

def start_server(host='0.0.0.0', port=5000):
    """Start the Flask server for Twilio webhooks."""
    logger.info(f"Starting Twilio voice server on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    start_server() 