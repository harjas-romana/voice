"""
QuantAI Restaurant Voice Call Integration (Fonoster)
===================================================
This module integrates the QuantAI Restaurant agent with Fonoster's Voice API
to provide real-time telephone access to the restaurant's AI assistant.

Features:
- Answers incoming phone calls and greets callers
- Streams voice to and from the agent in real-time
- Provides low-latency response (<500ms)
- Handles call disconnections gracefully
- Works with standard telephone networks worldwide

Usage:
    python fonoster_voice_app.py
"""

import os
import asyncio
import logging
import tempfile
import wave
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union, AsyncGenerator

# Import Fonoster SDK
from fonoster_sdk import Client as FonosterClient
from fonoster_voice import VoiceServer, VoiceRequest, VoiceResponse, GatherSource

# Import QuantAI agent
from agent import QuantAIRestaurantAgent
import voice

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/fonoster_voice.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

class QuantAIVoiceServer:
    """Voice server that handles telephone calls and connects to QuantAI agent."""
    
    def __init__(self):
        """Initialize the voice server with credentials from environment variables."""
        self.api_key = os.getenv("FONOSTER_API_KEY")
        self.api_secret = os.getenv("FONOSTER_API_SECRET")
        self.access_key_id = os.getenv("FONOSTER_ACCESS_KEY_ID")
        
        if not all([self.api_key, self.api_secret, self.access_key_id]):
            raise ValueError(
                "Missing Fonoster credentials. Set FONOSTER_API_KEY, "
                "FONOSTER_API_SECRET, and FONOSTER_ACCESS_KEY_ID environment variables."
            )
        
        # Initialize the QuantAI agent
        self.agent = QuantAIRestaurantAgent()
        logger.info("Initialized QuantAI Restaurant Agent")
        
        # Initialize voice server
        self.voice_server = VoiceServer()
        logger.info("Voice server initialized")

    async def start(self):
        """Start the voice server to listen for incoming calls."""
        logger.info("Starting QuantAI Voice Server...")
        
        # Login to Fonoster
        client = FonosterClient({"accessKeyId": self.access_key_id})
        await client.loginWithApiKey(self.api_key, self.api_secret)
        logger.info("Authenticated with Fonoster")
        
        # Start listening for calls
        self.voice_server.listen(self.handle_call)
        logger.info("Voice server is now listening for calls at tcp://0.0.0.0:50061")
        
        # Keep the server running
        try:
            # Keep the server running indefinitely
            while True:
                await asyncio.sleep(3600)  # Check every hour
        except KeyboardInterrupt:
            logger.info("Server shutting down...")

    async def handle_call(self, req: VoiceRequest, voice: VoiceResponse):
        """Handle an incoming call from a user."""
        caller = req.ingressNumber
        session_ref = req.sessionRef
        logger.info(f"Incoming call from {caller} (session: {session_ref})")
        
        try:
            # Answer the call
            await voice.answer()
            
            # Welcome message
            await voice.say("Welcome to QuantAI Restaurant. How can I help you today?")
            
            # Conversation loop
            conversation_active = True
            while conversation_active:
                # Listen for user input
                try:
                    # Using speech recognition to gather user input
                    speech_result = await voice.gather({
                        "source": GatherSource.SPEECH,
                        "speechTimeout": 2,  # Wait for 2 seconds of silence
                        "maxSpeechLength": 15  # Maximum 15 seconds of speech
                    })
                    
                    user_input = speech_result.get("speech", "")
                    
                    if user_input:
                        logger.info(f"User said: {user_input}")
                        
                        # If user wants to end the call
                        if any(phrase in user_input.lower() for phrase in 
                               ["goodbye", "bye", "end call", "hang up", "thank you"]):
                            await voice.say("Thank you for calling QuantAI Restaurant. Goodbye!")
                            conversation_active = False
                            break
                        
                        # Process with the QuantAI agent
                        agent_response = await self.agent.generate_response(user_input)
                        logger.info(f"Agent response: {agent_response}")
                        
                        # Speak the response back to the user
                        await voice.say(agent_response)
                    else:
                        # No speech detected
                        await voice.say("I didn't catch that. Could you please repeat?")
                
                except Exception as e:
                    logger.error(f"Error during conversation: {e}")
                    await voice.say("I'm sorry, I encountered a problem. Let me try to help you again.")
            
            # End the call gracefully
            await voice.hangup()
            logger.info(f"Call ended with {caller}")
            
        except Exception as e:
            logger.error(f"Error handling call: {e}")
            try:
                await voice.say("I'm sorry, we're experiencing technical difficulties. Please try calling again later.")
                await voice.hangup()
            except:
                # Attempt to clean up if even the error handler fails
                pass

async def main():
    """Main entry point for the QuantAI Voice Server."""
    voice_server = QuantAIVoiceServer()
    await voice_server.start()

if __name__ == "__main__":
    asyncio.run(main()) 