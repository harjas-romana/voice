"""
QuantAI Restaurant Outbound Calling System
==========================================
This module enables the QuantAI Restaurant system to make outbound calls
to customers for various purposes like reservation confirmations,
special promotions, or follow-ups.

Usage:
    python outbound_calls.py [phone_number] [purpose]

Example:
    python outbound_calls.py +6421234567 reservation_confirmation
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

from fonoster_sdk import Client
from fonoster_sdk.types import CallRequest
from dotenv import load_dotenv

# Import QuantAI agent for context if needed
from agent import QuantAIRestaurantAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/outbound_calls.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class OutboundCaller:
    """Manager for making outbound calls using Fonoster."""
    
    # Call purposes and their associated voice app references
    CALL_PURPOSES = {
        "reservation_confirmation": "res-confirm-app",
        "special_promotion": "promo-app",
        "feedback_request": "feedback-app",
        "general_inquiry": "inquiry-app",
    }
    
    def __init__(self):
        """Initialize the outbound caller with Fonoster credentials."""
        self.api_key = os.getenv("FONOSTER_API_KEY")
        self.api_secret = os.getenv("FONOSTER_API_SECRET")
        self.access_key_id = os.getenv("FONOSTER_ACCESS_KEY_ID")
        self.from_number = os.getenv("FONOSTER_FROM_NUMBER")
        
        # Voice app reference (default to general app)
        self.voice_app_ref = os.getenv("FONOSTER_VOICE_APP_REF", "quantai-restaurant-app")
        
        # Validate credentials
        if not all([self.api_key, self.api_secret, self.access_key_id, self.from_number]):
            raise ValueError(
                "Missing Fonoster credentials. Set FONOSTER_API_KEY, "
                "FONOSTER_API_SECRET, FONOSTER_ACCESS_KEY_ID, and "
                "FONOSTER_FROM_NUMBER environment variables."
            )
        
        # Initialize Fonoster client
        self.client = None
        self.agent = QuantAIRestaurantAgent()
        logger.info("OutboundCaller initialized")
    
    async def login(self):
        """Login to Fonoster to get authenticated client."""
        self.client = Client({"accessKeyId": self.access_key_id})
        await self.client.loginWithApiKey(self.api_key, self.api_secret)
        logger.info("Authenticated with Fonoster")
    
    async def make_call(
        self, 
        to_number: str, 
        purpose: str = "general_inquiry",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an outbound call to a customer.
        
        Args:
            to_number: The phone number to call
            purpose: The purpose of the call (must be in CALL_PURPOSES)
            context: Additional context for the call
            
        Returns:
            Dict with call details and status
        """
        if not self.client:
            await self.login()
        
        # Validate phone number (simple check)
        if not to_number.startswith("+"):
            raise ValueError("Phone number must start with '+' and country code")
        
        # Get the appropriate app ref for this purpose
        app_ref = self.CALL_PURPOSES.get(purpose, self.voice_app_ref)
        
        # Prepare context data for the voice app
        metadata = {
            "purpose": purpose,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        try:
            # Create Calls API instance
            calls_api = await self.client.calls()
            
            # Make the call
            call_request = {
                "from": self.from_number,
                "to": to_number,
                "appRef": app_ref,
                "metadata": json.dumps(metadata)
            }
            
            logger.info(f"Making outbound call to {to_number} for purpose: {purpose}")
            response = await calls_api.createCall(call_request)
            
            logger.info(f"Call initiated: {response}")
            return {
                "status": "initiated",
                "call_id": response.get("ref", "unknown"),
                "to": to_number,
                "purpose": purpose,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to make call: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "to": to_number,
                "purpose": purpose,
                "timestamp": datetime.now().isoformat()
            }
    
    async def make_bulk_calls(
        self, 
        numbers: list, 
        purpose: str = "general_inquiry",
        delay: int = 5  # seconds between calls
    ) -> list:
        """
        Make multiple outbound calls with a delay between them.
        
        Args:
            numbers: List of phone numbers to call
            purpose: The purpose of the calls
            delay: Seconds to wait between calls
            
        Returns:
            List of call results
        """
        results = []
        
        for number in numbers:
            result = await self.make_call(number, purpose)
            results.append(result)
            
            # Don't sleep after the last call
            if number != numbers[-1]:
                await asyncio.sleep(delay)
        
        return results

async def main():
    """Main entry point for outbound calling."""
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} [phone_number] [purpose]")
        print(f"Available purposes: {', '.join(OutboundCaller.CALL_PURPOSES.keys())}")
        return
    
    phone_number = sys.argv[1]
    purpose = sys.argv[2]
    
    # Validate purpose
    if purpose not in OutboundCaller.CALL_PURPOSES:
        print(f"Invalid purpose. Choose from: {', '.join(OutboundCaller.CALL_PURPOSES.keys())}")
        return
    
    # Create caller and make the call
    caller = OutboundCaller()
    result = await caller.make_call(phone_number, purpose)
    
    # Print result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 