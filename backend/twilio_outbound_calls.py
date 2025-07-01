"""
QuantAI Restaurant Outbound Calling System (Twilio)
==================================================
This module enables the QuantAI Restaurant system to make outbound calls
to customers for various purposes like reservation confirmations,
special promotions, or follow-ups using Twilio.

Usage:
    python twilio_outbound_calls.py [phone_number] [purpose]

Example:
    python twilio_outbound_calls.py +6421234567 reservation_confirmation
"""

import os
import sys
import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv

# Import QuantAI agent for context if needed
from agent import QuantAIRestaurantAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/twilio_outbound_calls.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TwilioOutboundCaller:
    """Manager for making outbound calls using Twilio."""
    
    # Call purposes and their associated TwiML Bin SIDs
    CALL_PURPOSES = {
        "reservation_confirmation": "reservation-confirmation",
        "special_promotion": "special-promotion",
        "feedback_request": "feedback-request",
        "general_inquiry": "general-inquiry",
    }
    
    def __init__(self):
        """Initialize the outbound caller with Twilio credentials."""
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_PHONE_NUMBER")
        self.base_url = os.getenv("TWILIO_WEBHOOK_BASE_URL", "https://your-webhook-url.com")
        
        # Validate credentials
        if not all([self.account_sid, self.auth_token, self.from_number]):
            raise ValueError(
                "Missing Twilio credentials. Set TWILIO_ACCOUNT_SID, "
                "TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER environment variables."
            )
        
        # Initialize Twilio client
        self.client = Client(self.account_sid, self.auth_token)
        self.agent = QuantAIRestaurantAgent()
        logger.info("TwilioOutboundCaller initialized")
    
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
        # Validate phone number (simple check)
        if not to_number.startswith("+"):
            raise ValueError("Phone number must start with '+' and country code")
        
        # Get the appropriate URL path for this purpose
        url_path = self.CALL_PURPOSES.get(purpose, "general-inquiry")
        
        # Construct webhook URL with context as query parameters
        webhook_url = f"{self.base_url}/voice/{url_path}"
        
        # Add context as query parameters if provided
        if context:
            query_params = "&".join([f"{k}={v}" for k, v in context.items()])
            webhook_url = f"{webhook_url}?{query_params}"
        
        try:
            # Make the call
            logger.info(f"Making outbound call to {to_number} for purpose: {purpose}")
            
            call = self.client.calls.create(
                to=to_number,
                from_=self.from_number,
                url=webhook_url,
                method="POST"
            )
            
            logger.info(f"Call initiated: {call.sid}")
            return {
                "status": "initiated",
                "call_id": call.sid,
                "to": to_number,
                "purpose": purpose,
                "timestamp": datetime.now().isoformat()
            }
            
        except TwilioRestException as e:
            logger.error(f"Twilio error making call: {e}")
            return {
                "status": "failed",
                "error": f"Twilio error: {str(e)}",
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
        numbers: List[str], 
        purpose: str = "general_inquiry",
        delay: int = 5  # seconds between calls
    ) -> List[Dict[str, Any]]:
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
        print(f"Available purposes: {', '.join(TwilioOutboundCaller.CALL_PURPOSES.keys())}")
        return
    
    phone_number = sys.argv[1]
    purpose = sys.argv[2]
    
    # Validate purpose
    if purpose not in TwilioOutboundCaller.CALL_PURPOSES:
        print(f"Invalid purpose. Choose from: {', '.join(TwilioOutboundCaller.CALL_PURPOSES.keys())}")
        return
    
    # Create caller and make the call
    caller = TwilioOutboundCaller()
    result = await caller.make_call(phone_number, purpose)
    
    # Print result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 