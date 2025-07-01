"""
QuantAI Restaurant Fonoster Configuration
=========================================
This module handles the configuration and setup of Fonoster services
for the QuantAI Restaurant voice system.

It provides:
- Functions to initialize and set up Fonoster resources
- Configuration validation and credential management
- Voice application deployment helpers
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from fonoster_sdk import Client
from fonoster_sdk.types import VoiceAppConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/fonoster_config.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FonosterConfig:
    """Manager for Fonoster configuration and resource setup."""
    
    def __init__(self, use_env: bool = True, config_path: Optional[str] = None):
        """
        Initialize Fonoster configuration manager.
        
        Args:
            use_env: Whether to use environment variables for credentials
            config_path: Path to JSON config file (alternative to env vars)
        """
        self.config_path = config_path
        self.use_env = use_env
        
        # Credentials
        self.api_key = None
        self.api_secret = None
        self.access_key_id = None
        
        # Resources
        self.phone_numbers = []
        self.voice_apps = {}
        
        # Load configuration
        self._load_config()
        
        # Client (will be initialized during login)
        self.client = None
    
    def _load_config(self):
        """Load configuration from environment or config file."""
        if self.use_env:
            self.api_key = os.getenv("FONOSTER_API_KEY")
            self.api_secret = os.getenv("FONOSTER_API_SECRET")
            self.access_key_id = os.getenv("FONOSTER_ACCESS_KEY_ID")
        elif self.config_path:
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.api_key = config.get("api_key")
                    self.api_secret = config.get("api_secret")
                    self.access_key_id = config.get("access_key_id")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
        
        # Validate credentials
        if not all([self.api_key, self.api_secret, self.access_key_id]):
            logger.warning(
                "Missing Fonoster credentials. Set FONOSTER_API_KEY, "
                "FONOSTER_API_SECRET, and FONOSTER_ACCESS_KEY_ID environment "
                "variables or provide a valid config file."
            )
    
    async def login(self) -> bool:
        """Login to Fonoster and initialize client."""
        try:
            self.client = Client({"accessKeyId": self.access_key_id})
            await self.client.loginWithApiKey(self.api_key, self.api_secret)
            logger.info("Successfully authenticated with Fonoster")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with Fonoster: {e}")
            return False
    
    async def create_voice_app(
        self, 
        name: str, 
        description: str = "QuantAI Restaurant Voice Application"
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new voice application in Fonoster.
        
        Args:
            name: Name of the voice application
            description: Description of the application
            
        Returns:
            Dict with application details or None if failed
        """
        if not self.client:
            if not await self.login():
                return None
        
        try:
            # Create Apps API instance
            apps_api = await self.client.apps()
            
            # Ensure valid name (alphanumeric and dashes only)
            safe_name = "".join([c if c.isalnum() else "-" for c in name.lower()])
            
            # Define app configuration
            app_config = {
                "name": safe_name,
                "description": description,
                "voice_url": "tcp://0.0.0.0:50061",  # Default Voice URL (can be changed later)
            }
            
            # Create the app
            response = await apps_api.createApp(app_config)
            
            logger.info(f"Created voice app: {safe_name} (ref: {response.get('ref')})")
            
            # Store app in local cache
            self.voice_apps[safe_name] = response
            
            return response
        except Exception as e:
            logger.error(f"Failed to create voice app: {e}")
            return None
    
    async def list_voice_apps(self) -> List[Dict[str, Any]]:
        """List all voice applications in the Fonoster account."""
        if not self.client:
            if not await self.login():
                return []
        
        try:
            # Create Apps API instance
            apps_api = await self.client.apps()
            
            # List apps
            response = await apps_api.listApps()
            
            # Update local cache
            apps = response.get("apps", [])
            for app in apps:
                self.voice_apps[app.get("name")] = app
            
            return apps
        except Exception as e:
            logger.error(f"Failed to list voice apps: {e}")
            return []
    
    async def list_phone_numbers(self) -> List[Dict[str, Any]]:
        """List all phone numbers in the Fonoster account."""
        if not self.client:
            if not await self.login():
                return []
        
        try:
            # Create Numbers API instance
            numbers_api = await self.client.numbers()
            
            # List numbers
            response = await numbers_api.listNumbers()
            
            # Update local cache
            self.phone_numbers = response.get("numbers", [])
            
            return self.phone_numbers
        except Exception as e:
            logger.error(f"Failed to list phone numbers: {e}")
            return []
    
    async def get_available_numbers(self, area_code: str = "404") -> List[Dict[str, Any]]:
        """
        Get available phone numbers for purchase.
        
        Args:
            area_code: Area code to search in
            
        Returns:
            List of available numbers
        """
        if not self.client:
            if not await self.login():
                return []
        
        try:
            # Create Numbers API instance
            numbers_api = await self.client.numbers()
            
            # Search for available numbers
            response = await numbers_api.getIngressInfo({
                "areaCode": area_code
            })
            
            return response.get("numbers", [])
        except Exception as e:
            logger.error(f"Failed to get available numbers: {e}")
            return []
    
    async def purchase_number(
        self, 
        phone_number: str,
        app_ref: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Purchase a phone number and optionally assign it to an app.
        
        Args:
            phone_number: Phone number to purchase
            app_ref: Optional voice app reference to assign to
            
        Returns:
            Dict with number details or None if failed
        """
        if not self.client:
            if not await self.login():
                return None
        
        try:
            # Create Numbers API instance
            numbers_api = await self.client.numbers()
            
            # Purchase config
            purchase_config = {
                "number": phone_number
            }
            
            # Add app reference if provided
            if app_ref:
                purchase_config["appRef"] = app_ref
            
            # Purchase the number
            response = await numbers_api.createNumber(purchase_config)
            
            logger.info(f"Purchased phone number: {phone_number}")
            
            # Update local cache
            self.phone_numbers.append(response)
            
            return response
        except Exception as e:
            logger.error(f"Failed to purchase number: {e}")
            return None
    
    async def setup_complete_system(
        self,
        app_name: str = "quantai-restaurant",
        purchase_new_number: bool = False,
        area_code: str = "404"
    ) -> Dict[str, Any]:
        """
        Set up a complete Fonoster voice system in one go.
        
        Args:
            app_name: Name for the voice application
            purchase_new_number: Whether to purchase a new number
            area_code: Area code if purchasing a new number
            
        Returns:
            Dict with setup results
        """
        results = {
            "success": False,
            "app": None,
            "number": None,
            "error": None
        }
        
        # Step 1: Login
        if not await self.login():
            results["error"] = "Failed to authenticate with Fonoster"
            return results
        
        # Step 2: Create voice app
        app = await self.create_voice_app(app_name)
        if not app:
            results["error"] = "Failed to create voice application"
            return results
        
        results["app"] = app
        app_ref = app.get("ref")
        
        # Step 3: Handle phone number
        if purchase_new_number:
            # Get available numbers
            available_numbers = await self.get_available_numbers(area_code)
            if not available_numbers:
                results["error"] = f"No available numbers found for area code {area_code}"
                return results
            
            # Purchase the first available number
            number = await self.purchase_number(
                available_numbers[0].get("e164Number"),
                app_ref
            )
            
            if not number:
                results["error"] = "Failed to purchase phone number"
                return results
            
            results["number"] = number
        else:
            # List existing numbers
            numbers = await self.list_phone_numbers()
            results["number"] = numbers[0] if numbers else None
        
        # Setup complete
        results["success"] = True
        return results

async def setup_fonoster():
    """
    Set up Fonoster resources for QuantAI Restaurant.
    This is the main function to be called for initial setup.
    """
    config = FonosterConfig()
    
    # Check if credentials are available
    if not all([config.api_key, config.api_secret, config.access_key_id]):
        logger.error("Missing Fonoster credentials. Please set environment variables.")
        return False
    
    # Set up the complete system
    result = await config.setup_complete_system(
        app_name="quantai-restaurant",
        purchase_new_number=False  # Set to True to purchase a new number
    )
    
    if result["success"]:
        logger.info(f"Fonoster setup completed successfully!")
        logger.info(f"App ref: {result['app'].get('ref')}")
        if result["number"]:
            logger.info(f"Phone number: {result['number'].get('e164Number')}")
        return True
    else:
        logger.error(f"Fonoster setup failed: {result['error']}")
        return False

if __name__ == "__main__":
    asyncio.run(setup_fonoster()) 