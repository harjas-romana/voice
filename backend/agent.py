"""
QuantAI Restaurant Assistant (Auckland, New Zealand)
===================================================
A sophisticated, context-aware conversational agent for **QuantAI Restaurant**.
This agent delegates knowledge retrieval and reasoning to the Retrieval-Augmented
Generation (RAG) layer implemented in `rag_layer.py`, and focuses on:

• Multilingual interaction with automatic language detection & translation.  
• Conversational memory for pronoun/entity resolution across turns.  
• Advanced Groq LLM integration with streaming, chain-of-thought reasoning
• Configurable generation parameters for optimal response quality
• Enhanced context management and prompt engineering

Run `python agent.py` to start the interactive assistant.
"""

from __future__ import annotations

# Fix OpenMP runtime conflict
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import logging
import pickle
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, AsyncGenerator, Any
from dataclasses import dataclass, field
from enum import Enum

from cachetools import TTLCache
from colorama import Back, Fore, Style, init
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from tqdm import tqdm
import groq

# --- Initialise logging ----------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "quantai_agent.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Third-party initialisation -------------------------------------------
init(autoreset=True)  # colour support for terminals

# --- RAG system -----------------------------------------------------------
import rag_layer  # pragma: no cover – imports & initialises rag_system
RAG_SYSTEM = rag_layer.rag_system  # exposed instance

# -------------------------------------------------------------------------
# Enhanced Generation Configuration
# -------------------------------------------------------------------------

class ReasoningMode(Enum):
    """Different reasoning modes for response generation."""
    STANDARD = "standard"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STEP_BY_STEP = "step_by_step"
    ANALYTICAL = "analytical"

@dataclass
class GenerationConfig:
    """Configuration for Groq LLM generation parameters."""
    
    # Model selection
    model: str = "llama-3.3-70b-versatile"  # Latest high-quality Groq model
    
    # Core generation parameters
    temperature: float = 0.7  # Controls creativity (0.0 = deterministic, 1.0 = very creative)
    top_p: float = 0.9  # Nucleus sampling parameter
    max_tokens: int = 1024  # Maximum response length
    
    # Advanced parameters
    frequency_penalty: float = 0.1  # Reduce repetition
    presence_penalty: float = 0.1  # Encourage new topics
    seed: Optional[int] = None  # For deterministic outputs
    
    # Streaming
    stream: bool = True  # Enable streaming responses
    
    # Reasoning
    reasoning_mode: ReasoningMode = ReasoningMode.STANDARD
    
    # Context management
    max_context_tokens: int = 4000  # Maximum context length
    include_conversation_history: bool = True
    include_chain_of_thought: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for Groq API."""
        config = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": self.stream,
        }
        if self.seed is not None:
            config["seed"] = self.seed
        return config

# -------------------------------------------------------------------------
# Enhanced System Prompts
# -------------------------------------------------------------------------

class PromptManager:
    """Manages system prompts for different use cases and reasoning modes."""
    
    @staticmethod
    def get_system_prompt(reasoning_mode: ReasoningMode = ReasoningMode.STANDARD) -> str:
        """Get the appropriate system prompt based on reasoning mode."""
        
        base_prompt = """You are Kai — QuantAI Restaurant's knowledgeable, personable AI host in Auckland, New Zealand.

CORE GUIDELINES:
• Provide warm, empathetic, and professional assistance
• Base all responses strictly on QuantAI Restaurant's data and policies
• Use clear, concise language appropriate for restaurant guests
• Maintain a friendly Kiwi tone while being professional
• Always offer helpful follow-up suggestions when appropriate

RESPONSE STYLE:
• Keep responses conversational and engaging
• Use bullet points or numbered lists for complex information
• Include relevant details from the restaurant's data
• Anticipate guest needs and offer proactive assistance
• End with an inviting question or offer for further help

CONTEXT AWARENESS:
• Use conversation history to maintain context
• Resolve pronouns and references appropriately
• Build on previous interactions naturally
• Remember guest preferences and past interactions

RESTAURANT FOCUS:
• Only provide information about QuantAI Restaurant
• Decline questions about other venues politely
• Focus on Auckland, New Zealand context
• Use local terminology and cultural references appropriately"""

        if reasoning_mode == ReasoningMode.CHAIN_OF_THOUGHT:
            return base_prompt + """

REASONING APPROACH:
• Think through the guest's request step by step
• Consider the context and available data
• Explain your reasoning process clearly
• Provide structured, logical responses
• Show your thought process for complex queries"""
        
        elif reasoning_mode == ReasoningMode.STEP_BY_STEP:
            return base_prompt + """

STEP-BY-STEP APPROACH:
• Break down complex requests into clear steps
• Provide numbered instructions when appropriate
• Explain each step briefly but clearly
• Ensure logical flow and completeness
• Make processes easy to follow"""
        
        elif reasoning_mode == ReasoningMode.ANALYTICAL:
            return base_prompt + """

ANALYTICAL APPROACH:
• Provide data-driven insights when relevant
• Use numbers and percentages appropriately
• Highlight patterns and trends
• Offer actionable recommendations
• Present information in an organized manner"""
        
        return base_prompt

# -------------------------------------------------------------------------
# Enhanced Conversation Context
# -------------------------------------------------------------------------

class EnhancedConversationContext:
    """Enhanced conversation memory with better context management."""

    def __init__(self, max_turns: int = 20, max_tokens: int = 4000):
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._history: List[Dict[str, Any]] = []
        self.last_reservation: Optional[str] = None
        self.last_order: Optional[str] = None
        self.last_dish: Optional[str] = None
        self.guest_preferences: Dict[str, Any] = {}
        self.conversation_summary: str = ""

    def _extract_entities(self, text: str) -> Dict[str, str]:
        """Enhanced entity extraction with more patterns."""
        ents = {}
        patterns = {
            "reservation": r"RES\d{5}",
            "order": r"ORD\d{5}",
            "dish": r"MENU\d{4}",
            "table": r"TABLE\s*\d+",
            "time": r"\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?",
            "date": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        }
        
        for entity_type, pattern in patterns.items():
            if match := re.search(pattern, text.upper()):
                ents[entity_type] = match.group(0)
        return ents

    def add_turn(self, user: str, assistant: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a conversation turn with enhanced metadata."""
        entities = self._extract_entities(user)
        
        # Update entity tracking
        self.last_reservation = entities.get("reservation", self.last_reservation)
        self.last_order = entities.get("order", self.last_order)
        self.last_dish = entities.get("dish", self.last_dish)
        
        # Extract and store preferences
        self._extract_preferences(user, assistant)
        
        # Add to history with metadata
        turn_data = {
            "user": user,
            "assistant": assistant,
            "entities": entities,
            "timestamp": asyncio.get_event_loop().time(),
            "metadata": metadata or {}
        }
        
        self._history.append(turn_data)
        
        # Maintain history size
        if len(self._history) > self._max_turns:
            self._history.pop(0)
        
        # Update conversation summary
        self._update_summary()

    def _extract_preferences(self, user: str, assistant: str):
        """Extract guest preferences from conversation."""
        # Extract dietary preferences
        dietary_keywords = ["vegetarian", "vegan", "gluten-free", "dairy-free", "halal", "kosher"]
        for keyword in dietary_keywords:
            if keyword.lower() in user.lower():
                self.guest_preferences["dietary"] = keyword.lower()
        
        # Extract seating preferences
        seating_keywords = ["window", "outdoor", "quiet", "private", "bar", "table"]
        for keyword in seating_keywords:
            if keyword.lower() in user.lower():
                self.guest_preferences["seating"] = keyword.lower()
        
        # Extract time preferences
        time_patterns = [r"lunch", r"dinner", r"breakfast", r"brunch"]
        for pattern in time_patterns:
            if re.search(pattern, user.lower()):
                self.guest_preferences["meal_time"] = pattern.replace(r"r", "")

    def _update_summary(self):
        """Update conversation summary for context."""
        if len(self._history) <= 3:
            self.conversation_summary = ""
            return
        
        recent_turns = self._history[-3:]
        summary_parts = []
        
        for turn in recent_turns:
            summary_parts.append(f"Q: {turn['user'][:50]}... A: {turn['assistant'][:50]}...")
        
        self.conversation_summary = " | ".join(summary_parts)

    def resolve(self, text: str) -> str:
        """Enhanced pronoun and reference resolution."""
        replacements = {
            r"\b(it|that reservation)\b": self.last_reservation,
            r"\b(it|that order)\b": self.last_order,
            r"\b(this dish|that dish|it)\b": self.last_dish,
            r"\b(my reservation|my booking)\b": self.last_reservation,
            r"\b(my order)\b": self.last_order,
        }
        
        for pattern, entity in replacements.items():
            if entity:
                text = re.sub(pattern, entity, text, flags=re.IGNORECASE)
        return text

    def get_context_for_llm(self, max_tokens: int = 1000) -> str:
        """Get formatted context for LLM input."""
        context_parts = []
        
        # Add conversation summary
        if self.conversation_summary:
            context_parts.append(f"Recent conversation: {self.conversation_summary}")
        
        # Add entity context
        entity_context = []
        if self.last_reservation:
            entity_context.append(f"Current reservation: {self.last_reservation}")
        if self.last_order:
            entity_context.append(f"Current order: {self.last_order}")
        if self.last_dish:
            entity_context.append(f"Current dish: {self.last_dish}")
        
        if entity_context:
            context_parts.append(" | ".join(entity_context))
        
        # Add preferences
        if self.guest_preferences:
            prefs = [f"{k}: {v}" for k, v in self.guest_preferences.items()]
            context_parts.append(f"Guest preferences: {', '.join(prefs)}")
        
        return " | ".join(context_parts)

    def get_full_history(self) -> List[Dict[str, str]]:
        """Get conversation history in a format suitable for LLM."""
        return [{"role": "user" if i % 2 == 0 else "assistant", 
                "content": turn["user"] if i % 2 == 0 else turn["assistant"]}
                for i, turn in enumerate(self._history)]

# -------------------------------------------------------------------------
# Enhanced Groq Client
# -------------------------------------------------------------------------

class EnhancedGroqClient:
    """Enhanced Groq client with streaming, advanced parameters, and chain-of-thought reasoning."""
    
    def __init__(self, api_key: str, config: GenerationConfig):
        self.api_key = api_key
        self.config = config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Groq client."""
        try:
            self.client = groq.AsyncGroq(api_key=self.api_key)
            logger.info(f"Groq client initialized with model: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    async def generate_response(
        self, 
        query: str, 
        context: str = "", 
        conversation_context: str = "",
        reasoning_mode: ReasoningMode = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response with enhanced context and reasoning."""
        
        if not self.client:
            self._initialize_client()
        
        # Use provided reasoning mode or default from config
        reasoning_mode = reasoning_mode or self.config.reasoning_mode
        
        # Build messages
        messages = self._build_messages(query, context, conversation_context, reasoning_mode)
        
        # Prepare generation parameters
        params = self.config.to_dict()
        
        try:
            # Generate streaming response
            stream = await self.client.chat.completions.create(
                messages=messages,
                **params
            )
            
            # Yield tokens as they arrive
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"I apologize, but I encountered an error: {str(e)}. Please try again."
    
    async def generate_complete_response(
        self, 
        query: str, 
        context: str = "", 
        conversation_context: str = "",
        reasoning_mode: ReasoningMode = None
    ) -> str:
        """Generate complete response (non-streaming)."""
        response_parts = []
        async for token in self.generate_response(query, context, conversation_context, reasoning_mode):
            response_parts.append(token)
        return "".join(response_parts)
    
    def _build_messages(
        self, 
        query: str, 
        context: str, 
        conversation_context: str,
        reasoning_mode: ReasoningMode
    ) -> List[Dict[str, str]]:
        """Build messages for the LLM with enhanced context and reasoning."""
        
        messages = []
        
        # System prompt
        system_prompt = PromptManager.get_system_prompt(reasoning_mode)
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation context if available
        if conversation_context:
            messages.append({
                "role": "system", 
                "content": f"Conversation context: {conversation_context}"
            })
        
        # Add restaurant data context
        if context:
            messages.append({
                "role": "system", 
                "content": f"Restaurant data context: {context}"
            })
        
        # Add chain-of-thought reasoning prompt if enabled
        if reasoning_mode == ReasoningMode.CHAIN_OF_THOUGHT:
            messages.append({
                "role": "system",
                "content": "Please think through this step by step and explain your reasoning."
            })
        
        # User query
        messages.append({"role": "user", "content": query})
        
        return messages

# -------------------------------------------------------------------------
# Language utilities
# -------------------------------------------------------------------------

class LanguageManager:
    """Persist and validate user language preferences."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(exist_ok=True)
        self._store_file = self._cache_dir / "lang_prefs.pkl"
        self._prefs: Dict[str, str] = self._load()

        try:
            self._supported = set(GoogleTranslator(source="auto", target="en").get_supported_languages())
        except Exception as exc:  # pragma: no cover – offline fallback
            logger.warning("Falling back to limited language set (%s)", exc)
            self._supported = {"english", "spanish", "french", "german", "italian", "portuguese"}

        self._aliases = {"cn": "chinese", "zh": "chinese", "español": "spanish", "français": "french"}

    # -------------------- internal helpers --------------------
    def _load(self) -> Dict[str, str]:
        if self._store_file.exists():
            try:
                with self._store_file.open("rb") as fh:
                    return pickle.load(fh)
            except Exception as exc:
                logger.warning("Could not load language prefs – %s", exc)
        return {}

    def _save(self) -> None:
        try:
            with self._store_file.open("wb") as fh:
                pickle.dump(self._prefs, fh)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not save language prefs – %s", exc)

    # -------------------- public API --------------------------
    def set_pref(self, user_id: str, lang: str) -> None:
        self._prefs[user_id] = lang
        self._save()

    def get_pref(self, user_id: str) -> Optional[str]:
        return self._prefs.get(user_id)

    def validate(self, lang: str) -> Tuple[bool, str]:
        lang = self._aliases.get(lang.lower().strip(), lang.lower().strip())
        return (lang in self._supported), lang

    # Compatibility methods for server
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of text with confidence score."""
        try:
            translator = GoogleTranslator(source="auto", target="en")
            detected = translator.detect(text)
            return detected.lang, detected.confidence
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return "english", 0.0

    def validate_language(self, lang: str) -> Tuple[bool, str]:
        """Alias for validate() method."""
        return self.validate(lang)

    @property
    def supported_languages(self):
        """Return supported languages set."""
        return self._supported

# -------------------------------------------------------------------------
# Lightweight translation (async + cached)
# -------------------------------------------------------------------------

class Translator:
    def __init__(self, ttl: int = 2 * 60 * 60):  # 2 hours cache
        self._cache = TTLCache(maxsize=1000, ttl=ttl)
        self._lock = asyncio.Lock()
        self._chunk = 800  # char

    async def translate(self, text: str, target: str) -> str:
        if target == "english" or not text:
            return text
        key = (text[:50], target)
        if key in self._cache:
            return self._cache[key]
        async with self._lock:
            translator = GoogleTranslator(source="english", target=target)
            parts = [text[i : i + self._chunk] for i in range(0, len(text), self._chunk)]
            translated = " ".join(translator.translate(p) for p in parts)
            self._cache[key] = translated
            return translated

    # Compatibility method for server
    async def translate_text(self, text: str, target: str) -> str:
        """Alias for translate() method."""
        return await self.translate(text, target)

# -------------------------------------------------------------------------
# Enhanced Main Agent
# -------------------------------------------------------------------------

class QuantAIRestaurantAgent:
    """Enhanced QuantAI Restaurant Agent with advanced Groq integration."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        load_dotenv()
        self._ensure_env()
        
        # Initialize components
        self.config = config or GenerationConfig()
        self.lang_mgr = LanguageManager()
        self.user_lang: str = "english"
        self.translator = Translator()
        self.context = EnhancedConversationContext()
        
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = EnhancedGroqClient(api_key, self.config)
        
        # Add compatibility attributes for server
        self.language_manager = self.lang_mgr
        self.enhanced_translator = self.translator
        self.translation_cache = self.translator._cache
        
        logger.info("Enhanced QuantAI Restaurant Agent ready")

    # ------------------- helpers ------------------------------
    @staticmethod
    def _ensure_env():
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY is missing – set it in your .env file.")

    async def _set_language(self):
        print(f"\n{Back.BLUE}{Fore.WHITE} Language Selection {Style.RESET_ALL}")
        languages = sorted(self.lang_mgr._supported)
        for idx, lang in enumerate(languages, 1):
            print(f"{idx:2}. {lang.title()}")
        while True:
            choice = input(f"\n{Fore.YELLOW}Enter language number or name: {Style.RESET_ALL}")
            if choice.isdigit():
                sel = int(choice) - 1
                if 0 <= sel < len(languages):
                    self.user_lang = languages[sel]
                    break
            else:
                valid, lang = self.lang_mgr.validate(choice)
                if valid:
                    self.user_lang = lang
                    break
            print(f"{Fore.RED}Invalid selection, please try again.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Language set to {self.user_lang.title()}{Style.RESET_ALL}\n")

    # ------------------- core logic ---------------------------
    async def answer(self, query: str, reasoning_mode: ReasoningMode = None) -> str:
        """Generate response with enhanced context and reasoning."""
        try:
            # Resolve pronouns and references
            resolved_query = self.context.resolve(query)
            
            # Get conversation context
            conv_context = self.context.get_context_for_llm()
            
            # Get restaurant data context from RAG system
            rag_response = await RAG_SYSTEM.process_query(resolved_query, conv_context)
            
            # Generate response using enhanced Groq client
            response = await self.groq_client.generate_complete_response(
                query=resolved_query,
                context=rag_response,
                conversation_context=conv_context,
                reasoning_mode=reasoning_mode or self.config.reasoning_mode
            )
            
            # Add to conversation history
            self.context.add_turn(query, response)
            
            # Translate if needed
            if self.user_lang != "english":
                response = await self.translator.translate(response, self.user_lang)
            
            return response
            
        except Exception as exc:
            logger.error("Error answering query – %s", exc, exc_info=True)
            return "Sorry, something went wrong. Please try again later."

    async def answer_streaming(
        self, 
        query: str, 
        reasoning_mode: ReasoningMode = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response with enhanced context and reasoning."""
        try:
            # Resolve pronouns and references
            resolved_query = self.context.resolve(query)
            
            # Get conversation context
            conv_context = self.context.get_context_for_llm()
            
            # Get restaurant data context from RAG system
            rag_response = await RAG_SYSTEM.process_query(resolved_query, conv_context)
            
            # Generate streaming response
            response_parts = []
            async for token in self.groq_client.generate_response(
                query=resolved_query,
                context=rag_response,
                conversation_context=conv_context,
                reasoning_mode=reasoning_mode or self.config.reasoning_mode
            ):
                response_parts.append(token)
                yield token
            
            # Add complete response to conversation history
            complete_response = "".join(response_parts)
            self.context.add_turn(query, complete_response)
            
        except Exception as exc:
            logger.error("Error in streaming response – %s", exc, exc_info=True)
            yield "Sorry, something went wrong. Please try again later."

    # Compatibility wrapper for external modules expecting generate_response()
    async def generate_response(self, user_query: str) -> str:  # noqa: D401
        """Alias for answer(); maintained for backwards-compatibility."""
        return await self.answer(user_query)

    # ------------------- configuration methods -------------------
    def update_config(self, **kwargs):
        """Update generation configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
        
        # Reinitialize Groq client with new config
        api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = EnhancedGroqClient(api_key, self.config)

    def set_reasoning_mode(self, mode: ReasoningMode):
        """Set the reasoning mode for responses."""
        self.config.reasoning_mode = mode
        logger.info(f"Reasoning mode set to: {mode.value}")

    def set_deterministic(self, seed: int = 42):
        """Enable deterministic outputs with a specific seed."""
        self.config.seed = seed
        self.config.temperature = 0.0
        logger.info(f"Deterministic mode enabled with seed: {seed}")

    # ------------------- CLI loop -----------------------------
    async def run_cli(self):
        print(f"\n{Back.BLUE}{Fore.WHITE} ENHANCED QUANTAI RESTAURANT ASSISTANT {Style.RESET_ALL}")
        print("Kia ora! I'm Kai, your QuantAI Restaurant host. How can I assist you today?")
        print(f"Model: {self.config.model} | Reasoning: {self.config.reasoning_mode.value}")

        await self._set_language()

        while True:
            user_input = input(f"\n{Fore.CYAN}You: {Style.RESET_ALL}")
            if user_input.lower() in {"quit", "exit", "bye", "q"}:
                print("\nHaere rā! We look forward to serving you again.")
                break
            if not user_input.strip():
                continue
            
            # Check for special commands
            if user_input.startswith("/"):
                await self._handle_command(user_input)
                continue
            
            print(f"\n{Fore.CYAN}Processing…{Style.RESET_ALL}")
            
            # Generate streaming response
            print(f"\n{Fore.GREEN}Kai:{Style.RESET_ALL} ", end="", flush=True)
            response_parts = []
            async for token in self.answer_streaming(user_input):
                print(token, end="", flush=True)
                response_parts.append(token)
            
            print(f"\n{Fore.CYAN}Anything else I can help with?{Style.RESET_ALL}")

    async def _handle_command(self, command: str):
        """Handle special CLI commands."""
        cmd_parts = command[1:].split()
        if not cmd_parts:
            return
        
        cmd = cmd_parts[0].lower()
        
        if cmd == "config":
            print(f"\n{Fore.YELLOW}Current Configuration:{Style.RESET_ALL}")
            for key, value in self.config.__dict__.items():
                print(f"  {key}: {value}")
        
        elif cmd == "mode":
            if len(cmd_parts) > 1:
                mode_name = cmd_parts[1].lower()
                try:
                    mode = ReasoningMode(mode_name)
                    self.set_reasoning_mode(mode)
                    print(f"{Fore.GREEN}Reasoning mode set to: {mode.value}{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Invalid mode. Available: {[m.value for m in ReasoningMode]}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Available modes: {[m.value for m in ReasoningMode]}{Style.RESET_ALL}")
        
        elif cmd == "temp":
            if len(cmd_parts) > 1:
                try:
                    temp = float(cmd_parts[1])
                    self.update_config(temperature=temp)
                    print(f"{Fore.GREEN}Temperature set to: {temp}{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Invalid temperature value{Style.RESET_ALL}")
        
        elif cmd == "help":
            print(f"\n{Fore.YELLOW}Available Commands:{Style.RESET_ALL}")
            print("  /config - Show current configuration")
            print("  /mode <mode> - Set reasoning mode")
            print("  /temp <value> - Set temperature")
            print("  /help - Show this help")
        
        else:
            print(f"{Fore.RED}Unknown command: {cmd}. Use /help for available commands.{Style.RESET_ALL}")

# -------------------------------------------------------------------------
# Script entry-point
# -------------------------------------------------------------------------

async def _main():
    # Create enhanced agent with custom configuration
    config = GenerationConfig(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024,
        reasoning_mode=ReasoningMode.STANDARD,
        stream=True
    )
    
    agent = QuantAIRestaurantAgent(config)
    await agent.run_cli()

if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nSession terminated. Ka kite anō!") 