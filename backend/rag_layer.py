"""
QuantAI Restaurant Assistant (Auckland, New Zealand)
This module implements a strictly domain-specific RAG system for QuantAI Restaurant in Auckland.
All responses are based solely on QuantAI Restaurant's datasets and operations in Auckland, New Zealand.
Features:
- Context-aware response generation using Groq (LLM) with strict QuantAI Restaurant focus
- Auckland-specific knowledge integration (customers, reservations, orders, menu, staff, inventory, infrastructure)
- Concise, accurate, and contextually rich response formatting
- No speculation or general advice – only respond based on QuantAI Restaurant's data
- All prompts and context reference QuantAI Restaurant in Auckland
"""

import os
# Fix OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import re
from dotenv import load_dotenv
import groq  # Add Groq SDK import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_layer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RestaurantDataLoader:
    """Loads and manages QuantAI Restaurant datasets with strict New Zealand context and terminology."""

    def __init__(self):
        self.data_dir = "data"
        self.data: Dict[str, Any] = {}
        self.load_data()

    def load_data(self):
        """Load QuantAI restaurant infrastructure and datasets for New Zealand context only."""
        try:
            infra_path = os.path.join(self.data_dir, "restaurant_infrastructure.json")
            if os.path.exists(infra_path):
                with open(infra_path, "r") as f:
                    self.data["infrastructure"] = json.load(f)
            else:
                logger.warning("restaurant_infrastructure.json not found in data directory")

            csv_files = [f for f in os.listdir(self.data_dir) if f.startswith("quantai_restaurant_") and f.endswith(".csv")]

            for file in csv_files:
                dataset_name = file.replace("quantai_restaurant_", "").replace(".csv", "")
                path = os.path.join(self.data_dir, file)
                try:
                    self.data[dataset_name] = pd.read_csv(path)
                    logger.info(f"Loaded {file}")
                except Exception as e:
                    logger.error(f"Failed to load {file}: {e}")

            if not self.data:
                logger.warning("No QuantAI restaurant datasets loaded – verify /data directory")
        except Exception as e:
            logger.error(f"RestaurantDataLoader error: {e}")
            raise

    def get_relevant_data(self, query_type: str, keywords: List[str]) -> Dict[str, Any]:
        """Retrieve only New Zealand restaurant data relevant to the query type and keywords."""
        try:
            # Map query types to relevant data sources
            mapping = {
                "reservation": ["reservations", "customers", "staff"],
                "order": ["orders", "customers", "menu"],
                "menu": ["menu"],
                "customer": ["customers"],
                "staff": ["staff"],
                "inventory": ["inventory"],
                "ambience": ["infrastructure"],
                "general": ["infrastructure"]
            }
            data_sources = mapping.get(query_type, ["infrastructure"])
            relevant_data = {}

            for source in data_sources:
                if source in self.data:
                    data_item = self.data[source]
                    
                    # Handle different data types
                    if isinstance(data_item, pd.DataFrame):
                        # Handle DataFrame data
                        df = data_item
                        filtered_data = df

                        # Filter based on keywords (AND logic for specificity)
                        if keywords:
                            combined_mask = None
                            for keyword in keywords:
                                keyword_mask = None
                                for col in df.columns:
                                    col_mask = df[col].astype(str).str.contains(
                                        keyword, case=False, na=False, regex=True
                                    )
                                    if keyword_mask is None:
                                        keyword_mask = col_mask
                                    else:
                                        keyword_mask |= col_mask
                                if combined_mask is None:
                                    combined_mask = keyword_mask
                                else:
                                    combined_mask &= keyword_mask
                            if combined_mask is not None:
                                filtered_data = filtered_data[combined_mask]

                        # Sort by date if available
                        date_columns = [col for col in filtered_data.columns if 'date' in col.lower() or 'time' in col.lower()]
                        if date_columns:
                            try:
                                filtered_data = filtered_data.sort_values(by=date_columns[0], ascending=False)
                            except Exception:
                                pass

                        if not filtered_data.empty:
                            relevant_data[source] = filtered_data.head(3).to_dict('records')
                    
                    elif isinstance(data_item, dict):
                        # Handle dictionary data (like infrastructure JSON)
                        if keywords:
                            # Simple keyword matching for dictionary data
                            matching_data = {}
                            for key, value in data_item.items():
                                value_str = str(value).lower()
                                if any(keyword.lower() in value_str for keyword in keywords):
                                    matching_data[key] = value
                            if matching_data:
                                relevant_data[source] = matching_data
                        else:
                            # If no keywords, include all data
                            relevant_data[source] = data_item
                    
                    elif isinstance(data_item, list):
                        # Handle list data
                        if keywords:
                            # Simple keyword matching for list data
                            matching_items = []
                            for item in data_item:
                                item_str = str(item).lower()
                                if any(keyword.lower() in item_str for keyword in keywords):
                                    matching_items.append(item)
                            if matching_items:
                                relevant_data[source] = matching_items[:3]  # Limit to 3 items
                        else:
                            # If no keywords, include first 3 items
                            relevant_data[source] = data_item[:3]

            return relevant_data
        except Exception as e:
            logger.error(f"Error getting relevant restaurant data: {e}")
            return {}

class NZGroqClient:
    """Client for interacting with Groq LLM for QuantAI Restaurant (Auckland) Assistant."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.model = "llama-3.3-70b-versatile"  # Update to a Groq model
    
    async def initialize(self):
        if not self.client:
            self.client = groq.AsyncGroq(api_key=self.api_key)
    
    async def close(self):
        # No explicit close needed for groq.AsyncGroq
        self.client = None
    
    async def generate_response(self, query: str, context: str, conversation_context: str = "", response_type: str = "standard") -> str:
        """
        Generate a brief, empathetic, and conversational response using Groq, with follow-up and context awareness.
        """
        try:
            if not self.client:
                await self.initialize()
            # Conversational, brief, and follow-up aware system prompts
            system_prompts = {
                "standard": (
                    "You are Kai — QuantAI Restaurant's knowledgeable, personable AI host in Auckland, NZ." \
                    "\nGuidelines:" \
                    "\n• Open with a warm, concise greeting." \
                    "\n• Provide clear, factual answers strictly based on QuantAI Restaurant data and policies." \
                    "\n• Where helpful, structure information using short bullet points or numbered steps." \
                    "\n• Anticipate the guest's next need: if clarification might help, politely ask a follow-up question." \
                    "\n• Always keep replies under ~5 sentences unless detail is explicitly requested." \
                    "\n• If required data is missing, apologise briefly and offer alternative help." \
                    "\n• Politely decline questions about other venues – you serve QuantAI Restaurant only." \
                    "\n• Maintain a friendly, professional, Kiwi tone. Use simple language and avoid jargon." \
                    "\n• Finish with an invitational prompt such as 'Is there anything else I can organise for you?'" \
                    "\nRemember: conversation_context may contain pronoun-resolved details — use them to avoid repetition."
                ),
                "operational": (
                    "You are Kai — QuantAI Restaurant's operational assistant (booking & order tasks)." \
                    "\nGuidelines:" \
                    "\n• Deliver concise, step-by-step instructions or confirmations." \
                    "\n• Prefer bullet points for options (timeslots, menu items, etc.)." \
                    "\n• Confirm key details (date, time, guest count, order number) and invite correction if needed." \
                    "\n• Close with a warm prompt for further help."
                ),
                "analytical": (
                    "You are Kai — QuantAI Restaurant's insights assistant." \
                    "\nGuidelines:" \
                    "\n• Present data-driven insights in clear everyday language." \
                    "\n• Use bullet points or short paragraphs; include numbers/percentages where useful." \
                    "\n• Highlight actionable takeaways (e.g., 'Consider extra staff on Friday dinner service')." \
                    "\n• End with an invitation for deeper analysis or related questions."
                )
            }
            messages = [
                {"role": "system", "content": system_prompts.get(response_type, system_prompts["standard"])}
            ]
            if conversation_context:
                messages.append({"role": "system", "content": f"Conversation context: {conversation_context}"})
            messages.append({"role": "system", "content": f"Context (QuantAI Restaurant data): {context}"})
            messages.append({"role": "user", "content": query})
            # Use Groq's async chat completion
            chat_completion = await self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=350,
                top_p=0.9,
                presence_penalty=0.7,
                frequency_penalty=0.4
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

class RestaurantAssistantRAG:
    """RAG system strictly for QuantAI Restaurant Assistant (New Zealand context only)."""

    def __init__(self):
        self.data_loader = None
        self.gemma_client = None
        self.load_environment()
        self.initialize_components()
        # Optionally, define staff roles for access control if needed
        self.roles = [
            "Manager", "Chef", "Sous Chef", "Waiter", "Host", "Bartender", "Kitchen Porter"
        ]

    def load_environment(self):
        load_dotenv()
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

    def initialize_components(self):
        self.data_loader = RestaurantDataLoader()
        self.gemma_client = NZGroqClient(self.api_key)

    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query for QuantAI Restaurant Auckland context."""
        q = query.lower()
        patterns = {
            "reservation": r"reservation|booking|book a table|table for",
            "order": r"order|takeaway|delivery|food to go|menu item",
            "menu": r"menu|dish|what do you have|food|drink|beverage|wine list|dessert|special",
            "customer": r"customer|my details|my account|my orders|loyalty",
            "staff": r"staff|chef|waiter|manager|who is working",
            "inventory": r"inventory|stock|ingredient|availability|out of stock",
            "hours": r"hours|opening|closing|open|close|when are you open",
            "location": r"location|address|directions|parking|where are you",
            "contact": r"contact|phone|email",
            "feedback": r"feedback|complaint|review|suggestion",
            "general": r"restaurant|about you|quantai restaurant"
        }
        for qt, pat in patterns.items():
            if re.search(pat, q):
                return qt
        return "general"

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from the query for QuantAI Restaurant context."""
        # Remove common stopwords and punctuation
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about',
            'show', 'tell', 'give', 'find', 'get', 'me', 'please', 'list', 'details', 'info', 'information',
            'restaurant', 'quantai', 'auckland', 'nz', 'new', 'zealand'
        }
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]

        # Add specific restaurant ID patterns
        id_patterns = [
            (r'CUST\d{5}', 'customer_id'),
            (r'RES\d{5}', 'reservation_id'),
            (r'ORD\d{5}', 'order_id'),
            (r'MENU\d{4}', 'menu_item_id'),
            (r'STF\d{4}', 'staff_id'),
            (r'INV\d{5}', 'inventory_id')
        ]
        for pattern, _ in id_patterns:
            ids = re.findall(pattern, query.upper())
            keywords.extend(ids)

        # Add Auckland-specific terms if mentioned
        auckland_terms = [
            "cbd", "central", "north shore", "south", "west", "east",
            "waitakere", "manukau", "waitematā", "dhb", "district"
        ]
        for term in auckland_terms:
            if term in query.lower():
                keywords.append(term)

        return list(set(keywords))

    def _determine_response_type(self, query: str) -> str:
        """Determine the appropriate response type based on query content."""
        q = query.lower()
        if any(x in q for x in ["booking", "order", "reservation"]):
            return "operational"
        if any(x in q for x in ["trend", "popular", "busiest", "statistics", "analysis", "report"]):
            return "analytical"
        return "standard"

    def _format_response(self, resp: str, q_type: str) -> str:
        """Format the response with explicit QuantAI Restaurant (Auckland) branding and context."""
        resp = resp.strip()
        for pat in [r"As an AI", r"I'm an AI", r"artificial intelligence"]:
            resp = re.sub(pat, "Kai, your QuantAI Restaurant host", resp, flags=re.IGNORECASE)

        prefixes = {
            "reservation": "QuantAI Restaurant Auckland | Reservation",
            "order": "QuantAI Restaurant Auckland | Order Status",
            "menu": "QuantAI Restaurant Auckland | Menu Information",
            "customer": "QuantAI Restaurant Auckland | Customer Information",
            "staff": "QuantAI Restaurant Auckland | Staff Information",
            "inventory": "QuantAI Restaurant Auckland | Stock Update",
            "general": "QuantAI Restaurant Auckland | Information",
            "location": "QuantAI Restaurant Auckland | Location",
            "hours": "QuantAI Restaurant Auckland | Opening Hours",
            "feedback": "QuantAI Restaurant Auckland | Feedback"
        }
        heading = prefixes.get(q_type, "QuantAI Restaurant Auckland | Information")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M NZST")
        return f"{heading} | {timestamp}\n\n{resp}"

    async def process_query(self, query: str, conversation_context: str = "") -> str:
        """
        Process a query and generate a strictly NZ restaurant-specific response as QuantAI Restaurant Assistant.
        """
        try:
            q_type = self._determine_query_type(query)
            keywords = self._extract_keywords(query)
            resp_type = self._determine_response_type(query)

            relevant = self.data_loader.get_relevant_data(q_type, keywords)
            context_parts: List[str] = [f"Information Type (NZ): {q_type}"]

            for src, rows in relevant.items():
                context_parts.append(f"\n{src.replace('_', ' ').title()}:")
                for r in rows:
                    row_str = ", ".join([f"{k}: {v}" for k, v in r.items() if k.lower() in {"id", "name", "type", "status", "date", "time", "customer_id", "reservation_id", "order_id", "item"}])
                    context_parts.append(row_str)

            context = "\n".join(context_parts)
            llm_resp = await self.gemma_client.generate_response(query, context, conversation_context, resp_type)
            return self._format_response(llm_resp, q_type)
        except Exception as e:
            logger.error(f"RestaurantRAG error: {e}")
            return "System Error: Unable to process query."

    async def close(self):
        if self.gemma_client:
            await self.gemma_client.close()

# Initialize the RAG system
rag_system = RestaurantAssistantRAG() 