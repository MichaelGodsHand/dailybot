import os
import sys
import asyncio
import aiohttp
import httpx
import time
import json
import atexit

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pyngrok import ngrok

from pipecat.frames.frames import EndFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.gemini_live.llm_vertex import GeminiLiveVertexLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from loguru import logger

# Configure logger
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Global variable to store the ngrok tunnel
ngrok_tunnel = None

# FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_VERTEX_CREDENTIALS = os.getenv("GOOGLE_VERTEX_CREDENTIALS", "")

if not DAILY_API_KEY:
    raise ValueError("DAILY_API_KEY must be set")
if not GOOGLE_CLOUD_PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT_ID must be set")
if not GOOGLE_VERTEX_CREDENTIALS:
    raise ValueError("GOOGLE_VERTEX_CREDENTIALS must be set")


def start_ngrok_tunnel(port=8000):
    """Start ngrok tunnel and return the public URL."""
    global ngrok_tunnel
    
    # Get ngrok auth token from environment or use default
    ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
    
    if ngrok_auth_token:
        # Set the authtoken
        ngrok.set_auth_token(ngrok_auth_token)
        logger.info("Using ngrok auth token from environment")
    else:
        logger.warning("NGROK_AUTH_TOKEN not set in environment, using free ngrok (may have limitations)")
    
    # Start the tunnel
    ngrok_tunnel = ngrok.connect(port, "http")
    
    # Get the public URL
    public_url = ngrok_tunnel.public_url
    
    logger.info("=" * 60)
    logger.info("🚀 ngrok tunnel started successfully!")
    logger.info(f"📞 Public URL: {public_url}")
    logger.info(f"🌐 Access your bot at: {public_url}/start")
    logger.info(f"💚 Health check: {public_url}/health")
    logger.info("=" * 60)
    
    # Register cleanup function
    atexit.register(cleanup_ngrok)
    
    return public_url


def cleanup_ngrok():
    """Clean up ngrok tunnel on exit."""
    global ngrok_tunnel
    if ngrok_tunnel:
        try:
            ngrok.disconnect(ngrok_tunnel.public_url)
            ngrok.kill()
            logger.info("ngrok tunnel closed")
        except Exception as e:
            logger.error(f"Error closing ngrok tunnel: {e}")


def fix_credentials():
    """
    Fix GOOGLE_VERTEX_CREDENTIALS so Pipecat can parse it.
    Supports both file paths and JSON strings (including quoted JSON strings from .env files).
    """
    creds = GOOGLE_VERTEX_CREDENTIALS
    
    if not creds:
        raise ValueError("GOOGLE_VERTEX_CREDENTIALS environment variable is not set")
    
    # Strip whitespace
    creds = creds.strip()
    
    # Remove surrounding quotes if present (handles .env files that quote the JSON string)
    if (creds.startswith('"') and creds.endswith('"')) or (creds.startswith("'") and creds.endswith("'")):
        creds = creds[1:-1]
    
    # Check if it looks like JSON (starts with { or [)
    if creds.startswith('{') or creds.startswith('['):
        # Try to parse as JSON first
        try:
            creds_dict = json.loads(creds)
            # Ensure proper newline formatting for private_key
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            return json.dumps(creds_dict)
        except json.JSONDecodeError:
            # If JSON parsing fails, continue to check if it's a file path
            pass
    
    # Determine the file path - try multiple locations
    file_path = None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if it's an absolute path
    if os.path.isabs(creds):
        if os.path.isfile(creds):
            file_path = creds
    # Check if it exists as a relative path from current working directory
    elif os.path.isfile(creds):
        file_path = os.path.abspath(creds)
    # Check if it exists relative to the script directory
    else:
        potential_path = os.path.join(script_dir, creds)
        if os.path.isfile(potential_path):
            file_path = potential_path
    
    # If it ends with .json but we haven't found it yet, assume it's a file path
    # and try relative to script directory
    if not file_path and creds.endswith('.json'):
        potential_path = os.path.join(script_dir, creds)
        if os.path.isfile(potential_path):
            file_path = potential_path
    
    # If we found a file path, read from it
    if file_path and os.path.isfile(file_path):
        try:
            with open(file_path, 'r') as f:
                creds_dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to read credentials from file '{file_path}': {e}") from e
    else:
        # Last attempt: try to parse as JSON (might be unquoted JSON string)
        try:
            creds_dict = json.loads(creds)
        except json.JSONDecodeError as e:
            raise ValueError(f"GOOGLE_VERTEX_CREDENTIALS is not valid JSON and not a valid file path. Value: '{creds[:50]}...' Error: {e}") from e
    
    # Ensure proper newline formatting for private_key
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

    return json.dumps(creds_dict)


async def fetch_property_details(params: FunctionCallParams):
    """Fetch detailed property information from the API"""
    try:
        query = params.arguments["query"]
        phone_number = params.arguments["phone_number"]
        
        # Ensure phone number is in correct format (+91XXXXXXXXXX)
        if not phone_number.startswith("+91"):
            phone_number = f"+91{phone_number}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://preprocessor-739298578243.us-central1.run.app/query",
                json={
                    "query": query,
                    "number": phone_number
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Return the summary from the API response
            if data.get("status") == "success":
                await params.result_callback({
                    "summary": data.get("summary", ""),
                    "whatsapp_sent": True
                })
            else:
                await params.result_callback({
                    "summary": "I apologize, but I couldn't fetch the detailed information at this moment. Please try again.",
                    "whatsapp_sent": False
                })
    except Exception as e:
        logger.error(f"Error fetching property details: {e}")
        await params.result_callback({
            "summary": "I apologize, but I'm having trouble accessing the detailed information right now. Let me help you with general information instead.",
            "whatsapp_sent": False
        })


async def create_daily_room() -> tuple[str, str]:
    """Create a Daily room and return the URL and token"""
    async with aiohttp.ClientSession() as session:
        # Create room
        async with session.post(
            "https://api.daily.co/v1/rooms",
            headers={
                "Authorization": f"Bearer {DAILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "properties": {
                    "exp": int(time.time()) + 3600,  # 1 hour from now
                    "enable_chat": False,
                    "enable_emoji_reactions": False,
                }
            },
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Failed to create room: {response.status} - {error_text}")
                raise Exception(f"Failed to create Daily room: {response.status}")
            
            room_data = await response.json()
            logger.debug(f"Room data: {room_data}")
            
            room_url = room_data.get("url")
            room_name = room_data.get("name")
            
            if not room_url or not room_name:
                logger.error(f"Missing url or name in room response: {room_data}")
                raise Exception("Invalid room data from Daily API")

        # Create token
        async with session.post(
            "https://api.daily.co/v1/meeting-tokens",
            headers={
                "Authorization": f"Bearer {DAILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "properties": {
                    "room_name": room_name,
                    "is_owner": True,
                }
            },
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Failed to create token: {response.status} - {error_text}")
                raise Exception(f"Failed to create Daily token: {response.status}")
            
            token_data = await response.json()
            logger.debug(f"Token data: {token_data}")
            token = token_data.get("token")
            
            if not token:
                logger.error(f"Missing token in response: {token_data}")
                raise Exception("Invalid token data from Daily API")

    logger.info(f"Successfully created room: {room_url}")
    return room_url, token


async def run_bot(room_url: str, token: str):
    """Run the voice bot in the Daily room"""
    transport = None
    try:
        logger.info(f"Starting bot for room: {room_url}")
        
        # Initialize transport with Silero VAD
        transport = DailyTransport(
            room_url,
            token,
            "Real Estate Agent Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=False,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        stop_secs=0.3,
                        min_volume=0.6,
                    )
                ),
                transcription_enabled=True,
            ),
        )

        # Get project configuration
        project_id = GOOGLE_CLOUD_PROJECT_ID
        location = GOOGLE_CLOUD_LOCATION
        model_id = "gemini-live-2.5-flash-preview-native-audio-09-2025"
        
        # Build the full model path
        model_path = f"projects/{project_id}/locations/{location}/publishers/google/models/{model_id}"
        
        logger.info(f"Using Vertex AI model: {model_path}")

        # System instruction for the bot
        system_instruction = """
You are a professional real estate agent representing two premium residential projects in India. Your role is to assist potential buyers with information about these properties.
IMPORTANT: Speak in a colloquial TAMIL tone if the user speaks in TAMIL!

**PROJECT DETAILS:**

**1. SHREE KRISHNA ASHREY - KOLKATA**
- Developer: Yaduka Group
- Type: Boutique high-rise apartment complex (G+12 structure)
- Scale: Only 31 luxury homes on 1 Bigha land with 68% open space
- Certification: IGBC Green Home Pre-Certified (Gold Rating)
- Location: 88, Satin Sen Sarani, Kolkata
- Connectivity: Well-connected to Sealdah, Airport, Mani Square Mall, Apollo Hospital
- Configuration: 2, 3, and 4 BHK apartments with 3-side open layout
- Special Features: Vastu compliant, earthquake-resistant structure
- Amenities: Club Upavan (dedicated amenities floor) with terrace garden, lounge, gym, community hall, indoor games, yoga area
- Security: 24x7 manned security with CCTV surveillance
- Power: Generator backup for apartments and common areas
- Registration: Registered with WBHIRA

**2. PALM PREMIERE - CHENNAI**
- Developer: Unitech
- Type: Villa project within Uniworld City township
- Scale: Part of massive 242-acre integrated township
- Location: Nallambakkam, Chennai (off Vandalur-Kelambakkam road)
- Connectivity: Near IT Corridor (Siruseri SIPCOT), VIT campus
- Configuration: Multiple villa types (A, B, C, D, E) with 2 or 3 floor options
- Special Features: Private terraces and gardens, serene green environment
- Amenities: Modern clubhouse with swimming pool, gym, multipurpose hall, badminton court, school, retail facilities
- Security: Gated community with 24x7 security and CCTV
- Power: 1 KW light load backup for each villa

**COMMON FEATURES:**
- Premium flooring (marble/vitrified tiles)
- Branded sanitaryware and fittings
- Concealed electrical wiring
- Earthquake-resistant structures

**YOUR CONVERSATION FLOW:**
1. Greet the caller warmly and introduce yourself as their real estate agent
MOST VERY VERY IMPORTANT: Wait for the user to give ALL 10 numbers do not PROCEED without receiving all 10 numbers. Some users might pause in the middle, ask them the remaining numbers and then proceed!!!!
2. Ask for their phone number (10 digits) - THIS IS MANDATORY before proceeding
3. Once you have the phone number, ask what specific information they would like to know about the properties
4. For simple queries about location, pricing, configurations, or general features, use the information provided above
5. For detailed queries about specific features (kitchen, bathrooms, flooring, technical specifications), use the get_property_details tool
6. **CRITICAL: When you need to use the get_property_details tool, you MUST do the following in a SINGLE response:**
   - First, SAY: "Let me fetch those detailed specifications for you. This will just take a moment, and I'll also send you the relevant brochure pages on WhatsApp."
   - Then, immediately make the tool call
   - After receiving the tool result, continue speaking in the SAME response with the detailed information
   - Do NOT wait for the user to respond between announcing the fetch and providing the information
   - This entire sequence (announcement + tool call + result delivery) must happen in ONE continuous exchange
7. Keep responses conversational, brief (1-2 sentences), and professional
8. Always be ready to schedule site visits or connect them with the sales team

IMPORTANT: 
- ALWAYS collect the phone number FIRST before answering any property-related questions
- Speak ONLY in English
- Be warm, professional, and helpful
- When using the tool, the phone number format should be +91 followed immediately by the 10-digit number (no spaces)
- The tool call workflow (announcement → fetch → response) must be completed in a single conversational turn without breaking into multiple exchanges
"""

        # Define the property details tool
        property_details_function = FunctionSchema(
            name="get_property_details",
            description="Fetch detailed information about property features from the brochure. Use this for specific queries about kitchens, bathrooms, flooring, technical specifications, or any detailed features not covered in the general information.",
            properties={
                "query": {
                    "type": "string",
                    "description": "The specific question or detail the user wants to know about the property (e.g., 'Tell me about the kitchen', 'What are the bathroom features?')",
                },
                "phone_number": {
                    "type": "string",
                    "description": "The user's 10-digit phone number (will be prefixed with +91 automatically)",
                },
            },
            required=["query", "phone_number"],
        )

        tools = ToolsSchema(standard_tools=[property_details_function])

        # Initialize Vertex AI LLM Service with tools
        llm = GeminiLiveVertexLLMService(
            credentials=fix_credentials(),
            project_id=project_id,
            location=location,
            model=model_path,
            system_instruction=system_instruction,
            voice_id="Puck",  # Options: Aoede, Charon, Fenrir, Kore, Puck
            tools=tools,
        )

        # Register the function
        llm.register_function("get_property_details", fetch_property_details)

        # Create context with initial greeting
        context = LLMContext(
            [
                {
                    "role": "user",
                    "content": "Greet the caller as a real estate agent and ask for their phone number ALL In a Colloquial TAMIL tone."
                }
            ]
        )

        # Use context aggregator for proper conversation flow
        context_aggregator = LLMContextAggregatorPair(context)

        # Build pipeline with context aggregator
        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        # Create task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        # Set up event handlers
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info(f"First participant joined: {participant}")
            # Start capturing transcription for the participant
            await transport.capture_participant_transcription(participant["id"])
            
            # Give a moment for audio to be ready, then start conversation
            await asyncio.sleep(0.5)
            try:
                # Use LLMRunFrame to immediately trigger the LLM with the initial context
                await task.queue_frames([LLMRunFrame()])
                logger.info("Initial greeting triggered with LLMRunFrame")
            except Exception as e:
                logger.error(f"Error sending greeting: {e}")

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant}, reason: {reason}")
            await task.queue_frame(EndFrame())

        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            logger.info(f"Participant joined: {participant}")
            # Capture transcription for any new participant
            await transport.capture_participant_transcription(participant["id"])

        logger.info("Starting pipeline runner")
        runner = PipelineRunner()
        await runner.run(task)
        
        logger.info("Pipeline runner completed")

    except Exception as e:
        logger.error(f"Error running bot: {e}", exc_info=True)
        raise
    finally:
        if transport:
            try:
                logger.info("Cleaning up transport")
                await transport.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up transport: {e}")


@app.post("/start")
async def start_session(request: Request):
    """Create a Daily room, start the bot, and return connection details"""
    try:
        logger.info("Creating Daily room and starting bot...")
        
        # Create room and token
        room_url, token = await create_daily_room()
        logger.info(f"Created room: {room_url}")

        # Start bot in background
        asyncio.create_task(run_bot(room_url, token))

        # Return connection details
        return JSONResponse(
            content={
                "room_url": room_url,
                "token": token,
            }
        )

    except Exception as e:
        logger.error(f"Error starting session: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    
    # Start ngrok tunnel before starting the server
    try:
        public_url = start_ngrok_tunnel(port)
        logger.info("🎉 Bot is ready to accept connections!")
        logger.info(f"📋 Make POST requests to: {public_url}/start")
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}")
        logger.warning("⚠️  Continuing without ngrok. Bot will only be accessible locally.")
    
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=port)