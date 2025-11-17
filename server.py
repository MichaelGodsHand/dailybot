"""
Simple Daily-based voice bot server using Vertex AI Gemini Live
"""

import os
import json
from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.gemini_live.llm_vertex import GeminiLiveVertexLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

# Load environment variables
load_dotenv(override=True)

# Daily API helpers
import aiohttp

DAILY_API_KEY = os.getenv("DAILY_API_KEY")
DAILY_API_URL = "https://api.daily.co/v1"

async def create_daily_room():
    """Create a Daily room and return room URL"""
    if not DAILY_API_KEY:
        raise ValueError("DAILY_API_KEY not set")
    
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {DAILY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Create room with default settings
        data = {
            "properties": {
                "exp": int(os.time()) + 3600,  # 1 hour expiry
            }
        }
        
        async with session.post(
            f"{DAILY_API_URL}/rooms",
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail="Failed to create Daily room")
            
            room_data = await response.json()
            return room_data["url"]


def fix_credentials():
    """Fix GOOGLE_VERTEX_CREDENTIALS for Pipecat"""
    creds = os.getenv("GOOGLE_VERTEX_CREDENTIALS")
    
    if not creds:
        raise ValueError("GOOGLE_VERTEX_CREDENTIALS not set")
    
    creds = creds.strip()
    
    # If it's a file path
    if creds.endswith('.json') and os.path.isfile(creds):
        with open(creds, 'r') as f:
            creds_dict = json.load(f)
    else:
        # Assume it's a JSON string
        try:
            creds_dict = json.loads(creds)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid GOOGLE_VERTEX_CREDENTIALS: {e}")
    
    # Fix newlines in private key
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    
    return json.dumps(creds_dict)


async def run_bot(room_url: str):
    """Run the bot in a Daily room"""
    logger.info(f"Starting bot in room: {room_url}")
    
    # Get Vertex AI config
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    model_id = "gemini-live-2.5-flash-preview-native-audio-09-2025"
    
    model_path = f"projects/{project_id}/locations/{location}/publishers/google/models/{model_id}"
    
    # System instruction for the bot
    system_instruction = """You are a friendly AI assistant. Keep your responses concise and natural. 
    You're having a voice conversation, so speak naturally as you would to a friend."""
    
    # Initialize Daily transport
    transport = DailyTransport(
        room_url,
        None,  # No token needed for public rooms
        "Voice Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.3,
                    min_volume=0.6,
                )
            ),
        )
    )
    
    # Initialize Vertex AI LLM
    llm = GeminiLiveVertexLLMService(
        credentials=fix_credentials(),
        project_id=project_id,
        location=location,
        model=model_path,
        system_instruction=system_instruction,
        voice_id="Aoede",  # Options: Aoede, Charon, Fenrir, Kore, Puck
    )
    
    # Create context with greeting
    context = LLMContext([
        {"role": "user", "content": "Please greet the user briefly and ask how you can help them."}
    ])
    
    context_aggregator = LLMContextAggregatorPair(context)
    
    # Build pipeline
    pipeline = Pipeline([
        transport.input(),
        context_aggregator.user(),
        llm,
        transport.output(),
        context_aggregator.assistant(),
    ])
    
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
        ),
    )
    
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"First participant joined: {participant}")
        await task.queue_frames([LLMRunFrame()])
    
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left: {participant}")
        await task.cancel()
    
    @transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport, state):
        if state == "left":
            logger.info("Bot left the call")
            await task.cancel()
    
    runner = PipelineRunner()
    await runner.run(task)


# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/start")
async def start_bot(request: Request):
    """Start a new bot session"""
    try:
        # Create a Daily room
        room_url = await create_daily_room()
        
        # Start the bot in the background
        import asyncio
        asyncio.create_task(run_bot(room_url))
        
        # Return room URL for client to join
        return {"room_url": room_url}
    
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
