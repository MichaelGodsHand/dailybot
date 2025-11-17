import os
import sys
import asyncio
import aiohttp
import time
from typing import AsyncGenerator

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipecat.frames.frames import EndFrame, LLMMessagesFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.processors.frame_processor import FrameProcessor

from loguru import logger

# Configure logger
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")

if not DAILY_API_KEY or not OPENAI_API_KEY:
    raise ValueError("DAILY_API_KEY and OPENAI_API_KEY must be set")


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
            
            # Daily API returns 'url' and 'name' fields
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


class TranscriptionLogger(FrameProcessor):
    """Simple processor to log transcriptions"""
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            logger.info(f"👤 User said: {frame.text}")
        
        await self.push_frame(frame, direction)


async def run_bot(room_url: str, token: str):
    """Run the voice bot in the Daily room"""
    transport = None
    try:
        logger.info(f"Starting bot for room: {room_url}")
        
        # Initialize transport with proper audio settings
        transport = DailyTransport(
            room_url,
            token,
            "Voice Bot",
            DailyParams(
                audio_out_enabled=True,
                audio_out_sample_rate=16000,
                transcription_enabled=True,
                vad_enabled=True,  # Enable VAD for better speech detection
                vad_analyzer=None,  # Use Daily's built-in VAD
                vad_audio_passthrough=True,  # Pass audio through even when not speaking
            ),
        )

        # Initialize TTS service with proper audio settings
        if CARTESIA_API_KEY:
            logger.info("Using Cartesia TTS")
            tts = CartesiaTTSService(
                api_key=CARTESIA_API_KEY,
                voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Conversational voice
                sample_rate=16000,
            )
        else:
            # Fallback to OpenAI TTS if Cartesia not available
            logger.info("Using OpenAI TTS")
            from pipecat.services.openai import OpenAITTSService
            tts = OpenAITTSService(
                api_key=OPENAI_API_KEY,
                voice="alloy",
            )

        # Initialize LLM service
        logger.info("Initializing OpenAI LLM")
        llm = OpenAILLMService(
            api_key=OPENAI_API_KEY,
            model="gpt-4o",
        )

        # Initialize context
        messages = [
            {
                "role": "system",
                "content": "You are a helpful and friendly voice assistant. Keep your responses concise and conversational, as this is a voice conversation. Be warm and engaging.",
            },
        ]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Create transcription logger
        transcription_logger = TranscriptionLogger()

        # Build pipeline
        logger.info("Building pipeline")
        pipeline = Pipeline(
            [
                transport.input(),
                transcription_logger,  # Log transcriptions
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        # Create task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        # Run the bot
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info(f"First participant joined: {participant}")
            try:
                # Greet the user
                await task.queue_frames([
                    LLMMessagesFrame(
                        [
                            {
                                "role": "system",
                                "content": "Greet the user warmly and ask how you can help them today.",
                            }
                        ]
                    )
                ])
            except Exception as e:
                logger.error(f"Error greeting user: {e}")

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant}, reason: {reason}")
            await task.queue_frame(EndFrame())

        @transport.event_handler("on_dialin_ready")
        async def on_dialin_ready(transport, cdata):
            logger.info("Dial-in ready")

        @transport.event_handler("on_transcription_message")
        async def on_transcription_message(transport, message):
            """Log raw transcription messages for debugging"""
            if message.get("is_final"):
                logger.debug(f"📝 Final transcription: {message.get('text', '')}")
            else:
                logger.debug(f"📝 Interim transcription: {message.get('text', '')}")

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
                "token": token,  # Optional: client can use this if needed
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
    uvicorn.run(app, host="0.0.0.0", port=port)
