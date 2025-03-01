from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import sys
import httpx
from .config import settings
from .assistant_logic import FestivalAssistant

# Setup logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('etnosur_bot')

app = FastAPI()

# Store assistant instance in app state
app.state.festival_assistant = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def send_whatsapp_message(recipient_id: str, message: str):
    """Send message using WhatsApp Cloud API"""
    if not settings.PHONE_NUMBER_ID:
        raise ValueError("PHONE_NUMBER_ID is not set")
    
    url = f"https://graph.facebook.com/v17.0/{settings.PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_id,
        "type": "text",
        "text": {"body": message}
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

@app.get("/")
async def health_check():
    """Basic health check endpoint"""
    logger.info("Health check requested")
    return {"status": "ok"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    """WhatsApp webhook verification endpoint"""
    try:
        mode = request.query_params.get('hub.mode')
        token = request.query_params.get('hub.verify_token')
        challenge = request.query_params.get('hub.challenge')
        
        if mode and token:
            if mode == 'subscribe' and token == settings.WEBHOOK_VERIFY_TOKEN:
                return Response(content=challenge, media_type="text/plain")
        raise HTTPException(status_code=403, detail="Invalid verification token")
    except Exception as e:
        logger.error(f"Error in verify_webhook: {str(e)}")
        raise

@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming messages from WhatsApp"""
    try:
        body = await request.json()
        logger.info(f"Received webhook data: {json.dumps(body, indent=2)}")
        
        if body.get("object") == "whatsapp_business_account":
            for entry in body.get("entry", []):
                for change in entry.get("changes", []):
                    if messages := change.get("value", {}).get("messages", []):
                        message = messages[0]
                        if message.get("type") == "text":
                            user_id = message["from"]
                            text = message["text"]["body"]
                            
                            logger.info(f"Processing message from {user_id}: {text}")
                            
                            # Use assistant from app state
                            if app.state.festival_assistant is None:
                                app.state.festival_assistant = FestivalAssistant()
                            
                            # Get response from assistant
                            response = await app.state.festival_assistant.handle_message(user_id, text)
                            
                            if response["type"] == "text":
                                # Send response back via WhatsApp
                                await send_whatsapp_message(user_id, response["content"])
                            else:
                                logger.error(f"Error: {response['content']}")
                                await send_whatsapp_message(
                                    user_id, 
                                    "Lo siento, ha ocurrido un error. ¿Podrías reformular tu pregunta?"
                                )
                            
                            return {"status": "success"}
        
        return {"status": "no_message"}
        
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize the assistant when the app starts"""
    try:
        logger.info("Starting application initialization...")
        app.state.festival_assistant = FestivalAssistant()
        logger.info("Assistant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize assistant: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when the app shuts down"""
    logger.info("Application shutting down...")