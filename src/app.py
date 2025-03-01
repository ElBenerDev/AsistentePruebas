from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import sys
from .config import settings
from .assistant_logic import initialize_assistant, handle_assistant_response  # Add this import

# Enhanced logging configuration
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('whatsapp_bot')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        logger.info(f"Webhook verification requested - Mode: {mode}, Token: {token}, Challenge: {challenge}")
        
        if mode and token:
            if mode == 'subscribe' and token == settings.WEBHOOK_VERIFY_TOKEN:
                logger.info("Webhook verified successfully")
                return Response(content=challenge, media_type="text/plain")
            logger.warning("Webhook verification failed - Invalid token")
        raise HTTPException(status_code=403, detail="Invalid verification token")
    except Exception as e:
        logger.error(f"Error in verify_webhook: {str(e)}", exc_info=True)
        raise

@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming messages from WhatsApp"""
    try:
        body = await request.json()
        logger.info(f"Received webhook data: {json.dumps(body, indent=2)}")
        
        # Extract message from WhatsApp payload
        if body.get("object") == "whatsapp_business_account":
            for entry in body.get("entry", []):
                for change in entry.get("changes", []):
                    if messages := change.get("value", {}).get("messages", []):
                        message = messages[0]
                        if message.get("type") == "text":
                            # Extract user ID and message text
                            user_id = message["from"]
                            text = message["text"]["body"]
                            
                            logger.info(f"Processing message from {user_id}: {text}")
                            
                            # Process with assistant and get response
                            response_text, error = await handle_assistant_response(text, user_id)
                            
                            if error:
                                logger.error(f"Error: {error}")
                            
                            return {"status": "success"}
        
        logger.info("No valid message found in webhook data")
        return {"status": "no_message"}
        
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize the assistant when the app starts"""
    try:
        logger.info("Starting application initialization...")
        await initialize_assistant()
        logger.info("Assistant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize assistant: {str(e)}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when the app shuts down"""
    logger.info("Application shutting down...")