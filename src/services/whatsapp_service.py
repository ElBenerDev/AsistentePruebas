import requests
import logging
from ..config import get_settings

class WhatsAppService:
    def __init__(self):
        settings = get_settings()
        self.api_url = "https://graph.facebook.com/v17.0"
        self.token = settings.WHATSAPP_TOKEN
        self.phone_number_id = settings.PHONE_NUMBER_ID
    
    async def send_message(self, recipient_id: str, message: str):
        url = f"{self.api_url}/{self.phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        data = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "text",
            "text": {"body": message}
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error sending WhatsApp message: {str(e)}")
            raise