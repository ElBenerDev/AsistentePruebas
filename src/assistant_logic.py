import os
import json
import asyncio
import logging
from datetime import datetime
from threading import Lock
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI
import pytz
import httpx
from dotenv import load_dotenv

# Basic configuration
load_dotenv()

class FestivalConfig:
    """Configuration for EtnoSur assistant"""
    SPAIN_TZ = pytz.timezone('Europe/Madrid')
    TODAY = datetime.now(SPAIN_TZ)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ASSISTANT_ID = os.getenv("ETNOSUR_ASSISTANT_ID")
    BASE_URL = "http://localhost:5000/api"
    logger = logging.getLogger('etnosur_assistant')  # Add this line

    @classmethod
    def get_or_create_assistant_id(cls) -> str:
        """Gets or creates assistant ID"""
        try:
            client = OpenAI(api_key=cls.OPENAI_API_KEY)
            assistants = client.beta.assistants.list(limit=100)
            
            # Search existing assistant
            for assistant in assistants.data:
                if assistant.name == "Asistente EtnoSur":
                    return assistant.id
            
            # Create new assistant if not found
            assistant = client.beta.assistants.create(
                name="Asistente EtnoSur",
                instructions=f"""Eres un asistente profesional para el festival EtnoSur 2024.
        
        La fecha actual es {FestivalConfig.TODAY.strftime('%Y-%m-%d')}.
        
        IMPORTANTE: 
        - Proporciona información precisa sobre el festival
        - Responde siempre en español
        - Si no tienes información específica sobre algo, indícalo
        - Usa la función get_festival_info para obtener datos actualizados
        
        Información disponible:
        - Fechas y horarios de actividades
        - Precios de entradas
        - Ubicación de eventos
        - Servicios disponibles
        - Programación completa
        """,
                model="gpt-3.5-turbo",
                tools=[{
                    "type": "file_search"  # Changed from "retrieval" to "file_search"
                }, {
                    "type": "function",
                    "function": {
                        "name": "get_festival_info",
                        "description": "Obtiene información específica del festival EtnoSur",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": ["evento", "programacion", "precios", "informacion_general"],
                                    "description": "Categoría de información a consultar"
                                },
                                "subcategory": {
                                    "type": "string",
                                    "description": "Subcategoría específica (opcional)"
                                }
                            },
                            "required": ["category"]
                        }
                    }
                }]
            )

            # Upload festival info file
            file = client.files.create(
                file=open("src/etnosur_info.json", "rb"),
                purpose='assistants'
            )

            # Associate file with assistant
            client.beta.assistants.update(
                assistant_id=assistant.id,
                file_ids=[file.id]
            )

            return assistant.id

        except Exception as e:
            cls.logger.error(f"Error in get_or_create_assistant_id: {str(e)}")
            raise

class ConversationManager:
    def __init__(self):
        self.threads: Dict[str, Dict] = {}
        self.lock = Lock()
        self.logger = logging.getLogger('etnosur_assistant')
        self.client = OpenAI()

    def get_thread(self, user_id: str) -> Dict:
        """Gets or creates a thread for a user"""
        with self.lock:
            if user_id in self.threads:
                return self.threads[user_id]

            try:
                thread = self.client.beta.threads.create()
                thread_data = {
                    'thread_id': thread.id,
                    'context': {
                        'last_interaction': datetime.now(FestivalConfig.SPAIN_TZ).isoformat()
                    }
                }
                self.threads[user_id] = thread_data
                return thread_data
            except Exception as e:
                self.logger.error(f"Error creating thread: {e}")
                raise

class FestivalAssistant:
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.logger = logging.getLogger('etnosur_assistant')
        self.client = OpenAI(api_key=FestivalConfig.OPENAI_API_KEY)
        
        try:
            self.assistant_id = FestivalConfig.get_or_create_assistant_id()
            self.logger.info(f"Initialized assistant with ID: {self.assistant_id}")
        except Exception as e:
            self.logger.error(f"Error initializing assistant: {str(e)}")
            raise

    async def handle_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """Handles user message"""
        try:
            thread_data = self.conversation_manager.get_thread(user_id)
            thread_id = thread_data['thread_id']
            
            # Add message to thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message
            )

            # Create and run assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id
            )

            # Wait for completion
            while True:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                
                if run_status.status == 'completed':
                    messages = self.client.beta.threads.messages.list(thread_id=thread_id)
                    return {
                        "type": "text",
                        "content": messages.data[0].content[0].text.value
                    }
                
                elif run_status.status == 'requires_action':
                    tool_outputs = []
                    for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                        result = await self._handle_tool_call(tool_call, user_id)
                        if result:
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": json.dumps(result)
                            })
                    
                    if tool_outputs:
                        self.client.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread_id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )
                        continue
                
                elif run_status.status in ['failed', 'expired', 'cancelled']:
                    raise Exception(f"Run failed with status: {run_status.status}")
                
                await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            return {
                "type": "error",
                "content": "Lo siento, ha ocurrido un error. ¿Podrías reformular tu pregunta?"
            }

    async def _handle_tool_call(self, tool_call: Any, user_id: str) -> Optional[Dict]:
        """Handles tool calls from the assistant"""
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "get_festival_info":
                try:
                    with open("src/etnosur_info.json", "r", encoding='utf-8') as f:
                        festival_data = json.load(f)
                    
                    category = function_args.get('category')
                    subcategory = function_args.get('subcategory')
                    
                    if category in festival_data:
                        if subcategory and subcategory in festival_data[category]:
                            return {"data": festival_data[category][subcategory]}
                        return {"data": festival_data[category]}
                    return {"error": "Categoría no encontrada"}
                except Exception as e:
                    self.logger.error(f"Error getting festival info: {str(e)}")
                    return {"error": f"Error al obtener información del festival: {str(e)}"}

        except Exception as e:
            self.logger.error(f"Error in _handle_tool_call: {str(e)}")
            return {"error": f"Error general: {str(e)}"}

# Global assistant instance
festival_assistant = None

async def initialize_assistant():
    """Initializes the assistant"""
    global festival_assistant
    festival_assistant = FestivalAssistant()
    return True

if __name__ == "__main__":
    async def main():
        try:
            print("Inicializando asistente EtnoSur...")
            if await initialize_assistant():
                print("¡Hola! Soy tu asistente de EtnoSur 2024. ¿En qué puedo ayudarte?")
                print("(Escribe 'salir' para terminar)")
                test_user_id = "test_user_123"
                
                while True:
                    user_input = input("\nTú: ")
                    if user_input.lower() == "salir":
                        print("\n¡Hasta luego! ¡Que disfrutes del festival!")
                        break
                    
                    response = await festival_assistant.handle_message(test_user_id, user_input)
                    if response["type"] == "text":
                        print(f"\nAsistente: {response['content']}")
                    else:
                        print(f"\nError: {response['content']}")
            else:
                print("Error: No se pudo inicializar el asistente")
        except Exception as e:
            print(f"Error: {str(e)}")

    # Run the program
    asyncio.run(main())