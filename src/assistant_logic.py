import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI
import pytz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import httpx
from .config import settings


# Configuración básica
load_dotenv()

class ConversationState(Enum):
    INITIAL = "initial"
    COLLECTING_INFO = "collecting_info"
    SCHEDULING = "scheduling"
    CONFIRMING = "confirming"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AppointmentData:
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    service_type: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la instancia a un diccionario"""
        return {
            'contact_name': self.contact_name,
            'contact_phone': self.contact_phone,
            'contact_email': self.contact_email,
            'service_type': self.service_type,
            'preferred_date': self.preferred_date,
            'preferred_time': self.preferred_time
        }

class Config:
    """Centraliza la configuración del asistente"""
    ARGENTINA_TZ = pytz.timezone('America/Argentina/Buenos_Aires')
    TODAY = datetime.now(ARGENTINA_TZ)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ASSISTANT_ID = os.getenv("DENTAL_ASSISTANT_ID")
    BASE_URL = "http://localhost:5000/api"
    
    # Log de la configuración
    logger = logging.getLogger('dental_assistant')
    logger.info(f"Current timezone: {ARGENTINA_TZ}")
    
    @classmethod
    async def test_api_connection(cls) -> bool:
        """Prueba la conexión a la API local"""
        try:
            test_url = f"{cls.BASE_URL}/health"
            cls.logger.info(f"Testing local API connection at: {test_url}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(test_url)
                if response.status_code == 200:
                    cls.logger.info("Local API connection successful")
                    return True
                else:
                    cls.logger.error(f"Local API connection failed with status: {response.status_code}")
                    return False
        except Exception as e:
            cls.logger.error(f"Local API connection test failed: {str(e)}")
            return False

    @classmethod
    def get_api_url(cls, endpoint: str) -> str:
        """Construye una URL completa para un endpoint de la API"""
        if not endpoint.startswith('/'):
            endpoint = f'/{endpoint}'
        return f"{cls.BASE_URL}{endpoint}"

    @classmethod
    async def test_api_connection(cls) -> bool:
        """Prueba la conexión a la API"""
        try:
            test_url = cls.get_api_url('/health')
            cls.logger.info(f"Testing API connection at: {test_url}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(test_url)
                if response.status_code == 200:
                    cls.logger.info("API connection successful")
                    return True
                else:
                    cls.logger.error(f"API connection failed with status: {response.status_code}")
                    return False
        except Exception as e:
            cls.logger.error(f"API connection test failed: {str(e)}")
            return False
    
    RETRY_STRATEGY = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
    )
    
    @classmethod
    def get_or_create_assistant_id(cls) -> str:
        """Obtiene o crea el ID del asistente"""
        try:
            # Primero intentar obtener el asistente existente
            client = OpenAI(api_key=cls.OPENAI_API_KEY)
            assistants = client.beta.assistants.list(limit=100)
            
            # Buscar un asistente con el nombre específico
            for assistant in assistants.data:
                if assistant.name == "Asistente Dental Laura":
                    cls.logger.info(f"Found existing assistant with ID: {assistant.id}")
                    return assistant.id
            
            cls.logger.info("No existing assistant found, creating new one...")
            
            # Si no se encuentra, crear uno nuevo
            assistant = client.beta.assistants.create(
                name="Asistente Dental Laura",
                instructions=f"""Eres un asistente dental profesional diseñado para programar citas.
                IMPORTANTE: 
                - NUNCA asumas fechas por tu cuenta
                - SIEMPRE pregunta al usuario por su fecha preferida
                - SIEMPRE recopila la información en orden específico:
                1. Tipo de servicio
                2. Nombre completo
                3. Teléfono (10 dígitos)
                4. Correo electrónico (requerido)
                5. Fecha preferida
                6. Y solo entonces mostrar horarios disponibles
                
                Detecta el idioma del usuario y responde siempre en el mismo idioma.
                
                La fecha actual es {Config.TODAY.strftime('%Y-%m-%d')}.
                
               Reglas adicionales:
                1. No sugieras fechas específicas a menos que el usuario las solicite
                2. Si el usuario no menciona una fecha, pregunta "¿Para qué fecha te gustaría agendar la cita?"
                3. Solo muestra horarios disponibles después de tener una fecha confirmada
                4. Valida cada fecha antes de mostrar horarios
                5. No agendar citas en fines de semana
                6. SIEMPRE recopila la siguiente información en este orden:
                - Tipo de servicio
                - Nombre completo
                - Teléfono (10 dígitos)
                - Correo electrónico (campo obligatorio)
                - Fecha preferida
                7. TODOS los campos son requeridos sin excepción
                8. No mencionar NUNCA que algún campo es opcional
                9. Si falta algún dato, solicitarlo antes de continuar
                10. Validar el formato del teléfono (10 dígitos) y correo electrónico
                """,
                model="gpt-3.5-turbo",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "schedule_appointment",
                            "description": "Programa una cita dental después de validar la disponibilidad",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "contact_name": {
                                        "type": "string",
                                        "description": "Nombre completo del paciente"
                                    },
                                    "contact_phone": {
                                        "type": "string",
                                        "description": "Número de teléfono de 10 dígitos"
                                    },
                                    "contact_email": {
                                        "type": "string",
                                        "description": "Correo electrónico del paciente"
                                    },
                                    "service_type": {
                                        "type": "string",
                                        "enum": ["Chequeo rutinario", "Limpieza", "Extracción", "Ortodoncia"],
                                        "description": "Tipo de servicio dental solicitado"
                                    },
                                    "preferred_date": {
                                        "type": "string",
                                        "description": "Fecha en formato YYYY-MM-DD"
                                    },
                                    "preferred_time": {
                                        "type": "string",
                                        "description": "Hora en formato HH:MM (se añadirán los segundos internamente)"
                                    }
                                },
                                "required": ["contact_name", "contact_phone", "contact_email", "service_type", "preferred_date", "preferred_time"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "validate_appointment_date",
                            "description": "Valida si una fecha está disponible para citas",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "date": {
                                        "type": "string",
                                        "description": "Fecha en formato YYYY-MM-DD"
                                    }
                                },
                                "required": ["date"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_available_times",
                            "description": "Obtiene los horarios disponibles para una fecha",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "date": {
                                        "type": "string",
                                        "description": "Fecha en formato YYYY-MM-DD"
                                    }
                                },
                                "required": ["date"]
                            }
                        }
                    }
                ]
            )
            
            cls.logger.info(f"Created new assistant with ID: {assistant.id}")
            return assistant.id

        except Exception as e:
            cls.logger.error(f"Error in get_or_create_assistant_id: {str(e)}")
            raise
        
        
    @classmethod
    def load_business_info(cls) -> Dict:
        """Carga y valida la información del negocio desde el archivo JSON"""
        try:
            file_path = os.path.join(os.path.dirname(__file__), 'dental_business_info.json')
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Validar estructura básica
            required_keys = ['business_info', 'services', 'policies']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Falta la sección '{key}' en el archivo de configuración")
            
            return data
        except FileNotFoundError:
            cls.logger.error("No se encontró el archivo dental_business_info.json")
            raise
        except json.JSONDecodeError as e:
            cls.logger.error(f"Error al decodificar el archivo JSON: {str(e)}")
            raise
        except Exception as e:
            cls.logger.error(f"Error al cargar la información del negocio: {str(e)}")
            raise
        
    @classmethod
    async def verify_api_connection(cls):
        """Verifica la conexión con la API probando varios endpoints"""
        client = HTTPClient()
        logger = logging.getLogger('dental_assistant')
        max_retries = 5  # Aumentado a 5 intentos
        retry_delay = 10  # Aumentado a 10 segundos
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Intento {attempt + 1}/{max_retries} - Verificando conexión API")
                
                # Primero intentar con la ruta principal
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(cls.RENDER_URL)
                    
                    if response.status_code == 200:
                        logger.info("Conexión API exitosa")
                        return True
                        
                    elif response.status_code == 502:
                        logger.warning(f"Bad Gateway (502) - Reintentando en {retry_delay} segundos...")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                            
                    else:
                        # Intentar con una ruta alternativa
                        try:
                            alt_url = f"{cls.BASE_URL}/health"
                            response = await client.get(alt_url)
                            if response.status_code == 200:
                                logger.info("Conexión API exitosa (ruta alternativa)")
                                return True
                        except Exception:
                            pass
                            
                        if attempt < max_retries - 1:
                            logger.warning(f"Reintentando en {retry_delay} segundos...")
                            await asyncio.sleep(retry_delay)
                            continue
                        
                        logger.error(f"API check failed: {response.status_code}")
                        return False
                        
            except Exception as e:
                logger.error(f"Error de conexión: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Reintentando en {retry_delay} segundos...")
                    await asyncio.sleep(retry_delay)
                else:
                    return False
        
        return False

def setup_logging():
    """Configura el sistema de logging"""
    logger = logging.getLogger('dental_assistant')
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler('dental_assistant.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class HTTPClient:
    """Cliente HTTP con reintentos usando httpx"""
    def __init__(self):
        self.logger = logging.getLogger('dental_assistant')
        self.timeout = 30.0
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

    async def get(self, url: str, **kwargs) -> httpx.Response:
        self.logger.info(f"Making GET request to: {url}")
        try:
            response = await self.client.get(url, **kwargs)
            self.logger.info(f"Response status: {response.status_code}")
            self.logger.debug(f"Response content: {response.text}")
            return response
        except Exception as e:
            self.logger.error(f"Error in GET request: {str(e)}")
            raise

    async def post(self, url: str, **kwargs) -> httpx.Response:
        self.logger.info(f"Making POST request to: {url}")
        try:
            response = await self.client.post(url, **kwargs)
            self.logger.info(f"Response status: {response.status_code}")
            self.logger.debug(f"Response content: {response.text}")
            return response
        except Exception as e:
            self.logger.error(f"Error in POST request: {str(e)}")
            raise

    async def close(self):
        await self.client.aclose()

class ConversationManager:
    def __init__(self):
        self.threads: Dict[str, Dict] = {}
        self.lock = Lock()
        self.logger = logging.getLogger('dental_assistant')
        self.client = OpenAI()  # Remove api_key parameter as it's read from env

    def get_thread(self, user_id: str) -> Dict:
        """Obtiene o crea un thread para un usuario"""
        with self.lock:
            if user_id in self.threads:
                thread_data = self.threads[user_id]
                # Actualizar última interacción
                thread_data['context']['last_interaction'] = datetime.now(Config.ARGENTINA_TZ).isoformat()
                # Asegurarse de que el contexto tenga toda la información necesaria
                if 'appointment_data' not in thread_data['context']:
                    thread_data['context']['appointment_data'] = {}
                if 'last_mentioned_date' not in thread_data['context']:
                    thread_data['context']['last_mentioned_date'] = None
                if 'last_mentioned_service' not in thread_data['context']:
                    thread_data['context']['last_mentioned_service'] = None
                return thread_data

            # Si no existe el thread, crear uno nuevo
            try:
                thread = self.client.beta.threads.create()
                thread_data = {
                    'thread_id': thread.id,
                    'context': {
                        'appointment_data': {},
                        'last_interaction': datetime.now(Config.ARGENTINA_TZ).isoformat(),
                        'state': ConversationState.INITIAL.value,
                        'available_times': [],
                        'last_mentioned_date': None,
                        'last_mentioned_service': None
                    }
                }
                self.threads[user_id] = thread_data
                self.logger.info(f"Created new thread {thread.id} for user {user_id}")
                return thread_data
            except Exception as e:
                self.logger.error(f"Error creating thread: {e}")
                raise

    def update_context(self, user_id: str, updates: Dict[str, Any]) -> None:
        """Actualiza el contexto de un thread"""
        with self.lock:
            if user_id in self.threads:
                # Convertir cualquier datetime a string ISO format
                serializable_updates = {}
                for key, value in updates.items():
                    if isinstance(value, datetime):
                        serializable_updates[key] = value.isoformat()
                    else:
                        serializable_updates[key] = value
                
                self.threads[user_id]['context'].update(serializable_updates)
                self.logger.debug(f"Updated context for user {user_id}: {serializable_updates}")

class DentalAssistant:
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.logger = logging.getLogger('dental_assistant')
        self.http_client = HTTPClient()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        try:
            self.assistant_id = Config.get_or_create_assistant_id()
            self.logger.info(f"Initialized assistant with ID: {self.assistant_id}")
        except Exception as e:
            self.logger.error(f"Error initializing assistant: {str(e)}")
            # Intentar crear un nuevo asistente si falla la obtención
            try:
                self.assistant_id = Config.get_or_create_assistant_id()
                self.logger.info(f"Created new assistant with ID: {self.assistant_id}")
            except Exception as create_error:
                self.logger.error(f"Critical error creating assistant: {str(create_error)}")
                raise

    async def handle_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """Maneja un mensaje del usuario"""
        self.logger.info(f"Processing message for user {user_id}: {message}")
        
        try:
            # Verificar que el asistente existe
            if not self.assistant_id:
                self.logger.error("No assistant ID available")
                return {
                    "type": "error",
                    "content": "Error del sistema. Por favor, intenta más tarde."
                }

            # Obtener o crear thread para el usuario
            thread_data = self.conversation_manager.get_thread(user_id)
            thread_id = thread_data['thread_id']
            context = thread_data['context']
            
            # Añadir el mensaje al thread
            try:
                await self._add_message_to_thread(thread_id, message)
            except Exception as e:
                self.logger.error(f"Error adding message to thread: {str(e)}")
                raise

            # Crear y procesar el run
            try:
                run = await self._create_run(thread_id, context)
                response = await self._process_run(thread_id, run.id, user_id)
            except Exception as e:
                self.logger.error(f"Error in run creation/processing: {str(e)}")
                raise

            # Actualizar el tiempo de última interacción
            try:
                current_time = datetime.now(Config.ARGENTINA_TZ)
                self.conversation_manager.update_context(user_id, {
                    'last_interaction': current_time.isoformat()
                })
            except Exception as e:
                self.logger.warning(f"Error updating context: {str(e)}")
                # No elevar esta excepción ya que no es crítica

            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", exc_info=True)
            error_message = (
                "Lo siento, ha ocurrido un error. "
                "¿Podrías reformular tu solicitud?"
            )
            if "No assistant found" in str(e):
                try:
                    # Intentar recrear el asistente
                    self.assistant_id = Config.get_or_create_assistant_id()
                    error_message = (
                        "Hubo un problema técnico, pero ya fue solucionado. "
                        "¿Podrías repetir tu mensaje?"
                    )
                except Exception as create_error:
                    self.logger.error(f"Failed to recreate assistant: {str(create_error)}")
                    
            return {
                "type": "error",
                "content": error_message
            }
    async def _add_message_to_thread(self, thread_id: str, message: str):
        """Añade un mensaje al thread"""
        max_retries = 5
        retry_delay = 2  # segundos
        
        for attempt in range(max_retries):
            try:
                return self.client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=message
                )
            except Exception as e:
                if "while a run is active" in str(e):
                    self.logger.info(f"Run activo detectado, esperando... (intento {attempt + 1})")
                    try:
                        # Obtener runs activos
                        runs = self.client.beta.threads.runs.list(thread_id=thread_id)
                        for run in runs.data:
                            if run.status in ['active', 'queued', 'in_progress']:
                                # Cancelar el run activo
                                self.logger.info(f"Cancelando run activo: {run.id}")
                                try:
                                    self.client.beta.threads.runs.cancel(
                                        thread_id=thread_id,
                                        run_id=run.id
                                    )
                                    await asyncio.sleep(1)  # Esperar a que se cancele
                                except Exception as cancel_error:
                                    self.logger.error(f"Error cancelando run: {str(cancel_error)}")
                    except Exception as list_error:
                        self.logger.error(f"Error listando runs: {str(list_error)}")
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise Exception("No se pudo agregar el mensaje después de múltiples intentos")
                else:
                    raise
        
        raise Exception("No se pudo agregar el mensaje al thread")

    async def _create_run(self, thread_id: str, context: Dict):
        try:
            serializable_context = {
                'state': context.get('state', ConversationState.INITIAL.value),
                'last_interaction': context.get('last_interaction', '').isoformat() if isinstance(context.get('last_interaction'), datetime) else context.get('last_interaction', ''),
                'appointment_data': context.get('appointment_data', {})
            }

            return self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id,
                instructions=f"""Contexto actual: {json.dumps(serializable_context)}
                
                Instrucciones:
                1. NUNCA asumas una fecha para la cita. SIEMPRE pregunta al usuario primero.
                2. Cuando el usuario proporcione información incompleta, solicita la información faltante.
                3. Solo usa validate_appointment_date cuando el usuario haya especificado una fecha.
                4. Solo usa get_available_times cuando tengas una fecha válida confirmada.
                5. Recopila la siguiente información en orden:
                - Tipo de servicio
                - Nombre completo
                - Teléfono
                - Email
                - Fecha preferida
                - Y solo entonces, muestra horarios disponibles
                6. Mantén un tono profesional y amable"""
            )
        except Exception as e:
            self.logger.error(f"Error creating run: {str(e)}")
            raise

    async def _process_run(self, thread_id: str, run_id: str, user_id: str) -> Dict[str, Any]:
        """Procesa un run hasta su completación"""
        try:
            max_retries = 10  # Número máximo de intentos
            retry_count = 0
            
            while retry_count < max_retries:
                run_status = await self._get_run_status(thread_id, run_id)
                
                if run_status.status == 'completed':
                    return await self._handle_completed_run(thread_id)
                    
                elif run_status.status == 'requires_action':
                    tool_response = await self._handle_required_action(thread_id, run_id, run_status, user_id)
                    if tool_response.get("type") != "error":
                        continue  # Continuar esperando la completación
                    return tool_response
                    
                elif run_status.status in ['failed', 'expired', 'cancelled']:
                    raise Exception(f"Run failed with status: {run_status.status}")
                    
                elif run_status.status == 'queued':
                    retry_count += 1
                    await asyncio.sleep(1)
                    continue
                    
                await asyncio.sleep(1)
                
            # Si llegamos aquí, se agotó el tiempo
            raise TimeoutError("Run processing timed out")
            
        except Exception as e:
            self.logger.error(f"Error processing run: {e}", exc_info=True)
            return {
                "type": "error",
                "content": f"Error procesando tu solicitud: {str(e)}"
            }

    async def _get_run_status(self, thread_id: str, run_id: str):
        """Obtiene el estado actual de un run"""
        return self.client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )

    async def _handle_completed_run(self, thread_id: str) -> Dict[str, Any]:
        """Maneja un run completado"""
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        return {
            "type": "text",
            "content": messages.data[0].content[0].text.value
        }

    async def _handle_required_action(self, thread_id: str, run_id: str, run_status: Any, user_id: str) -> Dict[str, Any]:
        """Maneja las acciones requeridas por el asistente"""
        try:
            tool_outputs = []
            
            for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                self.logger.info(f"Processing tool call: {tool_call.function.name}")
                result = await self._handle_tool_call(tool_call, user_id)
                
                if result is not None:
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps(result)
                    })
                    self.logger.info(f"Tool call result: {result}")

            if tool_outputs:
                try:
                    # Enviar los resultados de las herramientas
                    self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run_id,
                        tool_outputs=tool_outputs
                    )
                    
                    # Esperar un momento y verificar el estado
                    await asyncio.sleep(1)
                    return {"type": "tool_outputs_submitted"}
                    
                except Exception as e:
                    self.logger.error(f"Error submitting tool outputs: {str(e)}", exc_info=True)
                    return {
                        "type": "error",
                        "content": f"Error al procesar la respuesta: {str(e)}"
                    }
            
            return {
                "type": "error",
                "content": "No se pudieron procesar las acciones requeridas"
            }

        except Exception as e:
            self.logger.error(f"Error handling required action: {str(e)}", exc_info=True)
            return {
                "type": "error",
                "content": "Error procesando la acción requerida"
            }
    async def _handle_tool_call(self, tool_call: Any, user_id: str) -> Optional[Dict]:
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            self.logger.info(f"Processing {function_name} call with args: {function_args}")
            
            # Manejar get_business_info fuera del bloque httpx ya que no necesita hacer llamadas HTTP
            if function_name == "get_business_info":
                info_type = function_args.get('info_type')
                try:
                    # Cargar la información del negocio
                    business_info = Config.load_business_info()
                    
                    if info_type == "services":
                        return {"services": business_info["services"]}
                    elif info_type == "working_hours":
                        return {"working_hours": business_info["business_info"]["working_hours"]}
                    elif info_type == "policies":
                        return {"policies": business_info["policies"]}
                    elif info_type == "insurance":
                        return {"insurance": business_info["policies"]["insurance"]}
                    elif info_type == "payment_methods":
                        return {"payment_methods": business_info["policies"]["payment_methods"]}
                    else:
                        return {"error": "Tipo de información no válido"}
                except Exception as e:
                    self.logger.error(f"Error getting business info: {str(e)}")
                    return {"error": f"Error al obtener información del negocio: {str(e)}"}

            # Usar Config.BASE_URL correctamente
            base_url = "http://localhost:5000/api"  # URL fija para desarrollo local
            self.logger.info(f"Using base URL: {base_url}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    if function_name == "validate_appointment_date":
                        date_str = function_args.get('date')
                        if not date_str:
                            return {"error": "No date provided"}
                            
                        url = f"{base_url}/validate_date"
                        self.logger.info(f"Validating date {date_str} at URL: {url}")
                        
                        # Log completo de la solicitud
                        full_url = f"{url}?date={date_str}"
                        self.logger.info(f"Making request to: {full_url}")
                        
                        response = await client.get(
                            url,
                            params={"date": date_str},
                            headers={"Accept": "application/json"}
                        )
                        
                        self.logger.info(f"Response status: {response.status_code}")
                        self.logger.info(f"Response content: {response.text}")

                        if response.status_code == 200:
                            return response.json()
                        else:
                            error_msg = f"Error in {function_name}: Status {response.status_code} - {response.text}"
                            self.logger.error(error_msg)
                            return {"error": error_msg, "status_code": response.status_code}
                            
                    elif function_name == "get_available_times":
                        date_str = function_args.get('date')
                        url = f"{base_url}/get_availability"
                        response = await client.get(
                            url,
                            params={"date": date_str},
                            headers={"Accept": "application/json"}
                        )
                        
                    elif function_name == "schedule_appointment":
                        url = f"{base_url}/create_appointment"
                        time_str = function_args['preferred_time']
                        if ':' in time_str and len(time_str.split(':')) == 2:
                            time_str = f"{time_str}:00" 
                            
                        appointment_data = {
                            "contact_name": function_args['contact_name'],
                            "contact_phone": function_args['contact_phone'],
                            "contact_email": function_args['contact_email'],
                            "service_type": function_args['service_type'],
                            "activity_due_date": function_args['preferred_date'],
                            "activity_due_time": time_str
                        }
                        response = await client.post(
                            url, 
                            json=appointment_data,
                            headers={"Accept": "application/json", "Content-Type": "application/json"}
                        )

                    if response.status_code == 200:
                        return response.json()
                    else:
                        error_msg = f"Error in {function_name}: Status {response.status_code} - {response.text}"
                        self.logger.error(error_msg)
                        return {"error": error_msg, "status_code": response.status_code}

                except httpx.RequestError as e:
                    error_msg = f"Error making request in {function_name}: {str(e)}"
                    self.logger.error(error_msg)
                    return {"error": error_msg}

        except Exception as e:
            self.logger.error(f"Error in _handle_tool_call: {str(e)}", exc_info=True)
            return {"error": f"Error general: {str(e)}"}

    def _get_next_date(self, date_str: str) -> str:
        """Convierte una descripción de fecha en formato YYYY-MM-DD"""
        today = datetime.now(Config.ARGENTINA_TZ)
        
        # Si ya es una fecha en formato YYYY-MM-DD, verificar que sea correcta
        try:
            parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
            # Verificar que sea un día válido
            weekday = parsed_date.weekday()
            if weekday >= 5:  # Si es fin de semana
                days_to_add = 7 - weekday + 1  # Mover al próximo día hábil
                parsed_date += timedelta(days=days_to_add)
            return parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            pass  # No es una fecha en formato YYYY-MM-DD, continuar con el procesamiento

        # Procesar fechas en lenguaje natural
        date_str = date_str.lower()
        
        # Mapeo de días de la semana (0 = Lunes, 6 = Domingo)
        day_mapping = {
            'lunes': 0, 'martes': 1, 'miércoles': 2, 'miercoles': 2,
            'jueves': 3, 'viernes': 4, 'sábado': 5, 'sabado': 5, 'domingo': 6
        }

        # Encontrar el día de la semana mencionado
        mentioned_day = None
        for day, day_num in day_mapping.items():
            if day in date_str:
                mentioned_day = day_num
                break

        if mentioned_day is not None:
            current_day = today.weekday()
            days_until = (mentioned_day - current_day) % 7
            
            # Si se menciona "próximo" o "siguiente", añadir una semana
            if "próximo" in date_str or "siguiente" in date_str:
                if days_until == 0 or days_until < 0:
                    days_until += 7
            else:
                # Si el día ya pasó esta semana, ir a la próxima
                if days_until == 0 or days_until < 0:
                    days_until += 7

            target_date = today + timedelta(days=days_until)
            return target_date.strftime('%Y-%m-%d')

        return date_str

    async def _handle_appointment_scheduling(self, user_id: str, appointment_data: Dict) -> Dict[str, Any]:
        """Maneja la programación de citas"""
        try:
            thread_data = self.conversation_manager.get_thread(user_id)
            context = thread_data['context']
            
            # Crear o actualizar AppointmentData
            appt_data = AppointmentData(**appointment_data)
            context['appointment_data'] = appt_data.to_dict()  # Guardar como diccionario
            
            # Obtener horarios disponibles
            available_times = await self._get_available_times(appointment_data['preferred_date'])
            
            if available_times:
                self.conversation_manager.update_context(user_id, {
                    'state': ConversationState.SCHEDULING.value,  # Usar el valor del enum
                    'available_times': available_times,
                    'selected_date': appointment_data['preferred_date']
                })
                
                return {
                    "success": True,
                    "available_times": available_times,
                    "message": self._format_available_times_message(
                        appointment_data['preferred_date'],
                        available_times
                    )
                }
            else:
                next_dates = await self._get_next_available_dates(
                    datetime.strptime(appointment_data['preferred_date'], '%Y-%m-%d')
                )
                return {
                    "success": False,
                    "message": self._format_alternative_dates_message(
                        appointment_data['preferred_date'],
                        next_dates
                    )
                }

        except Exception as e:
            self.logger.error(f"Error scheduling appointment: {e}", exc_info=True)
            return {
                "success": False,
                "message": "Error al programar la cita. Por favor, intenta nuevamente."
            }

    async def _get_available_times(self, date: str) -> List[str]:
        """Obtiene los horarios disponibles para una fecha"""
        try:
            response = await self.http_client.get(
                f"{Config.BASE_URL}/get_availability",
                params={"date": date},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("available_times", [])
        except Exception as e:
            self.logger.error(f"Error getting available times: {e}")
            return []

    async def _get_next_available_dates(self, start_date: datetime, 
                                      num_dates: int = 3) -> List[str]:
        """Obtiene las próximas fechas disponibles"""
        available_dates = []
        current_date = start_date
        
        while len(available_dates) < num_dates:
            if current_date.weekday() < 5:  # Excluir fines de semana
                try:
                    response = await self.http_client.get(
                        f"{Config.BASE_URL}/validate_date",
                        params={"date": current_date.strftime('%Y-%m-%d')},
                        timeout=30
                    )
                    if response.status_code == 200:
                        available_dates.append(current_date.strftime('%Y-%m-%d'))
                except Exception as e:
                    self.logger.error(f"Error validating date: {e}")
                    
            current_date += timedelta(days=1)
        return available_dates

    def _format_available_times_message(self, date: str, times: List[str]) -> str:
        """Formatea el mensaje de horarios disponibles"""
        return (f"Para el {date}, tenemos los siguientes horarios disponibles:\n"
                f"{', '.join(times)}")

    def _format_alternative_dates_message(self, original_date: str, 
                                        alternative_dates: List[str]) -> str:
        """Formatea el mensaje de fechas alternativas"""
        return (f"Lo siento, no hay horarios disponibles para el {original_date}. "
                f"Fechas alternativas disponibles:\n{', '.join(alternative_dates)}")

# Crear una única instancia global del asistente
dental_assistant = None

async def initialize_assistant():
    """Inicializa el asistente"""
    global dental_assistant
    dental_assistant = DentalAssistant()
    return True

async def handle_assistant_response(message: str, user_id: str, user_context: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
    """Función principal para manejar las respuestas del asistente"""
    try:
        logger.info(f"Processing message for user {user_id}: {message}")
        
        # Si se proporciona contexto del usuario, actualizar el contexto del thread
        if user_context and dental_assistant:
            thread_data = dental_assistant.conversation_manager.get_thread(user_id)
            dental_assistant.conversation_manager.update_context(user_id, user_context)
        
        response = await dental_assistant.handle_message(user_id, message)
        
        if response["type"] == "text":
            return response["content"], None
        elif response["type"] == "error":
            logger.error(f"Error en la respuesta: {response['content']}")
            return None, response["content"]
        else:
            logger.warning(f"Tipo de respuesta no manejado: {response['type']}")
            return None, "Tipo de respuesta no manejado"

    except Exception as e:
        logger.error(f"Error en handle_assistant_response: {str(e)}", exc_info=True)
        return None, str(e)

async def test_endpoints():
    """Función para probar los endpoints"""
    client = HTTPClient()
    logger = logging.getLogger('dental_assistant')
    
    try:
        # Probar el endpoint de health check
        response = await client.get(f"{Config.BASE_URL}/health")
        logger.info(f"Health check response: {response.status_code}")
        logger.info(f"Health check content: {response.text}")
        
        # Probar validate_date
        response = await client.get(
            f"{Config.BASE_URL}/validate_date",
            params={"date": "2025-01-21"}
        )
        logger.info(f"Validate date response: {response.status_code}")
        logger.info(f"Validate date content: {response.text}")
        
        # Probar get_availability
        response = await client.get(
            f"{Config.BASE_URL}/get_availability",
            params={"date": "2025-01-21"}
        )
        logger.info(f"Get availability response: {response.status_code}")
        logger.info(f"Get availability content: {response.text}")
        
    except Exception as e:
        logger.error(f"Error testing endpoints: {str(e)}")

async def verify_api_connection():
    """Verifica la conexión con la API probando varios endpoints"""
    client = HTTPClient()
    logger = logging.getLogger('dental_assistant')
    
    try:
        # Intentar primero con la ruta raíz
        root_url = Config.BASE_URL.replace('/api', '')
        logger.info(f"Checking API root at: {root_url}")
        
        response = await client.get(root_url)
        if response.status_code == 200:
            logger.info("API root check passed")
            
            # Ahora intentar con /api/ping
            ping_url = f"{Config.BASE_URL}/ping"
            logger.info(f"Checking API ping at: {ping_url}")
            
            try:
                response = await client.get(ping_url)
                if response.status_code == 200:
                    logger.info("API ping check passed")
                    return True
            except Exception as e:
                logger.warning(f"Ping check failed: {e}")
            
            # Si el ping falla pero la raíz funciona, asumimos que la API está funcionando
            return True
            
        logger.error(f"API root check failed: {response.status_code}")
        return False
    
    except Exception as e:
        logger.error(f"Error connecting to API: {str(e)}")
        return False

async def send_whatsapp_message(recipient_id: str, message: str):
    """Send message using WhatsApp Cloud API"""
    if not settings.PHONE_NUMBER_ID:
        raise ValueError("PHONE_NUMBER_ID is not set")
    
    url = f"https://graph.facebook.com/v17.0/{settings.PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN.strip()}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_id,
        "type": "text",
        "text": {"body": message}
    }
    
    try:
        logger.info(f"Sending WhatsApp message to {recipient_id}")
        logger.debug(f"Request details - URL: {url}")
        logger.debug(f"Headers (Auth token length): {len(str(settings.WHATSAPP_TOKEN))}")
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        logger.info(f"WhatsApp message sent successfully to {recipient_id}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending WhatsApp message: {str(e)}")
        logger.error(f"Response status code: {e.response.status_code if hasattr(e, 'response') else 'N/A'}")
        logger.error(f"Response content: {e.response.text if hasattr(e, 'response') else 'N/A'}")
        raise

async def handle_assistant_response(text: str, user_id: str):
    try:
        logger.info(f"Processing message for user {user_id}: {text}")
        
        # Get response from assistant
        response = await dental_assistant.handle_message(user_id, text)
        
        if response and response.get("type") == "text":
            response_text = response["content"]
            logger.info(f"Assistant response: {response_text}")
            
            # Explicitly check environment variables before sending
            if not settings.PHONE_NUMBER_ID or not settings.WHATSAPP_TOKEN:
                raise ValueError("Missing required WhatsApp configuration")
            
            # Send the response via WhatsApp
            await send_whatsapp_message(user_id, response_text)
            
            return response_text, None
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

if __name__ == "__main__":
    async def main():
        try:
            print("Inicializando asistente dental...")
            if await initialize_assistant():
                print("¡Hola! Soy tu asistente dental. ¿En qué puedo ayudarte?")
                print("(Escribe 'salir' para terminar)")
                test_user_id = "test_user_123"
                
                while True:
                    user_input = input("\nTú: ")
                    if user_input.lower() == "salir":
                        print("\n¡Hasta luego! Que tengas un excelente día.")
                        break
                    
                    message, error = await handle_assistant_response(user_input, test_user_id)
                    if error:
                        print(f"\nError: {error}")
                    else:
                        print(f"\nAsistente: {message}")
            else:
                print("Error: No se pudo inicializar el asistente")
        except Exception as e:
            print(f"Error: {str(e)}")
            logger.error(f"Error en main: {str(e)}", exc_info=True)

    # Configurar logging
    logger = setup_logging()
    
    # Ejecutar el programa
    asyncio.run(main())