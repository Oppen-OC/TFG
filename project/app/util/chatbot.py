import os
import openai
from .db_manager import DBManager
from .document_handler import download_to_json
from .rag_operations import RAG
import logging

# Configuración del logger específico para el chatbot
chatbot_logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')  # Ruta al directorio de logs
chatbot_log_file_path = os.path.join(chatbot_logs_dir, 'chatbot.log')

# Crear el directorio de logs si no existe
os.makedirs(chatbot_logs_dir, exist_ok=True)
chatbot_logger = logging.getLogger("chatbot_logger")
chatbot_logger.setLevel(logging.INFO)
chatbot_handler = logging.FileHandler(chatbot_log_file_path, mode='w', encoding='utf-8')
chatbot_handler.setLevel(logging.INFO)
chatbot_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
chatbot_handler.setFormatter(chatbot_formatter)
chatbot_logger.addHandler(chatbot_handler)
chatbot_logger.propagate = False

first = True
data = []
rag_object = None


def chatbot_simple(user_input):
    client = openai.OpenAI(
        api_key=os.getenv("UPV_API_KEY"),
        base_url="https://api.poligpt.upv.es/v1"
    )

    messages = [{"role": "system", "content": "Eres un asistente útil."}]
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="llama3.3:70b",  # o el modelo que tengas disponible
            messages=messages
            )
        answer = response.choices[0].message.content
        print("Bot:", answer)
        messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        chatbot_logger.error("Error al contactar con la API:", e)
    
    return answer

def chatbot_with_context(user_input):     

    try:
        response = rag_object.QNA_RAG(user_input)
        print("Bot:", response)

    except Exception as e:
        chatbot_logger.error("Error al contactar con la API:", e)
    
    return response

def verify_document_code(code):
    global rag_object

    db = DBManager()
    db.openConnection()
    chatbot_logger.info("Buscando en la base de datos...")
    data = db.searchTable("Documentos", {"COD": code})
    db.closeConnection()

    if len(data) > 0:
        chatbot_logger.info("Cargando objeto RAG...")
        try:
            rag_object = RAG()
        except Exception as e:
            chatbot_logger.error("Error al cargar el objeto RAG:", e)

        if data[1]:
            # Esto es si contiene anexo I
            download_to_json(data[1])
        else:
            # Sino, va directamente al pliego administrativo
            download_to_json(data[2])
        
        return True

    return False

def handle_user_input(user_input, cod_documento):

    if cod_documento == "":
        chatbot_logger.info("Respondiendo pregunta simple")
        return chatbot_simple(user_input)

    else:
        chatbot_logger.info("Documento encontrado")
        return chatbot_with_context(user_input)

   

