from flask import Flask
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()  # Carga las variables del .env

public_ip = os.getenv("PUBLIC_IP")

origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://192.168.1.4:8000",
]
if public_ip:
    origins.append(public_ip)

app = Flask(__name__)
CORS(app, origins=origins, supports_credentials=True)

# Importa las rutas
from app import routes