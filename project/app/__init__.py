from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:8000"], supports_credentials=True)

# Importa las rutas
from app import routes