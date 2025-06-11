from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilita CORS para todos los orígenes

# Importa las rutas
from app import routes