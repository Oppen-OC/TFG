from django.shortcuts import render, HttpResponse
from util.db_manager import DBManager  # Importa tu clase DBManager

# Create your views here.

def home(request):
    # Funcion que devuelve un string a modo de respuesta
    return HttpResponse("hello world")

def licitaciones_view(request):
    db = DBManager()
    db.openConnection()
    data = db.viewTable("Licitaciones")  # Obtén los datos de la tabla
    db.closeConnection()

    # Renderiza la plantilla y pasa los datos al contexto
    return render(request, 'licitaciones.html', {'licitaciones': data})