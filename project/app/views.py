from django.shortcuts import render, HttpResponse
from app.util.db_manager import DBManager
from .models import TodoItem
# Create your views here.

def home(request):
    # Funcion que devuelve un string a modo de respuesta
    return render(request, 'base.html')

def licitaciones_view(request):
    db = DBManager()
    db.openConnection()
    data = db.viewTable("Licitaciones_test")  # Obtén los datos de la tabla
    db.closeConnection()

    # Renderiza la plantilla y pasa los datos al contexto
    return render(request, 'licitaciones.html', {'licitaciones': data})

def todos(request):
    items = TodoItem.objects.all()  # Obtiene todos los elementos de la base de datos
    return render(request, 'todos.html', {'todos': items})  # Renderiza la plantilla y pasa los elementos al contexto