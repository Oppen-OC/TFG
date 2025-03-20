from django.shortcuts import render, HttpResponse

# Create your views here.

def home(request):
    # Funcion que devuelve un string a modo de respuesta
    return HttpResponse("hello world")