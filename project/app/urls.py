from django.urls import path
from . import views

app_name = 'app'

urlpatterns = [
    path('', views.home, name='home'),
    path('licitaciones/', views.licitaciones_view, name='licitaciones'),
    path('licitaciones/<path:cod>/', views.detalles_licitacion, name='detalles_licitacion'),    
    path('chatbot/', views.chatbot_view, name='chatbot'),    
    path('check-document/', views.check_document_view, name='check_document'),
    #path('estadisticas', "", name=''),
    #path('cargar_licitacion', "", name=''), 
    #path('help', "", name=''),
]