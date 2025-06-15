from django.urls import path
from . import views

app_name = 'app'

urlpatterns = [
    path('', views.home, name='home'),
    path('licitaciones/', views.licitaciones_view, name='licitaciones'),
    path('licitaciones/<path:cod>/', views.detalles_licitacion, name='detalles_licitacion'),    
    path('chatbot/', views.chatbot_view, name='chatbot'),    
    path('check-document/', views.check_document_view, name='check_document'),
    path('user_settings/', views.user_settings, name='user_settings'),
    path('alertas/', views.alertas, name='alertas'),
    path('alertas/ajustar/<int:filtro_id>/', views.ajustar_filtro, name='ajustar_filtro'),
    path('crear-alerta/', views.crear_alerta, name='crear_alerta'),
    path('mis-licitaciones/', views.mis_licitaciones, name='mis_licitaciones'),
    path('update', views.update_database, name='update_database'),
    path('estadisticas', views.estadisticas, name='estadisticas'),
]