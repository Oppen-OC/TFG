from django.urls import path
from . import views

app_name = 'app'

urlpatterns = [
    path('', views.home, name='home'),
    path('licitaciones/', views.licitaciones_view, name='licitaciones'),
    path('todos/', views.todos, name='todos'), 
]