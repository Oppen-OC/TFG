from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path('licitaciones/', views.licitaciones_view, name='licitaciones'),
]