from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.db import models

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, label="Correo electrónico")

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    class Alerta(models.Model):
        user = models.ForeignKey(User, on_delete=models.CASCADE)
        nombre = models.CharField(max_length=255)
        activo = models.BooleanField(default=True)
        condiciones = models.TextField()
        ultima_notificacion = models.DateField(null=True, blank=True)

    def __str__(self):
        return self.nombre