from django.apps import AppConfig

class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'  # Este nombre debe coincidir con el nombre de la carpeta de la aplicación

    def ready(self):
        from .periodic import start_periodic_task
        start_periodic_task()
