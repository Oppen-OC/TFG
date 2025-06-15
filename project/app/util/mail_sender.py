import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
# Cargar las variables de entorno desde el archivo .env
load_dotenv()

def enviar_correo_licitacion(codigo_licitacion, destinatario, filtro):
    # Configura tus credenciales de Gmail
    remitente = os.getenv("GMAIL")
    password = os.getenv("CONTRASEÑAGMAIL")

    # Crea el mensaje
    asunto = f'Información sobre la licitación {codigo_licitacion}'
    cuerpo = f'Hola, \n\n Hemos encontrado una licitación que debería interesarte según tu filtro {filtro} .\n\nEl código de la licitación es: {codigo_licitacion}\n\nSaludos.'
    mensaje = MIMEMultipart()
    mensaje['From'] = remitente
    mensaje['To'] = destinatario
    mensaje['Subject'] = asunto
    mensaje.attach(MIMEText(cuerpo, 'plain'))

    try:
        # Conexión con el servidor SMTP de Gmail
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, password)
        servidor.sendmail(remitente, destinatario, mensaje.as_string())
        servidor.quit()
        print("Correo enviado correctamente.")
    except Exception as e:
        print(f"Error al enviar el correo: {e}")


if __name__ == "__main__":
    # Ejemplo de uso
    enviar_correo_licitacion('LIC12345', 'dantrox2015@gmail.com', 'Expedientes interesantes')