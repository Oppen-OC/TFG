from django.shortcuts import render
from app.util.db_manager import DBManager
from django.shortcuts import redirect
from app.util.main import main as util_main
from django.http import JsonResponse
from app.util.chatbot import handle_user_input
from django.views.decorators.csrf import csrf_exempt
import json

cod_documento = ""

def home(request):
    # Funcion que devuelve un string a modo de respuesta
    return redirect('app:licitaciones')

@csrf_exempt  # Desactiva la protección CSRF para esta vista (solo para desarrollo; usa tokens CSRF en producción)
def chatbot_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')

        # Procesar el mensaje del usuario
        bot_response = handle_user_input(user_message, cod_documento)

        # Devolver la respuesta como JSON
        return JsonResponse({'response': bot_response})
    
    elif request.method == 'GET':
        # Renderizar la plantilla del chatbot para solicitudes GET
        return render(request, 'chatbot.html')
    
    # Si el método no es GET ni POST, devolver un error
    return JsonResponse({'error': 'Método no permitido'}, status=405)

@csrf_exempt
def check_document_view(request):
    global cod_documento  # Declarar que se usará la variable global

    if request.method == 'POST':
        data = json.loads(request.body)
        document_code = data.get('code', '')

        # Llamar a la función en chatbot.py para verificar el código
        from app.util.chatbot import verify_document_code
        exists = verify_document_code(document_code)

        if exists:
            cod_documento = document_code  # Actualizar la variable global

        return JsonResponse({'exists': exists})
    
    return JsonResponse({'error': 'Método no permitido'}, status=405)

def licitaciones_view(request):
    # Crear una instancia de DBManager
    db = DBManager()
    
    # Abrir la conexión a la base de datos
    db.openConnection()
    
    # Obtener los datos de la tabla "Licitaciones_test"
    data = db.viewTable("Licitaciones_test")
    
    # Cerrar la conexión a la base de datos
    db.closeConnection()
    
    # Procesar los datos para que coincidan con el formato esperado
    licitaciones = [
        {
            'COD': row[0].replace("/", "-"),  # Ajusta los índices según las columnas de tu tabla
            'Organo_Contratacion': row[2],
            'Estado_Licitacion': row[4],
            'Presupuesto_Base_Licitacion': row[7],
            'Valor_Estimado_Contrato': row[8],
            'Lugar_Ejecucion': row[5],
        }
        for row in data
    ]

    num_licitaciones = len(licitaciones)

    total_euros = 0.0
    for price in data:
        # Convertir el precio a un número y sumarlo al total
        try:
            price = float(price[7].replace("€", "").replace(".", "").replace(",", ".").replace(" Euros", ""))
            total_euros += price
        except ValueError:
            continue

    total_euros = "{:,.2f}".format(total_euros).replace(",", "X").replace(".", ",").replace("X", ".")

    # Renderizar la plantilla y pasar los datos al contexto
    return render(request, 'licitaciones.html', {'licitaciones': licitaciones, 'num_licitaciones': num_licitaciones, 'total_euros':total_euros })

def mostrar_anexo_i(cod):

    db = DBManager()
    db.openConnection()

    data = db.searchTable("AnexoI", {"COD": cod})
    print(data)
    if data == []:
        # Carga el anexo I a la base de datos
        util_main(cod) 
        data = db.searchTable("AnexoI", {"COD": cod})

    if not data:
        return JsonResponse({'error': 'Licitación no encontrada'}, status=404)

    db.closeConnection()

    anexoI = {
            "Regulacion_armonizada": data[1],	
            "Sometido_recurso_especial": data[2],
            "Tipo_Tramitacion": data[3],
            "Clasificación_obra": data[4],
            "Nomenclatura": data[5],
            "Plazo_ejecución": data[6],
            "Presupuesto_sinIVA": data[7],
            "Presupuesto_conIVA": data[8],
            "Importe_IVA": data[9],
            "Valor_estimado": data[10],
            "Revision_precios": data[11],
            "Garantia_definitiva": data[12],
            "Garantia_provisional": data[13],
            "Criterio_1": data[14],
            "Criterio_2": data[15],
            "Criterio_3": data[16],
            "Admisibilidad_variantes": data[17],
            "Clasificacion": data[18],
            "Grupo": data[19],
            "Subgrupo": data[20],
            "Categoria": data[24],
            "Criterio_Solvencia_economica": data[25],
            "Criterio_Solvencia_tecnica": data[26],
            "Condiciones_especiales": data[27],
            "Penalidad_incumplimiento": data[28],
            "Forma_pago": data[29],
            "Causas_modificacion": data[30],
            "Plazo_garantia": data[31]
    }

    return anexoI

def chatbot(request):   
    return True

def detalles_licitacion(request, cod):
    db = DBManager()
    db.openConnection()

    # Reemplazar "-" por "/" en el código, sino no funciona bien el url
    cod = cod.replace("-", "/")

    # Obtener los datos de la licitación
    data = db.searchTable("Licitaciones_test", {"COD": cod})

    # Verificar si se encontró la licitación
    if data is None:
        db.closeConnection()
        return render(request, '404.html', {})  # Renderizar una página 404 si no se encuentra

    # Procesar los datos de la licitación
    licitacion = {
        'COD': data[0],
        'ID_publicacion_TED': data[1],
        'Organo_Contratacion': data[2],
        'ID_Organo_contratacion': data[3],
        'Estado_Licitacion': data[4],
        'Objeto_contrato': data[5],
        'Financiacion_UE': data[6],
        'Presupuesto_base': data[7],
        'Valor_Estimado_Contrato': data[8],
        'Tipo_contrato': data[9],
        'Codigo_CPV': data[10],
        'Lugar_ejecucion': data[11],
        'Sistema_contratacion': data[12],
        'Procedimiento_contratacion': data[13],
        'Tipo_contratacion': data[14],
        'Metodo_presentacion_oferta': data[15],
        'Resultado': data[16],
        'Adjudicatario': data[17],
        'Numero_licitaciones_presentadas': data[18],
        'Importe_adjudicacion': data[19],
    }

    # Obtener los documentos relacionados desde la tabla "Documentos"
    documentos = db.searchTable("Documentos", {"COD": cod})
    anexo = db.searchTable("Anexos", {"COD": cod})
    db.closeConnection()

    if len(anexo) == 0: anexo = None

    print(documentos)
    # Procesar los documentos
    pliego_administrativo_url = documentos[1]
    anexo_url = documentos[2]
    titulo_anexo = documentos[3]
    anexoI = mostrar_anexo_i(cod)
    # Renderizar la plantilla con los datos
    return render(request, 'detalles_licitacion.html', {
        'licitacion': licitacion,
        'pliego_administrativo_url': pliego_administrativo_url,
        'anexo_url': anexo_url,
        'titulo_anexo': titulo_anexo,
        'anexo': anexo,
        'anexoI': anexoI,
    })