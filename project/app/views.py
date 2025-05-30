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

    db = DBManager()
    db.openConnection()
    data = db.viewTable("Licitaciones_test")
    db.closeConnection()
    
    # Procesar los datos para que coincidan con el formato esperado
    licitaciones = [
        {
            'COD': row[0].replace("/", "_"),
            'ID_Publicacion_TED': row[1],
            'Organo_Contratacion': row[2],
            'ID_Organo_Contratacion': row[3],
            'Estado_Licitacion': row[4],
            'Objeto_Contrato': row[5],
            'Financiacion_UE': row[6],
            'Presupuesto_Base_Licitacion': row[7],
            'Valor_Estimado_Contrato': row[8],
            'Tipo_Contrato': row[9],
            'Codigo_CPV': row[10],
            'Lugar_Ejecucion': row[11],
            'Sistema_Contratacion': row[12],
            'Procedimiento_Contratacion': row[13],
            'Tipo_Tramitacion': row[14],
            'Metodo_Presentacion_Oferta': row[15],
            'Resultado': row[16],
            'Adjudicatario': row[17],
            'Num_Licitadores_Presentados': row[18],
            'Importe_Adjudicacion': row[19],
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
    data1 = db.searchTable("anexoI_fuentes", {"COD": cod})

    #print(data)
    if data == []:
        # Carga el anexo I a la base de datos
        util_main(cod) 
        data = db.searchTable("AnexoI", {"COD": cod})
        data1 = db.searchTable("anexoI_fuentes", {"COD": cod})

    if not data:
        return JsonResponse({'error': 'Licitación no encontrada'}, status=404)

    db.closeConnection()

    anexoI = {
        "Regulacion_armonizada": data[1],
        "Sometido_recurso_especial": data[2],
        "Plazo_ejecución": data[3],
        "Presupuesto_sinIVA": data[4],
        "Presupuesto_conIVA": data[5],
        "Importe_IVA": data[6],
        "Sistema_Retribucion": data[7],
        "Valor_estimado": data[8],
        "Revision_precios": data[9],
        "Garantia_definitiva": data[10],
        "Garantia_provisional": data[11],
        "Admisibilidad_variantes": data[12],
        "Clasificacion": data[13],
        "Grupo": data[14],
        "Subgrupo": data[15],
        "Categoria": data[16],
        "Criterio_Solvencia_economica": data[17],
        "Criterio_Solvencia_tecnica": data[18],
        "Condiciones_especiales": data[19],
        "Penalidad_incumplimiento": data[20],
        "Forma_pago": data[21],
        "Causas_modificacion": data[22],
        "Plazo_garantia": data[23],
        "Contratacion_precio_unit": data[24],
        "Tanto_alzado": data[25],
        "Tanto_alzado_con": data[26],
        "Criterio_juicio_val": data[27],
        "Evaluables_autom": data[28],
        "Criterio_ecónomico": data[29],
        "Obligaciones_ambientales": data[30],
        "Regimen_penalidades": data[31],
    }    
    anexoI_fuentes = {
        "Regulacion_armonizada": data1[1],
        "Sometido_recurso_especial": data1[2],
        "Plazo_ejecución": data1[3],
        "Presupuesto_sinIVA": data1[4],
        "Presupuesto_conIVA": data1[5],
        "Importe_IVA": data1[6],
        "Sistema_Retribucion": data1[7],
        "Valor_estimado": data1[8],
        "Revision_precios": data1[9],
        "Garantia_definitiva": data1[10],
        "Garantia_provisional": data1[11],
        "Admisibilidad_variantes": data1[12],
        "Clasificacion": data1[13],
        "Grupo": data1[14],
        "Subgrupo": data1[15],
        "Categoria": data1[16],
        "Criterio_Solvencia_economica": data1[17],
        "Criterio_Solvencia_tecnica": data1[18],
        "Condiciones_especiales": data1[19],
        "Penalidad_incumplimiento": data1[20],
        "Forma_pago": data1[21],
        "Causas_modificacion": data1[22],
        "Plazo_garantia": data1[23],
        "Contratacion_precio_unit": data1[24],
        "Tanto_alzado": data1[25],
        "Tanto_alzado_con": data1[26],
        "Criterio_juicio_val": data1[27],
        "Evaluables_autom": data1[28],
        "Criterio_ecónomico": data1[29],
        "Obligaciones_ambientales": data1[30],
        "Regimen_penalidades": data1[31],
    }

    return anexoI, anexoI_fuentes

def chatbot(request):   
    return True

def detalles_licitacion(request, cod):
    db = DBManager()
    db.openConnection()

    # Reemplazar "-" por "/" en el código, sino no funciona bien el url
    cod = cod.replace("-", "/")
    cod = cod.replace("%", "/")
    cod = cod.replace(" ", "/")
    cod = cod.replace("_", "/")
    print(cod)
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

    # Procesar los documentos
    pliego_administrativo_url = documentos[1]
    anexo_url = documentos[2]
    titulo_anexo = documentos[3]
    anexoI, anexoI_fuentes = mostrar_anexo_i(cod)

    context = {
        'licitacion': licitacion,
        'pliego_administrativo_url': pliego_administrativo_url,
        'anexo_url': anexo_url,
        'titulo_anexo': titulo_anexo,
        'anexo': anexo,
        'anexoI': anexoI,
        'anexoI_fuentes': json.dumps(anexoI_fuentes or {}),
    }
    return render(request, 'detalles_licitacion.html', context)