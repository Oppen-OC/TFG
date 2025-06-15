from django.shortcuts import render
from flask import request
from app.util.db_manager import DBManager
from django.shortcuts import redirect
from app.util.main import main as util_main
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
from app.util.scrapper import update
import datetime
from django.shortcuts import render, redirect
from .forms import CustomUserCreationForm
from django.contrib.auth.decorators import login_required

cod_documento = ""

def home(request):
    # Funcion que devuelve un string a modo de respuesta
    return redirect('app:licitaciones')

@csrf_exempt  # Desactiva la protección CSRF para esta vista (solo para desarrollo; usa tokens CSRF en producción)
def chatbot_view(request):
    # Solo renderiza la plantilla, el procesamiento lo hace Flask
    return render(request, 'chatbot.html')

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
    data = db.viewTable("Licitaciones")
    
    # Procesar los datos para que coincidan con el formato esperado
    licitaciones = [
        {
            'COD': row[0],
            'ID_Publicacion_TED': row[1],
            'Organo_Contratacion': row[2],
            'ID_Organo_Contratacion': row[3],
            'Estado_Licitacion': row[4],
            'Objeto_Contrato': row[5],
            'Financiacion_UE': row[6],
            'Presupuesto_Base': row[7],
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
            'Numero_Licitaciones_Presentadas': row[18],
            'Importe_Adjudicacion': row[19],
            'Fecha_Adjudicacion': row[20],
            'Fecha_Formalizacion': row[21],
            'Fecha_Desistimiento': row[22],
            'Fecha_Presentacion_Oferta': row[23],
            'Fecha_Presentacion_Solicitud': row[24],
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
    rows = db.getFullTable("actualizaciones")
    last_update = max(rows, key=lambda x: x[0])[1]

    # Procesar los datos de AnexoI
    for licitacion in licitaciones:
        cod = licitacion["COD"]
        data = db.searchTable("AnexoI", {"COD": cod})
        if not data:
            # Si no hay datos, agregar claves con valores vacíos
            licitacion.update({
                "Regulacion_armonizada": "",
                "Sometido_recurso_especial": "",
                "Plazo_ejecución": "",
                "Presupuesto_sinIVA": "",
                "Presupuesto_conIVA": "",
                "Importe_IVA": "",
                "Sistema_Retribucion": "",
                "Valor_estimado": "",
                "Revision_precios": "",
                "Garantia_definitiva": "",
                "Garantia_provisional": "",
                "Admisibilidad_variantes": "",
                "Clasificacion": "",
                "Grupo": "",
                "Subgrupo": "",
                "Categoria": "",
                "Criterio_Solvencia_economica": "",
                "Criterio_Solvencia_tecnica": "",
                "Condiciones_especiales": "",
                "Penalidad_incumplimiento": "",
                "Forma_pago": "",
                "Causas_modificacion": "",
                "Plazo_garantia": "",
                "Contratacion_precio_unit": "",
                "Tanto_alzado": "",
                "Tanto_alzado_con": "",
                "Criterio_juicio_val": "",
                "Evaluables_autom": "",
                "Criterio_ecónomico": "",
                "Obligaciones_ambientales": "",
                "Regimen_penalidades": "",
                "Ambiental": "",
                "Tecnico": "",
                "Precio": "",
                "Garantia": "",
                "Experiencia": "",
                "Calidad": "",
                "Social": "",
                "Seguridad": "",
                "Juicio_valor": "",
                "Formula_precio": "",
                "Formula_JuicioValor": "",
                "Criterios_valores_anormales": "",
                "Infraccion_grave": "",
                "Gastos_desistimiento": "",
                "Contratacion_control": "",
                "Tareas_criticas": "",
            })
        else:
            # Si hay datos, agregar los valores correspondientes
            licitacion.update({
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
                "Ambiental": data[32],
                "Tecnico": data[33],
                "Precio": data[34],
                "Garantia": data[35],
                "Experiencia": data[36],
                "Calidad": data[37],
                "Social": data[38],
                "Seguridad": data[39],
                "Juicio_valor": data[40],
                "Formula_precio": data[41],
                "Formula_JuicioValor": data[42],
                "Criterios_valores_anormales": data[43],
                "Infraccion_grave": data[44],
                "Gastos_desistimiento": data[45],
                "Contratacion_control": data[46],
                "Tareas_criticas": data[47],
            })

    db.closeConnection()

    # Renderizar la plantilla y pasar los datos al contexto
    return render(request, 'licitaciones.html', {
        'licitaciones': licitaciones,
        'num_licitaciones': num_licitaciones,
        'total_euros': total_euros,
        "last_update": invertir_dia_mes(last_update),
    })

def invertir_dia_mes(fecha):
    if isinstance(fecha, datetime.datetime):
        # Convertir el objeto datetime a una cadena en formato "YYYY-MM-DD HH:MM:SS"
        fecha = fecha.strftime("%Y-%m-%d %H:%M:%S")
    
    fecha_partes = fecha.split(" ")
    fecha_solo = fecha_partes[0]
    anio, mes, dia = fecha_solo.split("-")
    nueva_fecha = f"{anio}-{dia}-{mes}"
    if len(fecha_partes) > 1:
        nueva_fecha += f" {fecha_partes[1]}"

    return nueva_fecha

from django.contrib.auth.decorators import login_required

@login_required
def user_settings(request):
    db = DBManager()
    db.openConnection()
    user = request.user.username

    # Busca los datos del usuario en la tabla usuarios
    datos = db.searchTable("auth_user", {"username": user})

    db.closeConnection()
    # Ajusta los índices según el orden de tu tabla usuarios
    context = {
        'username': datos[4],      # username
        'email': datos[7],         # email
        'password': datos[1],      # password (¡no recomendable mostrarla!)
    }

    return render(request, 'user_settings.html', context)

def mostrar_anexo_i(cod):

    db = DBManager()
    db.openConnection()

    data = db.searchTable("AnexoI", {"COD": cod.replace("-", "/")})
    data1 = db.searchTable("anexoI_fuentes", {"COD":  cod.replace("-", "/")})

    if data == []:
        # Carga el anexo I a la base de datos
        util_main(cod) 
        data = db.searchTable("AnexoI", {"COD": cod})
        data1 = db.searchTable("AnexoI_fuentes", {"COD": cod})

    if not data:
        return JsonResponse({'error': 'Licitación no encontrada'}, status=404)

    db.closeConnection()

    anexoI = {
        "COD": data[0],
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
        "Ambiental": data[32],
        "Tecnico": data[33],
        "Precio": data[34],
        "Garantia": data[35],
        "Experiencia": data[36],
        "Calidad": data[37],
        "Social": data[38],
        "Seguridad": data[39],
        "Juicio_valor": data[40],
        "Formula_precio": data[41],
        "Formula_JuicioValor": data[42],
        "Criterios_valores_anormales": data[43],
        "Infraccion_grave": data[44],
        "Gastos_desistimiento": data[45],
        "Contratacion_control": data[46],
        "Tareas_criticas": data[47]
    }    

    anexoI_fuentes = {
        "COD": data1[0],
        "Regulacion_armonizada": data1[1],
        "Sometido_recurso_especial": data[2],
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
        "Ambiental": data1[32],
        "Tecnico": data1[33],
        "Precio": data1[34],
        "Garantia": data1[35],
        "Experiencia": data1[36],
        "Calidad": data1[37],
        "Social": data1[38],
        "Seguridad": data1[39],
        "Juicio_valor": data1[40],
        "Formula_precio": data1[41],
        "Formula_JuicioValor": data1[42],
        "Criterios_valores_anormales": data1[43],
        "Infraccion_grave": data1[44],
        "Gastos_desistimiento": data1[45],
        "Contratacion_control": data1[46],
        "Tareas_criticas": data1[47]
    }

    return anexoI, anexoI_fuentes

def chatbot(request):   
    return True


def detalles_licitacion(request, cod):
    db = DBManager()
    db.openConnection()

    # Obtener los datos de la licitación
    data = db.searchTable("Licitaciones", {"COD": cod})

    # Verificar si se encontró la licitación
    if data is None:
        db.closeConnection()
        return render(request, '404.html', {})

    # Procesar los datos de la licitación (igual que antes)
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
        'Fecha_Adjudicacion': data[20],
        'Fecha_Formalizacion': data[21],
        'Fecha_Desistimiento': data[22],
        'Fecha_Presentacion_Oferta': data[23],
        'Fecha_Presentacion_Solicitud': data[24],
    }

    # Obtener los documentos relacionados desde la tabla "Documentos"
    documentos = db.searchTable("Documentos", {"COD": cod})
    anexo = db.searchTable("Anexos", {"COD": cod})
    data1 = db.searchTable("Anexoi_fuentes", {"COD": cod})
    url_original = db.searchTable("Codigos", {"COD": cod})[1]

    # --- NUEVO: Comprobar si el anexo ya está guardado para el usuario ---
    user = request.user.username
    existe = db.searchTable("anexos_guardados", {"[user]": user, "cod_licitacion": cod})
    anexo_guardado = bool(existe)
    # ---------------------------------------------------------------------

    db.closeConnection()

    if data1: 
        anexoI_fuentes = {
            "COD": data1[0],
            "Regulacion_armonizada": data1[1],
            "Sometido_recurso_especial": data[2],
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
            "Ambiental": data1[32],
            "Tecnico": data1[33],
            "Precio": data1[34],
            "Garantia": data1[35],
            "Experiencia": data1[36],
            "Calidad": data1[37],
            "Social": data1[38],
            "Seguridad": data1[39],
            "Juicio_valor": data1[40],
            "Formula_precio": data1[41],
            "Formula_JuicioValor": data1[42],
            "Criterios_valores_anormales": data1[43],
            "Infraccion_grave": data1[44],
            "Gastos_desistimiento": data1[45],
            "Contratacion_control": data1[46],
            "Tareas_criticas": data1[47]
        }
    else:
        anexoI_fuentes = {
            "COD": "",
            "Regulacion_armonizada": "",
            "Sometido_recurso_especial": "",
            "Plazo_ejecución": "",
            "Presupuesto_sinIVA": "",
            "Presupuesto_conIVA": "",
            "Importe_IVA": "",
            "Sistema_Retribucion": "",
            "Valor_estimado": "",
            "Revision_precios": "",
            "Garantia_definitiva": "",
            "Garantia_provisional": "",
            "Admisibilidad_variantes": "",
            "Clasificacion": "",
            "Grupo": "",
            "Subgrupo": "",
            "Categoria": "",
            "Criterio_Solvencia_economica": "",
            "Criterio_Solvencia_tecnica": "",
            "Condiciones_especiales": "",
            "Penalidad_incumplimiento": "",
            "Forma_pago": "",
            "Causas_modificacion": "",
            "Plazo_garantia": "",
            "Contratacion_precio_unit": "",
            "Tanto_alzado": "",
            "Tanto_alzado_con": "",
            "Criterio_juicio_val": "",
            "Evaluables_autom": "",
            "Criterio_ecónomico": "",
            "Obligaciones_ambientales": "",
            "Regimen_penalidades:": "",
            "Ambiental": "",
            "Tecnico": "",
            "Precio": "",
            "Garantia": "",
            "Experiencia": "",
            "Calidad": "",
            "Social": "",
            "Seguridad": "",
            "Juicio_valor": "",
            "Formula_precio": "",
            "Formula_JuicioValor": "",
            "Criterios_valores_anormales": "",
            "Infraccion_grave": "",
            "Gastos_desistimiento": "",
            "Contratacion_control": "",
            "Tareas_criticas": ""
        }
    if len(anexo) == 0: anexo = None

    # Procesar los documentos
    pliego_administrativo_url = documentos[2]
    anexo_url = documentos[1]
    titulo_anexo = documentos[3]
    # anexoI, anexoI_fuentes = mostrar_anexo_i(cod)

    contexto = {
        'licitacion': licitacion,
        'pliego_administrativo_url': pliego_administrativo_url,
        'anexo_url': anexo_url,
        'titulo_anexo': titulo_anexo,
        'pliego_original_url': url_original,
        'anexo': anexo,
        'anexoI_fuentes': anexoI_fuentes,
        'anexo_guardado': anexo_guardado,  # <-- Añade esto al contexto
    }
   
    return render(request, 'detalles_licitacion.html', contexto)


import json

@login_required
def alertas(request):
    db = DBManager()
    db.openConnection()
    user = request.user.username
    data = db.searchAllTable("alertas", {"[user]": user})
    db.closeConnection()

    alertas = []
    for row in data:
        try:
            condiciones = json.loads(row[4]) if row[4] else []
        except Exception:
            condiciones = []
        alerta = {
            'id': row[0],
            '[user]': row[1],
            'nombre': row[2],
            'activo': row[3],
            'condiciones': condiciones,
        }
        alertas.append(alerta)

    return render(request, 'alertas.html', {'alertas': alertas})

def ajustar_filtro(request, filtro_id):
    if request.method == 'POST':
        # Obtén los datos del formulario
        nombre = request.POST.get('nombre')
        frecuencia = request.POST.get('frecuencia')
        activo = request.POST.get('activo') == 'on'

        # Aquí puedes actualizar la base de datos con los nuevos valores
        # Por ejemplo:
        # alerta = Alerta.objects.get(id=filtro_id)
        # alerta.nombre = nombre
        # alerta.frecuencia = frecuencia
        # alerta.activo = activo
        # alerta.save()

        # Responde con los datos actualizados
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'id': filtro_id,
                'nombre': nombre,
                'frecuencia': frecuencia,
                'activo': activo,
            })

    # Si no es una solicitud AJAX, redirige a la lista de alertas
    return redirect('app:alertas')

@login_required
def mis_licitaciones(request):
    db = DBManager()
    db.openConnection()
    user = request.user.username

    # 1. Busca los códigos de licitaciones guardados por el usuario en anexos_guardados
    guardadas = db.searchAllTable("anexos_guardados", {"[user]": user})
    codigos = [row[1] for row in guardadas]  # row[1] es cod_licitacion

    licitaciones = []
    if codigos:
        # 2. Busca los datos de cada licitación guardada
        for cod in codigos:
            data = db.searchTable("Licitaciones", {"COD": cod})
            if data:
                licitaciones.append({
                    'COD': data[0],
                    'ID_Publicacion_TED': data[1],
                    'Organo_Contratacion': data[2],
                    'ID_Organo_Contratacion': data[3],
                    'Estado_Licitacion': data[4],
                    'Objeto_Contrato': data[5],
                    'Financiacion_UE': data[6],
                    'Presupuesto_Base_Licitacion': data[7],
                    'Valor_Estimado_Contrato': data[8],
                    'Tipo_Contrato': data[9],
                    'Codigo_CPV': data[10],
                    'Lugar_Ejecucion': data[11],
                    'Sistema_Contratacion': data[12],
                    'Procedimiento_Contratacion': data[13],
                    'Tipo_Tramitacion': data[14],
                    'Metodo_Presentacion_Oferta': data[15],
                    'Resultado': data[16],
                    'Adjudicatario': data[17],
                    'Num_Licitadores_Presentados': data[18],
                    'Importe_Adjudicacion': data[19],
                })
    db.closeConnection()

    return render(request, 'mis_licitaciones.html', {
        'licitaciones': licitaciones,
        'num_licitaciones': len(licitaciones),
    })

def update_database(request):
    if request.method == 'POST':
        try:
            # Llama a la función `update` en scrapper.py
            update()
            return JsonResponse({"message": "Base de datos actualizada correctamente."}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Método no permitido."}, status=405)
    

def estadisticas(request):
    db = DBManager()
    db.openConnection()
    data = db.viewTable("Licitaciones")
    db.closeConnection()
    # Suponiendo que Estado_Licitacion es la columna 5 (índice 4)
    estados = [row[4] for row in data]
    financiacion = [row[6] for row in data]
    presupuesto = [row[7] for row in data]
    valor_estimado = [row[8] for row in data]
    tipo_contrato = [row[9] for row in data]
    cpv = [row[10] for row in data]
    lugar = [row[11] for row in data]
    procedimiento = [row[13] for row in data]

    contexto = {'estados': estados, 
                'financiacion': financiacion, 
                'presupuesto': presupuesto, 
                'valor_estimado': valor_estimado,
                'tipo_contrato': tipo_contrato,
                'cpv': cpv,
                'lugar': lugar,
                'procedimiento': procedimiento
                }

    return render(request, 'estadisticas.html', contexto)


def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/register.html', {'form': form})