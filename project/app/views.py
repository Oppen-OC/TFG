from django.shortcuts import render, HttpResponse
from app.util.db_manager import DBManager
from django.urls import reverse
from django.shortcuts import redirect


def home(request):
    # Funcion que devuelve un string a modo de respuesta
    return redirect('app:licitaciones')

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

def detalles_licitacion(request, cod):
    db = DBManager()
    db.openConnection()

    # Reemplazar "-" por "/" en el código
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
    # Renderizar la plantilla con los datos
    return render(request, 'detalles_licitacion.html', {
        'licitacion': licitacion,
        'pliego_administrativo_url': pliego_administrativo_url,
        'anexo_url': anexo_url,
        'titulo_anexo': titulo_anexo,
        'anexo': anexo,
    })