from flask import Flask, jsonify, request
from flask_cors import CORS
from app import app
from app.util.db_manager import DBManager
from app.util.scrapper import update
from app.util.main import main as util_main
import sqlite3  # Cambia esto según tu base de datos

# Habilitar CORS para toda la aplicación
CORS(app)

@app.route('/update-licitaciones', methods=['POST'])
def update_licitaciones():
    try:
        update()  # Llama a la función update
        return jsonify({'success': True, 'message': 'Licitaciones actualizadas correctamente.'}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/cargar-anexo', methods=['POST'])
def cargar_anexo():
    cod = request.json.get('cod')  # Obtener el código enviado desde el cliente
    print("Código enviado:", cod)
    if not cod:
        return jsonify({'error': 'Código no proporcionado'}), 400

    # Llamar a la función mostrar_anexo_i
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
        return jsonify({'error': 'Licitación no encontrada'}), 404

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

    return jsonify({
        'anexoI': anexoI,
        'anexoI_fuentes': anexoI_fuentes
    })

@app.route('/get-anexoI-data', methods=['POST'])
def get_anexoI_data():
    print("Endpoint /get-anexoI-data fue llamado")
    try:
        # Obtener la lista de códigos desde el cuerpo de la solicitud
        cod_list = request.json.get('cod_list', [])
        if not cod_list:
            return jsonify({'error': 'No se proporcionó una lista de códigos'}), 400

        # Crear una instancia de DBManager y abrir la conexión
        db = DBManager()
        db.openConnection()

        # Consultar la tabla "anexoI" para los códigos proporcionados
        data = []
        for cod in cod_list:
            rows = db.searchTable("anexoI", {"COD": cod})  # Filtrar por cada código
            if rows:
                # Obtener los nombres de las columnas
                column_names = db.getColumnNames("anexoI")
                # Convertir los datos a una lista de diccionarios y agregarlos a la lista
                data.extend([dict(zip(column_names, row)) for row in rows])

        # Cerrar la conexión
        db.closeConnection()

        # Verificar si se encontraron datos
        if not data:
            return jsonify({'error': 'No se encontraron datos para los códigos proporcionados'}), 404

        return jsonify(data)
    except Exception as e:
        print(f"Error al obtener los datos de anexoI: {e}")
        return jsonify({'error': str(e)}), 500

