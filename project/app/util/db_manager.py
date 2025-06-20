import pyodbc
import json
import logging
import csv
import os

cnxn_str_base = (
    r"DRIVER={ODBC Driver 17 for SQL Server};"
    r"SERVER=TATIOLD;"
    r"DATABASE=TFG;"
    r"Trusted_Connection=yes;"
)

# Construct the absolute path to Strings.json
base_dir = os.path.dirname(os.path.abspath(__file__))
strings_path = os.path.join(base_dir, 'JSON/Strings.json')

with open(strings_path, 'r', encoding='utf-8') as f:
    strings = json.load(f)
    strings_base = strings["dataBase"]

# Configuración del logger específico para db_manager
# Define the logs directory path
logs_dir = os.path.join(base_dir, 'logs')

# Create the logs directory if it doesn't exist
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Define the log file path inside the logs directory
log_file_path = os.path.join(logs_dir, 'db_manager.log')

db_logger = logging.getLogger('db_manager')
db_logger.setLevel(logging.INFO)

# Configure the file handler to write logs to the logs directory
db_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')  # Cambia la codificación a utf-8
db_handler.setLevel(logging.INFO)
db_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
db_handler.setFormatter(db_formatter)
db_logger.addHandler(db_handler)

class DBManager:
    def __init__(self, cnxn_str=cnxn_str_base, strings=strings_base):
        
        self.cnxn_str = cnxn_str
        self.strings = strings
        self.cnxn = None
        self.cursor = None

        self.logger = db_logger

    def openConnection(self):
        self.cnxn = pyodbc.connect(self.cnxn_str)
        self.cursor = self.cnxn.cursor()
        self.logger.info("Conectado a la base de datos exitosamente.")

    def ensure_connection(self):
        if self.cnxn is None or self.cursor is None:
            self.openConnection()

    def reset_cursor(self):
        self.cursor.close()
        self.cursor = self.cnxn.cursor()

    def closeConnection(self):
        self.cursor.close()
        self.cnxn.close()
        self.logger.info("Desconexión a la base de datos completada.")

    def insertDb(self, table, values):
        query = None  # <-- Añade esto
        if not all(value is not None for value in values):
            self.logger.error(f"Invalid data: {values}")
            return

        try:
            query = f"INSERT INTO {table} {self.strings[table]} {self.strings[table + 'V']}"
            self.cursor.execute(query, values)
            self.cnxn.commit()
            self.logger.info(f"Se ha insertado correctamente los valores en la tabla {table}.")

        except Exception as e:
            self.logger.error(f"Error inserting values into table {table}: {e}")
            self.logger.error(f"Query: {query}")
            self.logger.error(f"Values: {values}")

    def deleteObject(self, table, pk_name, pk_value):
        sql = f"DELETE FROM {table} WHERE {pk_name} = ?"
        try:
            self.cursor.execute(sql, (pk_value,))
            self.cnxn.commit()
            self.logger.info(f"Eliminado correctamente el elemento {pk_value} de la tabla {table}.")
        except pyodbc.Error as e:
            self.logger.error(f"Error al eliminar el elemento {pk_value} de la tabla {table}: {e}")

    def viewTable(self, table):
        query = f"SELECT * FROM {table}"
        self.cursor.execute(query)
        try:
            rows = self.cursor.fetchall()
            self.logger.info(f"{len(rows)} Elementos recuperados:")
            self.logger.info(f"Consulta exitosa, fin de la tabla {table}.")
            return rows  # Devuelve los datos obtenidos
        except pyodbc.Error as e:
            self.logger.error(f"Error al consultar la tabla {table}: {e}")
            return None  # Devuelve None en caso de error

    def getFullTable(self, table_name):
        """
        Devuelve todos los registros de una tabla.
        :param table_name: Nombre de la tabla.
        :return: Lista de registros de la tabla.
        """
        try:
            # Asegurarse de que la conexión esté activa
            self.ensure_connection()

            query = f"SELECT * FROM {table_name}"
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            self.logger.info(f"Se han recuperado {len(rows)} registros de la tabla {table_name}.")
            return rows
        except pyodbc.Error as e:
            self.logger.error(f"Error al obtener los registros de la tabla {table_name}: {e}")
            return []

    def updateObject(self, table, pk_name, pk_value, update_values):
        set_clause = ", ".join([f"{col} = ?" for col in update_values.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {pk_name} = ?"
        try:
            self.cursor.execute(sql, (*update_values.values(), pk_value))
            if self.cursor.rowcount == 0:
                self.logger.error(f"No se encontró el elemento {pk_value} en la tabla {table}.")
            else:
                self.cnxn.commit()
                self.logger.info(f"Actualizado correctamente el elemento {pk_value} de la tabla {table}.")
        except pyodbc.Error as e:
            self.logger.error(f"Error al actualizar el elemento {pk_value} de la tabla {table}: {e}")

    def searchTable(self, table, search_criteria) -> list:
        where_clause = " AND ".join([f"{col} = ?" for col in search_criteria.keys()])
        sql = f"SELECT * FROM {table} WHERE {where_clause}"
        try:
            self.cursor.execute(sql, tuple(search_criteria.values()))
            rows = self.cursor.fetchall()[0]
            self.logger.info(f"Resultados de la búsqueda en la tabla {table}: {rows}")
            #for cont, row in enumerate(rows):
                # self.logger.info(f"{cont}- {row}")
            return list(rows)
        except pyodbc.Error as e:
            self.logger.error(f"Error al buscar en la tabla {table}: {e}")
            return []
        
        except Exception as e:
            self.logger.warning(f"No se encontraron resultados en la tabla {table} para los criterios: {search_criteria}")
            return []
    """
    def searchAllTable(self, table, search_criteria) -> list:
        where_clause = " AND ".join([f"{col} = ?" for col in search_criteria.keys()])
        sql = f"SELECT * FROM {table} WHERE {where_clause}"
        try:
            self.cursor.execute(sql, tuple(search_criteria.values()))
            rows = self.cursor.fetchall()
            self.logger.info(f"Resultados de la búsqueda en la tabla {table}: {rows}")
            return [list(row) for row in rows]
        except pyodbc.Error as e:
            self.logger.error(f"Error al buscar en la tabla {table}: {e}")
            return []
        except Exception as e:
            self.logger.warning(f"No se encontraron resultados en la tabla {table} para los criterios: {search_criteria}")
            return []
    """
    def searchAllTable(self, table, search_criteria, columns="*") -> list:
        where_clause = " AND ".join([f"{col} = ?" for col in search_criteria.keys()])
        sql = f"SELECT {columns} FROM {table} WHERE {where_clause}"
        try:
            self.cursor.execute(sql, tuple(search_criteria.values()))
            rows = self.cursor.fetchall()
            self.logger.info(f"Resultados de la búsqueda en la tabla {table}: {rows}")
            return [list(row) for row in rows]
        except pyodbc.Error as e:
            self.logger.error(f"Error al buscar en la tabla {table}: {e}")
            return []
        except Exception as e:
            self.logger.warning(f"No se encontraron resultados en la tabla {table} para los criterios: {search_criteria}")
            return []

    def exportToCSV(self, table, filename):
        query = f"SELECT * FROM {table}"
        self.cursor.execute(query)
        try:
            rows = self.cursor.fetchall()
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([desc[0] for desc in self.cursor.description])  # Write headers
                csvwriter.writerows(rows)
            self.logger.info(f"Datos de la tabla {table} exportados correctamente a {filename}.")
        except pyodbc.Error as e:
            self.logger.error(f"Error al exportar los datos de la tabla {table} a {filename}: {e}")
        
    def fetchData(self, table, n):
        query = f"""
            WITH CTE AS (
                SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS rn 
                FROM {table}
            )
            SELECT * FROM CTE WHERE rn = ?
        """
        
        try:
            self.cursor.execute(query, (n,))  # Use n instead of i
            row = self.cursor.fetchone()
            
            if row:
                self.logger.info(f"Datos obtenidos de la fila en la posición {n} de la tabla {table}. Con PK = {row[0]}")
                return row  # Return the row at position n
            else:
                self.logger.warning(f"Índice {n} fuera de rango para la tabla {table}.")
                return None
        
        except pyodbc.Error as e:
            self.logger.error(f"Error al obtener datos de la tabla {table}: {str(e)}")
            return None
        except (TypeError, ValueError) as e:
            self.logger.error(f"Error de tipo o valor: {str(e)}")
            return None
        
    def truncateTable(self, table):
        sql = f"TRUNCATE TABLE {table}"
        try:
            self.cursor.execute(sql)
            self.cnxn.commit()
            self.logger.info(f"Todos los datos de la tabla {table} han sido eliminados.")
        except pyodbc.Error as e:
            self.logger.error(f"Error al truncar la tabla {table}: {e}")

    def createTable(self, table_name, columns_definition):
        sql = f"CREATE TABLE {table_name} ({columns_definition})"
        try:
            self.cursor.execute(sql)
            self.cnxn.commit()
            self.logger.info(f"Tabla {table_name} creada correctamente.")
        except pyodbc.Error as e:
            self.logger.error(f"Error al crear la tabla {table_name}: {e}")

    def dropTable(self, table_name):
        sql = f"DROP TABLE {table_name}"
        try:
            self.cursor.execute(sql)
            self.cnxn.commit()
            self.logger.info(f"Tabla {table_name} eliminada correctamente.")
        except pyodbc.Error as e:
            self.logger.error(f"Error al eliminar la tabla {table_name}: {e}")

    def getTableSize(self, table_name):
        sql = f"SELECT COUNT(*) FROM {table_name}"
        try:
            self.cursor.execute(sql)
            size = self.cursor.fetchone()[0]
            self.logger.info(f"El tamaño de la tabla {table_name} es {size} filas.")
            return size
        except pyodbc.Error as e:
            self.logger.error(f"Error al obtener el tamaño de la tabla {table_name}: {e}")
            return None
        
    def comp(self):
        for n in range(0, 8000):
            cod = self.fetchData("Codigos", n)
            exp = self.fetchData("Licitaciones", n)
            if cod[0] != exp[0]:
                print(f"Error in {n}: {cod[0]} | {exp[0]}")

    def deleteLog(self):
        try:
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
                self.logger.info("El archivo de log ha sido eliminado manualmente.")
            else:
                print("El archivo de log no existe.")
        except Exception as e:
            print(f"Error al intentar eliminar el archivo de log: {e}")

    def intersection(self, table1, table2):
        sql = f"""
            SELECT * FROM {table1}
            INTERSECT
            SELECT * FROM {table2}
        """
        try:
            self.cursor.execute(sql)
            rows = self.cursor.fetchall()
            self.logger.info(f"Intersección entre {table1} y {table2}: {rows}")
            return rows
        except pyodbc.Error as e:
            self.logger.error(f"Error al obtener la intersección entre {table1} y {table2}: {e}")
            return []
        
    def fetchMaxValue(self, table_name, column_name):
        query = f"SELECT MAX({column_name}) FROM {table_name}"
        self.cursor.execute(query)
        result = self.cursor.fetchone()
        return result[0] if result else None

    def insertNewUpdate(self):
        try:
            # Buscar el valor máximo de la columna
            query_max = f"SELECT MAX(id) FROM actualizaciones"
            self.cursor.execute(query_max)
            max_value = self.cursor.fetchone()[0]

            # Calcular el nuevo valor de la clave primaria
            new_pk = max_value + 1 if max_value is not None else 1

            # Obtener la fecha y hora actual
            from datetime import datetime
            current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Insertar el nuevo registro
            query_insert = f"INSERT INTO actualizaciones (id, fecha_actualizacion) VALUES (?, ?)"
            self.cursor.execute(query_insert, (new_pk, current_datetime))
            self.cnxn.commit()

            self.logger.info(f"Nuevo registro insertado en actualizaciones: id={new_pk}, fecha_actualizacion={current_datetime}")
            return new_pk, current_datetime

        except Exception as e:
            self.logger.error(f"Error al insertar un nuevo registro en actualizaciones: {e}")
            return None, None

    def complement_of_intersection(self, table1, table2, pk_column="COD"):
        sql = f"""
            SELECT {table1}.{pk_column} FROM {table1}
            LEFT JOIN {table2}
            ON TRIM(UPPER({table1}.{pk_column})) = TRIM(UPPER({table2}.{pk_column}))
            WHERE {table2}.{pk_column} IS NULL
        """
        try:
            self.cursor.execute(sql)
            rows = self.cursor.fetchall()
            self.logger.info(f"Complemento de la intersección entre las PK de {table1} y {table2}: {rows}")
            return [row[0] for row in rows]
        except Exception as e:
            self.logger.error(f"Error al obtener el complemento de la intersección entre las PK de {table1} y {table2}: {e}")
            return []


if __name__ == "__main__":   
    db = DBManager()
    db.openConnection()
    db.viewTable("Licitaciones_test")