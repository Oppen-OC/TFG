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
db_handler = logging.FileHandler(log_file_path, mode='w')
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

    def closeConnection(self):
        self.cursor.close()
        self.cnxn.close()
        self.logger.info("Desconexión a la base de datos completada.")

    def insertDb(self, table, values):
        try:
            query = f"INSERT INTO {table} {self.strings[table]} {self.strings[table + 'V']}"
            self.cursor.execute(query, values)
            self.cnxn.commit()
            self.logger.info(f"Se ha insertado correctamente los valores en la tabla {table}.")

        except Exception as e:
            self.logger.error(f"Error al insertar los valores en la tabla {table}: {e}")

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
            self.logger.info(f"Elementos de la tabla {table}:")
            for cont, row in enumerate(rows):
                self.logger.info(f"{cont}- {row}")
            self.logger.info(f"Consulta exitosa, fin de la tabla {table}.")
        except pyodbc.Error as e:
            self.logger.error(f"Error al consultar la tabla {table}: {e}")

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

    def searchTable(self, table, search_criteria):
        where_clause = " AND ".join([f"{col} = ?" for col in search_criteria.keys()])
        sql = f"SELECT * FROM {table} WHERE {where_clause}"
        try:
            self.cursor.execute(sql, tuple(search_criteria.values()))
            rows = self.cursor.fetchall()
            self.logger.info(f"Resultados de la búsqueda en la tabla {table}: {rows}")
            #for cont, row in enumerate(rows):
                # self.logger.info(f"{cont}- {row}")
            return rows
        except pyodbc.Error as e:
            self.logger.error(f"Error al buscar en la tabla {table}: {e}")
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


if __name__ == "__main__":   
    db = DBManager()
    db.openConnection()
    db.viewTable("Anexos")