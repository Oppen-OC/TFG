import unittest
import sys
import os

# Add the directory containing db_manager.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db_manager import DBManager

class TestDBManager(unittest.TestCase):

    def setUp(self):
        self.db = DBManager()
        self.db.openConnection()
        # Crear una tabla temporal de prueba
        columns_definition = """
            ID VARCHAR(100) PRIMARY KEY,
            URL VARCHAR(255)
        """
        self.table_name = "#TempTable"
        self.db.createTable(self.table_name, columns_definition)
        # Add structure for the temporary table
        self.db.strings[self.table_name] = "(ID, URL)"
        self.db.strings[self.table_name + 'V'] = "VALUES (?, ?)"

    def tearDown(self):
        # No es necesario borrar la tabla temporal explícitamente, se elimina automáticamente al cerrar la conexión
        self.db.closeConnection()

    def test_insertDb(self):
        valores = ('TestID', 'TestURL')
        self.db.insertDb(self.table_name, valores)
        result = self.db.searchTable(self.table_name, {"ID": "TestID"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 'TestID')
        self.assertEqual(result[0][1], 'TestURL')

    def test_updateObject(self):
        valores = ('TestID', 'TestURL')
        self.db.insertDb(self.table_name, valores)
        update_values = {"URL": "UpdatedURL"}
        self.db.updateObject(self.table_name, "ID", "TestID", update_values)
        result = self.db.searchTable(self.table_name, {"ID": "TestID"})
        self.assertEqual(result[0][1], 'UpdatedURL')

    def test_deleteObject(self):
        valores = ('TestID', 'TestURL')
        self.db.insertDb(self.table_name, valores)
        self.db.deleteObject(self.table_name, "ID", "TestID")
        result = self.db.searchTable(self.table_name, {"ID": "TestID"})
        self.assertEqual(len(result), 0)

    def test_exportToCSV(self):
        valores = ('TestID', 'TestURL')
        self.db.insertDb(self.table_name, valores)
        self.db.exportToCSV(self.table_name, "temp_table_test.csv")
        with open("temp_table_test.csv", 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn("TestID", content)
        self.assertIn("TestURL", content)

    def test_truncateTable(self):
        valores = ('TestID', 'TestURL')
        self.db.insertDb(self.table_name, valores)
        self.db.truncateTable(self.table_name)
        result = self.db.searchTable(self.table_name, {"ID": "TestID"})
        self.assertEqual(len(result), 0)

if __name__ == "__main__":
    unittest.main()