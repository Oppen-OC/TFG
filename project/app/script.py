from models import Codigo
from util.db_manager import DBManager

db = DBManager()
db.openConnection()
data = db.viewTable("Codigos")  # Obtén los datos de la tabla Codigos
db.closeConnection()

for row in data:
    Codigo.objects.create(
        cod=row[0],  # Ajusta los índices según las columnas de tu tabla
        dir_url=row[1]
    )