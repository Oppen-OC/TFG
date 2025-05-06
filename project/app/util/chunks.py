import json

class Chunk:
    def __init__(self, json_path):
        self.page = None            # Número de la página extraída
        self.chunks = []            # Chunks de texto, pueden contener info de la siguiente página
        self.tables = []            # Tablas extraídas de la página
        self.sections = []          # Secciones contenidas en la página
        self.json_path = json_path  # Ruta al archivo JSON

    def get_chunk(self):
        return self.chunks

    def get_tables(self):
        return self.tables    
    
    def get_sections(self):
        return self.sections
    
    def found_in_chunk(self, string):
        ans = []
        for chunk in self.chunks:
            if string in chunk:
                ans.append(chunk)
        return ans
    
    def load_page(self, n):
        """
        Carga los datos del elemento n del archivo JSON y los asigna a los atributos de la clase.
        
        :param json_path: Ruta al archivo JSON.
        :param n: Índice del elemento a cargar.
        """
        try:
            with open(self.json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if n < len(data):
                    element = data[n]
                    self.page = element.get("page", None)
                    self.chunks = element.get("chunks", [])
                    self.tables = element.get("tables", [])
                    self.sections = element.get("sections", [])
                else:
                    raise IndexError(f"El índice {n} está fuera del rango del archivo JSON.")
                
        except FileNotFoundError:
            print(f"El archivo {self.json_path} no existe.")
        except json.JSONDecodeError:
            print(f"Error al decodificar el archivo JSON: {self.json_path}.")
        except Exception as e:
            print(f"Error al cargar el archivo JSON: {e}")


