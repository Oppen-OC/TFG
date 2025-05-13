import os
import pandas as pd
from tqdm import tqdm  # Importar tqdm para la barra de carga
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rag_operations import RAG


def generate_dataset(rag_object):
    """
    Genera un conjunto de datos basado en los chunks procesados por la clase RAG,
    iterando sobre todas las claves de rag_object.semantics.
    """
    data = []

    # Iterar sobre las claves de rag_object.semantics
    for key in tqdm(rag_object.semantics.keys(), desc="Procesando claves", unit="key"):
        rag_object.key = key  # Establecer la clave actual en el objeto RAG

        for indx, chunk in enumerate(tqdm(rag_object.chunks, desc=f"Generando dataset para clave {key}", unit="chunk", leave=False)):
            # Extraer características
            interesting_strings = rag_object.semantics[key]
            chunk_lower = chunk.lower()
            matches = sum(1 for string in interesting_strings if string in chunk_lower)
            normalized_matches = matches / len(interesting_strings) if interesting_strings else 0
            sections = rag_object.metadata[indx][1]
            intersection = len(set(sections) & set(rag_object.sections[key]))
            chunk_length = len(chunk)
            score = rag_object.reranking(0.5, indx)  # Usa un puntaje inicial arbitrario

            # Etiqueta (debes definir manualmente si el chunk es relevante o no)
            label = 1 if "relevante" in chunk.lower() else 0

            # Agregar fila al conjunto de datos
            data.append({
                "key": key,  # Agregar la clave como una columna
                "matches": matches,
                "normalized_matches": normalized_matches,
                "intersection": intersection,
                "chunk_length": chunk_length,
                "score": score,
                "label": label
            })

    # Convertir a un DataFrame de pandas
    dataset = pd.DataFrame(data)
    return dataset


def train_data():
    dataset = pd.read_csv("dataset.csv")
    X = dataset[["matches", "normalized_matches", "intersection", "chunk_length", "score"]]
    y = dataset["label"]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


def main():
    # Crear una instancia de la clase RAG
    #rag_object = RAG()

    # Generar el conjunto de datos
    #dataset = generate_dataset(rag_object)

    # Guardar el conjunto de datos en un archivo CSV
    #output_path = os.path.join(os.getcwd(), "dataset.csv")
    #dataset.to_csv(output_path, index=False)
    #print(f"Conjunto de datos guardado en: {output_path}")
    train_data()


if __name__ == "__main__":
    main()
