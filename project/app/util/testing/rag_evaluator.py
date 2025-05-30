import os
import json
import sys
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_operations import RAG



class RAGEvaluator:
    def __init__(self, output_path, test_data_path):
        with open(output_path, 'r', encoding='utf-8') as file:
            self.responses_file = json.load(file)
        self.test_data_path = test_data_path
        self.test_data = self.load_test_data()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo para embeddings

    def load_test_data(self):
        with open(self.test_data_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def evaluate(self):
        y_true = []  # Respuestas esperadas
        y_pred = []  # Respuestas generadas por RAG
        context_similarities = []  # Similitud de contexto

        # Crear un diccionario para acceder rápido por prompt_key
        pred_dict = {item["prompt_key"]: item.get("response", "") for item in self.responses_file["prompts"]}

        for item in self.test_data["prompts"]:
            key = item["prompt_key"]
            expected_response = item.get("response", "")
            generated_response = pred_dict.get(key, "")

            y_true.append(expected_response)
            y_pred.append(generated_response)

            # Evaluar similitud de contexto
            if expected_response and generated_response:
                embedding_true = self.model.encode(expected_response, convert_to_tensor=True)
                embedding_pred = self.model.encode(generated_response, convert_to_tensor=True)
                similarity = util.cos_sim(embedding_true, embedding_pred).item()
                context_similarities.append(similarity)

        # Calcular métricas
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        avg_context_similarity = sum(context_similarities) / len(context_similarities) if context_similarities else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_context_similarity": avg_context_similarity
        }

def compare_strings(str1, str2, model=None):
    """
    Compara dos strings y devuelve precisión, recall, f1-score y similitud de contexto.
    """
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    y_true = [str1]
    y_pred = [str2]

    # Métricas exactas
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Similitud semántica
    if str1 and str2:
        embedding_true = model.encode(str1, convert_to_tensor=True)
        embedding_pred = model.encode(str2, convert_to_tensor=True)
        similarity = util.cos_sim(embedding_true, embedding_pred).item()
    else:
        similarity = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_context_similarity": similarity
    }

def main():
    file_name = "CMAYOR/2023/03Y05/19.json"
    file_name = file_name.replace("/", "_")
    # Construir la ruta al archivo JSON de prueba
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_path = os.path.join(base_dir, "JSON", file_name)
    output_path = os.path.join(base_dir, "temp_files", file_name)

    # Crear el evaluador
    evaluator = RAGEvaluator(output_path, test_data_path)

    # Evaluar el rendimiento
    # metrics = evaluator.evaluate()
    metrics = compare_strings("El plazo de ejecución es de doce (12) meses", "12 meses")

    # Mostrar los resultados
    print("Resultados de la evaluación:")
    # Of all the answers your system generated, % were correct (matched the expected answer).
    print(f"Precisión: {metrics['precision']:.2f}")
    # Of all the correct answers that should have been generated, your system found %.
    print(f"Recall: {metrics['recall']:.2f}")
    # This is the harmonic mean of precision and recall, giving a balanced measure of overall accuracy.
    print(f"F1-Score: {metrics['f1_score']:.2f}")
    # On average, the semantic similarity between the generated and expected answers is x (on a scale from 0 to 1), meaning the answers are moderately similar in meaning, even if not always exactly the same
    print(f"Similitud promedio de contexto: {metrics['avg_context_similarity']:.2f}")

if __name__ == "__main__":
    main()