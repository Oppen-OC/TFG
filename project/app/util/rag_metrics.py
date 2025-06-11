from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Tuple, Dict
import openai
import json
import os
from typing import List, Dict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import nltk

load_dotenv()

class RAGMetrics:
    def __init__(self, client, prompts: List[Dict]):
        self.client = client
        self.prompts = prompts
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_text(self, text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def calculate_response_score(self, generated_response: str, expected_response: str, prompt: str) -> Dict[str, float]:
        # Caso especial: si ambas respuestas están vacías, devolver métricas perfectas
        if generated_response.strip() == "" and expected_response.strip() == "":
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0
            }
        
        # Caso especial: si la respuesta generada contiene "si" y la esperada no contiene "no"
        if "si" in generated_response.lower() and "no" not in expected_response.lower():
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0
            }

        # Lematizar y tokenizar las respuestas y el prompt
        generated_tokens = self.lemmatize_text(generated_response)
        expected_tokens = self.lemmatize_text(expected_response)

        # Contar cuántos tokens de la respuesta generada coinciden con la respuesta esperada
        matching_tokens_in_expected = sum(1 for token in generated_tokens if token in expected_tokens)

        # Calcular precisión y recall
        precision = matching_tokens_in_expected / len(generated_tokens) if generated_tokens else 0
        recall = matching_tokens_in_expected / len(expected_tokens) if expected_tokens else 0

        # Calcular F1-Score
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

def calculate_metrics(generated_response:str, response: str, chunks: List[str]) -> dict:
    """
    Calcula varias métricas para evaluar la calidad de la respuesta generada en base a los chunks.

    :param response: Respuesta generada por el modelo.
    :param chunks: Lista de chunks devueltos por el sistema.
    :return: Diccionario con las métricas calculadas (ROUGE, Cosine Similarity).
    """

    # Manejar el caso en que la respuesta esté vacía
    if not response.strip():
        return {
            "rouge": {"rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                      "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                      "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}},
            "cosine_similarity": 0.0,
            "context_recall": 0.0
        }

    def calculate_rouge_score(response: str, generated_response: str) -> Dict[str, float]:
        rouge = Rouge()
        scores = rouge.get_scores(response, generated_response, avg=True)
        return scores

    def calculate_cosine_similarity(response: str, generated_response: str) -> float:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([response, generated_response])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def calculate_context_recall(response: str, generated_response) -> float:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # sims = [util.cos_sim(model.encode(response), model.encode(chunk))[0][0].item() for chunk in chunks]
        sims = util.cos_sim(model.encode(response), model.encode(generated_response))[0][0].item()
        return sims

    # Calcular las métricas
    #combined_chunks = " ".join(chunks).lower()  
    rouge_scores = calculate_rouge_score(response, generated_response)
    cosine_sim = calculate_cosine_similarity(response, generated_response)
    context_recall = calculate_context_recall(response, generated_response)


    # Devolver un diccionario con todas las métricas
    return {
        "rouge": rouge_scores,
        "cosine_similarity": cosine_sim,
        "context_recall": context_recall
    }

def simple_test():
    test = {
            "prompt_key": "Regulacion_armonizada",
            "prompt": "Indica si tiene regulación armonizada",
            "response": "No",
            "score": 0.8663049048858305,
            "index": [
                38,
                39,
                90,
                1
            ],
            "chunks": [
                "Direcció General de Ports, CIUTAT ADMINISTRATIVA 9 D'OCTUBRE – TORRE 1\nAeroports i Costes C/ Democràcia, 77 - 46018 VALÈNCIA - Tel. 012\nLa información que vaya a ser objeto de valoración en los criterios de adjudicación\ncuantificables mediante la mera aplicación de fórmulas matemáticas o aritméticas deberá\nincluirse EXCLUSIVAMENTE en el sobre Número3.\nLa documentación incluida en los sobres se presentará en formato pdf y firmada\nelectrónicamente.\nLa inclusión de información relativa a un sobre en otro diferente del que le corresponde según\nlos pliegos, supondrá la exclusión del proceso de licitación por acuerdo del Órgano de\nContratación a propuesta de la Mesa.\nCRITERIOS DE ADJUDICACIÓN DEL PROCEDIMIENTO ABIERTO\n Varios criterios\nSUJETO A REGULACIÓN ARMONIZADA\n No\nSUBASTA ELECTRÓNICA",
                "electrónicamente.\nLa inclusión de información relativa a un sobre en otro diferente del que le corresponde según\nlos pliegos, supondrá la exclusión del proceso de licitación por acuerdo del Órgano de\nContratación a propuesta de la Mesa.\nCRITERIOS DE ADJUDICACIÓN DEL PROCEDIMIENTO ABIERTO\n Varios criterios\nSUJETO A REGULACIÓN ARMONIZADA\n No\nSUBASTA ELECTRÓNICA\n No\nCARÁCTER VINCULANTE DE LAS RESPUESTAS A LAS SOLICITUDES DE\nACLARACIONES DEL PLIEGO:\n Sí\nMODIFICACION DEL PLAZO DE SOLICITUD DE INFORMACION ADICIONAL O\nDOCUMENTACION COMPLEMENTARIA (ART. 138.3 LCSP):\n No\nLOS INTERCAMBIOS DE INFORMACION PARA LOS QUE NO SE UTILICEN MEDIOS\nELECTRONICOS SE REALIZARA:\n Por correo\nLA PRESENTACION DE PROPOSICIONES PARA LAS QUE NO SE UTILICEN MEDIOS\nELECTRONICOS SE REALIZARA:",
                "ACREDITACIÓN DE LA SOLVENCIA POR LOS EMPRESARIOS NO ESPAÑOLES DE\nESTADOS MIEMBROS DE LA UNIÓN EUROPEA O DE ESTADOS SIGNATARIOS DEL\nACUERDO SOBRE EL ESPACIO ECONÓMICO EUROPEO:\nMEDIOS DE ACREDITACIÓN: relación de las obras ejecutadas en el curso de los cinco últimos\naños, avalada con certificados de buena ejecución.\nCRITERIOS DE ACREDITACIÓN: que la media anual del importe total de obras de los grupos\nindicados previamente, ejecutadas en el curso de los cinco últimos años, sea superior a\nOCHOCIENTOS MIL EUROS (800.000 €) o que durante al menos tres de los últimos cinco años\nel importe haya sido superior a UN MILLÓN DOSCIENTOS MIL EUROS (1.200.000 €)\nEN CONTRATOS SUJETOS A UNA REGULACION ARMONIZADA, CERTIFICADOS DE\nACREDITACION DEL CUMPLIMIENTO DE LAS NORMAS DE GARANTIA DE CALIDAD:\nNO",
                "apartado d medida ambiental valorar medida medioambiental conseguir mejora materia tal reutilización residuo reciclado protección marino calidad agua licitador indicar relación actuación prever ejecutar relacionado mejora medioambiental propuesta concreto actuación desarrollo contrato valorar él justificación mejora realizar comparativa solución convencional puntuación apartado realizar criterio recogido tabla puntuación criterio 4 ≤ p ≤ 5 relacionar forma exhaustivo medida ambiental implantar describir característica principal indicar claramente"
            ]
        }
    
    rag_metrics = RAGMetrics(client=None, prompts=[])
    
    metrics = rag_metrics.calculate_response_score(test["response"], "No, la respuesta no está contenida en el documento", test["prompt"])
    
    # Imprimir resultados
    print(f"Resultados para '{test["prompt_key"]}':")
    print(f"  Precisión: {metrics['precision']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")
    print(f"  F1-Score: {metrics['f1_score']:.2f}")
    print()

    metrics = calculate_metrics("No, la respuesta no está contenida en el documento", test["response"], test["chunks"])

    # Imprimir resultados individuales
    print(f"Resultados para '{test["prompt_key"]}':")    
    print(f"  ROUGE: {metrics['rouge']}")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.2f}")
    print(f"  Context Recall: {metrics['context_recall']:.2f}")

def main():
    # Ruta de los archivos JSON
    generated_file_path = r"D:\WINDOWS\Escritorio\TFG\project\app\util\temp_files\CMAYOR_2021_07Y10_134.json"
    expected_file_path = r"D:\WINDOWS\Escritorio\TFG\project\app\util\JSON\CMAYOR_2021_07Y10_134.json"

    # Leer los archivos JSON
    with open(generated_file_path, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)

    with open(expected_file_path, 'r', encoding='utf-8') as f:
        expected_data = json.load(f)

    # Crear el objeto RAGMetrics (sin cliente OpenAI ya que no se usa aquí)
    rag_metrics = RAGMetrics(client=None, prompts=[])

    # Variables para acumular las métricas
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    count = 0

    # Procesar cada entrada en la lista "prompts"
    for generated_entry in generated_data["prompts"]:
        prompt_key = generated_entry["prompt_key"]
        prompt = generated_entry["prompt"]
        generated_response = generated_entry["response"]

        # Buscar la respuesta esperada correspondiente
        expected_entry = next((e for e in expected_data["prompts"] if e["prompt_key"] == prompt_key), None)
        if not expected_entry:
            print(f"No se encontró respuesta esperada para '{prompt_key}'")
            continue

        expected_response = expected_entry["response"]

        # Calcular métricas
        metrics = rag_metrics.calculate_response_score(generated_response, expected_response, prompt)

        # Acumular las métricas
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_f1_score += metrics["f1_score"]
        count += 1

        # Imprimir resultados
        print(f"Resultados para '{prompt_key}':")
        print(f"  Precisión: {metrics['precision']:.2f}")
        print(f"  Recall: {metrics['recall']:.2f}")
        print(f"  F1-Score: {metrics['f1_score']:.2f}")
        print()

    # Calcular y mostrar la media de las métricas
    if count > 0:
        mean_precision = total_precision / count
        mean_recall = total_recall / count
        mean_f1_score = total_f1_score / count
        print("\nPromedio de métricas:")
        print(f"  Precisión media: {mean_precision:.2f}")
        print(f"  Recall medio: {mean_recall:.2f}")
        print(f"  F1-Score medio: {mean_f1_score:.2f}")
    else:
        print("\nNo se procesaron entradas válidas.")

def main2():
    # Ruta del archivo JSON
    json_file_path = r"D:\WINDOWS\Escritorio\TFG\project\app\util\temp_files\output.json"

    # Leer el archivo JSON
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Variables para acumular las métricas
    total_cosine_similarity = 0
    total_rouge_1_f = 0
    total_rouge_2_f = 0
    total_rouge_l_f = 0
    total_context_recall = 0
    count = 0

    # Procesar cada entrada en la lista "prompts"
    for entry in data["prompts"]:
        prompt_key = entry["prompt_key"]
        prompt = entry["prompt"]
        response = entry["response"]
        chunks = entry.get("chunks", [])

        # Calcular métricas
        metrics = calculate_metrics(response, chunks)

        # Acumular las métricas
        total_cosine_similarity += metrics["cosine_similarity"]
        total_rouge_1_f += metrics["rouge"]["rouge-1"]["f"]
        total_rouge_2_f += metrics["rouge"]["rouge-2"]["f"]
        total_rouge_l_f += metrics["rouge"]["rouge-l"]["f"]
        total_context_recall += metrics["context_recall"]
        count += 1

        # Imprimir resultados individuales
        print(f"Resultados para '{prompt_key}':")
        print(f"  ROUGE: {metrics['rouge']}")
        print(f"  Cosine Similarity: {metrics['cosine_similarity']:.2f}")
        print(f"  Context Recall: {metrics['context_recall']:.2f}")

    # Calcular y mostrar la media de las métricas
    if count > 0:
        mean_cosine_similarity = total_cosine_similarity / count
        mean_rouge_1_f = total_rouge_1_f / count
        mean_rouge_2_f = total_rouge_2_f / count
        mean_rouge_l_f = total_rouge_l_f / count
        mean_context_recall = total_context_recall / count

        print("\nPromedio de métricas:")
        print(f"  ROUGE-1 (F) medio: {mean_rouge_1_f:.2f}")
        print(f"  ROUGE-2 (F) medio: {mean_rouge_2_f:.2f}")
        print(f"  ROUGE-L (F) medio: {mean_rouge_l_f:.2f}")
        print(f"  Cosine Similarity media: {mean_cosine_similarity:.2f}")
        print(f"  Context Recall medio: {mean_context_recall:.2f}")
    else:
        print("\nNo se procesaron entradas válidas.")

if __name__ == "__main__":
    # Asegúrate de descargar los recursos necesarios de NLTK antes de ejecutar el programa
    import nltk
    #main2()
    simple_test()