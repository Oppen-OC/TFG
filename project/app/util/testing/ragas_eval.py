import json
from openai import OpenAI
from ragas import SingleTurnSample
from ragas.metrics import AspectCritic
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import asyncio

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Verificar la clave API
api_key = os.getenv("UPV_API_KEY")

# Configurar el cliente OpenAI
chat_client = ChatOpenAI(
    model="llama3.3:70b",
    api_key=api_key,
    base_url="https://api.poligpt.upv.es"
)

evaluator_llm = LangchainLLMWrapper(chat_client)

# Ruta del archivo JSON
file_path = r"D:\WINDOWS\Escritorio\TFG\project\app\util\temp_files\output.json"

# Cargar los datos desde el archivo JSON
with open(file_path, 'r', encoding='utf-8') as file:
    test_data = json.load(file)["prompts"]

# Configurar las métricas que deseas evaluar
metrics = [
    AspectCritic(
        name="Relevancia del contexto",
        llm=evaluator_llm,
        definition="Evalúa la calidad de la respuesta generada en relación con el contexto proporcionado."
    ),
    AspectCritic(
        name="Relevancia de la respuesta",
        llm=evaluator_llm,
        definition="Evalúa la calidad de la respuesta generada en relación con la pregunta del usuario."
    ),
    AspectCritic(
        name="Formato",
        llm=evaluator_llm,
        definition="Evalúa si la respuesta respesta el formato esperado."
    )
]

# Inicializar un diccionario para acumular los puntajes de cada métrica
metric_totals = {metric.name: 0 for metric in metrics}
metric_counts = {metric.name: 0 for metric in metrics}  # Para contar las muestras evaluadas

# Iterar sobre todos los elementos de test_data
for i, prompt_data in enumerate(test_data):
    # Extraer los datos relevantes del JSON
    sample_data = {
        "user_input": prompt_data["prompt"] + prompt_data["format"] ,  # El prompt del usuario
        "response": prompt_data["response"],  # La respuesta generada
        "retrieved_contexts": prompt_data["chunks"],  # El contexto asociado (si existe)
    }

    # Crear una instancia de SingleTurnSample
    sample = SingleTurnSample(**sample_data)

    # Evaluar cada métrica para la muestra actual
    print(f"Context: {prompt_data.get('context', 'No context provided')}")
    print(f"Prompt: {prompt_data['prompt']}")
    print(f"Response: {prompt_data['response']}")
    for metric in metrics:
        try:
            score = asyncio.run(metric.single_turn_ascore(sample))
            print(f"{metric.name} Score for sample {i + 1}: {score}")
            
            # Acumular el puntaje y contar la muestra
            metric_totals[metric.name] += score
            metric_counts[metric.name] += 1
        except Exception as e:
            print(f"Error processing {metric.name} for sample {i + 1}: {e}")
    print("-" * 50)

# Calcular y mostrar la media de cada métrica
print("\n=== Average Scores ===")
for metric_name, total_score in metric_totals.items():
    count = metric_counts[metric_name]
    average_score = total_score / count if count > 0 else 0
    print(f"{metric_name}: {average_score:.2f}")