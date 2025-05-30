import asyncio
import os
import json
from dotenv import load_dotenv
from ragas import SingleTurnSample
from context_support_metric import ContextSupportMetric  # Importa la métrica personalizada
import openai

# Configuración del cliente OpenAI
load_dotenv()
api_key = os.getenv("UPV_API_KEY")
base_url = "https://api.poligpt.upv.es"

# Configura el LLM usando LangChain y el wrapper de RAGAS
evaluator_llm = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )

async def main():

    # Carga los archivos JSON
    with open(r'd:\WINDOWS\Escritorio\TFG\project\app\util\temp_files\CMAYOR_2022_03Y03_101.json', encoding='utf-8') as f:
        output_data = json.load(f)["prompts"]
    with open(r'd:\WINDOWS\Escritorio\TFG\project\app\util\JSON\CMAYOR_2022_03Y03_101.json', encoding='utf-8') as f:
        reference_data = json.load(f)["prompts"]

    # Crea un diccionario para buscar la referencia por prompt_key
    ref_dict = {item["prompt_key"]: item.get("response", "") for item in reference_data}

    # Instancia la métrica personalizada
    metric = ContextSupportMetric(
        llm=evaluator_llm,
        name="context_support_metric"
    )

    scores = []
    callbacks = None  # No se utiliza Callbacks en este caso
    for item in output_data:
        prompt_key = item["prompt_key"]
        user_input = item.get("prompt", "")
        response = item.get("response", "")
        reference = ref_dict.get(prompt_key, "")


        if not response or not reference:
            continue  # Omite si faltan datos

        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            reference=reference
        )
        score = await metric.single_turn_ascore(sample, callbacks)
        print()
        print(f"{prompt_key}: {score}")
        print("Response:", response)
        print("Reference:", reference)  
        # if score == 0:
            # print(f"Prompt Key: {prompt_key}")
            # print("Response:", response)
            # print("Reference:", reference)        
        scores.append(score)

    if scores:
        print(f"\nAverage Context Support score: {sum(scores)/len(scores):.3f}")
    else:
        print("No comparable prompts found.")

if __name__ == "__main__":
    asyncio.run(main())