import re
from dataclasses import dataclass, field
import typing as t
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
from ragas.callbacks import Callbacks
from ragas.dataset_schema import SingleTurnSample
from dotenv import load_dotenv
import os


load_dotenv()
@dataclass
class ContextSupportMetric(MetricWithLLM, SingleTurnMetric):
    name: str = "context_support_metric"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference"}
        }
    )

    def normalize_text(self, text: str) -> str:
        """
        Normaliza el texto eliminando espacios adicionales, convirtiendo a minúsculas,
        reemplazando números escritos como palabras y aplicando otras reglas de limpieza.
        """
        text = text.lower().strip()  # Convertir a minúsculas y eliminar espacios
        text = re.sub(r"\s+", " ", text)  # Reemplazar múltiples espacios por uno solo
        text = re.sub(r"\((.*?)\)", "", text)  # Eliminar contenido entre paréntesis
        text = re.sub(r"[^\w\s]", "", text)  # Eliminar caracteres especiales
        text = text.replace("€", "euros")  # Reemplazar el símbolo de euro por 'euros'

        return text

    async def compare_with_openai(self, prompt: str, response: str, reference: str) -> float:
        """
        Utiliza el cliente configurado con LangchainLLMWrapper para comparar las respuestas.
        """
        if response == "" and reference != "":
            return "No"        
        if response != "" and reference == "":
            return "No"
        # Construir los mensajes como lista de diccionarios
        messages = [
            {"role": "system", "content": "Eres un evaluador que compara respuestas en función de un contexto. "
            "Si ambas respuestas pueden resumirse a Sí o No se consideran equivalentes. "
            "Si una respuesta está contenida dentro de la otra también se considerarán iguales. "
            "'No' y 'no procede' se consideran iguales"
            "Si la respuesta esperada está contenida dentro de la respuesta generada se considerará que son equivalentes."
            },
            {"role": "user", "content": f"Pregunta: {prompt}\nRespuesta generada: {response}\nRespuesta esperada: {reference}\n¿Son equivalentes? Responde con 'Sí' o 'No'."}
        ]

        # Llamar al modelo usando LangchainLLMWrapper
        result = self.llm.chat.completions.create(
            model="llama3.3:70b",
            messages=messages
        )

        # Interpreta la respuesta del modelo (ajusta según la estructura real de result)
        content = result.choices[0].message.content.lower()
        if "sí" in content:
            return 1.0
        return 0.0

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """
        Compara la respuesta generada con la respuesta esperada, basándose en el prompt.
        """
        # Normaliza las respuestas
        normalized_response = self.normalize_text(sample.response)
        normalized_reference = self.normalize_text(sample.reference)

        # Comparación directa
        if normalized_response == normalized_reference:
            return 1.0  # Respuesta perfecta

        # Comparación basada en OpenAI
        score = await self.compare_with_openai(
            prompt=sample.user_input,
            response=sample.response,
            reference=sample.reference
        )
        return score