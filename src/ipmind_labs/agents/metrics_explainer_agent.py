import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings, format_as_xml

load_dotenv()


class ModelMetrics(BaseModel):
    total_records: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    tn: int
    fp: int
    fn: int


SYSTEM_PROMPT = """
You are a helpful data science assistant. Review the provided binary classification metrics.
Provide a concise, easy-to-understand explanation using exactly three short bullet points:
- **Overall Performance:** Briefly praise the Accuracy and F1 score.
- **Precision:** Explain the precision score in plain English. Explicitly mention why it is good or bad based on the number of False Positives (false alarms).
- **Recall:** Explain the recall score in plain English. Explicitly mention why it is good or bad based on the number of False Negatives (missed cases).

Keep the total response under 100 words. Be clear and encouraging. Do not use tables or complex jargon.
"""

metrics_explainer_agent = Agent(
    "azure:gpt-4.1-mini",
    system_prompt=SYSTEM_PROMPT,
    retries=2,
    model_settings=ModelSettings(temperature=0.3, max_tokens=300),
    name="metrics_explainer_agent",
)


async def get_metrics_summary(metrics: ModelMetrics):
    user_prompt = f"Please analyze these metrics:\n{format_as_xml(metrics)}"
    result = await metrics_explainer_agent.run(user_prompt)
    return result.output


async def main():
    my_results = ModelMetrics(
        total_records=372,
        precision=0.9794,
        recall=0.9333,
        f1=0.9558,
        accuracy=0.9409,
        tp=238,
        tn=112,
        fp=5,
        fn=17,
    )

    print("Analysing metrics with GPT-4o-mini in Azure...\n")

    result = await get_metrics_summary(my_results)

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
