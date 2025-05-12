# pylint: disable=R0801
import asyncio
import os
import sys
from typing import Any, Dict

import evaluate as hf_evaluate
from dotenv import load_dotenv
from langchain import prompts
from langchain.schema import output_parser
from langsmith import Client, aevaluate
from langsmith.evaluation import LangChainStringEvaluator
from langsmith.schemas import Example, Run

from backend.src.constants import LlmModel
from backend.src.llm import LlmFactory

load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
MAX_OUTPUT_LENGTH = os.getenv("EVAL_MAX_OUTPUT_LENGTH")

dataset_name = sys.argv[1] if len(sys.argv) > 1 else "Natural Questions"

# Define the pipeline
prompt = prompts.ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Here is the information that you know: "
            + "<Info> {context} </Info> You only reply with information found in the above info. "
            + "If you can't get your answer from there, say you don't know. "
            + "Keep your answer in less than "
            + MAX_OUTPUT_LENGTH
            + " characters.",
        ),
        ("human", "{question}"),
    ]
)
llm = LlmFactory.create_llm(model=LlmModel.GPT4O_MINI, temperature=0.6)
eval_llm = LlmFactory.create_llm(model=LlmModel.GPT4O_MINI, temperature=0.0)
chain = prompt | llm | output_parser.StrOutputParser()


async def get_response(
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Get the response from the pipeline."""
    response = await chain.ainvoke(inputs)
    return {"output": response}


# Define the evaluators to apply
def f1_score(run: Run, example: Example):
    """Compute the F1 score."""
    squad_metric = hf_evaluate.load("squad")
    result = squad_metric.compute(
        predictions=[
            {
                "id": "1",
                "prediction_text": run.outputs["output"],
            }
        ],
        references=[
            {
                "id": "1",
                "answers": {
                    "text": example.outputs["answers"],
                    "answer_start": [0],
                },
            }
        ],
    )
    return {"score": result["f1"]}


def bleu_score(run: Run, example: Example):
    """Compute the BLEU score."""
    bleu_metric = hf_evaluate.load("bleu")
    result = bleu_metric.compute(
        predictions=[run.outputs["output"]], references=[example.outputs["answers"]]
    )
    return {"score": result["bleu"]}


def rouge_score(run: Run, example: Example):
    """Compute the ROUGE score."""
    rouge_metric = hf_evaluate.load("rouge")
    result = rouge_metric.compute(
        predictions=[run.outputs["output"]], references=[example.outputs["answers"]]
    )
    return {"score": result["rougeL"]}


def prepare_data(run: Run, example: Example):
    """Prepare data for the evaluator."""
    return {
        "prediction": run.outputs["output"],
        "reference": example.outputs["answers"],
        "input": (
            "You are a helpful AI assistant. Here is the information that you know: "
            + f"<Info> {example.inputs['context']} </Info> You only reply with information "
            + "found in the above info. If you can't get your answer from there, "
            "say you don't know."
        ),
    }


cot_qa_evaluator = LangChainStringEvaluator(
    "cot_qa",
    config={"llm": eval_llm},
    prepare_data=prepare_data,
)

labeled_criteria_evaluator = LangChainStringEvaluator(
    "labeled_criteria",
    config={
        "criteria": {
            "correctness": (
                "Is this prediction correct taking into account the correct reference answer?"
            )
        },
        "llm": eval_llm,
    },
    prepare_data=prepare_data,
)

client = Client(
    api_url="https://api.smith.langchain.com",
    api_key=LANGSMITH_API_KEY,
)


async def evaluate():
    """Evaluate the pipeline."""
    result = await aevaluate(
        get_response,
        client=client,
        data=dataset_name,
        evaluators=[
            cot_qa_evaluator,
            labeled_criteria_evaluator,
            f1_score,
            bleu_score,
            rouge_score,
        ],
        experiment_prefix="base-model",
        max_concurrency=10,
    )
    df = client.get_test_results(project_name=result.experiment_name)
    df.to_csv("base-model.csv")


asyncio.run(evaluate())
