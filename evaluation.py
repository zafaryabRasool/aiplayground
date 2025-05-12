# pylint: disable=R0801
import asyncio
import os
import sys
from typing import Any, Dict

import evaluate as hf_evaluate
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, aevaluate
from langsmith.schemas import Example, Run
from llama_index.core import Document

from backend.src.constants import LlmModel, RagTechnique, Technique
from backend.src.llm import LlmFactory
from backend.src.prompts.techniques import TechniqueFactory
from backend.src.services.etl import (
    delete_graph_data,
    delete_vector_data,
    insert_graph_data,
    insert_vector_data,
)

load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

dataset_name = sys.argv[1] if len(sys.argv) > 1 else "Natural Questions"
llm_model = LlmModel(sys.argv[2]) if len(sys.argv) > 2 else LlmModel.GPT4O_MINI
rag_technique = (
    (RagTechnique(sys.argv[3]) if sys.argv[3] != "none" else "none")
    if len(sys.argv) > 3
    else RagTechnique.VECTOR
)
prompt_technique = Technique(sys.argv[4]) if len(sys.argv) > 4 else Technique.COT
chunk_size = int(sys.argv[5]) if len(sys.argv) > 5 else 512
chunk_overlap = int(sys.argv[6]) if len(sys.argv) > 6 else 20
vector_top_k = int(sys.argv[7]) if len(sys.argv) > 7 else 3

# Define the pipeline
llm = LlmFactory.create_llm(model=llm_model)
eval_llm = LlmFactory.create_llm(model=llm_model, temperature=0.0)


async def get_response(
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Get the response from the pipeline."""
    if rag_technique == "none":
        chat = TechniqueFactory.create_technique(prompt_technique)
        response = await chat.ask(
            inputs["question"],
            [],
            llm_model,
            rag_technique,
            0,
            vector_top_k,
            [inputs["context"]],
        )
        return {"output": response}

    try:
        input_id = int(inputs["id"].replace("-", ""))
        documents = [Document(text=inputs["context"])]
        if rag_technique == RagTechnique.VECTOR:
            insert_vector_data(
                input_id,
                documents,
                llm_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            insert_graph_data(
                input_id,
                documents,
                llm_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        chat = TechniqueFactory.create_technique(prompt_technique)
        response = await chat.ask(
            inputs["question"], [], llm_model, rag_technique, input_id, vector_top_k
        )
        return {"output": response}
    finally:
        if rag_technique == RagTechnique.VECTOR:
            delete_vector_data(input_id)
        else:
            delete_graph_data(input_id)


# Define the evaluators to apply
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
    """Evaluate the system."""
    rag_technique_name = rag_technique.name if rag_technique != "none" else "none"
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
        experiment_prefix=f"{prompt_technique.name}-{rag_technique_name}-{llm_model.name}",
        max_concurrency=5,
    )
    await asyncio.sleep(60)

    df = client.get_test_results(project_name=result.experiment_name)
    df.to_csv(f"{prompt_technique.name}-{rag_technique_name}-{llm_model.name}.csv")


asyncio.run(evaluate())
