# pylint: disable=redefined-outer-name, missing-function-docstring
import json

from langsmith import Client

API_KEY = "LangSmith API Key"
client = Client(
    api_url="https://api.smith.langchain.com",
    api_key=API_KEY,
)

DATASET_NAME = "Dataset Name"
dataset = client.create_dataset(DATASET_NAME)
inputs = []
outputs = []


def extract_data(data):
    # modify this function to extract your data
    ids = []
    title = []
    questions = []
    answers = []
    contexts = []
    for item in data["rows"]:
        ids.append(item["row"]["id"])
        title.append(item["row"]["title"])
        questions.append(item["row"]["question"])
        answers.append(item["row"]["short_answers"])
        contexts.append(item["row"]["document"])
    return ids, title, questions, answers, contexts


# modify the below code to read your data
with open("./natural_question.json", encoding="utf-8") as f:
    data = json.load(f)

    ids, titles, questions, answers, contexts = extract_data(data)
    for row in zip(ids, titles, questions, answers, contexts):
        idd, title, question, answers, context = row
        # the input must have the keys: id, question, context
        # where id is a number, question is a string, and context is a string
        # title is optional
        inputs.append(
            {
                "id": idd,
                "title": title,
                "question": question,
                "context": context,
            }
        )
        # the output must have the key: answers
        # where answers is a list of strings
        outputs.append(
            {
                "answers": answers,
            }
        )

client.create_examples(
    inputs=inputs,
    outputs=outputs,
    dataset_id=dataset.id,
)
