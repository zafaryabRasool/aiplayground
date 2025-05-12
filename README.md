# Capstone

## Setup
Install Anaconda from [here](https://www.anaconda.com/products/distribution)

```bash
# setup conda environment
conda env create -f environment.yml
conda activate capstone
```
```bash
# install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

```bash
#Install the required browsers for testing
playwright install
```

## Run Web-app
```bash
# run the app
python3 app.py
```

## DB Migrations
```bash
# create migration
alembic revision --autogenerate -m "migration message"

# apply migration
alembic upgrade head
```

## Run Tests
```bash
# run unit tests
pytest --ignore=backend/tests_integration backend

# run tests with coverage
pytest --cov-report term-missing --cov-report annotate --cov=backend --ignore=backend/tests_integration backend

# run integration tests
pytest backend/tests_integration

#run front-end test with UI visualised
python -m test.main --headed
```

## Run evaluation script
Set evaluation environment variables in .env file
```bash
APP_ENV=evaluation
EVAL_MAX_OUTPUT_LENGTH=25
```

Run the evaluation script
```bash
python3 evaluation.py {dataset_name} {llm_model} {rag_technique} {prompt_technique} {chunk_size} {chunk_overlap} {vector_top_k}
```

- dataset_name: name of the dataset on Langsmith
- llm_model: LLM model to use for generation. Possible values are: `gpt-3.5-turbo`, `gpt-4-turbo`, `gpt-4o-mini`, `gpt-4o`, `gemini-1.5-pro`, `gemini-1.5-flash`
- rag_technique: RAG technique to use for generation. Possible values are: `vector`, `graph`, `none`
- prompt_technique: Technique to use for generating prompts. Possible values are: `cot`, `tot`, `got`
- chunk_size: The size of the chunks to split the input text into
- chunk_overlap: The overlap between the chunks
- vector_top_k: The number of top-k to use for the vector RAG technique

For example:
```bash
python3 evaluation.py "Natural Questions" gpt-4o-mini vector cot 512 20 3
```


## Deployment

```bash
# build the docker image

docker build -t {dockerhub_username}/capstone .
```

Rename the .env.example file to .env and fill in the required environment variables.

Replace "legacy107/capstone" with your dockerhub username in the dockercompose.yml file.

```bash
# run the docker containers using docker-compose
docker compose -f dockercompose.yml up -d

# stop the docker containers
docker compose -f dockercompose.yml stop

# or stop and remove the containers
docker compose -f dockercompose.yml down
```