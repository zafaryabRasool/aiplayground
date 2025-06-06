import os

import dotenv
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from llama_index.core.base.embeddings.base import BaseEmbedding
# from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingModelType,
)

from langchain_ollama import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from sentence_transformers import SentenceTransformer
# from typing import List, Optional
# from pydantic import PrivateAttr


from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

from backend.src.constants import LlmModel

dotenv.load_dotenv()


# class MiniLMEmbedding(BaseEmbedding):
#     """MiniLM embedding model - excellent balance of size and performance"""

#     _model = PrivateAttr()  # Private attribute for Pydantic
    
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
#         """Initialize the model. Downloads ~90MB on first run."""
#         self._model = SentenceTransformer(model_name, device='cpu')
#         super().__init__()

#     def _get_text_embedding(self, text: str) -> List[float]:
#         """Synchronous: Get embedding for a text passage."""
#         embeddings = self._model.encode(text, normalize_embeddings=True)
#         return embeddings.tolist()

#     def _get_query_embedding(self, query: str) -> List[float]:
#         """Synchronous: Get embedding for a search query."""
#         embeddings = self._model.encode(query, normalize_embeddings=True)
#         return embeddings.tolist()

#     async def _aget_text_embedding(self, text: str) -> List[float]:
#         """Asynchronous: Get embedding for a text passage."""
#         embeddings = self._model.encode(text, normalize_embeddings=True)
#         return embeddings.tolist()

#     async def _aget_query_embedding(self, query: str) -> List[float]:
#         """Asynchronous: Get embedding for a search query."""
#         embeddings = self._model.encode(query, normalize_embeddings=True)
#         return embeddings.tolist()


class LlmFactory:
    """Factory class for creating LLM chat models"""

    @staticmethod
    def create_llm(
        model=LlmModel.GPT4O_MINI, api_key="", max_tokens=4096, temperature=0.6
    ) -> BaseChatModel:
        """Creates an LLM chat model based on the specified model"""

        match model:
            case (
                LlmModel.GPT35 | LlmModel.GPT4 | LlmModel.GPT4O_MINI | LlmModel.GPT4O 
            ):
                return ChatOpenAI(
                    model=model.value,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_key if api_key else os.getenv("OPENAI_API_KEY"),
                )
            case (
                LlmModel.QUASARALPHA
                | LlmModel.GEMINI23_PRO_EXP
                | LlmModel.DEEPSEEKV3
                | LlmModel.DEEPSEEKR1
                | LlmModel.DEEPSEEKR1_ZERO
                | LlmModel.LLAMA4_MAVERICK
                | LlmModel.LLAMA4_SCOUT
            ):
            
                return ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    model=model.value,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_key if api_key else os.getenv("OPENROUTER_API_KEY"),
                )
            # case (
            #     #new models LlmModel
            # ):
            
            #     return ChatOpenAI(
            #         base_url="https://openrouter.ai/api/v1",
            #         model=model.value,
            #         temperature=temperature,
            #         max_tokens=max_tokens,
            #         openai_api_key=api_key if api_key else os.getenv("LOCAL_API_KEY"),
            #     )    

            case (LlmModel.LLAMA2_LOCAL | LlmModel.Qwen7B):
                if os.getenv("APP_ENV") == "production":
                    # Production configuration using vLLM
                    return ChatOpenAI(
                        base_url="http://vllm-service:8000/v1",  # Your vLLM service URL
                        model=model.value,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    try:
                    # Local development configuration using Ollama
                        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                        return ChatOllama(
                            model=model.value,
                            callback_manager=callback_manager,
                            temperature=temperature,
                            # verbose=True,
                            # stop=["\n\n"]  # Add stop sequence for better response formatting
                        )
                    
                    except Exception as e:
                        print(f"Error initializing Ollama: {e}")
                        print("Available models:", os.popen('ollama list').read())
                        raise

            case LlmModel.GEMINI15_FLASH | LlmModel.GEMINI15_PRO | LlmModel.GEMINI20_FLASH:
                return ChatGoogleGenerativeAI(
                    model=model.value,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    google_api_key=api_key if api_key else os.getenv("GOOGLE_API_KEY"),
                )
            case _:
                raise ValueError("Invalid model")

    @staticmethod
    def create_embedding_model(model=LlmModel.GPT4O_MINI) -> BaseEmbedding:
        """
        Creates an embedding model based on the specified LLM
        """
        match model:
            case (
                LlmModel.GPT35 
                | LlmModel.GPT4
                | LlmModel.GPT4O_MINI 
                | LlmModel.GPT4O 
                | LlmModel.GEMINI15_PRO 
                | LlmModel.GEMINI15_FLASH 
                | LlmModel.GEMINI20_FLASH
                | LlmModel.QUASARALPHA
                | LlmModel.GEMINI23_PRO_EXP
                | LlmModel.DEEPSEEKV3
                | LlmModel.DEEPSEEKR1
                | LlmModel.DEEPSEEKR1_ZERO
                | LlmModel.LLAMA4_MAVERICK
                | LlmModel.LLAMA4_SCOUT
                # | LlmModel.LLAMA2_LOCAL
                # | LlmModel.Qwen7B
                # LlmModel.QWEN7B
            ):
                return OpenAIEmbedding(
                    model_name=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
            case (LlmModel.LLAMA2_LOCAL | LlmModel.Qwen7B):
                # Create HuggingFace embeddings
                print("Using HuggingFace embeddings")
                hf_embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                # Wrap in LangchainEmbedding for LlamaIndex compatibility
                return LangchainEmbedding(
                    langchain_embeddings=hf_embeddings
                )
            
            # case LlmModel.GEMINI15_PRO | LlmModel.GEMINI15_FLASH | LlmModel.GEMINI20_FLASH:
            #     return GeminiEmbedding(
            #         model_name="models/embedding-001",
            #         api_key=os.getenv("GOOGLE_API_KEY"),
            #     )
            case _:
                raise ValueError("Invalid model")


