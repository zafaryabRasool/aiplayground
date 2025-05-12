from enum import Enum


class LlmModel(Enum):
    """Enum for LLM models"""

    GPT35 = "gpt-3.5-turbo"
    GPT4 = "gpt-4-turbo"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4O = "gpt-4o"
    GEMINI15_PRO = "gemini-1.5-pro"
    GEMINI15_FLASH = "gemini-1.5-flash"
    GEMINI20_FLASH = "gemini-2.0-flash"
    QUASARALPHA = "openrouter/quasar-alpha"
    GEMINI23_PRO_EXP= "google/gemini-2.5-pro-preview-03-25"
    # GEMINI23_PRO_EXP= "google/gemini-2.5-pro-exp-03-25:free"
    DEEPSEEKV3 = "deepseek/deepseek-chat-v3-0324:free"
    DEEPSEEKR1 = "deepseek/deepseek-r1:free"
    DEEPSEEKR1_ZERO = "deepseek/deepseek-r1-zero:free"
    LLAMA4_MAVERICK = "meta-llama/llama-4-maverick:free"
    LLAMA4_SCOUT = "meta-llama/llama-4-scout:free"
    # QWEN7B = "qwen/Qwen-7B-Chat"
    LLAMA2_LOCAL = "llama3.2:latest"
    Qwen7B = "qwen2.5:7b"
    # MISTRAL_LOCAL = "mistral"

