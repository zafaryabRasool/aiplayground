# Testing purpose! To be deleted
from backend.src.constants import LlmModel, Technique
from backend.src.services.ask import ChatService

if __name__ == "__main__":
    chat_service = ChatService()
    QUESTION = (
        "I went to the market and bought 10 apples. "
        "I gave 2 apples to the neighbor and 2 to the repairman. "
        "I then went and bought 5 more apples and ate 1. "
        "How many apples did I remain with?"
    )
    print("Question:", QUESTION)
    print(
        "Base output:",
        chat_service.get_response(QUESTION, Technique.NONE, LlmModel.GPT4).content,
    )
    print(
        "CoT output:",
        chat_service.get_response(QUESTION, Technique.COT, LlmModel.GPT4).content,
    )
