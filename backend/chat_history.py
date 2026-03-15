from bot.conversation.chat_history import ChatHistory
from core.config import settings


def init_chat_history(total_length: int = 2) -> ChatHistory:
    chat_history = ChatHistory(total_length=total_length)
    return chat_history


chat_history = init_chat_history(settings.CHAT_HISTORY_LENGTH)
