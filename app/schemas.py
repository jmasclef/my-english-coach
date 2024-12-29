from typing import List, Literal
from pydantic import BaseModel
import uuid


class Message(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str

    def is_assistant_message(self):
        return self.role == 'assistant'


class ChatResponseChunk(BaseModel):
    status: Literal['NOT_READY', 'STREAM', 'FINISHED'] = 'NOT_READY'
    text_chunk: str | None = None
    audio_chunk: str | None = None


class NewChat(BaseModel):
    session_id: str
    sample_rate: int
    format: str = 'audio/wav'


class ChatSession(BaseModel):
    session_id: str = str(uuid.uuid4())
    messages: List[Message] = []
    streaming_response: List[ChatResponseChunk] = []
    speaker: str | None

    def messages_to_dict(self):
        return [message.model_dump() for message in self.messages]
