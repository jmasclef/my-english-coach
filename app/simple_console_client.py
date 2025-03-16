import asyncio
from chatbot_client import ChatbotClient
from schemas import Message


if __name__ == '__main__':
    async def english_teaching():
        chatbot_client = ChatbotClient()
        first_message = Message(role='user', content="Hello let's discuss in English")
        chat_session = chatbot_client.create_chat_session(messages=[first_message])
        while (question := await chatbot_client.question_answer(chat_session=chat_session)) is not None:
            pass



    # Run the async main function
    asyncio.run(english_teaching())
