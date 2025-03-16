import asyncio
from chatbot_server import ChatbotServer
from schemas import ChatSession, NewChat, Message
from typing import List
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# https://pypi.org/project/SpeechRecognition/
# speech_recognition is a wrapper for followings

import speech_recognition
import sounddevice  # to play sounds

import logging

log_level = logging.CRITICAL

logger = logging.getLogger("CHAT-WITH-OLLAMA-CLIENT")
logging.basicConfig(level=log_level)
logger.info('Start')


class ChatbotClient:
    def __init__(self, chatbot_server: ChatbotServer = None, use_kb: bool = False,
                 use_print: bool = True,
                 lang='en'):
        self.audio_convert_rate = 16000
        self.audio_convert_width = 2

        self.recognizer = speech_recognition.Recognizer()
        self.recognizer.energy_threshold = 200
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 1  # seconds of non-speaking audio before a phrase is considered complete
        self.recognizer.phrase_threshold = 1  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)

        if lang == 'fr':
            self.model = 'llama3.2'
        else:
            self.model = 'english_coach_model:latest'

        self.use_kb = use_kb
        self.use_print = use_print
        self.messages = []
        self.language = lang
        self.chatbot_server = chatbot_server or ChatbotServer()

        self._print("Adjusting for ambient noise...", eol=True)
        with speech_recognition.Microphone() as source:
            self.source_device_index = source.device_index
            self.recognizer.adjust_for_ambient_noise(source)

    @classmethod
    def get_kb_question(cls):
        return input("You: ")

    def capture_audio_data(self):
        with speech_recognition.Microphone(device_index=self.source_device_index) as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self._print("Speak...")
            audio_data = self.recognizer.listen(source)
        self._print(" OK")
        return audio_data

    def _print(self, content: str, eol: bool = False):
        if content.strip() == "":
            return
        if self.use_print:
            if not eol:
                print(content, flush=True, end='')
            else:
                print(f'{content}', flush=True)

    def _possible_languages(self):
        return {'fr', 'en'}

    def create_chat_session(self, messages: List[Message]):
        # return ChatSession(speaker=self.chatbot_server.tts_speaker, messages=messages)
        return ChatSession(speaker='', messages=messages)

    async def question_answer(self, chat_session: ChatSession):
        # Store question in chat history => chat history is the material for ollama

        self._print("Assistant is talking...", eol=True)
        responses = []
        async for sentence, audio_data in self.chatbot_server.yield_response_messages(chat_session=chat_session):
            self._print(f"\"\"\" {sentence}", eol=True)
            # Play audio using sounddevice
            sounddevice.wait()  # Wait until previous playback finishes
            if audio_data is not None:
                sounddevice.play(audio_data, samplerate=self.chatbot_server.tts_output_sample_rate)
            responses.append(sentence)
        else:
            response = "".join(responses)
            sounddevice.wait()  # Wait until final playback finishes before capturing new question

        # Store response in chat history
        chat_session.messages.append(Message(role='assistant', content=response))
        while (question := self.chatbot_server.catch_user_question(audio_data=self.capture_audio_data())) == '':
            self._print(" blank", eol=True)
        self._print(f", User said: {question}", eol=True)

        if not isinstance(question, str):
            return None
        elif (question.lower() == 'stop') or (question == ''):
            return None
        else:
            logger.info(f"Start ollama process for question: '{question}'")
            chat_session.messages.append(Message(role='user', content=question))
            return question


