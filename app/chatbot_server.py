import io
import os.path
import uuid
import re
import soundfile  # To read streamed wave
from random import choice as random_choice
import asyncio
import base64

from ollama import AsyncClient as OllamaAsyncClient
import logging

from schemas import ChatResponseChunk, Message, NewChat, ChatSession
from typing import List

log_level = logging.FATAL

logger = logging.getLogger("CHAT-WITH-OLLAMA-SERVER")
logging.basicConfig(level=log_level)
logger.info('Start')
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())





"""STT MANAGERS"""
# https://github.com/openai/whisper
import whisper

whisper_logger= logging.getLogger("TTS")
whisper_logger.setLevel(level=log_level)

"""TTS MANAGERS"""
# Option 1 coqui-tts
from TTS.api import TTS
from TTS.api import logger as tts_logger
import numpy as np

import logging

log_level = logging.INFO

logger = logging.getLogger("STT-ollama-TTS")
logging.basicConfig(level=log_level)
logger.info('Start')

PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
COQUI_TTS_MODELS_PATH = os.path.normpath(os.path.join(PROJECT_ROOT_PATH, "../coqui_models"))

WHISPER_STT_MODELS_PATH = os.path.normpath(os.path.join(PROJECT_ROOT_PATH, "../whisper_models"))
os.environ['TTS_HOME'] = COQUI_TTS_MODELS_PATH

DEFAULT_LOCAL_OLLAMA_SERVER = "http://localhost:11434"

class ChatbotServer:
    def __init__(self, use_kb: bool = False,
                 use_print: bool = True, ollama_server: str =DEFAULT_LOCAL_OLLAMA_SERVER,
                 lang='en'):
        self.audio_convert_rate = 16000
        self.audio_convert_width = 2
        self.chat_sessions = dict()
        if lang == 'fr':
            self.model = 'llama3.2'
        else:
            self.model = 'english_teacher_model:latest'

        self._async_ollama_client = OllamaAsyncClient(host=ollama_server)

        self.whisper_model = None
        self.language = lang
        logger.info('Now loading WHISPER model')
        #self.whisper_model = whisper.load_model("base", download_root=WHISPER_STT_MODELS_PATH)   # Default path is "C:\Users\USER\.cache\whisper"
        self.whisper_model = whisper.load_model("turbo", download_root=WHISPER_STT_MODELS_PATH) # Turbo is a bit faster but x6 bigger
        logger.info('WHISPER model is loaded')

        speakers = ['Asya Anara', 'Gitta Nikolina', 'Sofia Hellen', 'Uta Obando',
                    'Dionisio Schuyler', 'Adde Michal', 'Ludvig Milivoj', 'Torcull Diarmuid']
        # Initialize TTS model
        # tts --list_models
        # Models are stored in 'C:\Users\USER\AppData\Local\tts'
        logger.info(f"Load coqui-tts {lang} model...")
        #  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        if lang == 'en':
            # speedy-speech ko and fast_pitch ok
            # GREAT but sound poor and one speaker
            # self.tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=True).to('cuda') # good 440Mo but single speaker

            # self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to('cuda') # Great sound with speakers but a bit slow
            self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC_ph", progress_bar=True, ).to(
                'cuda')  # good ~1go and single speaker
            # self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DCA", progress_bar=True).to('cuda') # glitch on first !  ~400Mo
            # self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=True) #bof but 111Mo

            # self.tts = TTS(model_name="tts_models/en/ek1/tacotron2", progress_bar=True).to('cuda') # slow! ~500Mo
            # self.tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=True) " didn't work
        elif lang == 'fr':
            self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to('cuda')
            # self.tts = TTS(model_name="tts_models/fr/mai/tacotron2-DDC", progress_bar=True).to('cuda')
        else:
            logger.critical(f"No known TTS model for lan='{lang}', check 'tts --list_models' ")

        logger.info("coqui-tts model loaded...")
        if self.tts.is_multi_speaker:
            # self.tts_speaker = random_choice(speakers)
            self.tts_speaker = random_choice(self.tts.speakers)
            logger.info(f"Multiple speakers, chose: {self.tts_speaker}")
        else:
            self.tts_speaker = None
            logger.info("Single speaker in the model")
        if self.tts.is_multi_lingual:
            self.tts_lang = self.language
        else:
            self.tts_lang = None


    async def _async_stream_response(self, messages):
        # This is the wrapped call to Ollama API
        streamed_chat = await self._async_ollama_client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={}
        )

        # Yield the chatbot's response in chunks
        async for chunk in streamed_chat:
            yield chunk['message']['content']

    def _get_whisper_speech2text_question(self, audio_data):
        # This is the wrapped call to openai whisper API
        wav_bytes = audio_data.get_wav_data(convert_rate=16000)
        wav_stream = io.BytesIO(wav_bytes)
        audio_array, sampling_rate = soundfile.read(wav_stream)
        audio_array = audio_array.astype(np.float32)

        # result = self.whisper_model.transcribe(audio_array)
        result = self.whisper_model.transcribe(audio_array, language=self.language, fp16=False)
        logger.info(f"Whisper STT result: {result}")
        return result['text']

    def _build_audio_response(self, content_to_output: str, speaker: str = None):
        # Set speaker if given or take server instance default
        target_speaker = speaker or self.tts_speaker

        # Remove repeated dots which generate cut audio issues
        content_to_output = content_to_output.replace('...', '. ')
        content_to_output = content_to_output.replace('..', '. ')
        # content_to_output = self.clean_string(string_to_clean=content_to_output)
        logger.debug(f"Content to output: {content_to_output}")

        # Generate TTS audio as a numpy array
        try:
            logger.debug(f"Sent to coqui: {content_to_output}")
            audio_data = self.tts.tts(content_to_output, speaker=target_speaker, language=self.tts_lang,
                                      split_sentences=True)
        except Exception as e:
            logger.error(f"Could not process audio for :'{content_to_output}'")
            logger.error(f"Exception :'{str(e)}'")
            return None

        # Normalize audio for playback
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

        return audio_data

    def _get_chat_session(self, session_id) -> ChatSession:
        try:
            return self.chat_sessions[session_id]
        except:
            return None

    def catch_user_question(self, audio_data):
        return self._get_whisper_speech2text_question(audio_data=audio_data)

    def catch_user_question_from_audio_browser(self, browser_audio_data, messages: list[Message] = None):
        """
        Process raw audio data (WAV, mono, 16-bit, 16kHz) from the browser and transcribe it using Whisper.

        Args:
            browser_audio_data (bytes): WAV audio data in mono, 16-bit, 16kHz format.

        Returns:
            str: Transcribed text from the audio.
        """
        if True:
            # Read audio bytes into a NumPy array
            audio_data = io.BytesIO(browser_audio_data)

            audio_array, sampling_rate = soundfile.read(audio_data)

            logger.info(
                f"Received audio format: {audio_array.dtype}, Sampling rate: {sampling_rate}, Shape: {audio_array.shape}")

            # Ensure the sample rate matches 16kHz
            if sampling_rate != 16000:
                raise ValueError("Input audio must have a sample rate of 16kHz.")

            # Ensure the audio array is in float32 format for Whisper
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # Log the processed audio array details
            logger.info(f"Processed audio shape: {audio_array.shape}, dtype: {audio_array.dtype}")

            # prepare prompt (for context)
            initial_prompt = " ".join(message.content for message in messages) if messages else None

            # Transcribe the audio with Whisper
            result = self.whisper_model.transcribe(audio_array, initial_prompt=initial_prompt, language=self.language,
                                                   fp16=False, temperature=0)
            logger.info(f"Whisper STT result: {result}")

            return result['text']

        # except Exception as e:
        #     logger.error(f"Error processing audio: {e}")
        #     return "Error processing audio"

    def _prepare_audio_response_from_tts_to_browser(self, tts_audio_data):
        # Normalize and convert to int16
        audio_data = np.int16(tts_audio_data / np.max(np.abs(tts_audio_data)) * 32767)

        # Write the audio to a BytesIO buffer
        audio_buffer = io.BytesIO()
        soundfile.write(audio_buffer, audio_data, 22050, format='WAV', subtype='PCM_16')  # PCM 16-bit subtype
        audio_buffer.seek(0)  # Reset cursor
        audio_chunk_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
        return audio_chunk_base64

    #
    # async def async_generate_response(self, messages: list):
    #     line = []
    #     async for chunk in self._async_stream_response(messages):
    #         line.append(chunk)
    #         if self._should_break_chunk(chunk=chunk) and len("".join(line)) > 50:
    #             audio_data = self._build_audio_response(content_to_output="".join(line), speaker=self.tts_speaker)
    #             yield "".join(line), audio_data, self.tts.synthesizer.output_sample_rate
    #             line = []
    #     else:
    #         if len(line) > 0:
    #             try:
    #                 audio_data = self._build_audio_response(content_to_output="".join(line), speaker=self.tts_speaker)
    #             except:
    #                 audio_data = None
    #             yield "".join(line), audio_data, self.tts.synthesizer.output_sample_rate

    @classmethod
    def _split_buffer_on_char(cls, text_buffer: str, split_on: str):
        split = text_buffer.split(split_on)
        text_data = split_on.join(split[:-1]) + split_on
        new_text_buffer = split[-1]
        return (text_data, new_text_buffer)

    @classmethod
    def _split_if_possible(cls, text_buffer):
        eol_str = '\n'
        question_str = '? '
        exclamation_str = '! '
        has_dot_str = '. '
        eol = (eol_str in text_buffer)
        question = (question_str in text_buffer)
        exclamation = (exclamation_str in text_buffer)
        has_dot = (has_dot_str in text_buffer)
        if not eol and not question and not exclamation and not has_dot:
            return (None, None)
        else:
            if re.findall(r'\b\d+\.', text_buffer) and not eol and not question and not exclamation:
                # Enumeration only
                return (None, None)
            if text_buffer.count('"') % 2 != 0:
                # Even number of "
                return (None, None)
            else:
                if eol:
                    return cls._split_buffer_on_char(text_buffer=text_buffer, split_on=eol_str)
                elif question:
                    return cls._split_buffer_on_char(text_buffer=text_buffer, split_on=question_str)
                elif exclamation:
                    return cls._split_buffer_on_char(text_buffer=text_buffer, split_on=exclamation_str)
                elif has_dot:
                    return cls._split_buffer_on_char(text_buffer=text_buffer, split_on=has_dot_str)

    async def yield_response_messages(self, chat_session):
        text_buffer = ""
        async for chunk in self._async_stream_response(chat_session.messages_to_dict()):
            text_buffer += chunk
            text_data, new_text_buffer = self._split_if_possible(text_buffer=text_buffer)
            if (text_data, new_text_buffer) != (None, None):
                text_buffer = new_text_buffer
                audio_data = self._build_audio_response(content_to_output=text_data, speaker=chat_session.speaker)
                yield text_data, audio_data

        else:
            if len(text_buffer) > 0:
                text_data = text_buffer
                audio_data = self._build_audio_response(content_to_output=text_data, speaker=chat_session.speaker)
                yield text_data, audio_data

    def start_chat(self, messages: List[Message]) -> NewChat:
        new_chat_session = ChatSession(speaker=self.tts_speaker, messages=messages)
        if self.tts.is_multi_speaker and not (any([message.role == 'assistant' for message in messages])):
            self.tts_speaker = random_choice(self.tts.speakers)
            logger.info(f"Multiple speakers, chose '{self.tts_speaker}' for session {new_chat_session.session_id}")
        self.chat_sessions[new_chat_session.session_id] = new_chat_session

        return NewChat(session_id=new_chat_session.session_id, sample_rate=self.tts.synthesizer.output_sample_rate)

    async def build_chat_response(self, session_id):
        """
        This method will be called as a background task
        """
        chat_session = self._get_chat_session(session_id)
        async for sentence, tts_audio in self.yield_response_messages(chat_session=chat_session):
            logger.debug(f"Background process for chat {session_id} is adding chunk")
            if tts_audio is not None:
                audio_for_browser = self._prepare_audio_response_from_tts_to_browser(tts_audio)
            else:
                logger.warning(f"No audio chunk for text:{sentence}")
                audio_for_browser = None
            response_chunk = ChatResponseChunk(status='STREAM', text_chunk=sentence, audio_chunk=audio_for_browser)
            chat_session.streaming_response.append(response_chunk)
        else:
            chat_session.streaming_response.append(ChatResponseChunk(status='FINISHED'))
            logger.info(f"Background process for chat {session_id} is finished.")

    async def stream_chat(self, session_id):
        chat_session = self._get_chat_session(session_id=session_id)
        if not chat_session:
            return None
        elif not chat_session.streaming_response:
            return ChatResponseChunk(status='NOT_READY')
        else:
            chunk = chat_session.streaming_response.pop(0)
            if chunk.status == 'FINISHED':
                del self.chat_sessions[session_id]
            return chunk
