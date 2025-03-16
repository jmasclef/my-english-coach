import telebot
from telebot.async_telebot import AsyncTeleBot
from telebot.formatting import format_text, mcode
import asyncio
import numpy as np
import io
from chatbot_server import ChatbotServer, PROJECT_ROOT_PATH
from schemas import Message
import ffmpeg
from my_conf import telegram_bot_id, DEFAULT_LOCAL_OLLAMA_SERVER

chatbot_server = ChatbotServer(ollama_server=DEFAULT_LOCAL_OLLAMA_SERVER)

bot = AsyncTeleBot(telegram_bot_id, parse_mode="MARKDOWN")


# To use sendVoice, the file must have the type audio/ogg and be no more than 1MB in size. 1-20MB voice notes will be sent as files.
# https://core.telegram.org/bots/api#sendvoice
@bot.message_handler(commands=['start', 'help'])
async def send_welcome(message):
    session_id = str(message.chat.id)
    if session_id in chatbot_server.chat_sessions.keys():
        del chatbot_server.chat_sessions[session_id]
        await bot.reply_to(message,
                           "Chat history cleared, send a voice message to introduce the new conversation")
    else:
        await bot.reply_to(message, "Send a voice message to introduce the conversation")


@bot.message_handler(content_types=['text'])
async def echo_all(message):
    chat_id = message.chat.id
    session_id = str(chat_id)

    if session_id not in chatbot_server.chat_sessions.keys():
        chatbot_server.start_chat(messages=[], set_chat_id_to=session_id)
    chat_session = chatbot_server.chat_sessions[session_id]
    chat_session.messages.append(Message(role='user', content=message.text))
    async for sentence, tts_audio_data in chatbot_server.yield_response_messages(chat_session=chat_session):
        # await bot.reply_to(message=message, text=sentence)
        chat_session.messages.append(Message(role='assistant', content=sentence))
        wav_buffer = chatbot_server.prepare_audio_response_for_client(tts_audio_data=tts_audio_data)
        if not wav_buffer:
            continue
        # Use FFmpeg to convert WAV to OGG
        ogg_data, _ = (ffmpeg.input('pipe:0', format='wav').output('pipe:1', format='ogg', acodec='libvorbis').run(
            input=wav_buffer.read(), capture_stdout=True, capture_stderr=True))
        # Now 'ogg_data' contains the OGG audio data
        await bot.send_voice(chat_id=chat_id, caption=sentence, voice=ogg_data)


# Handles all sent documents and audio files
@bot.message_handler(content_types=['voice'])
async def handle_docs_audio(message):
    # Get the file info from the bot
    file_info = await bot.get_file(file_id=message.voice.file_id)
    chat_id = message.chat.id
    session_id = str(chat_id)

    # Download the audio data
    audio_data = await bot.download_file(file_path=file_info.file_path)

    # Use BytesIO to keep the audio data in memory
    audio_buffer = io.BytesIO(audio_data)

    out, _ = (ffmpeg.input('pipe:0').output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar=16000).run(
        input=audio_buffer.read(), capture_stdout=True, capture_stderr=True))
    # Convert WAV bytes to a NumPy array
    audio_np = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
    transcribed_user_voice_msg = chatbot_server.whisper_model.transcribe(audio_np, language=chatbot_server.language,
                                                                         fp16=False)
    if session_id not in chatbot_server.chat_sessions.keys():
        chatbot_server.start_chat(messages=[], set_chat_id_to=session_id)
    chat_session = chatbot_server.chat_sessions[session_id]
    relevant_chatbot_message = Message(role='user', content=transcribed_user_voice_msg['text'])
    chat_session.messages.append(relevant_chatbot_message)
    transcription_quote = format_text(mcode(transcribed_user_voice_msg['text']))
    await bot.reply_to(message=message, text=transcription_quote)
    async for sentence, tts_audio_data in chatbot_server.yield_response_messages(chat_session=chat_session):
        # await bot.reply_to(message=message, text=sentence)
        chat_session.messages.append(Message(role='assistant', content=sentence))

        wav_buffer = chatbot_server.prepare_audio_response_for_client(tts_audio_data=tts_audio_data)
        if not wav_buffer:
            continue

        # Use FFmpeg to convert WAV to OGG
        ogg_data, _ = (ffmpeg.input('pipe:0', format='wav').output('pipe:1', format='ogg', acodec='libvorbis').run(
            input=wav_buffer.read(), capture_stdout=True, capture_stderr=True))

        # Now 'ogg_data' contains the OGG audio data
        await bot.send_voice(chat_id=chat_id, caption=sentence, voice=ogg_data)


if __name__ == '__main__':
    # Run the async main function
    asyncio.run(bot.polling())
