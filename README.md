**My personal english coach**
=============================

A fully self-hosted conversational chatbot built on a customized LLama 3.2 LLM and wrapped with OpenAI Whisper and Coqui-TTS for real-time speech-to-text and text-to-speech conversions.

**Overview**
------------

Enjoy instant, unlimited chat with a personalized conversational agent to help you develop your spoken English.  
It gives you personalized advice to help you improve your conversational skills.  
This chatbot is always available and powered for discussions in English, while benefiting from the general culture of an LLM. 
What's more, it's completely free and keeps no conversation history.     
This self-hosted chatbot integrates OpenAI Whisper for speech recognition and coqui-TTS for text-to-speech, providing excellent performance/resource ratios.  
Speech-to-text (STT) transcribes the user's speech into text.  
The transcribed text is analyzed by the customized LLM: it takes into account the speech-to-text limitations, suggests linguistic corrections and feeds the conversation.  
Chatbot responses are generated in real time using an asynchronous background process that parallelizes the generation of text and audio files.    
The application features a local client in console mode, as well as a web client/server solution containing a lightweight chat room website.  


**Dependencies**
----------------

This application is based on:
* A local Ollama server provided with Llama3.2 ~2Go
* NVIDIA GPU with driver
* Python packages including PyTorch with cuda ~6Go
* OpenAI whisper package with model (base ~150Mo, turbo ~1,6Go)
* Coqui-TTS package with model ~1Go

The whole installation use ~10Go


**Features**
------------

* Conversational audio real-time AI chatbot
* Speech-to-text technology using OpenAI Whisper
* Response generation using a Large Language Model (LLM) customised to act as an english coach
* Asynchronous and paralleled processes for LLM token generation and audio chunks preparation
* Integration with FastAPI framework for self-hosted split client/server

**Installation**
---------------
Install Ollama
Pull Llama3.2:latest
Build customised model
Remove Llama3.2
To build local virtual python environment:
```bash
python -m venv .venv
```
To install the dependencies required for this project, run the following command:
```bash
pip install -r requirements.txt
```
This will install all the necessary packages, including FastAPI, OpenAI Whisper, Coqui-TTS, and other dependencies.

**Running the Chatbot**
----------------------

To run the chatbot using web server, execute the following command:
```bash
cd app
uvicorn main:app --host 127.0.0.1 --port 8080
```
This will start the FastAPI development server on port 8000.

**Usage**
---------

To use the chatbot, simply send a message to the `/speech-to-text` endpoint with your audio input as a multipart/form-data attachment. The chatbot's response will be returned in real-time.

**License**
----------

This project is licensed under the GNU GPLv3 License. Please see the `LICENSE` file in the project root directory for more information.

**Contributing**
---------------

We welcome contributions to this project! If you'd like to contribute, please fork the repository and submit a pull request with your changes.

**Acknowledgments**
------------------

* OpenAI Whisper for speech-to-text technology
* Coqui-TTS for text-to-speech synthesis
* FastAPI for efficient and scalable development