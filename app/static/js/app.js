import { AudioProcessor } from './processor.js';
import { API_HOST } from '/host.js';

const recordBtn = document.getElementById("recordBtn");
const messagesDiv = document.getElementById("messages");
const processingMessage = document.getElementById("processingMessage");
const errorMessageDiv = document.getElementById("errorMessage");

const audioProcessor = new AudioProcessor();
let recording = false;
let chatHistory = [];
let audioQueue = Promise.resolve(); // Queue to serialize audio playback

recordBtn.addEventListener("click", async () => {
    if (!recording) {
        await audioProcessor.startRecording();
        recording = true;
        updateButtonState(recording); // Update to recording state
    } else {
        const wavBlob = await audioProcessor.stopRecording();
        recording = false;
        updateButtonState(recording); // Update to idle state

        try {
            // Send the audio to the server for transcription
            const response = await audioProcessor.sendWAVToServer(wavBlob, chatHistory);
            const { text } = response;

            if (!text) {
                console.error("No transcription text returned");
                return;
            }

            addMessage("user", text);

            // Add the user's question to the chat history
            chatHistory.push({ role: "user", content: text });

            // Start the chat session
            await startChat(chatHistory);
        } catch (error) {
            console.error("Error during transcription or chat start:", error);
            displayErrorMessage("Error processing your request.");
        }
    }
});

document.addEventListener("DOMContentLoaded", () => {
    updateButtonState(false); // Set to idle state on load
});

// Function to update button state
function updateButtonState(recording) {
    const recordBtn = document.getElementById("recordBtn");
    if (recording) {
        recordBtn.classList.add("recording");
        recordBtn.classList.remove("processing");
        recordBtn.innerHTML = "&#x1F534;"; // Stop recording icon
    } else {
        recordBtn.classList.remove("recording");
        recordBtn.classList.remove("processing");
        recordBtn.innerHTML = "&#x1F399;"; // Microphone icon
    }
}

// Function to update to the processing state (spinner)
function updateButtonToProcessingState() {
    const recordBtn = document.getElementById("recordBtn");
    recordBtn.classList.add("processing");
    recordBtn.classList.remove("recording");
    recordBtn.innerHTML = `<div class="spinner"></div>`; // Show spinner
}

async function startChat(messages) {
    try {
        const response = await fetch(`${API_HOST}/start_chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(messages), // Send messages array directly
        });

        if (!response.ok) {
            throw new Error(`Failed to start chat: ${response.statusText}`);
        }

        const { session_id } = await response.json();

        if (!session_id) {
            console.error("No session_id returned from server");
            return;
        }

        // Show the processing state on the button
        updateButtonToProcessingState();

        // Display the "Processing..." message during chat streaming
        toggleProcessingMessage(true);

        // Call streamChat to begin receiving and processing chunks
        await streamChat(session_id);

        // Hide the "Processing..." message once done
        toggleProcessingMessage(false);

        // Reset the button state to idle
        updateButtonState(false);
    } catch (error) {
        console.error("Error starting chat:", error);
        displayErrorMessage("Error starting chat.");
        updateButtonState(false); // Reset to idle state if error occurs
    }
}

async function streamChat(sessionId) {
    try {
        let fullResponse = "";
        const audioContext = new AudioContext();
        let audioChunks = []; // Array to hold audio chunks for queued playback

        let firstChunkPlayed = false; // Flag to track if the first chunk has started playing

        while (true) {
            const response = await fetch(`${API_HOST}/stream_chat?session_id=${encodeURIComponent(sessionId)}`, {
                method: "GET",
            });

            if (!response.ok) {
                throw new Error(`Failed to stream chat: ${response.statusText}`);
            }

            const data = await response.json();
            const { status, text_chunk, audio_chunk } = data;

            if (status === "NOT_READY") {
                await new Promise((resolve) => setTimeout(resolve, 200)); // Wait before retrying
                continue;
            }

            if (status === "STREAM") {
                console.log("Received new message");
                if (text_chunk) {
                    fullResponse += text_chunk;
                    addMessage("assistant", text_chunk); // Display text message immediately
                    console.log("Received text:"+text_chunk);
                }
                else
                {
                    console.error("Received no text from message");
                }
                if (audio_chunk) {
                    // If it's the first audio chunk, start playback immediately
                    if (!firstChunkPlayed) {
                        firstChunkPlayed = true;
                        await playAudioChunk(audio_chunk, audioContext); // Start playback immediately
                    } else {
                        // Enqueue subsequent audio chunks for playback
                        audioChunks.push(audio_chunk);
                    }
                }
            } else if (status === "FINISHED") {
                chatHistory.push({ role: "assistant", content: fullResponse });
                break;
            }
        }

        // After receiving all chunks, continue to play the queued audio chunks
        for (const chunk of audioChunks) {
            await playAudioChunk(chunk, audioContext);
        }
    } catch (error) {
        console.error("Error initializing chat stream:", error);
    }
}

// Function to play audio chunk (plays and continues with the next after the current one finishes)
function playAudioChunk(base64Audio, audioContext) {
    return new Promise((resolve, reject) => {
        // Decode the audio chunk from base64
        const binaryAudio = atob(base64Audio); // Decode base64 to binary string
        const buffer = new Uint8Array(binaryAudio.length);

        for (let i = 0; i < binaryAudio.length; i++) {
            buffer[i] = binaryAudio.charCodeAt(i);
        }

        // Decode the audio data
        audioContext.decodeAudioData(buffer.buffer, (decodedData) => {
            console.log("Decoded audio duration:", decodedData.duration); // Log duration to debug

            const source = audioContext.createBufferSource();
            source.buffer = decodedData;
            source.connect(audioContext.destination);
            source.start(0); // Start the audio immediately

            // Resolve when audio has finished playing
            source.onended = () => {
                console.log("Audio chunk finished.");
                resolve();
            };
        }, (error) => {
            console.error("Error decoding audio:", error);
            reject(error); // Reject if there is an error decoding the audio
        });
    });
}

function addMessage(role, content) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;
    messageDiv.textContent = content;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to the latest message
}

// Function to display an error message
function displayErrorMessage(message) {
    const errorMessageDiv = document.createElement('div');
    errorMessageDiv.classList.add('error-message');
    errorMessageDiv.textContent = message;
    document.body.appendChild(errorMessageDiv); // Or append it to a specific container

    // Optional: Style the error message (You can define styles for `.error-message` in CSS)
    errorMessageDiv.style.backgroundColor = "#f44336"; // Red background for error
    errorMessageDiv.style.color = "white";
    errorMessageDiv.style.padding = "10px";
    errorMessageDiv.style.position = "absolute";
    errorMessageDiv.style.top = "20px";
    errorMessageDiv.style.left = "50%";
    errorMessageDiv.style.transform = "translateX(-50%)";
    errorMessageDiv.style.zIndex = "1000"; // Ensure it's on top
}

// Function to show/hide the "Processing..." message
function toggleProcessingMessage(show) {
    if (processingMessage) {
        processingMessage.style.display = show ? 'block' : 'none';
    }
}
