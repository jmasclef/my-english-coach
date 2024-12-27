// processor.js
import { API_HOST } from '/host.js'; // Import API host

export class AudioProcessor {
    constructor() {
        this.audioChunks = [];
        this.mediaRecorder = null;
        this.audioContext = null; // AudioContext is created during recording
    }

    async startRecording() {
        if (!this.audioContext) {
            // Create and resume the AudioContext during the first recording start
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            await this.audioContext.resume();
        }

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.mediaRecorder = new MediaRecorder(stream);

        this.audioChunks = [];
        this.mediaRecorder.ondataavailable = event => {
            this.audioChunks.push(event.data);
        };

        this.mediaRecorder.start();
    }

    async stopRecording() {
        return new Promise(resolve => {
            this.mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                const audioBuffer = await this.decodeAudioBlob(audioBlob);
                const resampledBuffer = await this.resampleAudio(audioBuffer, 16000);
                const wavBlob = this.exportToWAV(resampledBuffer);

                resolve(wavBlob);
            };

            this.mediaRecorder.stop();
        });
    }

    async decodeAudioBlob(audioBlob) {
        const arrayBuffer = await audioBlob.arrayBuffer();
        return await this.audioContext.decodeAudioData(arrayBuffer);
    }

    async resampleAudio(audioBuffer, targetSampleRate) {
        const offlineContext = new OfflineAudioContext(
            audioBuffer.numberOfChannels,
            (audioBuffer.duration * targetSampleRate),
            targetSampleRate
        );

        const source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start(0);

        return await offlineContext.startRendering();
    }

    exportToWAV(audioBuffer) {
        const numOfChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const length = audioBuffer.length * numOfChannels * 2 + 44; // WAV Header size
        const buffer = new ArrayBuffer(length);
        const view = new DataView(buffer);

        this.writeUTFBytes(view, 0, 'RIFF');
        view.setUint32(4, length - 8, true);
        this.writeUTFBytes(view, 8, 'WAVE');
        this.writeUTFBytes(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numOfChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numOfChannels * 2, true);
        view.setUint16(32, numOfChannels * 2, true);
        view.setUint16(34, 16, true);
        this.writeUTFBytes(view, 36, 'data');
        view.setUint32(40, length - 44, true);

        let offset = 44;
        for (let i = 0; i < audioBuffer.length; i++) {
            for (let channel = 0; channel < numOfChannels; channel++) {
                const sample = audioBuffer.getChannelData(channel)[i];
                const intSample = Math.max(-1, Math.min(1, sample)) * 0x7FFF;
                view.setInt16(offset, intSample, true);
                offset += 2;
            }
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    writeUTFBytes(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

async sendWAVToServer(wavBlob, chatHistory, endpoint = '/speech-to-text') {
    const formData = new FormData();

    // Append the audio file (WAV)
    formData.append('sound_file', wavBlob, 'audio.wav'); // This works as before

    // Serialize chatHistory to JSON before appending
    formData.append('messages', JSON.stringify(chatHistory)); // Serialize chatHistory to JSON string

    try {
        const response = await fetch(`${API_HOST}${endpoint}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Failed to send audio to server: ${response.statusText}`);
        }

        return await response.json(); // Return the server response as a JSON object
    } catch (error) {
        console.error('Error during fetch:', error);
        throw error; // Rethrow the error after logging it
    }
}

}
