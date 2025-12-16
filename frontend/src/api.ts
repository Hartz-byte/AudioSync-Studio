import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export interface TTSResponse {
    status: string;
    audio_url: string;
    filename: string;
    path: string;
}

export interface UploadResponse {
    status: string;
    filename: string;
    path: string;
    url: string;
}

export interface ProcessResponse {
    status: string;
    output_url: string;
    filename: string;
}

export const api = {
    generateAudio: async (text: string, voice: string = "pyttsx3") => {
        const res = await axios.post<TTSResponse>(`${API_BASE}/api/tts`, { text, voice });
        return res.data;
    },

    uploadVideo: async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        const res = await axios.post<UploadResponse>(`${API_BASE}/api/upload`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return res.data;
    },

    processVideo: async (videoFilename: string, audioFilename: string) => {
        const formData = new FormData();
        formData.append('video_filename', videoFilename);
        formData.append('audio_filename', audioFilename);
        const res = await axios.post<ProcessResponse>(`${API_BASE}/api/process`, formData);
        return res.data;
    }
};
