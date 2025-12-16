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
    checkHealth: async () => {
        try {
            const res = await axios.get(`${API_BASE}/health`, { timeout: 2000 });
            return res.status === 200;
        } catch {
            return false;
        }
    },

    generateAudio: async (text: string, gender: string) => {
        const res = await axios.post<TTSResponse>(`${API_BASE}/api/tts`, { text, gender });
        return res.data;
    },

    generateScript: async (topic: string, tone: string = "professional") => {
        const res = await axios.post<{ script: string }>(`${API_BASE}/api/generate-script`, { topic, tone });
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

    processVideo: async (videoFilename: string, audioFilename: string, enhanceFace: boolean = false) => {
        const formData = new FormData();
        formData.append('video_filename', videoFilename);
        formData.append('audio_filename', audioFilename);
        formData.append('enhance_face', enhanceFace.toString());

        const res = await axios.post<ProcessResponse>(`${API_BASE}/api/process`, formData);
        return res.data;
    }
};
