import React, { useState, useRef } from 'react';
import { api } from '../api';
import { Upload, Loader2 } from 'lucide-react';

interface Props {
    onUploaded: (filename: string) => void;
}

export const VideoUploader = ({ onUploaded }: Props) => {
    const [isUploading, setIsUploading] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsUploading(true);
        try {
            const res = await api.uploadVideo(file);
            onUploaded(res.filename);
        } catch (e) {
            alert("Upload failed");
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="flex flex-col items-center justify-center p-10 border-2 border-dashed border-slate-700 rounded-xl bg-slate-800/50 hover:bg-slate-800 transition-colors cursor-pointer group"
            onClick={() => fileInputRef.current?.click()}>

            <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="video/mp4,video/avi"
                onChange={handleFileChange}
            />

            <div className="w-16 h-16 bg-slate-700 rounded-full flex items-center justify-center mb-4 group-hover:bg-blue-600 transition-colors">
                {isUploading ? (
                    <Loader2 className="w-8 h-8 text-white animate-spin" />
                ) : (
                    <Upload className="w-8 h-8 text-white" />
                )}
            </div>

            <h3 className="text-lg font-medium text-white mb-2">Upload Reference Face Video</h3>
            <p className="text-slate-400 text-sm">Click to browse or drag and drop MP4 file</p>
        </div>
    );
};
