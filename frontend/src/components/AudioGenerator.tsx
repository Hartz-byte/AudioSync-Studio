import React, { useState } from 'react';
import { api } from '../api';
import { Play, Loader2, User } from 'lucide-react';

interface Props {
    onGenerated: (filename: string) => void;
}

export const AudioGenerator = ({ onGenerated }: Props) => {
    const [text, setText] = useState("Hello! Welcome to AudioSync Studio.");
    const [isLoading, setIsLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [gender, setGender] = useState<string | null>(null);

    React.useEffect(() => {
        let interval: any;
        if (isLoading) {
            setProgress(0);
            interval = setInterval(() => {
                setProgress(prev => Math.min(prev + 10, 90));
            }, 100);
        } else {
            setProgress(100);
        }
        return () => clearInterval(interval);
    }, [isLoading]);

    const handleGenerate = async () => {
        if (!gender) return;
        setIsLoading(true);
        try {
            const res = await api.generateAudio(text, gender);
            // Pass filename to parent
            onGenerated(res.filename);
        } catch (e) {
            alert("Error generating audio");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">1. Enter Text for Speech</label>
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    className="w-full h-32 bg-slate-900 border border-slate-700 rounded-lg p-4 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none"
                    placeholder="Type what you want the avatar to say..."
                />
            </div>

            <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">2. Select Voice Gender</label>
                <div className="flex gap-4">
                    <button
                        onClick={() => setGender('male')}
                        className={`flex-1 p-4 rounded-xl border flex items-center justify-center gap-3 transition-all ${gender === 'male'
                            ? 'bg-blue-600/20 border-blue-500 text-blue-200 shadow-lg shadow-blue-900/20'
                            : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-750 hover:border-slate-600'
                            }`}
                    >
                        <User className="w-5 h-5" /> Male (David)
                    </button>
                    <button
                        onClick={() => setGender('female')}
                        className={`flex-1 p-4 rounded-xl border flex items-center justify-center gap-3 transition-all ${gender === 'female'
                            ? 'bg-pink-600/20 border-pink-500 text-pink-200 shadow-lg shadow-pink-900/20'
                            : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-750 hover:border-slate-600'
                            }`}
                    >
                        <User className="w-5 h-5" /> Female (Zira)
                    </button>
                </div>
            </div>

            <div className="flex flex-col items-end gap-3 pt-4">
                <button
                    onClick={handleGenerate}
                    disabled={isLoading || !text.trim() || !gender}
                    className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-500"
                >
                    {isLoading ? <Loader2 className="animate-spin w-4 h-4" /> : <Play className="w-4 h-4" />}
                    Generate Audio
                </button>

                {isLoading && (
                    <div className="w-full h-1 bg-slate-700 rounded-full overflow-hidden max-w-[200px]">
                        <div
                            className="h-full bg-blue-500 transition-all duration-300 ease-out"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                )}
            </div>
        </div>
    );
};
