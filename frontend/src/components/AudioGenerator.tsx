import React, { useState } from 'react';
import { api } from '../api';
import { Play, Loader2, User, Sparkles, PenTool, Edit3, Mic } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
    onGenerated: (filename: string) => void;
}

export const AudioGenerator = ({ onGenerated }: Props) => {
    const [mode, setMode] = useState<'manual' | 'ai' | 'clone'>('manual');
    const [text, setText] = useState("Hello! Welcome to AudioSync Studio.");

    // AI Script State
    const [topic, setTopic] = useState("");
    const [tone, setTone] = useState("professional");
    const [isGeneratingScript, setIsGeneratingScript] = useState(false);

    // Clone State
    const [cloneFile, setCloneFile] = useState<File | null>(null);

    // Audio Gen State
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

    const handleGenerateScript = async () => {
        if (!topic.trim()) return;
        setIsGeneratingScript(true);
        try {
            const res = await api.generateScript(topic, tone);
            setText(res.script);
            setMode('manual'); // Switch to manual to review/edit
        } catch (e) {
            alert("Error generating script: " + e);
        } finally {
            setIsGeneratingScript(false);
        }
    };

    const handleGenerateAudio = async () => {
        setIsLoading(true);
        try {
            if (mode === 'clone') {
                if (!cloneFile) return;
                const res = await api.cloneVoice(text, cloneFile);
                onGenerated(res.filename);
            } else {
                if (!gender) return;
                const res = await api.generateAudio(text, gender);
                onGenerated(res.filename);
            }
        } catch (e) {
            alert("Error generating audio");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            {/* Mode Selection Tabs */}
            <div className="flex bg-slate-900 p-1 rounded-lg gap-1">
                {/* Manual */}
                <button
                    onClick={() => setMode('manual')}
                    className={`flex-1 py-2 px-2 rounded-md text-sm font-medium transition-all flex items-center justify-center gap-2 ${mode === 'manual' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-white'}`}
                >
                    <Edit3 className="w-4 h-4" /> Manual
                </button>
                {/* AI Script */}
                <button
                    onClick={() => setMode('ai')}
                    className={`flex-1 py-2 px-2 rounded-md text-sm font-medium transition-all flex items-center justify-center gap-2 ${mode === 'ai' ? 'bg-indigo-600 text-white shadow shadow-indigo-500/20' : 'text-slate-400 hover:text-white'}`}
                >
                    <Sparkles className="w-4 h-4" /> AI Script
                </button>
                {/* Voice Clone */}
                <button
                    onClick={() => setMode('clone')}
                    className={`flex-1 py-2 px-2 rounded-md text-sm font-medium transition-all flex items-center justify-center gap-2 ${mode === 'clone' ? 'bg-emerald-600 text-white shadow shadow-emerald-500/20' : 'text-slate-400 hover:text-white'}`}
                >
                    <Mic className="w-4 h-4" /> Voice Clone
                </button>
            </div>

            <div className="min-h-[200px]">
                <AnimatePresence mode="wait">
                    {mode === 'manual' || mode === 'clone' ? (
                        <motion.div
                            key={mode}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                        >
                            <label className="block text-sm font-medium text-slate-400 mb-2">
                                {mode === 'clone' ? '1. Enter Script for Cloning' : '1. Enter or Edit Script'}
                            </label>
                            <textarea
                                value={text}
                                onChange={(e) => setText(e.target.value)}
                                className="w-full h-40 bg-slate-900 border border-slate-700 rounded-lg p-4 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none font-sans leading-relaxed"
                                placeholder="Type what you want the avatar to say..."
                            />
                        </motion.div>
                    ) : (
                        <motion.div
                            key="ai"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="space-y-4"
                        >
                            {/* AI Script UI (same as before) */}
                            <div>
                                <label className="block text-sm font-medium text-slate-400 mb-2">Topic / Context</label>
                                <input
                                    type="text"
                                    value={topic}
                                    onChange={(e) => setTopic(e.target.value)}
                                    className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                                    placeholder="e.g. Sales pitch for organic coffee"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-slate-400 mb-2">Tone</label>
                                <div className="flex gap-2">
                                    {['Professional', 'Casual', 'Energetic', 'Funny'].map(t => (
                                        <button
                                            key={t}
                                            onClick={() => setTone(t.toLowerCase())}
                                            className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${tone === t.toLowerCase()
                                                ? 'bg-indigo-600 border-indigo-500 text-white'
                                                : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500'
                                                }`}
                                        >
                                            {t}
                                        </button>
                                    ))}
                                </div>
                            </div>
                            <button
                                onClick={handleGenerateScript}
                                disabled={isGeneratingScript || !topic.trim()}
                                className="w-full py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 rounded-lg font-bold text-white shadow-lg shadow-indigo-900/20 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                            >
                                {isGeneratingScript ? <Loader2 className="animate-spin w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
                                Generate Script with AI
                            </button>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            <div className="pt-4 border-t border-slate-700/50">
                {mode === 'clone' ? (
                    <div className="mb-6">
                        <label className="block text-sm font-medium text-emerald-400 mb-3">2. Upload Reference Voice (3-10s wav/mp3)</label>
                        <div className="border-2 border-dashed border-emerald-500/30 rounded-xl p-8 bg-emerald-900/10 hover:bg-emerald-900/20 transition-colors text-center cursor-pointer relative">
                            <input
                                type="file"
                                accept="audio/*"
                                onChange={(e) => setCloneFile(e.target.files?.[0] || null)}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                            />
                            <div className="flex flex-col items-center gap-2">
                                <Mic className="w-8 h-8 text-emerald-400" />
                                <p className="text-emerald-200 font-medium">
                                    {cloneFile ? cloneFile.name : "Click to Upload Sample"}
                                </p>
                                <p className="text-emerald-500/70 text-xs">
                                    {cloneFile ? "Ready for cloning" : "Must be clear speech (wav, mp3, m4a)"}
                                </p>
                            </div>
                        </div>
                    </div>
                ) : (
                    <>
                        <label className="block text-sm font-medium text-slate-400 mb-3">2. Select Voice Gender</label>
                        <div className="flex gap-4 mb-6">
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
                    </>
                )}

                <div className="flex flex-col items-end gap-3">
                    <button
                        onClick={handleGenerateAudio}
                        disabled={isLoading || !text.trim() || (mode === 'clone' ? !cloneFile : !gender)}
                        className={`px-8 py-3 rounded-lg font-medium transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg 
                            ${mode === 'clone'
                                ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/20'
                                : 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-900/20'}`}
                    >
                        {isLoading ? <Loader2 className="animate-spin w-5 h-5" /> : (mode === 'clone' ? <Mic className="w-5 h-5" /> : <Play className="w-5 h-5" />)}
                        {mode === 'clone' ? 'Clone & Generate' : 'Generate Audio'}
                    </button>

                    {isLoading && (
                        <div className="w-full h-1 bg-slate-700 rounded-full overflow-hidden max-w-[200px]">
                            <div
                                className={`h-full transition-all duration-300 ease-out ${mode === 'clone' ? 'bg-emerald-500' : 'bg-blue-500'}`}
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
