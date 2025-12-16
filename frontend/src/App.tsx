import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AudioGenerator } from './components/AudioGenerator';
import { VideoUploader } from './components/VideoUploader';
import { ResultViewer } from './components/ResultViewer';
import { Activity, CheckCircle, Music, Video, Wand2, Loader2 } from 'lucide-react';
import { api } from './api';

function App() {
  const [step, setStep] = useState<number>(1);
  const [audioFilename, setAudioFilename] = useState<string | null>(null);
  const [videoFilename, setVideoFilename] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [resultFilename, setResultFilename] = useState<string | null>(null);
  const [isServerReady, setIsServerReady] = useState(false);

  // Health Check Poll
  React.useEffect(() => {
    const check = async () => {
      const ready = await api.checkHealth();
      if (ready) setIsServerReady(true);
    };
    check();

    const interval = setInterval(async () => {
      const ready = await api.checkHealth();
      if (ready) {
        setIsServerReady(true);
        clearInterval(interval);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []); // Only on mount

  const handleAudioGenerated = (filename: string) => {
    setAudioFilename(filename);
    setStep(2);
  };

  const handleVideoUploaded = (filename: string) => {
    setVideoFilename(filename);
    setStep(3);
  };

  // Progress Simulation
  React.useEffect(() => {
    let interval: any;
    if (isProcessing) {
      setProgress(0);
      // Simulate progress: Reach 90% in approx 60 seconds
      interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 95) return 95;
          const increment = prev < 50 ? 2 : prev < 80 ? 1 : 0.5;
          return Math.min(prev + increment, 95);
        });
      }, 800);
    } else {
      setProgress(100);
    }
    return () => clearInterval(interval);
  }, [isProcessing]);

  const handleProcess = async () => {
    if (!audioFilename || !videoFilename) return;
    setIsProcessing(true);
    try {
      const res = await api.processVideo(videoFilename, audioFilename);
      setResultFilename(res.output_url); // URL from backend
      setStep(4);
    } catch (e) {
      alert("Processing failed: " + e);
    } finally {
      setIsProcessing(false);
    }
  };

  // SPLASH SCREEN (Server starting)
  if (!isServerReady) {
    return (
      <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center p-4">
        <div className="text-center space-y-6 animate-pulse">
          <div className="flex justify-center mb-4">
            <Activity className="w-20 h-20 text-blue-500" />
          </div>
          <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
            AudioSync Studio
          </h1>
          <p className="text-slate-400 text-xl">AI-Powered Lip Synchronization</p>
          <div className="mt-8 flex items-center justify-center gap-3 text-slate-500">
            <Loader2 className="animate-spin w-5 h-5" />
            <span>Server is starting... (Loading AI Models)</span>
          </div>
        </div>
      </div>
    );
  }

  // MAIN APP
  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center py-10 px-4">
      <header className="mb-10 text-center">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400 flex items-center gap-3 justify-center">
          <Activity className="w-10 h-10 text-blue-400" /> AudioSync Studio
        </h1>
        <p className="text-slate-400 mt-2">AI-Powered Lip Synchronization</p>
      </header>

      {/* Progress Bar */}
      <div className="w-full max-w-2xl bg-slate-800 rounded-full h-2 mb-12 flex overflow-hidden">
        <div className={`h-full bg-blue-500 transition-all duration-500 ${step >= 1 ? 'w-1/4' : 'w-0'}`} />
        <div className={`h-full bg-blue-500 transition-all duration-500 ${step >= 2 ? 'w-1/4' : 'w-0'}`} />
        <div className={`h-full bg-purple-500 transition-all duration-500 ${step >= 3 ? 'w-1/4' : 'w-0'}`} />
        <div className={`h-full bg-emerald-500 transition-all duration-500 ${step >= 4 ? 'w-1/4' : 'w-0'}`} />
      </div>

      <main className="w-full max-w-4xl bg-slate-800/50 backdrop-blur-lg rounded-2xl p-8 border border-slate-700 shadow-2xl relative min-h-[500px]">
        <AnimatePresence mode="wait">

          {step === 1 && (
            <motion.div key="step1" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }}>
              <StepHeader icon={<Music />} title="Step 1: Generate Audio" />
              <AudioGenerator onGenerated={handleAudioGenerated} />
            </motion.div>
          )}

          {step === 2 && (
            <motion.div key="step2" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }}>
              <StepHeader icon={<Video />} title="Step 2: Upload Video" />
              <VideoUploader onUploaded={handleVideoUploaded} />
              <button onClick={() => setStep(1)} className="text-sm text-slate-500 mt-4 hover:text-white">Back to Audio</button>
            </motion.div>
          )}

          {step === 3 && (
            <motion.div key="step3" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }} className="flex flex-col items-center text-center justify-center h-full pt-10">
              <StepHeader icon={<Wand2 />} title="Step 3: Process" />
              <div className="flex gap-10 mb-8">
                <div className="p-4 bg-slate-900 rounded border border-slate-600">
                  <p className="text-xs text-slate-400 mb-1">Audio</p>
                  <p className="font-mono text-blue-300">{audioFilename}</p>
                </div>
                <div className="p-4 bg-slate-900 rounded border border-slate-600">
                  <p className="text-xs text-slate-400 mb-1">Video</p>
                  <p className="font-mono text-purple-300">{videoFilename}</p>
                </div>
              </div>

              <button
                onClick={handleProcess}
                disabled={isProcessing}
                className="group relative px-8 py-4 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-lg shadow-lg hover:shadow-blue-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed mb-6"
              >
                {isProcessing ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="animate-spin h-5 w-5 text-white" />
                    Processing...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">Start Lip-Sync <Wand2 className="w-5 h-5" /></span>
                )}
              </button>

              {isProcessing && (
                <div className="w-full max-w-md space-y-2">
                  <div className="flex justify-between text-xs text-slate-400">
                    <span>Generating Video...</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-blue-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                  <p className="text-xs text-slate-500 animate-pulse">This may take 1-2 minutes depending on GPU...</p>
                </div>
              )}
            </motion.div>
          )}

          {step === 4 && (
            <motion.div key="step4" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}>
              <StepHeader icon={<CheckCircle />} title="Result" />
              <ResultViewer videoUrl={`http://localhost:8000${resultFilename}`} />
              <div className="flex justify-center mt-6">
                <button onClick={() => window.location.reload()} className="px-6 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm transition-colors">Start Over</button>
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </main>
    </div>
  );
}

const StepHeader = ({ icon, title }: { icon: any, title: string }) => (
  <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3 border-b border-slate-700 pb-4">
    <span className="p-2 bg-slate-800 rounded-lg text-blue-400">{icon}</span>
    {title}
  </h2>
)

export default App;
