// import React from 'react';

interface Props {
    videoUrl: string;
}

export const ResultViewer = ({ videoUrl }: Props) => {
    return (
        <div className="rounded-xl overflow-hidden bg-black shadow-2xl border border-slate-700">
            <video
                src={videoUrl}
                controls
                autoPlay
                className="w-full h-auto max-h-[600px]"
            />
        </div>
    );
};
