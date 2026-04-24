import { useEffect, useRef } from 'react';
import type { LossLandscapeData } from '../../types';

interface Props {
  data: LossLandscapeData | null;
}

export function LossLandscape({ data }: Props) {
  const divRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!divRef.current) return;
    if (!data) return;

    const Plotly = (window as any).Plotly;
    if (!Plotly) return;

    const surface = {
      type: 'surface',
      x: data.w1,
      y: data.w2,
      z: data.loss,
      colorscale: [
        [0, '#1a1f35'],
        [0.25, '#1e40af'],
        [0.5, '#7c3aed'],
        [0.75, '#db2777'],
        [1, '#ef4444'],
      ],
      contours: {
        z: { show: true, usecolormap: true, highlightcolor: '#fff', project: { z: true } },
      },
      opacity: 0.9,
      showscale: true,
      colorbar: {
        tickfont: { color: '#9ca3af', size: 9 },
        bgcolor: 'rgba(0,0,0,0)',
        bordercolor: '#374151',
        title: { text: 'Loss', font: { color: '#9ca3af', size: 11 } },
      },
    };

    const layout = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      scene: {
        bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
          title: { text: 'Weight 1', font: { color: '#9ca3af', size: 11 } },
          tickfont: { color: '#9ca3af', size: 9 },
          gridcolor: '#1f2937',
          backgroundcolor: 'rgba(0,0,0,0)',
        },
        yaxis: {
          title: { text: 'Weight 2', font: { color: '#9ca3af', size: 11 } },
          tickfont: { color: '#9ca3af', size: 9 },
          gridcolor: '#1f2937',
          backgroundcolor: 'rgba(0,0,0,0)',
        },
        zaxis: {
          title: { text: 'Loss', font: { color: '#9ca3af', size: 11 } },
          tickfont: { color: '#9ca3af', size: 9 },
          gridcolor: '#1f2937',
          backgroundcolor: 'rgba(0,0,0,0)',
        },
        camera: { eye: { x: 1.4, y: 1.4, z: 1.0 } },
      },
      margin: { t: 20, r: 20, b: 20, l: 20 },
      font: { family: 'Inter, system-ui, sans-serif' },
    };

    Plotly.react(divRef.current, [surface], layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
    });
  }, [data]);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm">
        Build the network to see the loss landscape.
      </div>
    );
  }

  return <div ref={divRef} className="w-full h-full" />;
}
