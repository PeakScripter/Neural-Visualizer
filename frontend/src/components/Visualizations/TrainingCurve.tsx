import { useEffect, useRef } from 'react';
import type { TrainingResult } from '../../types';

interface Props {
  data: TrainingResult | null;
}

export function TrainingCurve({ data }: Props) {
  const divRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!divRef.current) return;
    if (!data) return;

    const Plotly = (window as any).Plotly;
    if (!Plotly) return;

    const lossTrace = {
      type: 'scatter',
      mode: 'lines+markers',
      x: data.epochs,
      y: data.loss_history,
      name: 'Loss',
      line: { color: '#ef4444', width: 2.5, shape: 'spline', smoothing: 1.2 },
      marker: { color: '#ef4444', size: 6, symbol: 'circle' },
      fill: 'tozeroy',
      fillcolor: 'rgba(239,68,68,0.08)',
    };

    const accTrace = {
      type: 'scatter',
      mode: 'lines+markers',
      x: data.epochs,
      y: data.accuracy_history,
      name: 'Accuracy',
      yaxis: 'y2',
      line: { color: '#10b981', width: 2.5, shape: 'spline', smoothing: 1.2 },
      marker: { color: '#10b981', size: 6, symbol: 'diamond' },
    };

    const layout = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { t: 30, r: 60, b: 50, l: 50 },
      xaxis: {
        title: { text: 'Epoch', font: { color: '#6b7280', size: 12 } },
        tickfont: { color: '#9ca3af', size: 10 },
        gridcolor: '#1f2937',
        zerolinecolor: '#374151',
      },
      yaxis: {
        title: { text: 'Loss', font: { color: '#ef4444', size: 12 } },
        tickfont: { color: '#9ca3af', size: 10 },
        gridcolor: '#1f2937',
        zerolinecolor: '#374151',
        rangemode: 'tozero',
      },
      yaxis2: {
        title: { text: 'Accuracy', font: { color: '#10b981', size: 12 } },
        tickfont: { color: '#9ca3af', size: 10 },
        overlaying: 'y',
        side: 'right',
        range: [0, 1.05],
        showgrid: false,
      },
      legend: {
        bgcolor: 'rgba(17,24,39,0.8)',
        bordercolor: '#374151',
        borderwidth: 1,
        font: { color: '#d1d5db', size: 11 },
        x: 0.5,
        xanchor: 'center',
        y: 1.05,
        orientation: 'h',
      },
      font: { family: 'Inter, system-ui, sans-serif' },
      hovermode: 'x unified',
      hoverlabel: {
        bgcolor: '#111827',
        bordercolor: '#374151',
        font: { color: '#e2e8f0', size: 12 },
      },
    };

    Plotly.react(divRef.current, [lossTrace, accTrace], layout, {
      responsive: true,
      displayModeBar: false,
    });
  }, [data]);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm">
        Run training to see the learning curves.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Summary stats */}
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: 'Final Loss', value: data.loss_history.at(-1)?.toFixed(4), color: 'text-red-400' },
          { label: 'Final Acc', value: `${((data.accuracy_history.at(-1) ?? 0) * 100).toFixed(1)}%`, color: 'text-green-400' },
          { label: 'Best Loss', value: Math.min(...data.loss_history).toFixed(4), color: 'text-blue-400' },
          { label: 'Best Acc', value: `${(Math.max(...data.accuracy_history) * 100).toFixed(1)}%`, color: 'text-purple-400' },
        ].map((stat) => (
          <div key={stat.label} className="bg-gray-800/60 rounded-lg p-2.5 text-center border border-gray-700/50">
            <div className={`text-base font-bold ${stat.color}`}>{stat.value}</div>
            <div className="text-xs text-gray-500 mt-0.5">{stat.label}</div>
          </div>
        ))}
      </div>
      <div ref={divRef} className="flex-1" />
    </div>
  );
}
