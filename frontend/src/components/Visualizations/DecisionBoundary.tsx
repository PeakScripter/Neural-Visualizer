import { useEffect, useRef } from 'react';
import type { DecisionBoundaryData } from '../../types';

declare const Plotly: any;

interface Props {
  data: DecisionBoundaryData | null;
}

export function DecisionBoundary({ data }: Props) {
  const divRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!divRef.current) return;
    if (!data) {
      // @ts-ignore
      if (window.Plotly) window.Plotly.purge(divRef.current);
      return;
    }

    const Plotly = (window as any).Plotly;
    if (!Plotly) return;

    const { xx, yy, zz, X, y } = data;

    const flatX = xx[0];
    const flatY = yy.map((row: number[]) => row[0]);

    const heatmap = {
      type: 'heatmap',
      x: flatX,
      y: flatY,
      z: zz,
      colorscale: [
        [0, 'rgba(239,68,68,0.4)'],
        [0.5, 'rgba(30,30,50,0.2)'],
        [1, 'rgba(59,130,246,0.4)'],
      ],
      showscale: false,
      hoverinfo: 'none',
    };

    const class0X = X.filter((_, i) => y[i] === 0).map((p: number[]) => p[0]);
    const class0Y = X.filter((_, i) => y[i] === 0).map((p: number[]) => p[1]);
    const class1X = X.filter((_, i) => y[i] === 1).map((p: number[]) => p[0]);
    const class1Y = X.filter((_, i) => y[i] === 1).map((p: number[]) => p[1]);

    const scatter0 = {
      type: 'scatter',
      mode: 'markers',
      x: class0X,
      y: class0Y,
      name: 'Class 0',
      marker: { color: '#ef4444', size: 7, opacity: 0.85, line: { color: '#fff', width: 1 } },
    };

    const scatter1 = {
      type: 'scatter',
      mode: 'markers',
      x: class1X,
      y: class1Y,
      name: 'Class 1',
      marker: { color: '#3b82f6', size: 7, opacity: 0.85, line: { color: '#fff', width: 1 } },
    };

    const layout = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { t: 20, r: 20, b: 40, l: 40 },
      xaxis: {
        gridcolor: '#1f2937',
        zerolinecolor: '#374151',
        tickfont: { color: '#9ca3af', size: 10 },
        title: { text: 'Feature 1', font: { color: '#6b7280', size: 11 } },
      },
      yaxis: {
        gridcolor: '#1f2937',
        zerolinecolor: '#374151',
        tickfont: { color: '#9ca3af', size: 10 },
        title: { text: 'Feature 2', font: { color: '#6b7280', size: 11 } },
      },
      legend: {
        bgcolor: 'rgba(17,24,39,0.8)',
        bordercolor: '#374151',
        borderwidth: 1,
        font: { color: '#d1d5db', size: 11 },
      },
      font: { family: 'Inter, system-ui, sans-serif' },
    };

    Plotly.react(divRef.current, [heatmap, scatter0, scatter1], layout, {
      responsive: true,
      displayModeBar: false,
    });
  }, [data]);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm">
        Run training to see the decision boundary.
      </div>
    );
  }

  return <div ref={divRef} className="w-full h-full" />;
}
