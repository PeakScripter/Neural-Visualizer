import { useState, useMemo } from 'react';
import { Scissors } from 'lucide-react';
import { NetworkGraph } from './NetworkGraph';
import type { NetworkGraph as GraphData } from '../../types';

interface Props { graph: GraphData }

export function PruningView({ graph }: Props) {
  const [threshold, setThreshold] = useState(0.1);

  // Nodes whose |value| is below threshold are "pruned"
  const { pruned, prunedGraph, stats } = useMemo(() => {
    const prunedIds = new Set(
      graph.nodes.filter(n => n.layer_type !== 'input' && n.layer_type !== 'output' && Math.abs(n.value ?? 0) < threshold).map(n => n.id)
    );
    const prunedEdges = graph.edges.filter(e => !prunedIds.has(e.source) && !prunedIds.has(e.target));
    const prunedNodes = graph.nodes.map(n => ({
      ...n,
      value: prunedIds.has(n.id) ? 0 : n.value,
    }));
    return {
      pruned: prunedIds,
      prunedGraph: { nodes: prunedNodes, edges: prunedEdges },
      stats: {
        removedNodes: prunedIds.size,
        removedEdges: graph.edges.length - prunedEdges.length,
        compressionRatio: graph.edges.length > 0
          ? ((graph.edges.length - prunedEdges.length) / graph.edges.length * 100).toFixed(1)
          : '0',
      },
    };
  }, [graph, threshold]);

  // Active = surviving nodes, inactive = pruned
  const activeNodeIds = useMemo(
    () => new Set(graph.nodes.filter(n => !pruned.has(n.id)).map(n => n.id)),
    [graph.nodes, pruned]
  );

  if (!graph.nodes.length) {
    return (
      <div className="flex items-center justify-center h-full text-sm" style={{ color: 'var(--text-muted)' }}>
        Build a network to visualize pruning.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <Scissors size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
          <label className="text-xs font-medium uppercase tracking-wide whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
            Prune threshold
          </label>
          <input
            type="range" min={0} max={1} step={0.01}
            value={threshold}
            onChange={e => setThreshold(+e.target.value)}
            className="flex-1"
          />
          <span className="text-xs font-bold w-10 text-right" style={{ color: 'var(--accent)' }}>
            {threshold.toFixed(2)}
          </span>
        </div>

        {/* Stats pills */}
        {[
          { label: 'Pruned neurons', value: stats.removedNodes, color: '#ef4444' },
          { label: 'Pruned edges',   value: stats.removedEdges, color: '#f59e0b' },
          { label: 'Compression',    value: `${stats.compressionRatio}%`, color: '#10b981' },
          { label: 'Surviving',      value: graph.nodes.length - stats.removedNodes, color: '#3b82f6' },
        ].map(s => (
          <div key={s.label} className="rounded-lg px-3 py-1.5 border text-center" style={{ background: s.color + '12', borderColor: s.color + '44' }}>
            <div className="text-sm font-bold" style={{ color: s.color }}>{s.value}</div>
            <div className="text-xs" style={{ color: 'var(--text-faint)' }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Explanation */}
      <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
        Neurons with |activation| &lt; {threshold.toFixed(2)} are pruned (greyed out). Their incoming/outgoing edges are removed.
        Input and output neurons are never pruned.
      </p>

      {/* Graph — use the pruned graph but highlight surviving nodes */}
      <div className="flex-1 min-h-0 rounded-xl overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
        <NetworkGraph
          graph={prunedGraph}
          activeNodeIds={activeNodeIds}
          mode="forward"
        />
      </div>
    </div>
  );
}
