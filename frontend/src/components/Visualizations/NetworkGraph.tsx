import { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import type { NetworkGraph as GraphData, NetworkNode } from '../../types';

const LAYER_TYPE_COLORS: Record<string, string> = {
  input: '#3b82f6',
  hidden: '#10b981',
  output: '#ef4444',
  conv: '#8b5cf6',
  fc: '#6366f1',
  rnn: '#f59e0b',
  lstm: '#ec4899',
  generator: '#14b8a6',
  discriminator: '#f97316',
  gen_output: '#06b6d4',
  latent: '#a78bfa',
  attention: '#fb923c',
  embedding: '#818cf8',
  feedforward: '#34d399',
  encoder: '#2dd4bf',
  decoder: '#fb7185',
  bottleneck: '#c084fc',
};

interface Props {
  graph: GraphData;
  activeNodeIds?: Set<number>;
  activeEdgeIds?: Set<number>;
  gradients?: Record<string, number>;
  mode?: 'architecture' | 'forward' | 'backward';
}

export function NetworkGraph({ graph, activeNodeIds, activeEdgeIds, gradients, mode = 'architecture' }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Compute scale to fit all nodes
  const { minX, maxX, minY, maxY } = useMemo(() => {
    if (!graph.nodes.length) return { minX: 0, maxX: 10, minY: -5, maxY: 5 };
    const xs = graph.nodes.map((n) => n.x);
    const ys = graph.nodes.map((n) => n.y);
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
  }, [graph.nodes]);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !graph.nodes.length) return;

    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 500;
    const pad = 60;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    // Defs for glow filters and arrowheads
    const defs = svg.append('defs');

    const glowFilter = defs.append('filter').attr('id', 'glow').attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    glowFilter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
    const feMerge = glowFilter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    const activeGlow = defs.append('filter').attr('id', 'activeGlow').attr('x', '-100%').attr('y', '-100%').attr('width', '300%').attr('height', '300%');
    activeGlow.append('feGaussianBlur').attr('stdDeviation', '5').attr('result', 'coloredBlur');
    const feMerge2 = activeGlow.append('feMerge');
    feMerge2.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge2.append('feMergeNode').attr('in', 'SourceGraphic');

    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 10)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#4b5563');

    // Scales
    const xScale = d3.scaleLinear().domain([minX, maxX]).range([pad, width - pad]);
    const yScale = d3.scaleLinear().domain([minY, maxY]).range([height - pad, pad]);

    const nodeById = new Map<number, NetworkNode>(graph.nodes.map((n) => [n.id, n]));

    // Background grid
    const gridG = svg.append('g').attr('class', 'grid').attr('opacity', 0.15);
    const gridSpacing = 40;
    for (let x = 0; x < width; x += gridSpacing) {
      gridG.append('line').attr('x1', x).attr('y1', 0).attr('x2', x).attr('y2', height).attr('stroke', '#374151').attr('stroke-width', 0.5);
    }
    for (let y = 0; y < height; y += gridSpacing) {
      gridG.append('line').attr('x1', 0).attr('y1', y).attr('x2', width).attr('y2', y).attr('stroke', '#374151').attr('stroke-width', 0.5);
    }

    const g = svg.append('g');

    // Zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => g.attr('transform', event.transform));
    svg.call(zoom);

    // Edges
    const edgeG = g.append('g').attr('class', 'edges');
    graph.edges.forEach((edge, i) => {
      const src = nodeById.get(edge.source);
      const tgt = nodeById.get(edge.target);
      if (!src || !tgt) return;

      const isActive = activeEdgeIds ? activeEdgeIds.has(i) : true;
      const isSkip = edge.type === 'skip';
      const isRecurrent = edge.type === 'recurrent';
      const isAttention = edge.type === 'attention';
      const weight = edge.weight;

      // Gradient flow colouring in backward mode
      let edgeGradMag = 0;
      if (mode === 'backward' && gradients && isActive) {
        const gSrc = gradients[String(edge.source)] ?? 0;
        const gTgt = gradients[String(edge.target)] ?? 0;
        edgeGradMag = (Math.abs(gSrc) + Math.abs(gTgt)) / 2;
      }

      const gradFlowColor = (() => {
        if (mode !== 'backward' || !gradients || !isActive) return null;
        if (edgeGradMag < 0.01) return '#3b82f6';   // vanishing — blue
        if (edgeGradMag < 0.1)  return '#10b981';   // healthy  — green
        if (edgeGradMag < 0.5)  return '#f59e0b';   // large    — yellow
        return '#ef4444';                             // exploding — red
      })();

      const baseColor = weight > 0 ? '#10b981' : '#ef4444';
      const opacity = isActive ? (mode === 'architecture' ? Math.min(0.8, 0.2 + Math.abs(weight) * 0.6) : 0.9) : 0.08;
      const strokeWidth = mode === 'architecture'
        ? Math.max(0.5, Math.min(3, Math.abs(weight) * 2))
        : isActive ? (mode === 'backward' ? Math.max(1, Math.min(4, edgeGradMag * 20)) : 2.5) : 0.5;

      const strokeColor = gradFlowColor
        ?? (isSkip ? '#3b82f6' : isRecurrent ? '#f59e0b' : isAttention ? '#fb923c' : isActive ? '#60a5fa' : baseColor);

      const line = edgeG.append('line')
        .attr('x1', xScale(src.x))
        .attr('y1', yScale(src.y))
        .attr('x2', xScale(tgt.x))
        .attr('y2', yScale(tgt.y))
        .attr('stroke', strokeColor)
        .attr('stroke-width', strokeWidth)
        .attr('stroke-opacity', opacity)
        .attr('stroke-dasharray', isSkip || isRecurrent ? '4,3' : isAttention ? '2,3' : 'none');

      if (isActive && mode !== 'architecture') {
        line.attr('filter', 'url(#glow)');
        // Animated signal dot
        const circle = g.append('circle').attr('r', 3).attr('fill', '#60a5fa').attr('opacity', 0);
        circle.append('animateMotion')
          .attr('dur', `${0.8 + Math.random() * 0.4}s`)
          .attr('repeatCount', 'indefinite')
          .attr('path', `M${xScale(src.x)},${yScale(src.y)} L${xScale(tgt.x)},${yScale(tgt.y)}`);
        circle.append('animate')
          .attr('attributeName', 'opacity')
          .attr('values', '0;1;0')
          .attr('dur', `${0.8 + Math.random() * 0.4}s`)
          .attr('repeatCount', 'indefinite');
      }
    });

    // Nodes
    const nodeG = g.append('g').attr('class', 'nodes');
    const tooltip = d3.select(container).append('div')
      .style('position', 'absolute')
      .style('background', 'rgba(17, 24, 39, 0.95)')
      .style('border', '1px solid #374151')
      .style('border-radius', '8px')
      .style('padding', '8px 12px')
      .style('font-size', '12px')
      .style('color', '#e2e8f0')
      .style('pointer-events', 'none')
      .style('opacity', '0')
      .style('transition', 'opacity 0.15s')
      .style('z-index', '10')
      .style('max-width', '200px');

    graph.nodes.forEach((node) => {
      const cx = xScale(node.x);
      const cy = yScale(node.y);
      const isActive = activeNodeIds ? activeNodeIds.has(node.id) : true;
      const color = LAYER_TYPE_COLORS[node.layer_type] ?? '#6b7280';
      const radius = node.layer_type === 'output' ? 14 : node.layer_type === 'input' ? 12 : 10;

      const nodeGroup = nodeG.append('g')
        .attr('transform', `translate(${cx}, ${cy})`)
        .attr('cursor', 'pointer');

      // Outer glow ring for active nodes
      if (isActive && mode !== 'architecture') {
        nodeGroup.append('circle')
          .attr('r', radius + 6)
          .attr('fill', 'none')
          .attr('stroke', color)
          .attr('stroke-width', 2)
          .attr('opacity', 0.3)
          .attr('filter', 'url(#activeGlow)');
      }

      // Value-based fill opacity
      const fillOpacity = mode === 'architecture' ? 0.85 : isActive ? 1.0 : 0.2;

      // Main circle
      nodeGroup.append('circle')
        .attr('r', radius)
        .attr('fill', color)
        .attr('fill-opacity', fillOpacity)
        .attr('stroke', isActive ? '#ffffff' : '#374151')
        .attr('stroke-width', isActive && mode !== 'architecture' ? 2 : 1)
        .attr('filter', isActive && mode !== 'architecture' ? 'url(#glow)' : 'none');

      // Value bar inside node
      if (node.value !== undefined) {
        const barH = Math.abs(node.value) * (radius - 2);
        nodeGroup.append('rect')
          .attr('x', -3)
          .attr('y', node.value >= 0 ? -barH : 0)
          .attr('width', 6)
          .attr('height', barH)
          .attr('fill', 'white')
          .attr('fill-opacity', 0.4)
          .attr('rx', 2);
      }

      // Gradient label (backward mode)
      if (mode === 'backward' && gradients && gradients[node.id] !== undefined) {
        const grad = gradients[node.id];
        nodeGroup.append('text')
          .attr('y', radius + 14)
          .attr('text-anchor', 'middle')
          .attr('font-size', '9px')
          .attr('fill', '#fb923c')
          .text(`∇${grad.toFixed(3)}`);
      }

      // Node label
      nodeGroup.append('text')
        .attr('dy', '0.35em')
        .attr('text-anchor', 'middle')
        .attr('font-size', '8px')
        .attr('font-weight', '600')
        .attr('fill', 'white')
        .attr('pointer-events', 'none')
        .text(node.name.length > 8 ? node.name.slice(0, 6) + '…' : node.name);

      // Tooltip
      nodeGroup
        .on('mouseenter', (_event) => {
          tooltip.style('opacity', '1');
          const lines = [
            `<strong style="color:#60a5fa">${node.name}</strong>`,
            `Type: <span style="color:#a3e635">${node.layer_type}</span>`,
            `Value: <span style="color:#34d399">${node.value?.toFixed(4)}</span>`,
            node.activation ? `Activation: <span style="color:#fb923c">${node.activation}</span>` : null,
            node.bias !== undefined ? `Bias: <span style="color:#c084fc">${node.bias.toFixed(4)}</span>` : null,
          ].filter(Boolean).join('<br/>');
          tooltip.html(lines);
        })
        .on('mousemove', (event) => {
          const rect = container.getBoundingClientRect();
          tooltip
            .style('left', `${event.clientX - rect.left + 12}px`)
            .style('top', `${event.clientY - rect.top - 10}px`);
        })
        .on('mouseleave', () => tooltip.style('opacity', '0'));
    });

    return () => {
      tooltip.remove();
    };
  }, [graph, activeNodeIds, activeEdgeIds, gradients, mode, minX, maxX, minY, maxY]);

  if (!graph.nodes.length) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-3">
        <div className="w-16 h-16 rounded-2xl bg-gray-800/50 flex items-center justify-center">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="5" cy="12" r="2" /><circle cx="19" cy="5" r="2" /><circle cx="19" cy="19" r="2" />
            <line x1="7" y1="12" x2="17" y2="6" /><line x1="7" y1="12" x2="17" y2="18" />
          </svg>
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-gray-400">No network built yet</p>
          <p className="text-xs text-gray-600 mt-1">Configure and click "Build Network"</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <svg ref={svgRef} className="w-full h-full" />
      <div className="absolute bottom-3 left-3 flex flex-wrap gap-1.5">
        {Object.entries(LAYER_TYPE_COLORS)
          .filter(([type]) => graph.nodes.some((n) => n.layer_type === type))
          .map(([type, color]) => (
            <div key={type} className="flex items-center gap-1 text-xs text-gray-400 bg-gray-900/80 px-2 py-1 rounded-md border border-gray-800">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
              <span className="capitalize">{type}</span>
            </div>
          ))}
      </div>
      {mode === 'backward' && (
        <div className="absolute top-3 right-3 flex items-center gap-2 text-xs bg-gray-900/90 px-2.5 py-1.5 rounded-lg border border-gray-800">
          {[['#3b82f6','vanishing'],['#10b981','healthy'],['#f59e0b','large'],['#ef4444','exploding']].map(([c,l]) => (
            <span key={l} className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full inline-block" style={{ background: c }} />
              <span style={{ color: 'var(--text-faint)' }}>{l}</span>
            </span>
          ))}
        </div>
      )}
      {mode !== 'backward' && (
        <div className="absolute top-3 right-3 text-xs bg-gray-900/80 px-2 py-1 rounded-md border border-gray-800" style={{ color: 'var(--text-faint)' }}>
          Scroll to zoom · Drag to pan
        </div>
      )}
    </div>
  );
}
