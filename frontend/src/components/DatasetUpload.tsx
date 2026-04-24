import { useState, useRef, useCallback } from 'react';
import { Upload, X, Table } from 'lucide-react';

export interface ParsedDataset {
  headers: string[];
  rows: number[][];
  X: number[][];
  y: number[];
}

interface Props {
  onLoad: (ds: ParsedDataset) => void;
  onClear: () => void;
  loaded: boolean;
}

export function DatasetUpload({ onLoad, onClear, loaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<ParsedDataset | null>(null);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const parse = useCallback((text: string) => {
    setError('');
    const lines = text.trim().split('\n').filter(Boolean);
    if (lines.length < 2) { setError('CSV must have a header row and data.'); return; }

    const headers = lines[0].split(',').map((h) => h.trim().replace(/^"/, '').replace(/"$/, ''));
    const rows: number[][] = [];

    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(',').map((v) => parseFloat(v.trim()));
      if (vals.some(isNaN)) continue;
      rows.push(vals);
    }

    if (!rows.length) { setError('No numeric rows found.'); return; }
    if (rows[0].length < 2) { setError('Need at least 2 columns (features + label).'); return; }

    const X = rows.map((r) => r.slice(0, -1));
    const y = rows.map((r) => (r.at(-1)! > 0.5 ? 1 : 0));

    const ds: ParsedDataset = { headers, rows, X, y };
    setPreview(ds);
    onLoad(ds);
  }, [onLoad]);

  const handleFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => parse(e.target?.result as string);
    reader.readAsText(file);
  }, [parse]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const clear = () => {
    setPreview(null);
    setError('');
    onClear();
    if (inputRef.current) inputRef.current.value = '';
  };

  if (loaded && preview) {
    return (
      <div className="rounded-lg border p-3" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
            <Table size={12} />
            <span>{preview.rows.length} rows · {preview.headers.length} cols</span>
          </div>
          <button onClick={clear} className="p-1 rounded hover:bg-red-900/20 transition-colors">
            <X size={12} style={{ color: 'var(--text-muted)' }} />
          </button>
        </div>
        <div className="text-xs overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                {preview.headers.map((h) => (
                  <th key={h} className="text-left pr-3 pb-1" style={{ color: 'var(--accent)' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.slice(0, 3).map((row, i) => (
                <tr key={i}>
                  {row.map((v, j) => (
                    <td key={j} className="pr-3" style={{ color: 'var(--text-muted)' }}>{v.toFixed(3)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {preview.rows.length > 3 && (
            <p className="mt-1" style={{ color: 'var(--text-faint)' }}>…{preview.rows.length - 3} more rows</p>
          )}
        </div>
        <div className="mt-2 text-xs px-2 py-1 rounded" style={{ background: 'rgba(16,185,129,0.1)', color: '#34d399' }}>
          ✓ Custom dataset active — last column used as label
        </div>
      </div>
    );
  }

  return (
    <div>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className="border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-all"
        style={{
          borderColor: dragging ? 'var(--accent)' : 'var(--border-soft)',
          background: dragging ? 'rgba(59,130,246,0.06)' : 'transparent',
        }}
      >
        <Upload size={18} className="mx-auto mb-1.5" style={{ color: dragging ? 'var(--accent)' : 'var(--text-faint)' }} />
        <p className="text-xs font-medium" style={{ color: dragging ? 'var(--accent)' : 'var(--text-muted)' }}>
          Drop a CSV file here
        </p>
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>or click to browse</p>
      </div>
      {error && <p className="text-xs mt-1.5 text-red-400">{error}</p>}
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
      />
    </div>
  );
}
