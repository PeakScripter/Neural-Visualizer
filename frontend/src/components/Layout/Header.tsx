import { Activity, Film, Sun, Moon, Terminal } from 'lucide-react';
import { useTheme, type Theme } from '../../contexts/ThemeContext';

const THEMES: { id: Theme; label: string; icon: React.ReactNode; dot: string }[] = [
  { id: 'dark',      label: 'Dark',      icon: <Moon size={11} />,     dot: '#3b82f6' },
  { id: 'cyberpunk', label: 'Cyber',     icon: <Terminal size={11} />, dot: '#e879f9' },
  { id: 'matrix',    label: 'Matrix',    icon: <Activity size={11} />, dot: '#10b981' },
  { id: 'paper',     label: 'Paper',     icon: <Sun size={11} />,      dot: '#2563eb' },
];

interface Props {
  onCinema: () => void;
  cinemaDisabled: boolean;
}

export function Header({ onCinema, cinemaDisabled }: Props) {
  const { theme, setTheme } = useTheme();

  return (
    <header
      className="border-b sticky top-0 z-50 backdrop-blur"
      style={{
        borderColor: 'var(--border)',
        background: 'rgba(var(--bg-card-rgb, 17,24,39), 0.85)',
        backgroundColor: 'color-mix(in srgb, var(--bg-card) 85%, transparent)',
      }}
    >
      <div className="flex items-center gap-3 px-5 py-2.5">
        {/* Logo */}
        <div className="flex items-center gap-2.5 flex-shrink-0">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--accent)', boxShadow: `0 0 12px var(--accent-glow)` }}
          >
            <Activity size={16} className="text-white" />
          </div>
          <span className="text-gradient font-bold text-lg tracking-tight">Neural Visualizer</span>
        </div>

        <span className="hidden sm:block text-sm mx-1" style={{ color: 'var(--border-soft)' }}>|</span>
        <span className="hidden sm:block text-sm" style={{ color: 'var(--text-muted)' }}>
          Interactive deep learning exploration
        </span>

        <div className="ml-auto flex items-center gap-2">
          {/* Theme switcher */}
          <div
            className="flex items-center gap-0.5 p-0.5 rounded-lg border"
            style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
          >
            {THEMES.map((t) => (
              <button
                key={t.id}
                onClick={() => setTheme(t.id)}
                title={t.label}
                className="flex items-center gap-1 px-2 py-1 rounded-md text-xs transition-all duration-200"
                style={{
                  background: theme === t.id ? t.dot + '22' : 'transparent',
                  color: theme === t.id ? t.dot : 'var(--text-faint)',
                  border: theme === t.id ? `1px solid ${t.dot}44` : '1px solid transparent',
                }}
              >
                {t.icon}
                <span className="hidden sm:inline">{t.label}</span>
              </button>
            ))}
          </div>

          {/* Cinema button */}
          <button
            onClick={onCinema}
            disabled={cinemaDisabled}
            title={cinemaDisabled ? 'Build a network first' : 'Launch Cinema Mode'}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border"
            style={{
              background: cinemaDisabled ? 'transparent' : 'rgba(139,92,246,0.15)',
              borderColor: cinemaDisabled ? 'var(--border)' : '#8b5cf6',
              color: cinemaDisabled ? 'var(--text-faint)' : '#c4b5fd',
              cursor: cinemaDisabled ? 'not-allowed' : 'pointer',
            }}
          >
            <Film size={13} />
            <span>Cinema</span>
          </button>

          <span
            className="text-xs px-2 py-0.5 rounded font-medium"
            style={{ background: 'rgba(16,185,129,0.15)', color: '#6ee7b7', border: '1px solid rgba(16,185,129,0.3)' }}
          >
            v3.0
          </span>
        </div>
      </div>
    </header>
  );
}
