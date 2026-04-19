export const ROUTES = [
  { id: 'intro', label: 'Intro' },
  { id: 'session', label: 'Sessions' },
  { id: 'alignment', label: 'Alignment' },
  { id: 'coefficients', label: 'Coefficients' },
  { id: 'decay', label: 'Decay' },
  { id: 'performance', label: 'Performance' },
] as const;

export type RouteId = (typeof ROUTES)[number]['id'];

export default function Navigation({ current }: { current: RouteId }) {
  return (
    <nav className="flex gap-1">
      {ROUTES.map((r) => (
        <a
          key={r.id}
          href={`#/${r.id}`}
          className={`nav-tab ${r.id === current ? 'active' : ''}`}
        >
          {r.label}
        </a>
      ))}
    </nav>
  );
}
