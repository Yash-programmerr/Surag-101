export default function TopBar({ statusInfo, canReload, onReload }) {
  const count = statusInfo?.tiles_embedded ?? statusInfo?.tiles_indexed ?? 0
  return (
    <header className="topbar">
      <div>
        <h1>SURAG Satellite Tile Explorer</h1>
        <p>{count.toLocaleString()} tiles searchable</p>
      </div>
      {canReload ? (
        <button type="button" className="secondary" onClick={onReload}>
          Reload data
        </button>
      ) : null}
    </header>
  )
}
