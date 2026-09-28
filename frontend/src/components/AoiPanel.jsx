import { useEffect, useState } from 'react'
import { previewPathToUrl } from '../api'

function decisionText(row) {
  if (row.confirmed === true) return 'confirmed'
  if (row.confirmed === false) return 'rejected'
  return 'pending'
}

export default function AoiPanel({
  drawPoints,
  drawnGeoJSON,
  onFinish,
  onClear,
  aoiFile,
  setAoiFile,
  onRun,
  onRunTemporal,
  jobStatus,
  jobResults,
  selectedTileId,
  onSelectResult,
  onDecision,
  reviewLog,
  error,
}) {
  const [reviewer, setReviewer] = useState('analyst')
  const [notes, setNotes] = useState({})
  const [yearStart, setYearStart] = useState(2021)
  const [yearEnd, setYearEnd] = useState(2025)

  useEffect(() => {
    if (!selectedTileId) return
    document.getElementById(`aoi-row-${selectedTileId}`)?.scrollIntoView({ block: 'nearest' })
  }, [selectedTileId])

  const canRun = Boolean(drawnGeoJSON || aoiFile)
  return (
    <section className="aoi-panel">
      <section className="panel">
        <h2>AOI input</h2>
        <p className="muted">Click the map to add points, then finish the polygon, or upload a GeoJSON file.</p>
        <div className="button-row">
          <button type="button" disabled={drawPoints.length < 3} onClick={onFinish}>Finish polygon</button>
          <button type="button" className="secondary" onClick={onClear}>Clear</button>
        </div>
        <p className="muted">{drawPoints.length} point{drawPoints.length === 1 ? '' : 's'} drawn {drawnGeoJSON ? '(closed)' : ''}</p>
        <input type="file" accept=".geojson,application/geo+json,application/json" onChange={(event) => setAoiFile(event.target.files?.[0] || null)} />
        {aoiFile ? <p className="muted">Selected file: {aoiFile.name}</p> : null}
        <button type="button" disabled={!canRun} onClick={onRun}>Run analysis</button>
        <div className="temporal-controls">
          <label>From year<input type="number" value={yearStart} onChange={(event) => setYearStart(Number(event.target.value))} /></label>
          <label>To year<input type="number" value={yearEnd} onChange={(event) => setYearEnd(Number(event.target.value))} /></label>
        </div>
        <button type="button" className="secondary" disabled={!canRun || yearStart >= yearEnd} onClick={() => onRunTemporal({ yearStart, yearEnd })}>
          Run fresh temporal ML analysis
        </button>
        <p className="muted">Runs AnyChange on available consecutive observations in this AOI. Current imagery years: 2021 and 2025.</p>
        {error ? <p className="error">{error}</p> : null}
      </section>

      {jobStatus ? (
        <section className="panel">
          <h2>Job status</h2>
          <p>
            {jobStatus.status}
            {jobStatus.status === 'running' ? `: ${jobStatus.done}/${jobStatus.total} tiles` : ''}
            {jobStatus.error ? `: ${jobStatus.error}` : ''}
          </p>
          {jobStatus.kind === 'temporal' ? (
            <p className="muted">
              {jobStatus.done || 0}/{jobStatus.total || 0} image pairs processed
              {jobStatus.candidate_pairs != null ? ` · ${jobStatus.candidate_pairs} candidate pairs` : ''}
              {jobStatus.truncated ? ' · capped by max_pairs; narrow the AOI or raise the API limit' : ''}
            </p>
          ) : null}
        </section>
      ) : null}

      {jobResults.length ? (
        <section className="panel">
          <h2>AOI results</h2>
          <label className="reviewer">Reviewer<input value={reviewer} onChange={(event) => setReviewer(event.target.value)} /></label>
          <div className="aoi-results">
            {jobResults.map((row) => (
              <article
                id={`aoi-row-${row.tile_id}`}
                key={row.tile_id}
                className={`aoi-row ${selectedTileId === row.tile_id ? 'selected' : ''}`}
                onClick={() => onSelectResult(row.tile_id)}
              >
                <img src={previewPathToUrl(row.preview_path)} alt="" loading="lazy" />
                <div>
                  <strong>{row.tile_id}</strong>
                  <p>{row.category} · {Number(row.total_change_ha || 0).toFixed(3)} ha · {decisionText(row)}</p>
                  {row.earliest_supported_observation ? <p>First supported observation: {row.earliest_supported_observation}</p> : null}
                  {row.change_types?.length ? <p>Supported types: {row.change_types.join(', ')}</p> : null}
                  {row.temporal_evidence_note ? <p className="muted">{row.temporal_evidence_note}</p> : null}
                  <input
                    placeholder="Review note"
                    value={notes[row.tile_id] || ''}
                    onChange={(event) => setNotes((current) => ({ ...current, [row.tile_id]: event.target.value }))}
                    onClick={(event) => event.stopPropagation()}
                  />
                  <div className="button-row">
                    <button type="button" onClick={(event) => { event.stopPropagation(); onDecision(row.tile_id, 'confirmed', reviewer, notes[row.tile_id] || '') }}>Confirm</button>
                    <button type="button" className="danger" onClick={(event) => { event.stopPropagation(); onDecision(row.tile_id, 'rejected', reviewer, notes[row.tile_id] || '') }}>Reject</button>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      <section className="panel">
        <h2>Review log</h2>
        {reviewLog.length === 0 ? <p className="muted">No decisions yet.</p> : null}
        {reviewLog.length ? (
          <table>
            <thead><tr><th>Tile</th><th>Decision</th><th>Reviewer</th><th>Note</th><th>Time</th></tr></thead>
            <tbody>
              {reviewLog.slice(0, 10).map((row, index) => (
                <tr key={`${row.job_id}-${row.tile_id}-${index}`}>
                  <td>{row.tile_id}</td><td>{row.decision}</td><td>{row.reviewer}</td><td>{row.note}</td><td>{row.timestamp_utc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </section>
    </section>
  )
}
