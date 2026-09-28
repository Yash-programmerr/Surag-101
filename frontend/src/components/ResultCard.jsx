const categoryColor = {
  city_structures: '#e53935',
  urban_areas: '#fb8c00',
  open_land: '#8d8d8d',
  agriculture: '#43a047',
}

function pct(value) {
  return value == null ? 'n/a' : `${Math.round(value * 100)}%`
}

export default function ResultCard({ result, selected, onSelect, onSimilar }) {
  return (
    <article className={`result-card ${selected ? 'selected' : ''}`} onClick={() => onSelect(result.tile_id)}>
      <img src={result.thumb_url} alt="" loading="lazy" />
      <div className="result-body">
        <div className="result-topline">
          <span>#{result.rank} {Number(result.score).toFixed(3)}</span>
          <span className="dot" style={{ background: categoryColor[result.category] || '#555' }} />
        </div>
        <strong>{result.category?.replace('_', ' ')}</strong>
        <dl>
          <div><dt>Built</dt><dd>{pct(result.built_frac)}</dd></div>
          <div><dt>Crop</dt><dd>{pct(result.crop_frac)}</dd></div>
          <div><dt>Open</dt><dd>{pct(result.open_frac)}</dd></div>
        </dl>
        <button type="button" className="link-button" onClick={(event) => { event.stopPropagation(); onSimilar(result.tile_id) }}>
          Similar
        </button>
      </div>
    </article>
  )
}
