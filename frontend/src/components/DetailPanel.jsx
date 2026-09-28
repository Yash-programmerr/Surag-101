function fmt(value, digits = 3) {
  return value == null ? 'n/a' : Number(value).toFixed(digits)
}

export default function DetailPanel({ detail, onSimilar, onShowChange }) {
  if (!detail) return null
  return (
    <section className="panel detail-panel">
      <h2>{detail.tile_id}</h2>
      <img src={detail.thumb_url} alt="" />
      <dl className="metadata">
        <div><dt>Category</dt><dd>{detail.category}</dd></div>
        <div><dt>ML category</dt><dd>{detail.ml_category || 'n/a'} {detail.ml_category_confidence ? `(${fmt(detail.ml_category_confidence, 2)})` : ''}</dd></div>
        <div><dt>Location</dt><dd>{fmt(detail.lat, 4)}, {fmt(detail.lon, 4)}</dd></div>
        <div><dt>Years</dt><dd>{detail.years_available?.join(', ') || 'n/a'}</dd></div>
        <div><dt>Built/Crop/Open/Water</dt><dd>{fmt(detail.built_frac)} / {fmt(detail.crop_frac)} / {fmt(detail.open_frac)} / {fmt(detail.water_frac)}</dd></div>
        <div><dt>NDVI/NDBI</dt><dd>{fmt(detail.ndvi_mean)} / {fmt(detail.ndbi_mean)}</dd></div>
      </dl>
      <div className="button-row">
        <button type="button" onClick={() => onSimilar(detail.tile_id)}>Find similar</button>
        <a className="button secondary" href={detail.maps_url} target="_blank" rel="noreferrer">Google Maps</a>
        {detail.change ? <button type="button" className="secondary" onClick={() => onShowChange(detail.tile_id)}>Show change</button> : null}
      </div>
    </section>
  )
}
