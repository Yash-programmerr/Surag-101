import ResultCard from './ResultCard'

export default function ResultsGrid({ results, loading, error, selectedTileId, onSelect, onSimilar }) {
  return (
    <section className="panel">
      <h2>Results</h2>
      {loading ? <p className="muted">Searching...</p> : null}
      {error ? <p className="error">{error}</p> : null}
      {!loading && !error && results.length === 0 ? <p className="muted">Run a text search or pick an example.</p> : null}
      <div className="results-grid">
        {results.map((result) => (
          <ResultCard
            key={result.tile_id}
            result={result}
            selected={selectedTileId === result.tile_id}
            onSelect={onSelect}
            onSimilar={onSimilar}
          />
        ))}
      </div>
    </section>
  )
}
