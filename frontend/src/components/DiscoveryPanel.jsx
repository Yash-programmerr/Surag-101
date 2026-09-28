import { useState } from 'react'
import { getDiscoveryClusters, getDiscoverySimilar } from '../api'

export default function DiscoveryPanel({ selectedTileId, onSelectTile, onResultsChange }) {
  const [clustersCount, setClustersCount] = useState(12)
  const [refresh, setRefresh] = useState(false)
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function findSimilar() {
    if (!selectedTileId) return
    setLoading(true)
    setError('')
    try {
      const result = await getDiscoverySimilar(selectedTileId, 30)
      setPayload({ kind: 'similar', ...result })
      onResultsChange(result.results || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function clusterSites() {
    setLoading(true)
    setError('')
    try {
      const result = await getDiscoveryClusters({ n_clusters: Number(clustersCount), refresh })
      setPayload({ kind: 'clusters', ...result })
      onResultsChange(result.clusters.flatMap((group) => group.sites || []))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <section className="panel">
        <h2>Discover similar sites</h2>
        <p className="muted">Find other locations with similar Sentinel-2 visual embeddings.</p>
        <button type="button" disabled={!selectedTileId || loading} onClick={findSimilar}>
          {loading ? 'Searching…' : selectedTileId ? `Find sites like ${selectedTileId}` : 'Select a tile first'}
        </button>
      </section>
      <section className="panel">
        <h2>Group sites across the dataset</h2>
        <p className="muted">All satellite embeddings are L2 normalized, reduced to 64 PCA dimensions, then grouped. Every tile receives a saved cluster ID.</p>
        <div className="cluster-controls">
          <label>Number of groups<input type="number" min="2" max="20" value={clustersCount} onChange={(event) => setClustersCount(event.target.value)} /></label>
          <button type="button" disabled={loading} onClick={clusterSites}>{loading ? 'Grouping…' : 'Discover site groups'}</button>
        </div>
        <label className="cluster-refresh"><input type="checkbox" checked={refresh} onChange={(event) => setRefresh(event.target.checked)} /> Recompute saved clusters</label>
      </section>
      {error ? <p className="error">{error}</p> : null}
      {payload?.kind === 'similar' ? (
        <section className="panel">
          <h2>Similar locations</h2>
          <p className="muted">{payload.method} · {payload.searched_tiles.toLocaleString()} tiles searched</p>
          <div className="discovery-list">
            {payload.results.map((site) => <button type="button" key={site.tile_id} onClick={() => onSelectTile(site.tile_id)}>
              <span>{site.tile_id}</span><small>{site.category} · similarity {Number(site.score).toFixed(3)}</small>
            </button>)}
          </div>
        </section>
      ) : null}
      {payload?.kind === 'clusters' ? (
        <section className="panel">
          <h2>{payload.n_clusters} visual groups · {payload.n_tiles.toLocaleString()} tiles</h2>
          <p className="muted">{payload.method} · {payload.fit_embedding_rows.toLocaleString()} embedding rows fitted</p>
          <p className="muted">Silhouette (2,000 tiles): {payload.quality?.silhouette_2000_tile_sample ?? 'n/a'} · weighted category purity: {payload.quality?.overall_weighted_purity ?? 'n/a'}</p>
          <p className="muted">k comparison: {Object.entries(payload.comparisons || {}).map(([k, item]) => `${k}: silhouette ${item.silhouette_2000_tile_sample}, purity ${item.overall_weighted_purity}`).join(' · ')}</p>
          {payload.clusters.map((group) => (
            <details className="cluster-group" key={group.cluster_id}>
              <summary>Group {group.cluster_id + 1} · {group.count} tiles · {Object.entries(group.category_counts).map(([name, count]) => `${name} ${count}`).join(' / ')}</summary>
              <div className="discovery-list">
                {group.sites.map((site) => <button type="button" key={site.tile_id} onClick={() => onSelectTile(site.tile_id)}>
                  <span>{site.tile_id}</span><small>{site.category} · {Number(site.score).toFixed(3)} centrality</small>
                </button>)}
              </div>
            </details>
          ))}
        </section>
      ) : null}
    </>
  )
}
