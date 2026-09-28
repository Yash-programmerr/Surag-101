async function request(path, options = {}) {
  const response = await fetch(path, options)
  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json') ? await response.json() : await response.text()
  if (!response.ok) {
    const detail = payload && typeof payload === 'object' ? payload.detail : payload
    throw new Error(detail || `Request failed with ${response.status}`)
  }
  return payload
}

export function getHealth() {
  return request('/api/health')
}

export function getStats() {
  return request('/api/stats')
}

export function reloadData() {
  return request('/api/reload', { method: 'POST' })
}

export function searchTiles(body) {
  return request('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function getTile(tileId) {
  return request(`/api/tiles/${encodeURIComponent(tileId)}`)
}

export function getTileChange(tileId) {
  return request(`/api/tiles/${encodeURIComponent(tileId)}/change`)
}

export function createAoiJob(aoi, name = 'Drawn AOI') {
  return request('/api/aoi/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ aoi, name }),
  })
}

export function uploadAoiJob(file) {
  const body = new FormData()
  body.append('file', file)
  return request('/api/aoi/jobs/upload', { method: 'POST', body })
}

export function getAoiJob(jobId) {
  return request(`/api/aoi/jobs/${encodeURIComponent(jobId)}`)
}

export function getAoiResults(jobId) {
  return request(`/api/aoi/jobs/${encodeURIComponent(jobId)}/results`)
}

export function decideAoiResult(jobId, body) {
  return request(`/api/aoi/jobs/${encodeURIComponent(jobId)}/decision`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function getAoiReviews() {
  return request('/api/aoi/reviews')
}

export function createTemporalAnalysis(body) {
  return request('/api/aoi/change-analysis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function getDiscoverySimilar(tileId, k = 20, category) {
  const query = new URLSearchParams({ k: String(k) })
  if (category) query.set('category', category)
  return request(`/api/discovery/similar/${encodeURIComponent(tileId)}?${query}`)
}

export function getDiscoveryClusters(body) {
  return request('/api/discovery/clusters', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function previewPathToUrl(path) {
  if (!path) return ''
  if (path.startsWith('/files/')) return path
  const marker = 'india_tiles/change/'
  const index = path.indexOf(marker)
  if (index >= 0) return `/files/change/${path.slice(index + marker.length)}`
  return `/files/change/${path.replace(/^\/+/, '')}`
}
