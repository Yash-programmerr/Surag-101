import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  createAoiJob,
  createTemporalAnalysis,
  decideAoiResult,
  getAoiJob,
  getAoiResults,
  getAoiReviews,
  getHealth,
  getTile,
  getTileChange,
  reloadData,
  searchTiles,
  uploadAoiJob,
} from './api'
import AoiPanel from './components/AoiPanel'
import ChangePanel from './components/ChangePanel'
import Chips from './components/Chips'
import DetailPanel from './components/DetailPanel'
import DiscoveryPanel from './components/DiscoveryPanel'
import Filters from './components/Filters'
import Legend from './components/Legend'
import MapView from './components/MapView'
import ResultsGrid from './components/ResultsGrid'
import SearchBar from './components/SearchBar'
import Tabs from './components/Tabs'
import TopBar from './components/TopBar'

const defaultFilters = {
  category: ['agriculture', 'city_structures', 'open_land', 'urban_areas'],
  min_built: '',
  max_built: '',
  min_crop: '',
  min_open: '',
  max_water: '',
  min_valid: 90,
}

function toFraction(value) {
  return value === '' || value === null || Number.isNaN(Number(value)) ? null : Number(value) / 100
}

function apiFilters(filters) {
  return {
    category: filters.category.length === 4 ? null : filters.category,
    min_built: toFraction(filters.min_built),
    max_built: toFraction(filters.max_built),
    min_crop: toFraction(filters.min_crop),
    min_open: toFraction(filters.min_open),
    max_water: toFraction(filters.max_water),
    min_valid: toFraction(filters.min_valid) ?? 0.9,
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState('search')
  const [statusInfo, setStatusInfo] = useState(null)
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState(defaultFilters)
  const [k, setK] = useState(12)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedTileId, setSelectedTileId] = useState('')
  const [selectedTileDetail, setSelectedTileDetail] = useState(null)
  const [changeData, setChangeData] = useState(null)
  const [drawnGeoJSON, setDrawnGeoJSON] = useState(null)
  const [drawPoints, setDrawPoints] = useState([])
  const [aoiFile, setAoiFile] = useState(null)
  const [currentJobId, setCurrentJobId] = useState('')
  const [jobStatus, setJobStatus] = useState(null)
  const [jobResults, setJobResults] = useState([])
  const [discoveryMapResults, setDiscoveryMapResults] = useState([])
  const [reviewLog, setReviewLog] = useState([])
  const [aoiError, setAoiError] = useState('')
  const [focusedAoiTileId, setFocusedAoiTileId] = useState('')

  const refreshHealth = useCallback(async () => setStatusInfo(await getHealth()), [])
  const refreshReviews = useCallback(async () => {
    const payload = await getAoiReviews()
    setReviewLog(payload.decisions || [])
  }, [])

  useEffect(() => {
    refreshHealth().catch((err) => setError(err.message))
    refreshReviews().catch(() => {})
  }, [refreshHealth, refreshReviews])

  const selectTile = useCallback(async (tileId) => {
    setSelectedTileId(tileId)
    setChangeData(null)
    try {
      setSelectedTileDetail(await getTile(tileId))
    } catch (err) {
      setError(err.message)
    }
  }, [])

  const runSearch = useCallback(async (override = {}) => {
    setLoading(true)
    setError('')
    try {
      const body = {
        text: override.text ?? query,
        like_tile_id: override.like_tile_id,
        k: Number(k),
        filters: apiFilters(filters),
      }
      if (body.like_tile_id) delete body.text
      const payload = await searchTiles(body)
      setResults(payload.results || [])
      if (payload.results?.[0]) await selectTile(payload.results[0].tile_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [filters, k, query, selectTile])

  const findSimilar = useCallback((tileId) => runSearch({ like_tile_id: tileId }), [runSearch])

  const showChange = useCallback(async (tileId) => {
    setError('')
    try {
      setChangeData(await getTileChange(tileId))
    } catch (err) {
      setError(err.message)
    }
  }, [])

  useEffect(() => {
    if (!currentJobId) return undefined
    let active = true
    const poll = async () => {
      try {
        const status = await getAoiJob(currentJobId)
        if (!active) return
        setJobStatus(status)
        if (status.status === 'done') {
          const payload = await getAoiResults(currentJobId)
          if (!active) return
          setJobResults(payload.results || [])
        }
      } catch (err) {
        if (active) setAoiError(err.message)
      }
    }
    poll()
    if (jobStatus?.status === 'done' || jobStatus?.status === 'error') {
      return () => { active = false }
    }
    const id = window.setInterval(poll, 2000)
    return () => {
      active = false
      window.clearInterval(id)
    }
  }, [currentJobId, jobStatus?.status])

  const finishPolygon = useCallback(() => {
    if (drawPoints.length < 3) return
    const ring = [...drawPoints.map(([lat, lon]) => [lon, lat]), [drawPoints[0][1], drawPoints[0][0]]]
    setDrawnGeoJSON({ type: 'Polygon', coordinates: [ring] })
  }, [drawPoints])

  const clearAoi = useCallback(() => {
    setDrawnGeoJSON(null)
    setDrawPoints([])
    setAoiFile(null)
    setFocusedAoiTileId('')
  }, [])

  const startAoiJob = useCallback(async () => {
    setAoiError('')
    setJobResults([])
    try {
      const payload = aoiFile ? await uploadAoiJob(aoiFile) : await createAoiJob(drawnGeoJSON)
      setCurrentJobId(payload.job_id)
      setJobStatus(payload)
    } catch (err) {
      setAoiError(err.message)
    }
  }, [aoiFile, drawnGeoJSON])

  const startTemporalAnalysis = useCallback(async ({ yearStart, yearEnd }) => {
    setAoiError('')
    setJobResults([])
    try {
      let aoi = drawnGeoJSON
      if (!aoi && aoiFile) {
        const uploaded = JSON.parse(await aoiFile.text())
        if (uploaded.type === 'Feature') aoi = uploaded.geometry
        else if (uploaded.type === 'FeatureCollection') aoi = uploaded.features?.find((feature) => feature.geometry?.type === 'Polygon' || feature.geometry?.type === 'MultiPolygon')?.geometry
        else aoi = uploaded
      }
      if (!aoi) throw new Error('Draw or upload a Polygon/MultiPolygon AOI first')
      const payload = await createTemporalAnalysis({ aoi, year_start: yearStart, year_end: yearEnd, max_pairs: 100,
        name: 'AOI multi-temporal change analysis' })
      setCurrentJobId(payload.job_id)
      setJobStatus(payload)
    } catch (err) {
      setAoiError(err.message)
    }
  }, [aoiFile, drawnGeoJSON])

  const decideResult = useCallback(async (tileId, decision, reviewer, note) => {
    const updated = await decideAoiResult(currentJobId, { tile_id: tileId, decision, reviewer, note })
    setJobResults((rows) => rows.map((row) => (row.tile_id === tileId ? updated : row)))
    await refreshReviews()
  }, [currentJobId, refreshReviews])

  const selectedAoiTile = useMemo(
    () => jobResults.find((row) => row.tile_id === focusedAoiTileId) || null,
    [focusedAoiTileId, jobResults],
  )

  return (
    <div className="app-shell">
      <TopBar statusInfo={statusInfo} canReload onReload={async () => setStatusInfo(await reloadData())} />
      <main className="workspace">
        <aside className="side-panel">
          <Tabs activeTab={activeTab} onChange={setActiveTab} />
          {activeTab === 'search' ? (
            <>
              <SearchBar query={query} setQuery={setQuery} onSearch={() => runSearch()} />
              <Chips onPick={(text) => { setQuery(text); runSearch({ text }) }} />
              <Filters filters={filters} setFilters={setFilters} k={k} setK={setK} />
              <ResultsGrid
                results={results}
                loading={loading}
                error={error}
                selectedTileId={selectedTileId}
                onSelect={selectTile}
                onSimilar={findSimilar}
              />
              <DetailPanel detail={selectedTileDetail} onSimilar={findSimilar} onShowChange={showChange} />
              {changeData ? <ChangePanel changeData={changeData} /> : null}
              {changeData ? <Legend /> : null}
            </>
          ) : activeTab === 'discover' ? (
            <DiscoveryPanel selectedTileId={selectedTileId} onSelectTile={selectTile} onResultsChange={setDiscoveryMapResults} />
          ) : (
            <AoiPanel
              drawPoints={drawPoints}
              drawnGeoJSON={drawnGeoJSON}
              onFinish={finishPolygon}
              onClear={clearAoi}
              aoiFile={aoiFile}
              setAoiFile={setAoiFile}
              onRun={startAoiJob}
              onRunTemporal={startTemporalAnalysis}
              jobStatus={jobStatus}
              jobResults={jobResults}
              selectedTileId={focusedAoiTileId}
              selectedResult={selectedAoiTile}
              onSelectResult={setFocusedAoiTileId}
              onDecision={decideResult}
              reviewLog={reviewLog}
              error={aoiError}
            />
          )}
        </aside>
        <section className="map-pane">
          <MapView
            mode={activeTab === 'aoi' ? 'aoi' : 'search'}
            results={activeTab === 'discover' ? discoveryMapResults : results}
            selectedTileId={selectedTileId}
            onSelectTile={selectTile}
            changePolygons={changeData?.polygons || null}
            drawPoints={drawPoints}
            setDrawPoints={setDrawPoints}
            drawnGeoJSON={drawnGeoJSON}
            onFinishPolygon={finishPolygon}
            onClearAoi={clearAoi}
            aoiResults={jobResults}
            focusedAoiTileId={focusedAoiTileId}
            onSelectAoiTile={setFocusedAoiTileId}
          />
        </section>
      </main>
    </div>
  )
}
