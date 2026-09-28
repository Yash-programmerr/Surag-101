import { GeoJSON, LayersControl, MapContainer, Polygon, Polyline, Rectangle, TileLayer, useMap, useMapEvents } from 'react-leaflet'
import { useEffect } from 'react'

const categoryColor = {
  city_structures: '#e53935',
  urban_areas: '#fb8c00',
  open_land: '#8d8d8d',
  agriculture: '#43a047',
}

const changeColor = {
  1: '#ff0000',
  2: '#ff8c00',
  3: '#00c800',
  4: '#ffeb00',
  5: '#0064ff',
}

function FitResults({ results }) {
  const map = useMap()
  useEffect(() => {
    const bounds = results.map((result) => result.bounds).filter(Boolean)
    if (bounds.length) map.fitBounds(bounds, { padding: [28, 28] })
  }, [map, results])
  return null
}

function DrawingEvents({ mode, drawPoints, setDrawPoints, onFinishPolygon }) {
  useMapEvents({
    click(event) {
      if (mode !== 'aoi') return
      setDrawPoints((points) => [...points, [event.latlng.lat, event.latlng.lng]])
    },
    dblclick() {
      if (mode === 'aoi' && drawPoints.length >= 3) onFinishPolygon()
    },
  })
  return null
}

function geoJsonStyle(feature) {
  const code = feature?.properties?.class_code
  return { color: changeColor[code] || '#ffeb00', weight: 2, fillOpacity: 0.45 }
}

function decisionColor(row) {
  if (row.confirmed === true) return '#2e7d32'
  if (row.confirmed === false) return '#c62828'
  return '#fbc02d'
}

export default function MapView({
  mode,
  results,
  selectedTileId,
  onSelectTile,
  changePolygons,
  drawPoints,
  setDrawPoints,
  drawnGeoJSON,
  onFinishPolygon,
  onClearAoi,
  aoiResults,
  focusedAoiTileId,
  onSelectAoiTile,
}) {
  return (
    <MapContainer center={[22.5, 79.0]} zoom={5} doubleClickZoom={mode !== 'aoi'} className="map">
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name="Satellite">
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attribution="Tiles &copy; Esri"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Streets">
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution="&copy; OpenStreetMap contributors"
          />
        </LayersControl.BaseLayer>
      </LayersControl>

      <DrawingEvents mode={mode} drawPoints={drawPoints} setDrawPoints={setDrawPoints} onFinishPolygon={onFinishPolygon} />

      {mode === 'search' ? (
        <>
          <FitResults results={results} />
          {results.map((result) => (
            <Rectangle
              key={result.tile_id}
              bounds={result.bounds}
              pathOptions={{
                color: selectedTileId === result.tile_id ? '#111827' : categoryColor[result.category] || '#555',
                fillColor: categoryColor[result.category] || '#555',
                weight: selectedTileId === result.tile_id ? 3 : 1,
                fillOpacity: selectedTileId === result.tile_id ? 0.25 : 0.12,
              }}
              eventHandlers={{ click: () => onSelectTile(result.tile_id) }}
            />
          ))}
          {changePolygons ? (
            <GeoJSON
              key={JSON.stringify(changePolygons).slice(0, 80)}
              data={changePolygons}
              style={geoJsonStyle}
              onEachFeature={(feature, layer) => {
                const props = feature.properties || {}
                layer.bindPopup(`${props.class_name || 'change'} ${props.area_m2 ? `${Math.round(props.area_m2)} m2` : ''}`)
              }}
            />
          ) : null}
        </>
      ) : (
        <>
          {drawPoints.length > 1 ? <Polyline positions={drawPoints} pathOptions={{ color: '#1565c0', weight: 3 }} /> : null}
          {drawnGeoJSON ? (
            <Polygon positions={drawnGeoJSON.coordinates[0].map(([lon, lat]) => [lat, lon])} pathOptions={{ color: '#1565c0', fillOpacity: 0.12 }} />
          ) : null}
          <div className="leaflet-top leaflet-left draw-actions">
            <div className="leaflet-control">
              <button type="button" disabled={drawPoints.length < 3} onClick={onFinishPolygon}>Finish polygon</button>
              <button type="button" onClick={onClearAoi}>Clear</button>
            </div>
          </div>
          {aoiResults.map((row) => (
            row.bounds ? (
              <Rectangle
                key={row.tile_id}
                bounds={row.bounds}
                pathOptions={{
                  color: focusedAoiTileId === row.tile_id ? '#111827' : decisionColor(row),
                  fillColor: decisionColor(row),
                  weight: focusedAoiTileId === row.tile_id ? 3 : 1,
                  fillOpacity: 0.22,
                }}
                eventHandlers={{ click: () => onSelectAoiTile(row.tile_id) }}
              />
            ) : null
          ))}
        </>
      )}
    </MapContainer>
  )
}
