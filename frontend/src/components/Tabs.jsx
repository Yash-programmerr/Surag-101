export default function Tabs({ activeTab, onChange }) {
  return (
    <div className="tabs">
      <button type="button" className={activeTab === 'search' ? 'active' : ''} onClick={() => onChange('search')}>Search</button>
      <button type="button" className={activeTab === 'discover' ? 'active' : ''} onClick={() => onChange('discover')}>Discovery</button>
      <button type="button" className={activeTab === 'aoi' ? 'active' : ''} onClick={() => onChange('aoi')}>AOI Analysis</button>
    </div>
  )
}
