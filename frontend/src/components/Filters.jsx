const categories = ['agriculture', 'city_structures', 'open_land', 'urban_areas']

export default function Filters({ filters, setFilters, k, setK }) {
  const toggleCategory = (category) => {
    setFilters((current) => ({
      ...current,
      category: current.category.includes(category)
        ? current.category.filter((item) => item !== category)
        : [...current.category, category],
    }))
  }
  const setField = (field, value) => setFilters((current) => ({ ...current, [field]: value }))
  return (
    <section className="panel">
      <h2>Filters</h2>
      <div className="category-grid">
        {categories.map((category) => (
          <label key={category}>
            <input type="checkbox" checked={filters.category.includes(category)} onChange={() => toggleCategory(category)} />
            {category.replace('_', ' ')}
          </label>
        ))}
      </div>
      <div className="filter-grid">
        <label>Min built %<input type="number" min="0" max="100" value={filters.min_built} onChange={(event) => setField('min_built', event.target.value)} /></label>
        <label>Max built %<input type="number" min="0" max="100" value={filters.max_built} onChange={(event) => setField('max_built', event.target.value)} /></label>
        <label>Min crop %<input type="number" min="0" max="100" value={filters.min_crop} onChange={(event) => setField('min_crop', event.target.value)} /></label>
        <label>Min open %<input type="number" min="0" max="100" value={filters.min_open} onChange={(event) => setField('min_open', event.target.value)} /></label>
        <label>Max water %<input type="number" min="0" max="100" value={filters.max_water} onChange={(event) => setField('max_water', event.target.value)} /></label>
        <label>Min valid %<input type="number" min="0" max="100" value={filters.min_valid} onChange={(event) => setField('min_valid', event.target.value)} /></label>
        <label>Results<input type="number" min="1" max="60" value={k} onChange={(event) => setK(event.target.value)} /></label>
      </div>
    </section>
  )
}
