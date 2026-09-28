export default function SearchBar({ query, setQuery, onSearch }) {
  return (
    <form className="searchbar" onSubmit={(event) => { event.preventDefault(); onSearch() }}>
      <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search satellite tiles..." />
      <button type="submit">Search</button>
    </form>
  )
}
