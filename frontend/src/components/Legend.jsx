const items = [
  ['#ff0000', 'New built'],
  ['#ff8c00', 'Vegetation loss'],
  ['#00c800', 'Vegetation gain'],
  ['#ffeb00', 'Other change'],
  ['#0064ff', 'Water change'],
]

export default function Legend() {
  return (
    <section className="panel legend">
      <h2>Change classes</h2>
      {items.map(([color, label]) => (
        <div key={label}><span style={{ background: color }} />{label}</div>
      ))}
    </section>
  )
}
