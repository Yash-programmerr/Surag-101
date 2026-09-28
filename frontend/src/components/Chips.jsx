const examples = [
  'dense buildings in a city',
  'farmland with crop fields',
  'barren open land',
  'suburban residential area',
  'river or water body',
  'road network',
  'new construction site',
]

export default function Chips({ onPick }) {
  return (
    <div className="chips">
      {examples.map((text) => (
        <button type="button" key={text} onClick={() => onPick(text)}>{text}</button>
      ))}
    </div>
  )
}
