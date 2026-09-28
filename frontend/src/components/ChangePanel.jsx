import { previewPathToUrl } from '../api'

const rows = [
  ['new_built_ha', 'New built ha'],
  ['vegetation_loss_ha', 'Vegetation loss ha'],
  ['vegetation_gain_ha', 'Vegetation gain ha'],
  ['other_change_ha', 'Other change ha'],
  ['total_change_ha', 'Total change ha'],
  ['mean_confidence', 'Mean confidence'],
]

export default function ChangePanel({ changeData }) {
  const summary = changeData.summary || {}
  const suspicious = summary.suspicious || Number(summary.changed_fraction || 0) > 0.65
  return (
    <section className="panel change-panel">
      <h2>Change detection</h2>
      <img src={changeData.preview_url || previewPathToUrl(summary.preview_path)} alt="" />
      <p className="muted">2021 RGB, 2025 RGB, and class-coloured overlay.</p>
      {suspicious ? <p className="warning">Large or suspicious change fraction. Review visually before using in a demo.</p> : null}
      <table>
        <tbody>
          {rows.map(([key, label]) => (
            <tr key={key}><th>{label}</th><td>{summary[key] == null ? 'n/a' : Number(summary[key]).toFixed(3)}</td></tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
