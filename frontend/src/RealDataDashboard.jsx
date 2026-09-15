import { useCallback, useEffect, useState } from 'react'
import { Shield, Play, Pause, RotateCcw, AlertTriangle, ExternalLink } from 'lucide-react'

const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`
const money = value => value == null ? '—' : Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })

export default function RealDataDashboard() {
  const [metadata, setMetadata] = useState(null)
  const [rows, setRows] = useState([])
  const [cursor, setCursor] = useState(-1)
  const [hasMore, setHasMore] = useState(true)
  const [playing, setPlaying] = useState(false)
  const [knownFraudOnly, setKnownFraudOnly] = useState(false)
  const [loading, setLoading] = useState(false)
  const [viewedCount, setViewedCount] = useState(0)
  const [error, setError] = useState('')

  const load = useCallback(async (after, replace, fraudOnly) => {
    setLoading(true)
    try {
      const response = await fetch(`/api/v1/archive/rows?after=${after}&limit=20&known_fraud_only=${fraudOnly}`)
      if (!response.ok) throw new Error(`Archive unavailable (${response.status}). Run ml/train_real.py and restart the API.`)
      const data = await response.json()
      setRows(previous => replace ? data.rows : [...previous, ...data.rows].slice(-100))
      setViewedCount(previous => replace ? data.rows.length : previous + data.rows.length)
      setCursor(data.next_cursor)
      setHasMore(data.has_more)
      if (!data.has_more) setPlaying(false)
      setError('')
    } catch (failure) {
      setPlaying(false)
      setError(failure.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetch('/api/v1/archive/metadata').then(response => {
      if (!response.ok) throw new Error(`Archive unavailable (${response.status}). Run ml/train_real.py and restart the API.`)
      return response.json()
    }).then(setMetadata).catch(failure => setError(failure.message))
    load(-1, true, false)
  }, [load])

  useEffect(() => {
    if (!playing || !hasMore || loading) return undefined
    const timer = window.setTimeout(() => load(cursor, false, knownFraudOnly), 1500)
    return () => window.clearTimeout(timer)
  }, [playing, hasMore, loading, cursor, knownFraudOnly, load])

  const reset = fraudOnly => {
    setPlaying(false)
    setKnownFraudOnly(fraudOnly)
    load(-1, true, fraudOnly)
  }

  const metrics = metadata?.metrics
  const shownFraud = rows.filter(row => row.actual_fraud === 1).length
  const shownFlagged = rows.filter(row => row.model_flagged).length

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/70">
        <div className="mx-auto max-w-7xl px-6 py-5 flex items-center gap-3">
          <Shield className="h-7 w-7 text-emerald-400" />
          <div>
            <h1 className="text-xl font-bold">Fraud Intelligence Platform</h1>
            <p className="text-sm text-slate-400">Real-world historical card fraud benchmark</p>
          </div>
          <span className="ml-auto rounded-full border border-amber-500/40 bg-amber-500/10 px-3 py-1 text-xs font-semibold text-amber-300">HISTORICAL REPLAY · NOT LIVE PAYMENTS</span>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8 space-y-7">
        <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/10 p-5">
          <p className="font-semibold text-cyan-200">Real anonymized transactions, retrospective labels</p>
          <p className="mt-2 text-sm leading-6 text-slate-300">The rows below come from the ULB/Worldline archive of European card transactions from September 2013. This replay shows the time-ordered model holdout. It does not receive new payments, and the archive contains no merchant, customer, device, or country identifiers.</p>
          {metadata?.source_url && <a href={metadata.source_url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1 text-sm text-cyan-300 underline">Dataset source <ExternalLink className="h-3 w-3" /></a>}
        </div>

        {error && <div role="alert" className="flex items-center gap-2 rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-red-200"><AlertTriangle className="h-5 w-5" />{error}</div>}

        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Card label="Held-out transactions" value={metrics?.test_rows?.toLocaleString() ?? '—'} detail="Original time order" />
          <Card label="Known fraud in holdout" value={metrics?.test_fraud_rows?.toLocaleString() ?? '—'} detail="Retrospective labels" />
          <Card label="Average precision" value={percent(metrics?.average_precision)} detail="Measured on holdout" />
          <Card label="Recall" value={percent(metrics?.recall)} detail="At validation-selected threshold" />
        </section>

        <section className="rounded-xl border border-slate-800 bg-slate-900 p-5">
          <div className="flex flex-wrap items-center gap-3">
            <div className="mr-auto">
              <h2 className="text-lg font-semibold">Held-out transaction replay</h2>
              <p className="text-sm text-slate-400">The archive has {metrics?.test_rows?.toLocaleString() ?? '—'} held-out rows. This table fetches 20 at a time and retains the latest 100 displayed; Play fetches the next page every 1.5 seconds.</p>
            </div>
            <button onClick={() => reset(!knownFraudOnly)} className="rounded-lg border border-slate-700 px-3 py-2 text-sm hover:bg-slate-800">{knownFraudOnly ? 'Show all rows' : 'Show known fraud only'}</button>
            <button onClick={() => reset(knownFraudOnly)} className="rounded-lg border border-slate-700 p-2 hover:bg-slate-800" aria-label="Restart replay"><RotateCcw className="h-4 w-4" /></button>
            <button onClick={() => load(cursor, false, knownFraudOnly)} disabled={!hasMore || loading} className="rounded-lg border border-slate-700 px-3 py-2 text-sm hover:bg-slate-800 disabled:opacity-40">Load next 20</button>
            <button onClick={() => setPlaying(value => !value)} disabled={!hasMore || loading} className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 px-4 py-2 text-sm font-semibold text-slate-950 disabled:opacity-40">{playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}{playing ? 'Pause' : 'Play replay'}</button>
          </div>
          <div className="mt-5 flex flex-wrap gap-3 text-xs text-slate-400">
            <span>{viewedCount.toLocaleString()} of {(knownFraudOnly ? metrics?.test_fraud_rows : metrics?.test_rows)?.toLocaleString() ?? '—'} rows fetched</span><span>{rows.length} displayed</span><span>{shownFlagged} model-flagged in display</span><span>{shownFraud} known fraud in display</span>
            {knownFraudOnly && <span className="text-amber-300">Filtered using retrospective labels</span>}
          </div>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[650px] text-left text-sm">
              <thead className="border-b border-slate-700 text-slate-400"><tr><th className="py-3">Archive row</th><th>Time offset</th><th>Amount</th><th>XGBoost score</th><th>Model flag</th><th>Known fraud</th></tr></thead>
              <tbody>{[...rows].reverse().map(row => <tr key={row.row_id} className="border-b border-slate-800/80 hover:bg-slate-800/40">
                <td className="py-3 font-mono text-slate-300">#{row.row_id}</td>
                <td>{(row.time_offset_seconds / 3600).toFixed(1)} h</td>
                <td>{money(row.amount)}</td>
                <td className="font-mono">{percent(row.fraud_score)}</td>
                <td><Badge active={row.model_flagged} /></td>
                <td><Badge active={row.actual_fraud === 1} /></td>
              </tr>)}</tbody>
            </table>
            {!rows.length && !loading && !error && <p className="py-8 text-center text-slate-400">No archive rows found.</p>}
          </div>
          <p className="mt-4 text-xs text-slate-500">Time is seconds elapsed since the first transaction; exact customer identities and calendar timestamps are unavailable. Scores are model outputs, not calibrated probabilities.</p>
        </section>
      </main>
    </div>
  )
}

function Card({ label, value, detail }) {
  return <div className="rounded-xl border border-slate-800 bg-slate-900 p-5"><p className="text-sm text-slate-400">{label}</p><p className="mt-2 text-2xl font-bold text-white">{value}</p><p className="mt-1 text-xs text-slate-500">{detail}</p></div>
}

function Badge({ active }) {
  return <span className={`rounded-full px-2 py-1 text-xs ${active ? 'bg-red-500/15 text-red-300' : 'bg-slate-700/40 text-slate-400'}`}>{active ? 'Yes' : 'No'}</span>
}
