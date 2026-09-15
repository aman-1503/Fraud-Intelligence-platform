import { useEffect, useRef, useState } from 'react'
import { Activity, ExternalLink, RefreshCw, AlertTriangle } from 'lucide-react'
import { accumulateTransactions } from './accumulateTransactions'

const btc = satoshis => `${(satoshis / 100_000_000).toFixed(8)} BTC`
const feeRate = row => `${(row.fee / row.vsize).toFixed(1)} sat/vB`

export default function LiveBitcoinDashboard() {
  const [feed, setFeed] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [observed, setObserved] = useState([])
  const [newCount, setNewCount] = useState(0)
  const seenIds = useRef(new Set())
  const observedRows = useRef([])

  useEffect(() => {
    let active = true
    const refresh = async () => {
      try {
        const response = await fetch('/api/v1/live/bitcoin/recent')
        if (!response.ok) throw new Error(`Live source unavailable (${response.status}). Check the API logs or try again later.`)
        const data = await response.json()
        if (active) {
          const firstSeen = new Date().toISOString()
          const accumulated = accumulateTransactions(observedRows.current, data.transactions, seenIds.current, firstSeen)
          observedRows.current = accumulated.rows
          setObserved(accumulated.rows)
          setNewCount(accumulated.newCount)
          setFeed(data)
          setError('')
        }
      } catch (failure) {
        if (active) setError(failure.message)
      } finally {
        if (active) setLoading(false)
      }
    }
    refresh()
    const timer = window.setInterval(refresh, 10_000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])

  return <div className="min-h-screen bg-slate-950 text-slate-100">
    <header className="border-b border-slate-800 bg-slate-900/70">
      <div className="mx-auto flex max-w-7xl items-center gap-3 px-6 py-5">
        <Activity className="h-7 w-7 text-emerald-400" />
        <div><h1 className="text-xl font-bold">Live Bitcoin transaction activity</h1><p className="text-sm text-slate-400">Public mainnet mempool · unconfirmed transactions</p></div>
        <span className="ml-auto rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-300">LIVE API · REFRESHES EVERY 10S</span>
      </div>
    </header>
    <main className="mx-auto max-w-7xl space-y-7 px-6 py-8">
      <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/10 p-5">
        <p className="font-semibold text-cyan-200">Public Bitcoin mempool snapshots</p>
        <p className="mt-2 text-sm leading-6 text-slate-300">mempool.space returns only the latest ten transactions to enter its mempool per request. This view polls every ten seconds and retains up to 100 distinct IDs first seen while the tab is open. “New since last poll” means an ID was absent from this tab’s earlier snapshots; it is not a transaction creation timestamp. Polling can miss transactions between requests. These are blockchain transactions, not card payments or fraud-labeled records.</p>
        <a href="https://mempool.space/docs/api/rest" target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1 text-sm text-cyan-300 underline">API documentation <ExternalLink className="h-3 w-3" /></a>
      </div>
      {error && <div role="alert" className="flex items-center gap-2 rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-red-200"><AlertTriangle className="h-5 w-5" />{error}</div>}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card label="Latest API snapshot" value={feed ? `${feed.transactions.length} transactions` : '—'} detail="The source endpoint is limited to ten" />
        <Card label="Seen in this tab" value={observed.length.toLocaleString()} detail="Up to 100 distinct IDs retained" />
        <Card label="New since last poll" value={feed ? newCount.toLocaleString() : '—'} detail="Not a creation or confirmation count" />
        <Card label="Server fetched" value={feed ? new Date(feed.observed_at).toLocaleTimeString() : '—'} detail="Snapshot time, not transaction time" />
      </section>
      <section className="rounded-xl border border-slate-800 bg-slate-900 p-5">
        <div className="flex items-center gap-2"><h2 className="text-lg font-semibold">Transactions first seen in this tab</h2><RefreshCw className="ml-auto h-4 w-4 text-emerald-400" /></div>
        <p className="mt-1 text-sm text-slate-400">Value is reported by the API, not an inferred payment or fraud loss. Earlier entries may have confirmed or left the mempool since first observation.</p>
        <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[650px] text-left text-sm">
          <thead className="border-b border-slate-700 text-slate-400"><tr><th className="py-3">Transaction ID</th><th>Reported value</th><th>Fee</th><th>Virtual size</th><th>Fee rate</th><th>First seen here</th></tr></thead>
          <tbody>{observed.map(row => <tr key={row.txid} className="border-b border-slate-800/80 hover:bg-slate-800/40">
            <td className="py-3 font-mono"><a className="inline-flex items-center gap-1 text-cyan-300 hover:underline" href={`https://mempool.space/tx/${row.txid}`} target="_blank" rel="noreferrer">{row.txid.slice(0, 12)}…{row.txid.slice(-8)}<ExternalLink className="h-3 w-3" /></a></td>
            <td className="font-mono">{btc(row.value)}</td><td>{row.fee.toLocaleString()} sats</td><td>{row.vsize.toLocaleString()} vB</td><td>{feeRate(row)}</td><td>{new Date(row.firstSeen).toLocaleTimeString()}</td>
          </tr>)}</tbody>
        </table>{loading && <p className="py-8 text-center text-slate-400">Loading the public API…</p>}</div>
      </section>
    </main>
  </div>
}

function Card({ label, value, detail }) {
  return <div className="rounded-xl border border-slate-800 bg-slate-900 p-5"><p className="text-sm text-slate-400">{label}</p><p className="mt-2 text-2xl font-bold text-white">{value}</p><p className="mt-1 text-xs text-slate-500">{detail}</p></div>
}
