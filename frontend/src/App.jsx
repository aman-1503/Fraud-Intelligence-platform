import { useState } from 'react'
import LiveBitcoinDashboard from './LiveBitcoinDashboard'
import RealDataDashboard from './RealDataDashboard'

export default function App() {
  const [tab, setTab] = useState('live')
  return <>
    <nav className="flex gap-2 border-b border-slate-800 bg-slate-950 px-6 py-3 text-sm text-slate-100" aria-label="Data views">
      <button onClick={() => setTab('live')} className={`rounded-lg px-4 py-2 ${tab === 'live' ? 'bg-emerald-500 text-slate-950 font-semibold' : 'bg-slate-800 hover:bg-slate-700'}`}>Live Bitcoin API</button>
      <button onClick={() => setTab('archive')} className={`rounded-lg px-4 py-2 ${tab === 'archive' ? 'bg-emerald-500 text-slate-950 font-semibold' : 'bg-slate-800 hover:bg-slate-700'}`}>Real card fraud archive</button>
    </nav>
    {tab === 'live' ? <LiveBitcoinDashboard /> : <RealDataDashboard />}
  </>
}
