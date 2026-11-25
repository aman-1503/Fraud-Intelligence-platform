import { useCallback, useEffect, useState } from 'react'
import { ShieldCheck, AlertTriangle, RefreshCw } from 'lucide-react'

const actions = ['APPROVE', 'BLOCK', 'REQUEST_VERIFICATION', 'ESCALATE_FOR_REVIEW']
const initial = { transaction_id: '', user_id: '', amount: '', merchant_category: '', country: '', device_id: '' }

const token = import.meta.env.VITE_API_TOKEN || ''

async function api(path, method = 'GET', body) {
  const headers = {}
  if (body) headers['Content-Type'] = 'application/json'
  if (token) headers.Authorization = `Bearer ${token}`
  const response = await fetch(`/api/v1${path}`, {
    method, headers, body: body ? JSON.stringify(body) : undefined,
  })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    if (response.status === 401) throw new Error('Not authenticated. Set VITE_API_TOKEN to an admin bearer token.')
    throw new Error(typeof data.detail === 'string' ? data.detail : `API request failed (${response.status})`)
  }
  return data
}

export default function InvestigationDashboard() {
  const [form, setForm] = useState(initial)
  const [score, setScore] = useState(null)
  const [caseData, setCaseData] = useState(null)
  const [reviewCases, setReviewCases] = useState([])
  const [providerStatus, setProviderStatus] = useState(null)
  const [finalAction, setFinalAction] = useState('REQUEST_VERIFICATION')
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const refreshQueue = useCallback(async () => {
    try { setReviewCases(await api('/review/cases')) } catch (failure) { setError(failure.message) }
  }, [])
  useEffect(() => {
    refreshQueue()
    api('/investigations/provider-status').then(setProviderStatus).catch(failure => setError(failure.message))
  }, [refreshQueue])

  useEffect(() => {
    if (caseData?.status !== 'PENDING') return undefined
    let active = true
    const timer = window.setInterval(async () => {
      try {
        const updated = await api(`/investigations/${caseData.case_id}`)
        if (active) {
          setCaseData(updated)
          if (updated.status === 'REVIEW_REQUIRED') refreshQueue()
        }
      } catch (failure) { if (active) setError(failure.message) }
    }, 1000)
    return () => { active = false; window.clearInterval(timer) }
  }, [caseData?.case_id, caseData?.status, refreshQueue])

  const submit = async event => {
    event.preventDefault()
    setBusy(true); setError(''); setScore(null); setCaseData(null)
    try {
      const transaction = { ...form, amount: Number(form.amount), currency: 'USD' }
      const scored = await api('/score', 'POST', transaction)
      setScore(scored)
      setCaseData(await api(`/investigations/${encodeURIComponent(form.transaction_id)}`, 'POST'))
    } catch (failure) { setError(failure.message) }
    finally { setBusy(false) }
  }

  const review = async event => {
    event.preventDefault()
    if (!caseData) return
    setBusy(true); setError('')
    try {
      const updated = await api(`/review/cases/${caseData.case_id}`, 'POST',
        { final_action: finalAction, note })
      setCaseData(updated)
      await refreshQueue()
    } catch (failure) { setError(failure.message) }
    finally { setBusy(false) }
  }

  return <div className="min-h-screen bg-slate-950 text-slate-100">
    <header className="border-b border-slate-800 bg-slate-900/70"><div className="mx-auto flex max-w-7xl items-center gap-3 px-6 py-5">
      <ShieldCheck className="h-7 w-7 text-emerald-400" /><div><h1 className="text-xl font-bold">Investigate a submitted card transaction</h1>
        <p className="text-sm text-slate-400">Immediate score → flagged-case investigation → human review</p></div></div></header>
    <main className="mx-auto max-w-7xl space-y-6 px-6 py-8">
      <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/10 p-5 text-sm leading-6 text-slate-300">
        Enter a transaction you are permitted to use. This API is not connected to Stripe or a live card-payment feed. The LLM runs only for flagged cases with available evidence when the backend is configured with <code>INVESTIGATION_PROVIDER=openai</code> and an API key. Each saved case states which mode actually ran. No automatic email or payment action is sent.
        {providerStatus && <p className="mt-3 font-semibold text-cyan-200">Current backend: {providerStatus.llm_configured
          ? `OpenAI configured (${providerStatus.model}); check the case mode for the actual result.`
          : `Deterministic investigation (${providerStatus.configured_provider}${providerStatus.configured_provider === 'openai' ? ', API key missing' : ''}).`}</p>}
      </div>
      {error && <div role="alert" className="flex items-center gap-2 rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-red-200"><AlertTriangle className="h-5 w-5" />{error}</div>}
      <div className="grid gap-6 lg:grid-cols-2">
        <form onSubmit={submit} className="space-y-4 rounded-xl border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-lg font-semibold">Submit transaction</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {[
              ['transaction_id', 'Transaction ID'], ['user_id', 'Account ID'],
              ['amount', 'Amount (USD)'], ['merchant_category', 'Merchant category'],
              ['country', 'Country code'], ['device_id', 'Device ID (optional)'],
            ].map(([name, label]) => <label key={name} className="text-sm text-slate-300">{label}
              <input className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" name={name}
                type={name === 'amount' ? 'number' : 'text'} min={name === 'amount' ? '0.01' : undefined}
                step={name === 'amount' ? '0.01' : undefined} required={name !== 'device_id'} value={form[name]}
                onChange={event => setForm(previous => ({ ...previous, [name]: event.target.value }))} />
            </label>)}
          </div>
          <button disabled={busy} className="rounded-lg bg-emerald-500 px-4 py-2 font-semibold text-slate-950 disabled:opacity-40">{busy ? 'Working…' : 'Score and investigate'}</button>
        </form>
        <section className="space-y-4 rounded-xl border border-slate-800 bg-slate-900 p-5">
          <div className="flex items-center"><h2 className="text-lg font-semibold">Case result</h2>
            {caseData && <button onClick={async () => { try { setCaseData(await api(`/investigations/${caseData.case_id}`)) } catch (failure) { setError(failure.message) } }}
              className="ml-auto rounded-lg border border-slate-700 p-2" aria-label="Refresh case"><RefreshCw className="h-4 w-4" /></button>}</div>
          {!caseData && <p className="text-sm text-slate-400">Submit a transaction to see its score, evidence, recommendation, and review status.</p>}
          {score && <p className="text-sm">Immediate score: <strong>{(score.fraud_score * 100).toFixed(1)}%</strong> · {score.risk_level} · scorer: <code>{score.model_version}</code></p>}
          {caseData && <div className="space-y-3 text-sm">
            <p>Status: <strong>{caseData.status}</strong> · Investigation mode: <strong>{caseData.investigation_mode ?? 'pending'}</strong>{caseData.provider_fallback && ' (model fallback)'}</p>
            {caseData.error && <p className="text-red-300">{caseData.error}</p>}
            {caseData.result && <><p>Recommendation: <strong>{caseData.result.recommended_action}</strong> · confidence: {(caseData.result.confidence * 100).toFixed(0)}%</p>
              <p className="text-slate-300">{caseData.result.risk_summary}</p>
              <p>Evidence cited: {caseData.result.evidence_ids.join(', ')}</p></>}
            {!!caseData.review_reasons?.length && <p>Human review required because: <strong>{caseData.review_reasons.join(', ')}</strong></p>}
            {!!caseData.evidence?.gaps?.length && <div><p className="font-semibold">Missing evidence and next checks</p><ul className="mt-2 list-disc space-y-2 pl-5 text-slate-300">{caseData.evidence.gaps.map(gap => <li key={gap.signal}><strong>{gap.signal}</strong>: {gap.reason} {gap.next_check}</li>)}</ul></div>}
            {caseData.status === 'REVIEWED' && <p>Saved human action: <strong>{caseData.final_action}</strong> by {caseData.reviewer}</p>}
          </div>}
          {caseData?.status === 'REVIEW_REQUIRED' && <form onSubmit={review} className="space-y-2 border-t border-slate-700 pt-4 text-sm">
            <h3 className="font-semibold">Complete human review</h3>
            <p className="text-slate-400">The reviewer name is the subject of the signed-in token.</p>
            <select className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2" value={finalAction} onChange={event => setFinalAction(event.target.value)}>{actions.map(action => <option key={action}>{action}</option>)}</select>
            <input className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2" placeholder="Reason for final action" required value={note} onChange={event => setNote(event.target.value)} />
            <button disabled={busy} className="rounded-lg bg-emerald-500 px-4 py-2 font-semibold text-slate-950 disabled:opacity-40">Save review</button>
          </form>}
        </section>
      </div>
      <section className="rounded-xl border border-slate-800 bg-slate-900 p-5"><div className="flex items-center"><h2 className="text-lg font-semibold">Cases waiting for review</h2>
        <button onClick={refreshQueue} className="ml-auto rounded-lg border border-slate-700 px-3 py-2 text-sm">Refresh</button></div>
        <p className="mt-1 text-sm text-slate-400">These are saved investigation cases, not customer notifications.</p>
        <div className="mt-4 space-y-2">{reviewCases.map(item => <button key={item.case_id} onClick={() => { setScore(null); setCaseData(item) }}
          className="block w-full rounded-lg border border-slate-700 p-3 text-left text-sm hover:bg-slate-800">{item.transaction_id} · {item.result?.recommended_action ?? item.status} · {item.investigation_mode ?? 'unknown mode'}</button>)}
          {!reviewCases.length && <p className="text-sm text-slate-400">No cases currently require review.</p>}</div>
      </section>
    </main>
  </div>
}
