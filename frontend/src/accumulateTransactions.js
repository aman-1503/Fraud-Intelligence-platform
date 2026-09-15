/** Keep a bounded view of distinct IDs first seen while this tab is open. */
export function accumulateTransactions(previous, snapshot, seenIds, firstSeen, maxRows = 100) {
  const fresh = snapshot.filter((row, index) =>
    !seenIds.has(row.txid) && snapshot.findIndex(item => item.txid === row.txid) === index)
  fresh.forEach(row => seenIds.add(row.txid))
  const details = new Map(snapshot.map(row => [row.txid, row]))
  return {
    rows: [...fresh.map(row => ({ ...row, firstSeen })), ...previous]
      .map(row => ({ ...row, ...details.get(row.txid), firstSeen: row.firstSeen }))
      .slice(0, maxRows),
    newCount: fresh.length,
  }
}

/** The documented low-bandwidth WebSocket event provides IDs only. */
export function parseMempoolTxidsMessage(raw) {
  const event = raw?.['mempool-txids']
  if (!event || !Array.isArray(event.added)) return []
  return event.added.filter(id => typeof id === 'string' && /^[0-9a-f]{64}$/.test(id))
    .map(txid => ({ txid, fee: null, vsize: null, value: null }))
}
