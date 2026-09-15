/** Keep a bounded view of distinct IDs first seen while this tab is open. */
export function accumulateTransactions(previous, snapshot, seenIds, firstSeen, maxRows = 100) {
  const fresh = snapshot.filter(row => !seenIds.has(row.txid))
  fresh.forEach(row => seenIds.add(row.txid))
  return {
    rows: [...fresh.map(row => ({ ...row, firstSeen })), ...previous].slice(0, maxRows),
    newCount: fresh.length,
  }
}
