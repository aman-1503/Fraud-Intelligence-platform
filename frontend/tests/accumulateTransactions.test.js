import test from 'node:test'
import assert from 'node:assert/strict'
import { accumulateTransactions } from '../src/accumulateTransactions.js'

test('distinct transactions accumulate across ten-item snapshots without counting repeats', () => {
  const seen = new Set()
  const first = Array.from({ length: 10 }, (_, i) => ({ txid: `tx-${i}` }))
  let result = accumulateTransactions([], first, seen, 'first-poll')
  assert.equal(result.newCount, 10)
  assert.equal(result.rows.length, 10)
  result = accumulateTransactions(result.rows, first, seen, 'repeat-poll')
  assert.equal(result.newCount, 0)
  assert.equal(result.rows.length, 10)
  const second = Array.from({ length: 10 }, (_, i) => ({ txid: `tx-${i + 10}` }))
  result = accumulateTransactions(result.rows, second, seen, 'second-poll')
  assert.equal(result.newCount, 10)
  assert.equal(result.rows.length, 20)
  assert.equal(result.rows[0].firstSeen, 'second-poll')
})

test('display is bounded at 100 while distinct-ID counts remain accurate', () => {
  const seen = new Set()
  let rows = []
  for (let poll = 0; poll < 12; poll++) {
    const snapshot = Array.from({ length: 10 }, (_, i) => ({ txid: `tx-${poll * 10 + i}` }))
    const result = accumulateTransactions(rows, snapshot, seen, `poll-${poll}`)
    assert.equal(result.newCount, 10)
    rows = result.rows
  }
  assert.equal(rows.length, 100)
  assert.equal(seen.size, 120)
  assert.equal(rows[0].txid, 'tx-110')
})
