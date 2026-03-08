import React, { useState, useEffect, useCallback } from 'react';
import { 
  Shield, AlertTriangle, CheckCircle, XCircle, 
  Activity, TrendingUp, Clock, Users, 
  BarChart3, RefreshCw, Search, Filter
} from 'lucide-react';

// Fraud Intelligence Platform Dashboard
export default function FraudDashboard() {
  const [stats, setStats] = useState({
    totalTransactions: 0,
    fraudDetected: 0,
    fraudRate: 0,
    avgLatency: 0,
    throughput: 0,
  });
  
  const [recentTransactions, setRecentTransactions] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [isLive, setIsLive] = useState(true);
  const [selectedTab, setSelectedTab] = useState('overview');
  const [searchQuery, setSearchQuery] = useState('');

  // Simulate real-time data updates
  useEffect(() => {
    if (!isLive) return;
    
    const interval = setInterval(() => {
      // Update stats
      setStats(prev => ({
        totalTransactions: prev.totalTransactions + Math.floor(Math.random() * 50),
        fraudDetected: prev.fraudDetected + (Math.random() > 0.97 ? 1 : 0),
        fraudRate: 2.3 + (Math.random() - 0.5) * 0.4,
        avgLatency: 35 + Math.random() * 20,
        throughput: 2800 + Math.random() * 500,
      }));
      
      // Add new transaction
      const newTxn = generateTransaction();
      setRecentTransactions(prev => [newTxn, ...prev.slice(0, 49)]);
      
      // Random alert
      if (Math.random() > 0.95) {
        setAlerts(prev => [{
          id: Date.now(),
          type: Math.random() > 0.5 ? 'high_risk' : 'velocity',
          message: Math.random() > 0.5 
            ? 'High-risk transaction detected from new device'
            : 'Unusual velocity pattern detected for user',
          timestamp: new Date().toISOString(),
          transactionId: newTxn.id,
        }, ...prev.slice(0, 9)]);
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }, [isLive]);

  // Initialize with sample data
  useEffect(() => {
    setStats({
      totalTransactions: 1247893,
      fraudDetected: 3124,
      fraudRate: 2.5,
      avgLatency: 42,
      throughput: 3150,
    });
    
    setRecentTransactions(Array(20).fill(null).map(() => generateTransaction()));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-xl">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                  Fraud Intelligence Platform
                </h1>
                <p className="text-xs text-slate-500">Real-time fraud detection & analytics</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-sm">
                <span className={`w-2 h-2 rounded-full ${isLive ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'}`} />
                <span className="text-slate-400">{isLive ? 'Live' : 'Paused'}</span>
              </div>
              
              <button
                onClick={() => setIsLive(!isLive)}
                className={`p-2 rounded-lg transition-colors ${
                  isLive ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-400'
                }`}
              >
                <RefreshCw className={`w-4 h-4 ${isLive ? 'animate-spin' : ''}`} style={{ animationDuration: '3s' }} />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-5 gap-4 mb-8">
          <StatCard
            label="Total Transactions"
            value={stats.totalTransactions.toLocaleString()}
            icon={<Activity className="w-5 h-5" />}
            color="blue"
          />
          <StatCard
            label="Fraud Detected"
            value={stats.fraudDetected.toLocaleString()}
            icon={<AlertTriangle className="w-5 h-5" />}
            color="red"
          />
          <StatCard
            label="Fraud Rate"
            value={`${stats.fraudRate.toFixed(2)}%`}
            icon={<TrendingUp className="w-5 h-5" />}
            color="amber"
          />
          <StatCard
            label="Avg Latency"
            value={`${stats.avgLatency.toFixed(0)}ms`}
            icon={<Clock className="w-5 h-5" />}
            color="emerald"
            highlight={stats.avgLatency < 50}
          />
          <StatCard
            label="Throughput"
            value={`${stats.throughput.toFixed(0)} RPS`}
            icon={<BarChart3 className="w-5 h-5" />}
            color="purple"
            highlight={stats.throughput > 3000}
          />
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-3 gap-6">
          {/* Transactions Table */}
          <div className="col-span-2 bg-slate-900/50 rounded-2xl border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <h2 className="font-semibold">Recent Transactions</h2>
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                  <input
                    type="text"
                    placeholder="Search transactions..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9 pr-4 py-1.5 bg-slate-800 rounded-lg text-sm border border-slate-700 focus:border-emerald-500 focus:outline-none w-64"
                  />
                </div>
                <button className="p-2 bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors">
                  <Filter className="w-4 h-4" />
                </button>
              </div>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-xs text-slate-500 uppercase tracking-wider">
                    <th className="px-4 py-3">Transaction ID</th>
                    <th className="px-4 py-3">User</th>
                    <th className="px-4 py-3">Amount</th>
                    <th className="px-4 py-3">Category</th>
                    <th className="px-4 py-3">Score</th>
                    <th className="px-4 py-3">Decision</th>
                    <th className="px-4 py-3">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {recentTransactions
                    .filter(txn => 
                      searchQuery === '' || 
                      txn.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
                      txn.userId.toLowerCase().includes(searchQuery.toLowerCase())
                    )
                    .slice(0, 15)
                    .map((txn) => (
                      <TransactionRow key={txn.id} transaction={txn} />
                    ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Alerts Panel */}
          <div className="bg-slate-900/50 rounded-2xl border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800">
              <h2 className="font-semibold flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-500" />
                Active Alerts
                <span className="ml-auto bg-red-500/20 text-red-400 px-2 py-0.5 rounded-full text-xs">
                  {alerts.length}
                </span>
              </h2>
            </div>
            
            <div className="p-4 space-y-3 max-h-96 overflow-y-auto">
              {alerts.length === 0 ? (
                <div className="text-center py-8 text-slate-500">
                  <CheckCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No active alerts</p>
                </div>
              ) : (
                alerts.map((alert) => (
                  <AlertCard key={alert.id} alert={alert} />
                ))
              )}
            </div>
          </div>
        </div>

        {/* Performance Chart */}
        <div className="mt-6 bg-slate-900/50 rounded-2xl border border-slate-800 p-6">
          <h2 className="font-semibold mb-4">Latency Distribution (Last Hour)</h2>
          <LatencyChart />
        </div>

        {/* Model Info */}
        <div className="mt-6 grid grid-cols-3 gap-4">
          <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4">
            <h3 className="text-sm text-slate-400 mb-2">Model Version</h3>
            <p className="text-lg font-mono">v1.0.0 (XGBoost)</p>
            <p className="text-xs text-slate-500 mt-1">Last updated: 2024-01-15</p>
          </div>
          <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4">
            <h3 className="text-sm text-slate-400 mb-2">Model Performance</h3>
            <p className="text-lg font-mono">AUC: 0.967</p>
            <p className="text-xs text-slate-500 mt-1">Precision: 94.2% | Recall: 89.7%</p>
          </div>
          <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4">
            <h3 className="text-sm text-slate-400 mb-2">System Status</h3>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-emerald-500 rounded-full" />
              <p className="text-lg">All Systems Operational</p>
            </div>
            <p className="text-xs text-slate-500 mt-1">Uptime: 99.97% (30d)</p>
          </div>
        </div>
      </main>
    </div>
  );
}

// Stat Card Component
function StatCard({ label, value, icon, color, highlight }) {
  const colorClasses = {
    blue: 'from-blue-500/20 to-blue-600/5 border-blue-500/30 text-blue-400',
    red: 'from-red-500/20 to-red-600/5 border-red-500/30 text-red-400',
    amber: 'from-amber-500/20 to-amber-600/5 border-amber-500/30 text-amber-400',
    emerald: 'from-emerald-500/20 to-emerald-600/5 border-emerald-500/30 text-emerald-400',
    purple: 'from-purple-500/20 to-purple-600/5 border-purple-500/30 text-purple-400',
  };

  return (
    <div className={`
      bg-gradient-to-br ${colorClasses[color]} 
      rounded-xl border p-4 
      ${highlight ? 'ring-2 ring-emerald-500/50' : ''}
    `}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-slate-400 text-sm">{label}</span>
        {icon}
      </div>
      <p className="text-2xl font-bold">{value}</p>
    </div>
  );
}

// Transaction Row Component
function TransactionRow({ transaction }) {
  const getDecisionStyle = (decision) => {
    switch (decision) {
      case 'APPROVE':
        return 'bg-emerald-500/20 text-emerald-400';
      case 'REVIEW':
        return 'bg-amber-500/20 text-amber-400';
      case 'DECLINE':
        return 'bg-red-500/20 text-red-400';
      default:
        return 'bg-slate-500/20 text-slate-400';
    }
  };

  const getScoreColor = (score) => {
    if (score < 0.3) return 'text-emerald-400';
    if (score < 0.6) return 'text-amber-400';
    return 'text-red-400';
  };

  return (
    <tr className="hover:bg-slate-800/50 transition-colors">
      <td className="px-4 py-3 font-mono text-sm">{transaction.id}</td>
      <td className="px-4 py-3 text-sm">{transaction.userId}</td>
      <td className="px-4 py-3 font-mono">${transaction.amount.toFixed(2)}</td>
      <td className="px-4 py-3 text-sm capitalize">{transaction.category.replace('_', ' ')}</td>
      <td className={`px-4 py-3 font-mono ${getScoreColor(transaction.score)}`}>
        {transaction.score.toFixed(3)}
      </td>
      <td className="px-4 py-3">
        <span className={`px-2 py-0.5 rounded-full text-xs ${getDecisionStyle(transaction.decision)}`}>
          {transaction.decision}
        </span>
      </td>
      <td className="px-4 py-3 text-sm text-slate-400">{transaction.latency}ms</td>
    </tr>
  );
}

// Alert Card Component
function AlertCard({ alert }) {
  const getAlertStyle = (type) => {
    switch (type) {
      case 'high_risk':
        return { bg: 'bg-red-500/10', border: 'border-red-500/30', icon: <XCircle className="w-4 h-4 text-red-400" /> };
      case 'velocity':
        return { bg: 'bg-amber-500/10', border: 'border-amber-500/30', icon: <AlertTriangle className="w-4 h-4 text-amber-400" /> };
      default:
        return { bg: 'bg-slate-500/10', border: 'border-slate-500/30', icon: <Activity className="w-4 h-4 text-slate-400" /> };
    }
  };

  const style = getAlertStyle(alert.type);

  return (
    <div className={`${style.bg} ${style.border} border rounded-lg p-3`}>
      <div className="flex items-start gap-2">
        {style.icon}
        <div className="flex-1 min-w-0">
          <p className="text-sm">{alert.message}</p>
          <p className="text-xs text-slate-500 mt-1">
            {new Date(alert.timestamp).toLocaleTimeString()} • {alert.transactionId}
          </p>
        </div>
      </div>
    </div>
  );
}

// Latency Chart Component
function LatencyChart() {
  const data = Array(60).fill(null).map((_, i) => ({
    minute: i,
    p50: 30 + Math.random() * 15,
    p95: 60 + Math.random() * 30,
    p99: 90 + Math.random() * 40,
  }));

  const maxValue = 150;
  const height = 200;
  const width = 1000;

  return (
    <div className="relative h-52">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
        {/* Grid lines */}
        {[0, 50, 100, 150].map((val) => (
          <g key={val}>
            <line
              x1="0"
              y1={height - (val / maxValue) * height}
              x2={width}
              y2={height - (val / maxValue) * height}
              stroke="rgb(51, 65, 85)"
              strokeWidth="1"
              strokeDasharray="4"
            />
            <text
              x="0"
              y={height - (val / maxValue) * height - 5}
              fill="rgb(100, 116, 139)"
              fontSize="10"
            >
              {val}ms
            </text>
          </g>
        ))}

        {/* Target line at 150ms */}
        <line
          x1="0"
          y1={height - (150 / maxValue) * height}
          x2={width}
          y2={height - (150 / maxValue) * height}
          stroke="rgb(239, 68, 68)"
          strokeWidth="2"
          strokeDasharray="8"
        />
        <text
          x={width - 60}
          y={height - (150 / maxValue) * height - 5}
          fill="rgb(239, 68, 68)"
          fontSize="10"
        >
          SLA: 150ms
        </text>

        {/* P99 area */}
        <path
          d={`
            M 0 ${height}
            ${data.map((d, i) => `L ${(i / 59) * width} ${height - (d.p99 / maxValue) * height}`).join(' ')}
            L ${width} ${height}
            Z
          `}
          fill="url(#p99Gradient)"
          opacity="0.3"
        />

        {/* P95 line */}
        <path
          d={data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${(i / 59) * width} ${height - (d.p95 / maxValue) * height}`).join(' ')}
          fill="none"
          stroke="rgb(251, 191, 36)"
          strokeWidth="2"
        />

        {/* P50 line */}
        <path
          d={data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${(i / 59) * width} ${height - (d.p50 / maxValue) * height}`).join(' ')}
          fill="none"
          stroke="rgb(52, 211, 153)"
          strokeWidth="2"
        />

        {/* Gradient definition */}
        <defs>
          <linearGradient id="p99Gradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgb(239, 68, 68)" />
            <stop offset="100%" stopColor="rgb(239, 68, 68)" stopOpacity="0" />
          </linearGradient>
        </defs>
      </svg>

      {/* Legend */}
      <div className="absolute bottom-0 right-0 flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-emerald-400 rounded" />
          <span className="text-slate-400">P50</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-amber-400 rounded" />
          <span className="text-slate-400">P95</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-red-400 rounded" />
          <span className="text-slate-400">P99</span>
        </div>
      </div>
    </div>
  );
}

// Data generation helper
function generateTransaction() {
  const categories = ['grocery', 'gas_station', 'restaurant', 'online_retail', 'electronics', 'travel'];
  const decisions = ['APPROVE', 'APPROVE', 'APPROVE', 'APPROVE', 'REVIEW', 'DECLINE'];
  
  const score = Math.random();
  let decision = 'APPROVE';
  if (score > 0.85) decision = 'DECLINE';
  else if (score > 0.6) decision = 'REVIEW';

  return {
    id: `TXN-${Math.random().toString(36).substr(2, 8).toUpperCase()}`,
    userId: `U${Math.floor(Math.random() * 10000).toString().padStart(6, '0')}`,
    amount: Math.random() * 2000 + 10,
    category: categories[Math.floor(Math.random() * categories.length)],
    score: score,
    decision: decision,
    latency: Math.floor(20 + Math.random() * 60),
    timestamp: new Date().toISOString(),
  };
}
