import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Loader2, Search, RefreshCw } from 'lucide-react'
import { getPatients, getModelStats } from '@/api/client'
import { MetricCard } from '@/components/MetricCard'
import { RiskBadge } from '@/components/RiskBadge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { Patient } from '@/types'

type SortKey = 'risk_score' | 'stay_id' | 'age' | 'first_careunit'
type SortDir = 'asc' | 'desc'
type TierFilter = 'ALL' | 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW'

const TIER_PILLS: { label: TierFilter; active: string }[] = [
  { label: 'ALL',      active: 'bg-slate-200 text-slate-800 border-slate-400' },
  { label: 'CRITICAL', active: 'bg-red-950/10 text-red-900 border-red-400' },
  { label: 'HIGH',     active: 'bg-red-100 text-red-700 border-red-300' },
  { label: 'MODERATE', active: 'bg-amber-100 text-amber-700 border-amber-300' },
  { label: 'LOW',      active: 'bg-emerald-100 text-emerald-700 border-emerald-300' },
]

function SortArrow({ col, sortKey, sortDir }: { col: SortKey; sortKey: SortKey; sortDir: SortDir }) {
  if (sortKey !== col) return <span className="ml-1 opacity-30">↕</span>
  return <span className="ml-1">{sortDir === 'asc' ? '↑' : '↓'}</span>
}

export function LiveMonitor() {
  const navigate = useNavigate()
  const [search, setSearch]         = useState('')
  const [tierFilter, setTierFilter] = useState<TierFilter>('ALL')
  const [unitFilter, setUnitFilter] = useState('ALL')
  const [sortKey, setSortKey]       = useState<SortKey>('risk_score')
  const [sortDir, setSortDir]       = useState<SortDir>('desc')

  const {
    data: patients = [],
    isLoading: patientsLoading,
    dataUpdatedAt,
    refetch,
    isFetching,
  } = useQuery({
    queryKey: ['patients'],
    queryFn: getPatients,
    refetchInterval: 30_000,
  })

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: getModelStats,
  })

  const criticalCount = patients.filter((p) => p.risk_label === 'CRITICAL').length
  const highCount     = patients.filter((p) => p.risk_label === 'HIGH').length
  const modCount      = patients.filter((p) => p.risk_label === 'MODERATE').length
  const lowCount      = patients.filter((p) => p.risk_label === 'LOW').length

  const distData = [
    { label: 'CRITICAL', count: criticalCount, fill: '#7f1d1d' },
    { label: 'HIGH',     count: highCount,     fill: '#e74c3c' },
    { label: 'MODERATE', count: modCount,      fill: '#f39c12' },
    { label: 'LOW',      count: lowCount,      fill: '#27ae60' },
  ]

  const careUnits = useMemo(() => {
    const units = [...new Set(patients.map((p) => p.first_careunit).filter(Boolean))]
    return units.sort()
  }, [patients])

  const filtered = useMemo(() => {
    let list: Patient[] = [...patients]
    if (search.trim()) {
      const q = search.trim().toLowerCase()
      list = list.filter(
        (p) =>
          String(p.stay_id).includes(q) ||
          (p.first_careunit ?? '').toLowerCase().includes(q),
      )
    }
    if (tierFilter !== 'ALL') list = list.filter((p) => p.risk_label === tierFilter)
    if (unitFilter !== 'ALL') list = list.filter((p) => p.first_careunit === unitFilter)

    list.sort((a, b) => {
      const va = a[sortKey] ?? ''
      const vb = b[sortKey] ?? ''
      if (typeof va === 'string' && typeof vb === 'string') {
        return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va)
      }
      return sortDir === 'asc'
        ? (va as number) - (vb as number)
        : (vb as number) - (va as number)
    })
    return list
  }, [patients, search, tierFilter, unitFilter, sortKey, sortDir])

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('desc') }
  }

  const lastUpdated = dataUpdatedAt
    ? new Date(dataUpdatedAt).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
    : null

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight">ICU Live Monitor</h1>
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 border border-amber-200">
            Demo mode · static snapshot
          </span>
          {/* Last-updated badge */}
          <div className="flex items-center gap-1.5 ml-auto text-xs text-muted-foreground">
            <button
              onClick={() => refetch()}
              disabled={isFetching}
              className="hover:text-foreground transition-colors"
              title="Refresh patient list"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isFetching ? 'animate-spin' : ''}`} />
            </button>
            {lastUpdated && <span>Updated {lastUpdated}</span>}
          </div>
        </div>
        <p className="text-muted-foreground text-sm mt-1">
          Sepsis risk assessment for sampled ICU patients — updates hourly in production with live hospital feeds
        </p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard
          title="Active Patients"
          value={patients.length}
          description="Sampled cohort"
        />
        <MetricCard
          title="Critical / High"
          value={criticalCount + highCount}
          description={`${criticalCount} critical · ${highCount} high`}
          valueClassName="text-destructive"
        />
        <MetricCard
          title="Moderate Risk"
          value={modCount}
          description="Risk score 0.40–0.60"
          valueClassName="text-warning"
        />
        <MetricCard
          title="Model AUROC"
          value={stats ? stats.auroc.toFixed(3) : '—'}
          description="SepsisAlert vs NEWS2"
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Risk distribution chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Risk Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={distData} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {distData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Patient table */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">
                  Patient List
                  {filtered.length !== patients.length && (
                    <span className="ml-2 text-xs font-normal text-muted-foreground">
                      {filtered.length} of {patients.length}
                    </span>
                  )}
                </CardTitle>
              </div>

              {/* Search + filters */}
              <div className="space-y-2 pt-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
                  <input
                    type="text"
                    placeholder="Search stay ID or care unit…"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="w-full pl-8 pr-3 py-1.5 text-sm rounded-md border border-input bg-background focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>

                <div className="flex items-center gap-2 flex-wrap">
                  {TIER_PILLS.map(({ label, active }) => (
                    <button
                      key={label}
                      onClick={() => setTierFilter(label)}
                      className={[
                        'text-xs px-2.5 py-0.5 rounded-full border font-medium transition-colors',
                        tierFilter === label
                          ? active
                          : 'bg-background text-muted-foreground border-border hover:bg-muted',
                      ].join(' ')}
                    >
                      {label}
                      {label !== 'ALL' && (
                        <span className="ml-1 opacity-60">
                          {label === 'CRITICAL' ? criticalCount
                            : label === 'HIGH' ? highCount
                            : label === 'MODERATE' ? modCount
                            : lowCount}
                        </span>
                      )}
                    </button>
                  ))}

                  <select
                    value={unitFilter}
                    onChange={(e) => setUnitFilter(e.target.value)}
                    className="ml-auto text-xs rounded-md border border-input bg-background px-2 py-1 focus:outline-none focus:ring-2 focus:ring-ring"
                  >
                    <option value="ALL">All units</option>
                    {careUnits.map((u) => (
                      <option key={u} value={u}>{u}</option>
                    ))}
                  </select>
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-0">
              {patientsLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : filtered.length === 0 ? (
                <div className="py-10 text-center text-sm text-muted-foreground">
                  No patients match the current filters.
                </div>
              ) : (
                <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 z-10">
                      <tr className="border-b bg-muted/90 backdrop-blur-sm">
                        <th
                          className="px-4 py-3 text-left font-medium text-muted-foreground cursor-pointer hover:text-foreground select-none"
                          onClick={() => toggleSort('stay_id')}
                        >
                          Stay ID <SortArrow col="stay_id" sortKey={sortKey} sortDir={sortDir} />
                        </th>
                        <th
                          className="px-4 py-3 text-left font-medium text-muted-foreground cursor-pointer hover:text-foreground select-none"
                          onClick={() => toggleSort('first_careunit')}
                        >
                          Care Unit <SortArrow col="first_careunit" sortKey={sortKey} sortDir={sortDir} />
                        </th>
                        <th
                          className="px-4 py-3 text-left font-medium text-muted-foreground cursor-pointer hover:text-foreground select-none"
                          onClick={() => toggleSort('age')}
                        >
                          Age <SortArrow col="age" sortKey={sortKey} sortDir={sortDir} />
                        </th>
                        <th className="px-4 py-3 text-center font-medium text-muted-foreground">
                          Sex
                        </th>
                        <th
                          className="px-4 py-3 text-right font-medium text-muted-foreground cursor-pointer hover:text-foreground select-none"
                          onClick={() => toggleSort('risk_score')}
                        >
                          Risk Score <SortArrow col="risk_score" sortKey={sortKey} sortDir={sortDir} />
                        </th>
                        <th className="px-4 py-3 text-center font-medium text-muted-foreground">
                          Risk Level
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.map((p) => (
                        <tr
                          key={p.stay_id}
                          onClick={() => navigate(`/patient/${p.stay_id}`)}
                          className={[
                            'border-b cursor-pointer transition-colors hover:bg-muted/60',
                            p.risk_label === 'CRITICAL' ? 'bg-red-100/70' : '',
                            p.risk_label === 'HIGH'     ? 'bg-red-50/60'  : '',
                            p.risk_label === 'MODERATE' ? 'bg-amber-50/40' : '',
                          ]
                            .filter(Boolean)
                            .join(' ')}
                        >
                          <td className="px-4 py-2.5 font-mono text-xs">{p.stay_id}</td>
                          <td className="px-4 py-2.5 text-xs text-muted-foreground max-w-[140px] truncate">
                            {p.first_careunit}
                          </td>
                          <td className="px-4 py-2.5">{p.age ? Math.round(p.age) : '—'}</td>
                          <td className="px-4 py-2.5 text-center text-xs text-muted-foreground">
                            {p.gender ?? '—'}
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono font-semibold">
                            {p.risk_score.toFixed(3)}
                          </td>
                          <td className="px-4 py-2.5 text-center">
                            <RiskBadge label={p.risk_label} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
