import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Loader2 } from 'lucide-react'
import { getPatients, getModelStats } from '@/api/client'
import { MetricCard } from '@/components/MetricCard'
import { RiskBadge } from '@/components/RiskBadge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function LiveMonitor() {
  const navigate = useNavigate()

  const { data: patients = [], isLoading: patientsLoading } = useQuery({
    queryKey: ['patients'],
    queryFn: getPatients,
    refetchInterval: 30_000,
  })

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: getModelStats,
  })

  const highCount = patients.filter((p) => p.risk_label === 'HIGH' || p.risk_label === 'CRITICAL').length
  const modCount = patients.filter((p) => p.risk_label === 'MODERATE').length
  const lowCount = patients.filter((p) => p.risk_label === 'LOW').length

  const distData = [
    { label: 'HIGH', count: highCount, fill: '#e74c3c' },
    { label: 'MODERATE', count: modCount, fill: '#f39c12' },
    { label: 'LOW', count: lowCount, fill: '#27ae60' },
  ]

  const sorted = [...patients].sort((a, b) => b.risk_score - a.risk_score)

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight">ICU Live Monitor</h1>
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 border border-amber-200">
            Demo mode · static snapshot
          </span>
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
          title="High Risk"
          value={highCount}
          description="Risk score ≥ 0.60"
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
                <XAxis dataKey="label" tick={{ fontSize: 12 }} />
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
            <CardHeader>
              <CardTitle className="text-base">Patient List</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {patientsLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="px-4 py-3 text-left font-medium text-muted-foreground">Stay ID</th>
                        <th className="px-4 py-3 text-left font-medium text-muted-foreground">Care Unit</th>
                        <th className="px-4 py-3 text-left font-medium text-muted-foreground">Age</th>
                        <th className="px-4 py-3 text-right font-medium text-muted-foreground">Risk Score</th>
                        <th className="px-4 py-3 text-center font-medium text-muted-foreground">Risk Level</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sorted.map((p) => (
                        <tr
                          key={p.stay_id}
                          onClick={() => navigate(`/patient/${p.stay_id}`)}
                          className={[
                            'border-b cursor-pointer transition-colors hover:bg-muted/60',
                            (p.risk_label === 'CRITICAL' || p.risk_label === 'HIGH') ? 'bg-red-50/60' : '',
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
