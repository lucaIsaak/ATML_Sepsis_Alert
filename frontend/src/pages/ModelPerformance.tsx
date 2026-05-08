import { useQuery } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts'
import { Loader2 } from 'lucide-react'
import { getModelStats } from '@/api/client'
import { MetricCard } from '@/components/MetricCard'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function ModelPerformance() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: getModelStats,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!stats) {
    return (
      <div className="flex items-center justify-center py-24 text-muted-foreground text-sm">
        Failed to load statistics. Make sure the API is running.
      </div>
    )
  }

  const aurocBarData = [
    { name: 'SepsisAlert', auroc: stats.auroc },
    { name: 'NEWS2', auroc: stats.news2_auroc },
  ]

  // Sample ROC data to ~50 points each for display
  const step = Math.max(1, Math.floor(stats.roc_sepsis.length / 50))
  const rocData = stats.roc_sepsis
    .filter((_, i) => i % step === 0)
    .map((pt, i) => ({
      fpr: parseFloat(pt.fpr.toFixed(3)),
      sepsis_tpr: parseFloat(pt.tpr.toFixed(3)),
      news2_tpr:
        stats.roc_news2[Math.min(i * step, stats.roc_news2.length - 1)]?.tpr != null
          ? parseFloat(stats.roc_news2[Math.min(i * step, stats.roc_news2.length - 1)].tpr.toFixed(3))
          : undefined,
    }))

  const datasetRows = [
    { label: 'Dataset', value: 'MIMIC-IV v3.1' },
    { label: 'Total ICU stays', value: stats.total_stays.toLocaleString() },
    { label: 'Sepsis cases', value: stats.sepsis_cases.toLocaleString() },
    { label: 'Sepsis prevalence', value: `${((stats.sepsis_cases / stats.total_stays) * 100).toFixed(1)}%` },
    { label: 'Features', value: stats.features },
    { label: 'Prediction horizon', value: '6 hours before onset' },
    { label: 'Lookback window', value: '24 hours' },
    { label: 'Model', value: 'LightGBM (Optuna-tuned)' },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Model Performance</h1>
        <p className="text-muted-foreground text-sm mt-1">
          SepsisAlert vs. NEWS2 on MIMIC-IV holdout set
        </p>
      </div>

      {/* Top metrics */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          title="SepsisAlert AUROC"
          value={stats.auroc.toFixed(3)}
          description="+28.1pp vs NEWS2"
          valueClassName="text-primary"
        />
        <MetricCard
          title="NEWS2 AUROC"
          value={stats.news2_auroc.toFixed(3)}
          description="Clinical baseline"
          valueClassName="text-muted-foreground"
        />
        <MetricCard
          title="AUPRC"
          value={stats.auprc.toFixed(3)}
          description="Average precision (imbalanced)"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* ROC curve */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">ROC Curve</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={rocData} margin={{ top: 4, right: 16, bottom: 16, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="fpr"
                  label={{ value: 'FPR', position: 'insideBottom', offset: -8, fontSize: 12 }}
                  tick={{ fontSize: 11 }}
                  domain={[0, 1]}
                />
                <YAxis
                  label={{ value: 'TPR', angle: -90, position: 'insideLeft', fontSize: 12 }}
                  tick={{ fontSize: 11 }}
                  domain={[0, 1]}
                />
                <Tooltip
                  formatter={(val: number) => val.toFixed(3)}
                  labelFormatter={(l: number) => `FPR: ${l.toFixed(3)}`}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {/* Diagonal reference */}
                <ReferenceLine
                  segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  label={{ value: 'Random', fontSize: 10, fill: '#94a3b8' }}
                />
                <Line
                  type="monotone"
                  dataKey="sepsis_tpr"
                  name="SepsisAlert"
                  stroke="#0284c7"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="news2_tpr"
                  name="NEWS2"
                  stroke="#94a3b8"
                  dot={false}
                  strokeWidth={2}
                  strokeDasharray="5 5"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* AUROC bar */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">AUROC Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={aurocBarData} margin={{ top: 16, right: 16, bottom: 16, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v: number) => v.toFixed(3)} />
                <Bar dataKey="auroc" radius={[6, 6, 0, 0]}>
                  {aurocBarData.map((_entry, i) => (
                    <Cell
                      key={i}
                      fill={i === 0 ? '#0284c7' : '#94a3b8'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Training cohort summary */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          title="Total ICU Stays"
          value={stats.total_stays.toLocaleString()}
          description="MIMIC-IV v3.1"
        />
        <MetricCard
          title="Sepsis Cases"
          value={stats.sepsis_cases.toLocaleString()}
          description={`${((stats.sepsis_cases / stats.total_stays) * 100).toFixed(1)}% prevalence`}
        />
        <MetricCard
          title="Features"
          value={stats.features}
          description="Vitals + labs + demographics"
        />
      </div>

      {/* Dataset info table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Dataset & Methodology</CardTitle>
        </CardHeader>
        <CardContent>
          <table className="w-full text-sm">
            <tbody>
              {datasetRows.map((row) => (
                <tr key={row.label} className="border-b last:border-0">
                  <td className="py-2.5 pr-4 font-medium text-muted-foreground w-48">{row.label}</td>
                  <td className="py-2.5 font-medium">{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  )
}
