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
import { useState, useEffect, useRef } from 'react'
import { Loader2, RefreshCw, Play } from 'lucide-react'
import { getModelStats, getFeedbackAgentStatus, triggerRetrain, getRetrainStatus } from '@/api/client'
import { MetricCard } from '@/components/MetricCard'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { FeedbackAgentStatus } from '@/types'

// ------------------------------------------------------------------ //
// Feedback Loop Agent status card                                      //
// ------------------------------------------------------------------ //

const DECISION_STYLES: Record<FeedbackAgentStatus['decision'], { badge: string; bar: string; label: string }> = {
  WAIT:    { badge: 'bg-slate-100 text-slate-700',   bar: 'bg-slate-400',  label: 'Monitoring' },
  FLAG:    { badge: 'bg-amber-100 text-amber-800',   bar: 'bg-amber-500',  label: 'Review needed' },
  RETRAIN: { badge: 'bg-red-100   text-red-800',     bar: 'bg-red-500',    label: 'Retrain recommended' },
}

function FeedbackAgentCard() {
  const { data: status, isLoading, refetch, isFetching } = useQuery({
    queryKey: ['feedback-agent-status'],
    queryFn: getFeedbackAgentStatus,
    refetchInterval: 60_000,
  })

  const [retrainStatus, setRetrainStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [retrainLog, setRetrainLog] = useState('')
  const [launching, setLaunching] = useState(false)
  const logRef = useRef<HTMLPreElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Auto-scroll log to bottom as new lines arrive
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [retrainLog])

  // Clean up polling on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const startPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      const s = await getRetrainStatus()
      setRetrainLog(s.log)
      setRetrainStatus(s.status)
      if (s.status === 'done' || s.status === 'error') {
        clearInterval(pollRef.current!)
        pollRef.current = null
        // Refresh agent decision — model may have improved
        refetch()
      }
    }, 1500)
  }

  const handleRetrain = async () => {
    setLaunching(true)
    setRetrainLog('')
    try {
      await triggerRetrain()
      setRetrainStatus('running')
      startPolling()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setRetrainLog(`Failed to start retraining: ${msg}`)
      setRetrainStatus('error')
    } finally {
      setLaunching(false)
    }
  }

  const style = status ? DECISION_STYLES[status.decision] : DECISION_STYLES['WAIT']

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-base">Feedback Loop Agent</CardTitle>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="text-muted-foreground hover:text-foreground transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
        </button>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Evaluating feedback logs…
          </div>
        ) : !status ? (
          <p className="text-sm text-muted-foreground">Unable to reach feedback agent.</p>
        ) : (
          <>
            {/* Decision badge */}
            <div className="flex items-center gap-3">
              <div className={`h-2 w-2 rounded-full ${style.bar}`} />
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${style.badge}`}>
                {status.decision} — {style.label}
              </span>
            </div>

            {/* Reason text */}
            <p className="text-sm text-muted-foreground leading-relaxed">{status.reason}</p>

            {/* Metrics grid */}
            <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm border-t pt-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Clinical records</span>
                <span className="font-medium">{status.clinical_total}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Confirmed sepsis</span>
                <span className="font-medium">{status.confirmed_sepsis}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Flagged wrong</span>
                <span className="font-medium">{status.flagged_wrong}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">False positive rate</span>
                <span className="font-medium">
                  {status.fp_rate != null ? `${(status.fp_rate * 100).toFixed(0)}%` : '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Narrative ratings</span>
                <span className="font-medium">{status.narrative_total}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Mean star rating</span>
                <span className="font-medium">
                  {status.mean_rating != null ? `${status.mean_rating.toFixed(1)} / 5` : '—'}
                </span>
              </div>
            </div>

            {/* Recent clinician corrections */}
            {status.correction_notes.length > 0 && (
              <div className="border-t pt-3">
                <p className="text-xs font-semibold text-muted-foreground mb-1.5 uppercase tracking-wide">
                  Recent corrections
                </p>
                <ul className="space-y-1">
                  {status.correction_notes.map((note, i) => (
                    <li key={i} className="text-xs text-muted-foreground italic">
                      &ldquo;{note}&rdquo;
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* ── Retrain button — only shown when RETRAIN is recommended ── */}
            {status.decision === 'RETRAIN' && (
              <div className="border-t pt-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">Model retraining available</p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Enough labelled data has accumulated to improve the model.
                    </p>
                  </div>
                  <button
                    onClick={handleRetrain}
                    disabled={retrainStatus === 'running' || launching}
                    className={[
                      'flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors',
                      retrainStatus === 'running' || launching
                        ? 'bg-muted text-muted-foreground cursor-not-allowed'
                        : 'bg-primary text-primary-foreground hover:bg-primary/90',
                    ].join(' ')}
                  >
                    {retrainStatus === 'running' || launching ? (
                      <><Loader2 className="h-4 w-4 animate-spin" /> Retraining…</>
                    ) : (
                      <><Play className="h-4 w-4" /> Retrain model</>
                    )}
                  </button>
                </div>

                {/* Log output — shown once retraining starts */}
                {retrainLog && (
                  <div className="rounded-md border bg-muted/40 overflow-hidden">
                    <div className="flex items-center justify-between px-3 py-1.5 border-b bg-muted/60">
                      <span className="text-xs font-mono font-medium text-muted-foreground">
                        retrain_with_feedback.py
                      </span>
                      {retrainStatus === 'done' && (
                        <span className="text-xs text-green-600 font-medium">✓ Complete</span>
                      )}
                      {retrainStatus === 'error' && (
                        <span className="text-xs text-red-600 font-medium">✗ Error</span>
                      )}
                      {retrainStatus === 'running' && (
                        <span className="flex items-center gap-1 text-xs text-amber-600 font-medium">
                          <Loader2 className="h-3 w-3 animate-spin" /> Running
                        </span>
                      )}
                    </div>
                    <pre
                      ref={logRef}
                      className="text-xs font-mono p-3 max-h-56 overflow-y-auto whitespace-pre-wrap text-muted-foreground leading-relaxed"
                    >
                      {retrainLog}
                    </pre>
                  </div>
                )}
              </div>
            )}

            {/* Evaluated at */}
            <p className="text-xs text-muted-foreground/60 text-right">
              Evaluated {new Date(status.evaluated_at).toLocaleTimeString()}
            </p>
          </>
        )}
      </CardContent>
    </Card>
  )
}

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

      {/* Feedback Loop Agent status */}
      <FeedbackAgentCard />
    </div>
  )
}
