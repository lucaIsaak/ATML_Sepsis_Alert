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
import { Loader2, RefreshCw, Play, ChevronDown, ChevronRight } from 'lucide-react'
import { getModelStats, getFeedbackAgentStatus, triggerRetrain, getRetrainStatus, getDriftStatus, getAuditLog } from '@/api/client'
import { MetricCard } from '@/components/MetricCard'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { FeedbackAgentStatus, DriftStatus } from '@/types'

// ------------------------------------------------------------------ //
// Feedback Loop Agent status card                                      //
// ------------------------------------------------------------------ //

const DECISION_STYLES: Record<FeedbackAgentStatus['decision'], { badge: string; bar: string; label: string }> = {
  WAIT:    { badge: 'bg-slate-100 text-slate-700',   bar: 'bg-slate-400',  label: 'Collecting data' },
  STABLE:  { badge: 'bg-emerald-100 text-emerald-800', bar: 'bg-emerald-500', label: 'All metrics healthy' },
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

            {/* Progress toward retrain thresholds */}
            <div className="border-t pt-3 space-y-2">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                Progress toward retraining
              </p>
              {/* Confirmed sepsis labels */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Confirmed sepsis labels</span>
                  <span className="font-medium">{status.confirmed_sepsis} / 20</span>
                </div>
                <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                  <div
                    className="h-full rounded-full bg-primary transition-all"
                    style={{ width: `${Math.min(100, (status.confirmed_sepsis / 20) * 100)}%` }}
                  />
                </div>
              </div>
              {/* FP rate */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">False positive rate</span>
                  <span className="font-medium">
                    {status.fp_rate != null ? `${(status.fp_rate * 100).toFixed(0)}%` : '—'} / 30% threshold
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      status.fp_rate != null && status.fp_rate > 0.3 ? 'bg-red-500' : 'bg-amber-400'
                    }`}
                    style={{ width: `${Math.min(100, ((status.fp_rate ?? 0) / 0.3) * 100)}%` }}
                  />
                </div>
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

            {/* ── Retrain section — always visible ── */}
            {(status.decision === 'RETRAIN' || true) && (
              <div className="border-t pt-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    {status.decision === 'RETRAIN' ? (
                      <>
                        <p className="text-sm font-medium">Model retraining recommended</p>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          Enough labelled data has accumulated to improve the model.
                        </p>
                      </>
                    ) : (
                      <>
                        <p className="text-sm font-medium text-muted-foreground">Manual retrain</p>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          Thresholds not yet met — proceed only if you have clinical reason.
                        </p>
                      </>
                    )}
                    <p className="text-xs text-muted-foreground mt-1 italic">
                      Note: at demo scale (&lt;50 feedback labels) AUROC change after retraining is
                      expected to be near-zero — this is correct behaviour, not a bug.
                      The feedback labels are incorporated and weighted; the effect becomes
                      visible at production scale.
                    </p>
                  </div>
                  <button
                    onClick={handleRetrain}
                    disabled={retrainStatus === 'running' || launching}
                    className={[
                      'flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors',
                      retrainStatus === 'running' || launching
                        ? 'bg-muted text-muted-foreground cursor-not-allowed'
                        : status.decision === 'RETRAIN'
                        ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                        : 'bg-muted text-muted-foreground border border-border hover:bg-muted/80',
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

// ------------------------------------------------------------------ //
// Drift Monitor components                                            //
// ------------------------------------------------------------------ //

const DRIFT_STYLES: Record<DriftStatus['overall_status'], {
  bg: string; border: string; dot: string; text: string; label: string
}> = {
  stable:      { bg: 'bg-emerald-50', border: 'border-emerald-200', dot: 'bg-emerald-500', text: 'text-emerald-700', label: 'No drift detected' },
  moderate:    { bg: 'bg-amber-50',   border: 'border-amber-200',   dot: 'bg-amber-500',   text: 'text-amber-700',   label: 'Moderate drift'    },
  significant: { bg: 'bg-red-50',     border: 'border-red-200',     dot: 'bg-red-500',     text: 'text-red-700',     label: 'Significant drift' },
  unknown:     { bg: 'bg-slate-50',   border: 'border-slate-200',   dot: 'bg-slate-400',   text: 'text-slate-600',   label: 'Insufficient data' },
}

const PSI_STATUS_STYLES: Record<string, string> = {
  stable:      'text-emerald-600',
  moderate:    'text-amber-600',
  significant: 'text-red-600',
  unknown:     'text-slate-400',
}

const RISK_COLORS: Record<string, string> = {
  CRITICAL: '#ef4444',
  HIGH:     '#f97316',
  MODERATE: '#f59e0b',
  LOW:      '#22c55e',
}

function DriftMonitorSection({ drift }: { drift: DriftStatus }) {
  const style = DRIFT_STYLES[drift.overall_status]

  // PSI sparkline data
  const sparkData = drift.psi_history.map((h) => ({
    ts:  new Date(h.ts).toLocaleDateString(),
    psi: h.psi ?? 0,
  }))

  // Risk distribution bar data
  const riskOrder = ['CRITICAL', 'HIGH', 'MODERATE', 'LOW']
  const riskBarData = riskOrder.map((tier) => ({
    tier,
    live:     Math.round((drift.risk_distribution.live[tier] ?? 0) * 100),
    expected: Math.round((drift.risk_distribution.expected[tier] ?? 0) * 100),
  }))

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">Data Drift Monitor</h2>
        <p className="text-muted-foreground text-sm mt-0.5">
          Live patient features vs. MIMIC-IV training distribution (PSI)
        </p>
      </div>

      {/* Banner */}
      <div className={`rounded-lg border px-4 py-3 flex items-center gap-3 ${style.bg} ${style.border}`}>
        <div className={`h-2.5 w-2.5 rounded-full flex-shrink-0 ${style.dot}`} />
        <div className="flex-1">
          <span className={`font-semibold text-sm ${style.text}`}>{style.label}</span>
          {drift.overall_psi != null && (
            <span className={`text-sm ml-2 ${style.text} opacity-75`}>
              — max PSI {drift.overall_psi.toFixed(3)} across {drift.features.length} features
              ({drift.live_patients} live patients)
            </span>
          )}
        </div>
      </div>

      {/* Data source note — always shown */}
      <div className="rounded-md border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-600 leading-relaxed">
        <span className="font-semibold">Note on interpretation:</span>{' '}
        The &ldquo;live&rdquo; patients shown here are sampled from the same MIMIC-IV dataset used for training,
        so PSI values reflect sampling noise rather than true distribution shift.
        This monitor becomes meaningful once genuinely new patient data flows in —
        values below 0.10 are stable, 0.10–0.20 indicate moderate drift, and above 0.20 warrant investigation.
      </div>

      {/* Feature table + Risk distribution side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* Feature drift table */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Feature PSI — worst first</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/30">
                  <th className="text-left py-2 px-4 text-xs font-semibold text-muted-foreground">Feature</th>
                  <th className="text-right py-2 px-3 text-xs font-semibold text-muted-foreground">Train mean</th>
                  <th className="text-right py-2 px-3 text-xs font-semibold text-muted-foreground">Live mean</th>
                  <th className="text-right py-2 px-4 text-xs font-semibold text-muted-foreground">PSI</th>
                </tr>
              </thead>
              <tbody>
                {drift.features.map((f) => (
                  <tr key={f.feature} className="border-b last:border-0 hover:bg-muted/20">
                    <td className="py-2 px-4 font-medium text-xs">{f.label}</td>
                    <td className="py-2 px-3 text-right text-xs text-muted-foreground">
                      {f.train_mean != null ? f.train_mean.toFixed(1) : '—'}
                    </td>
                    <td className="py-2 px-3 text-right text-xs text-muted-foreground">
                      {f.live_mean != null ? f.live_mean.toFixed(1) : '—'}
                    </td>
                    <td className="py-2 px-4 text-right">
                      <span className={`text-xs font-semibold ${PSI_STATUS_STYLES[f.status]}`}>
                        {f.psi != null ? f.psi.toFixed(3) : '—'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        {/* Risk distribution + PSI history */}
        <div className="space-y-4">
          {/* Risk distribution */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Risk tier distribution — live vs. expected</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={riskBarData} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="tier" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} unit="%" />
                  <Tooltip formatter={(v: number) => `${v}%`} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="live" name="Live" radius={[3, 3, 0, 0]}>
                    {riskBarData.map((entry) => (
                      <Cell key={entry.tier} fill={RISK_COLORS[entry.tier]} />
                    ))}
                  </Bar>
                  <Bar dataKey="expected" name="Expected" fill="#94a3b8" radius={[3, 3, 0, 0]} opacity={0.5} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* PSI sparkline */}
          {sparkData.length > 1 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Max PSI trend</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={100}>
                  <LineChart data={sparkData} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="ts" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} domain={[0, 'auto']} />
                    <Tooltip formatter={(v: number) => v.toFixed(3)} />
                    <ReferenceLine y={0.1} stroke="#f59e0b" strokeDasharray="4 4" />
                    <ReferenceLine y={0.2} stroke="#ef4444" strokeDasharray="4 4" />
                    <Line type="monotone" dataKey="psi" name="Max PSI" stroke="#0284c7" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
                <p className="text-xs text-muted-foreground mt-1">
                  Dashed lines: 0.1 (moderate) and 0.2 (significant) thresholds
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

export function ModelPerformance() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: getModelStats,
  })

  const { data: drift } = useQuery({
    queryKey: ['drift-status'],
    queryFn: getDriftStatus,
    refetchInterval: 5 * 60_000,
  })

  const [auditOpen, setAuditOpen] = useState(false)
  const { data: auditLog = [], refetch: refetchAudit } = useQuery({
    queryKey: ['audit-log'],
    queryFn: () => getAuditLog(20),
    enabled: auditOpen,   // only fetch when expanded
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
    { label: 'Model', value: 'HistGradientBoostingClassifier (sklearn) — initial train Optuna-tuned' },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Model Performance</h1>
        <p className="text-muted-foreground text-sm mt-1">
          SepsisAlert vs. NEWS2 on MIMIC-IV holdout set
        </p>
      </div>

      {/* Top metrics — 4 cards */}
      <div className="grid grid-cols-4 gap-4">
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
        {/* Drift status card */}
        {drift ? (() => {
          const s = DRIFT_STYLES[drift.overall_status]
          return (
            <div className={`rounded-xl border p-4 flex flex-col gap-2 ${s.bg} ${s.border}`}>
              <p className="text-sm font-medium text-muted-foreground">Data Drift</p>
              <div className="flex items-center gap-2">
                <div className={`h-2.5 w-2.5 rounded-full ${s.dot}`} />
                <p className={`text-lg font-bold leading-tight ${s.text}`}>{s.label}</p>
              </div>
              <p className={`text-xs ${s.text} opacity-75`}>
                {drift.overall_psi != null ? `Max PSI ${drift.overall_psi.toFixed(3)}` : 'Evaluating…'}
              </p>
            </div>
          )
        })() : (
          <div className="rounded-xl border p-4 flex flex-col gap-2 bg-slate-50 border-slate-200">
            <p className="text-sm font-medium text-muted-foreground">Data Drift</p>
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Loading…</p>
            </div>
          </div>
        )}
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

      {/* Feedback Loop Agent */}
      <FeedbackAgentCard />

      {/* Drift Monitor */}
      {drift && <DriftMonitorSection drift={drift} />}

      {/* Audit Log */}
      <Card>
        <button
          className="w-full flex items-center justify-between px-6 py-4 text-left"
          onClick={() => {
            setAuditOpen((v) => {
              if (!v) refetchAudit()
              return !v
            })
          }}
        >
          <span className="font-medium text-sm">Audit Log <span className="text-muted-foreground font-normal">(last 20 alerts)</span></span>
          {auditOpen
            ? <ChevronDown className="h-4 w-4 text-muted-foreground" />
            : <ChevronRight className="h-4 w-4 text-muted-foreground" />
          }
        </button>
        {auditOpen && (
          <CardContent className="pt-0">
            {auditLog.length === 0 ? (
              <p className="text-sm text-muted-foreground py-4 text-center">No audit entries yet.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <th className="px-3 py-2 text-left font-semibold text-muted-foreground">Timestamp</th>
                      <th className="px-3 py-2 text-left font-semibold text-muted-foreground">Stay ID</th>
                      <th className="px-3 py-2 text-left font-semibold text-muted-foreground">Risk tier</th>
                      <th className="px-3 py-2 text-right font-semibold text-muted-foreground">Risk score</th>
                      <th className="px-3 py-2 text-left font-semibold text-muted-foreground">OOD flag</th>
                    </tr>
                  </thead>
                  <tbody>
                    {auditLog.map((entry, i) => (
                      <tr key={i} className="border-b last:border-0 hover:bg-muted/20">
                        <td className="px-3 py-2 text-muted-foreground font-mono">
                          {entry.timestamp ? new Date(entry.timestamp as string).toLocaleString() : '—'}
                        </td>
                        <td className="px-3 py-2 font-medium">{String(entry.stay_id ?? '—')}</td>
                        <td className="px-3 py-2">
                          <span className={[
                            'px-1.5 py-0.5 rounded text-xs font-semibold',
                            entry.risk_tier === 'HIGH' || entry.risk_tier === 'CRITICAL'
                              ? 'bg-red-100 text-red-700'
                              : entry.risk_tier === 'MODERATE'
                              ? 'bg-amber-100 text-amber-700'
                              : 'bg-emerald-100 text-emerald-700',
                          ].join(' ')}>
                            {String(entry.risk_tier ?? '—')}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-right font-mono">
                          {entry.risk_score != null ? Number(entry.risk_score).toFixed(3) : '—'}
                        </td>
                        <td className="px-3 py-2 text-muted-foreground">
                          {String(entry.ood_flag ?? 'NORMAL')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        )}
      </Card>

      {/* Dataset & Methodology — bottom */}
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
