import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { ChevronDown, ChevronRight, ArrowLeft, CheckCircle, XCircle, Loader2, AlertTriangle, ShieldAlert, Brain } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@radix-ui/react-tabs'
import { getPatientDetail, getClinicalFeedback, saveClinicalFeedback } from '@/api/client'
import { MetricCard } from '@/components/MetricCard'
import { RiskBadge } from '@/components/RiskBadge'
import { GaugeChart } from '@/components/GaugeChart'
import { ShapChart } from '@/components/ShapChart'
import { NarrativePanel } from '@/components/NarrativePanel'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function PatientDetailPage() {
  const { stayId } = useParams<{ stayId: string }>()
  const id = parseInt(stayId ?? '0', 10)
  const [rawExpanded, setRawExpanded] = useState(false)
  const queryClient = useQueryClient()

  const { data: patient, isLoading } = useQuery({
    queryKey: ['patient', id],
    queryFn: () => getPatientDetail(id),
    enabled: !!id,
  })

  const { data: feedback } = useQuery({
    queryKey: ['feedback', id],
    queryFn: () => getClinicalFeedback(id),
    enabled: !!id,
  })

  const feedbackMutation = useMutation({
    mutationFn: ({ type }: { type: string }) =>
      saveClinicalFeedback(id, type, patient?.risk_score ?? 0),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['feedback', id] })
    },
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!patient) {
    return (
      <div className="text-center py-24 text-muted-foreground">
        <p>Patient not found.</p>
        <Link to="/" className="text-primary underline mt-2 inline-block">
          Back to monitor
        </Link>
      </div>
    )
  }

  const gaugePct = Math.round(patient.risk_score * 100)

  return (
    <div className="space-y-6">
      {/* Back nav */}
      <Link
        to="/"
        className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Live Monitor
      </Link>

      {/* Clinical feedback banner */}
      {feedback && (
        <div
          className={[
            'rounded-md px-4 py-3 text-sm flex items-center gap-2',
            feedback.feedback_type === 'confirmed_sepsis'
              ? 'bg-green-50 text-green-800 border border-green-200'
              : 'bg-amber-50 text-amber-800 border border-amber-200',
          ].join(' ')}
        >
          {feedback.feedback_type === 'confirmed_sepsis' ? (
            <CheckCircle className="h-4 w-4 shrink-0" />
          ) : (
            <XCircle className="h-4 w-4 shrink-0" />
          )}
          <span>
            Clinician feedback:{' '}
            <strong>
              {feedback.feedback_type === 'confirmed_sepsis'
                ? 'Sepsis confirmed'
                : 'Alert flagged wrong'}
            </strong>{' '}
            (risk score at time: {feedback.risk_score.toFixed(3)})
          </span>
        </div>
      )}

      {/* OOD warning — individual features: 'NORMAL' | 'CAUTION' | 'LOW_CONFIDENCE' */}
      {patient.ood_flag !== 'NORMAL' && (
        <div className={[
          'rounded-md px-4 py-3 text-sm flex items-start gap-3 border',
          patient.ood_flag === 'LOW_CONFIDENCE'
            ? 'bg-red-50 text-red-800 border-red-200'
            : 'bg-amber-50 text-amber-800 border-amber-200',
        ].join(' ')}>
          {patient.ood_flag === 'LOW_CONFIDENCE'
            ? <ShieldAlert className="h-4 w-4 shrink-0 mt-0.5" />
            : <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />}
          <div>
            <p className="font-semibold">
              {patient.ood_flag === 'LOW_CONFIDENCE'
                ? 'Out-of-distribution input — risk score may be unreliable'
                : 'Borderline input (1–2 outlier features) — interpret with caution'}
            </p>
            {patient.outlier_features.length > 0 && (
              <p className="mt-0.5 text-xs opacity-80">
                Outlier features: {patient.outlier_features.join(', ')}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Multivariate novelty warning — unusual combination of individually normal vitals */}
      {patient.multivariate_novel && patient.ood_flag === 'NORMAL' && (
        <div className="rounded-md px-4 py-3 text-sm flex items-start gap-3 border bg-violet-50 text-violet-800 border-violet-200">
          <Brain className="h-4 w-4 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">
              Novel feature combination — each vital is normal, but this pattern is unusual
            </p>
            <p className="mt-0.5 text-xs opacity-80">
              Mahalanobis distance from training centroid:{' '}
              <strong>{patient.mahalanobis_distance?.toFixed(1) ?? '—'}</strong>.
              The model has rarely seen this specific combination. Clinical assessment recommended.
            </p>
          </div>
        </div>
      )}

      {/* Overreliance on LOW — uncertain LOW scores can be misleading */}
      {patient.risk_label === 'LOW' && patient.epistemic_uncertainty?.uncertainty_flag === 'HIGH' && (
        <div className="rounded-md px-4 py-3 text-sm flex items-start gap-3 border bg-amber-50 text-amber-800 border-amber-200">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">LOW score — but model confidence is low</p>
            <p className="mt-0.5 text-xs opacity-80">
              90% CI: [{patient.epistemic_uncertainty.ci_lower?.toFixed(2) ?? '—'},{' '}
              {patient.epistemic_uncertainty.ci_upper?.toFixed(2) ?? '—'}].
              The upper bound reaches into elevated-risk territory. Do not rule out sepsis on this score alone.
            </p>
          </div>
        </div>
      )}

      {/* Epistemic uncertainty warning — model sensitivity near decision boundary */}
      {patient.epistemic_uncertainty?.is_uncertain && !patient.multivariate_novel && !(patient.risk_label === 'LOW' && patient.epistemic_uncertainty?.uncertainty_flag === 'HIGH') && (
        <div className={[
          'rounded-md px-4 py-3 text-sm flex items-start gap-3 border',
          patient.epistemic_uncertainty.uncertainty_flag === 'HIGH'
            ? 'bg-violet-50 text-violet-800 border-violet-200'
            : 'bg-blue-50 text-blue-800 border-blue-200',
        ].join(' ')}>
          <Brain className="h-4 w-4 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">
              Model uncertainty: {patient.epistemic_uncertainty.uncertainty_flag.toLowerCase()} —
              confidence interval is wide
            </p>
            <p className="mt-0.5 text-xs opacity-80">
              90% CI: [{patient.epistemic_uncertainty.ci_lower?.toFixed(2) ?? '—'},{' '}
              {patient.epistemic_uncertainty.ci_upper?.toFixed(2) ?? '—'}].
              The prediction is sensitive to small input changes. Consider bedside reassessment.
            </p>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex flex-wrap items-start gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl font-bold tracking-tight">Stay #{patient.stay_id}</h1>
            <RiskBadge label={patient.risk_label} />
          </div>
          <div className="flex gap-6 mt-2 text-sm text-muted-foreground">
            <span>Age: <strong className="text-foreground">{patient.age ? Math.round(patient.age) : '—'}</strong></span>
            <span>Unit: <strong className="text-foreground">{patient.first_careunit}</strong></span>
            {patient.gender && (
              <span>Gender: <strong className="text-foreground">{patient.gender}</strong></span>
            )}
          </div>
        </div>

        {/* Gauge */}
        <div className="shrink-0">
          <GaugeChart value={gaugePct} label="Sepsis Risk" />
        </div>
      </div>

      {/* Metric row */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          title="Risk Score"
          value={patient.risk_score.toFixed(3)}
          description={
            patient.epistemic_uncertainty?.is_uncertain &&
            patient.epistemic_uncertainty.ci_lower != null &&
            patient.epistemic_uncertainty.ci_upper != null
              ? `90% CI [${patient.epistemic_uncertainty.ci_lower.toFixed(2)}, ${patient.epistemic_uncertainty.ci_upper.toFixed(2)}]`
              : 'Model probability (0–1)'
          }
        />
        <MetricCard
          title="Risk Level"
          value={patient.risk_label}
          description="HIGH ≥ 0.60 · MODERATE ≥ 0.40"
          valueClassName={
            patient.risk_label === 'CRITICAL' || patient.risk_label === 'HIGH'
              ? 'text-destructive'
              : patient.risk_label === 'MODERATE'
              ? 'text-warning'
              : 'text-green-600'
          }
        />
        <div className="flex flex-col gap-2">
          <span className="text-xs font-medium text-muted-foreground">Clinical feedback</span>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="outline"
              className="border-green-500 text-green-700 hover:bg-green-50"
              onClick={() => feedbackMutation.mutate({ type: 'confirmed_sepsis' })}
              disabled={feedbackMutation.isPending}
            >
              <CheckCircle className="h-3.5 w-3.5" />
              Confirm Sepsis
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="border-amber-500 text-amber-700 hover:bg-amber-50"
              onClick={() => feedbackMutation.mutate({ type: 'flagged_wrong' })}
              disabled={feedbackMutation.isPending}
            >
              <XCircle className="h-3.5 w-3.5" />
              Flag Wrong
            </Button>
          </div>
        </div>
      </div>

      {/* Main two-column: SHAP + Narrative */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* SHAP chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Feature Importance (SHAP)</CardTitle>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              SHAP values are computed on the uncalibrated base model. The displayed risk score uses
              isotonic calibration — small divergences between SHAP attribution and displayed score are expected.
            </p>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="top">
              <TabsList className="flex gap-1 mb-4">
                <TabsTrigger
                  value="top"
                  className="px-3 py-1.5 text-sm rounded-md data-[state=active]:bg-primary data-[state=active]:text-white text-muted-foreground hover:text-foreground transition-colors"
                >
                  Most Responsible
                </TabsTrigger>
                <TabsTrigger
                  value="bottom"
                  className="px-3 py-1.5 text-sm rounded-md data-[state=active]:bg-primary data-[state=active]:text-white text-muted-foreground hover:text-foreground transition-colors"
                >
                  Least Responsible
                </TabsTrigger>
              </TabsList>
              <TabsContent value="top">
                <ShapChart
                  features={patient.shap_top.slice(0, 8)}
                  color="#0284c7"
                  title="Top SHAP features"
                />
              </TabsContent>
              <TabsContent value="bottom">
                <ShapChart
                  features={patient.shap_bottom}
                  color="#94a3b8"
                  title="Bottom SHAP features"
                />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* Narrative */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Clinical Narrative</CardTitle>
          </CardHeader>
          <CardContent>
            <NarrativePanel stayId={id} patientDetail={patient} />
          </CardContent>
        </Card>
      </div>

      {/* Raw features expander */}
      <Card>
        <button
          className="w-full flex items-center justify-between px-6 py-4 text-left"
          onClick={() => setRawExpanded((v) => !v)}
        >
          <span className="font-medium text-sm">Raw Feature Values</span>
          {rawExpanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
        </button>
        {rawExpanded && (
          <CardContent className="pt-0">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="px-3 py-2 text-left font-medium text-muted-foreground">Feature</th>
                    <th className="px-3 py-2 text-right font-medium text-muted-foreground">Value</th>
                    <th className="px-3 py-2 text-right font-medium text-muted-foreground">SHAP</th>
                  </tr>
                </thead>
                <tbody>
                  {patient.shap_top.map((f) => (
                    <tr key={f.feature} className="border-b hover:bg-muted/30">
                      <td className="px-3 py-1.5 text-muted-foreground">{f.label}</td>
                      <td className="px-3 py-1.5 text-right font-mono">
                        {f.value != null ? f.value.toFixed(2) : 'N/A'}
                      </td>
                      <td
                        className={[
                          'px-3 py-1.5 text-right font-mono',
                          f.shap > 0 ? 'text-destructive' : 'text-green-600',
                        ].join(' ')}
                      >
                        {f.shap > 0 ? '+' : ''}{f.shap.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        )}
      </Card>
    </div>
  )
}
