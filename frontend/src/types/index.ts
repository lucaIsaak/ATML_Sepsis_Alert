export interface Patient {
  stay_id: number
  risk_score: number
  risk_label: 'HIGH' | 'MODERATE' | 'LOW'
  age: number
  first_careunit: string
  gender?: string
}

export interface ShapFeature {
  label: string
  shap: number
  value: number
  feature: string
}

export interface PatientDetail extends Patient {
  shap_top: ShapFeature[]      // top 16 by |shap|, used as top 8 in UI
  shap_bottom: ShapFeature[]   // bottom 8 by |shap|
  // Values from backend guardrails.py: 'NORMAL' | 'CAUTION' | 'LOW_CONFIDENCE'
  ood_flag: 'NORMAL' | 'CAUTION' | 'LOW_CONFIDENCE'
  outlier_features: string[]
}

export interface ClinicalFeedback {
  feedback_type: 'confirmed_sepsis' | 'flagged_wrong'
  risk_score: number
}

export interface ModelStats {
  auroc: number
  news2_auroc: number
  auprc: number
  total_stays: number
  sepsis_cases: number
  features: number
  roc_sepsis: Array<{ fpr: number; tpr: number }>
  roc_news2: Array<{ fpr: number; tpr: number }>
}

export interface DriftFeature {
  feature: string
  label: string
  train_mean: number | null
  live_mean: number | null
  psi: number | null
  status: 'stable' | 'moderate' | 'significant' | 'unknown'
}

export interface DriftStatus {
  overall_status: 'stable' | 'moderate' | 'significant' | 'unknown'
  overall_psi: number | null
  features: DriftFeature[]
  risk_distribution: {
    live: Record<string, number>
    expected: Record<string, number>
    live_counts: Record<string, number>
    total_live: number
  }
  psi_history: Array<{ ts: string; psi: number | null; status: string }>
  evaluated_at: string
  live_patients: number
  note: string | null
}

export interface FeedbackAgentStatus {
  decision: 'WAIT' | 'STABLE' | 'FLAG' | 'RETRAIN'
  reason: string
  evaluated_at: string
  clinical_total: number
  confirmed_sepsis: number
  flagged_wrong: number
  fp_rate: number | null
  narrative_total: number
  mean_rating: number | null
  std_rating: number | null
  low_rated_pct: number | null
  correction_notes: string[]
  details: Record<string, unknown>
}
