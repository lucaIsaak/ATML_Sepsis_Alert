import axios from 'axios'
import type { Patient, PatientDetail, ClinicalFeedback, ModelStats } from '@/types'

const api = axios.create({
  baseURL: '/api',
})

export function getPatients(): Promise<Patient[]> {
  return api.get<Patient[]>('/patients').then((r) => r.data)
}

export function getPatientDetail(stayId: number): Promise<PatientDetail> {
  return api.get<PatientDetail>(`/patients/${stayId}`).then((r) => r.data)
}

export function getModels(): Promise<string[]> {
  return api.get<string[]>('/narrative/models').then((r) => r.data)
}

export function getClinicalFeedback(stayId: number): Promise<ClinicalFeedback | null> {
  return api.get<ClinicalFeedback | null>(`/feedback/clinical/${stayId}`).then((r) => r.data)
}

export function saveClinicalFeedback(
  stayId: number,
  feedbackType: string,
  riskScore: number,
): Promise<void> {
  return api
    .post('/feedback/clinical', {
      stay_id: stayId,
      feedback_type: feedbackType,
      risk_score: riskScore,
    })
    .then(() => undefined)
}

export function saveNarrativeFeedback(data: {
  stay_id: number
  rating: number
  correction_note: string
  narrative_text: string
  model_used: string
}): Promise<void> {
  return api.post('/feedback/narrative', data).then(() => undefined)
}

export function transcribeAudio(blob: Blob): Promise<string> {
  const form = new FormData()
  form.append('file', blob, 'audio.webm')
  return api
    .post<{ text: string }>('/feedback/transcribe', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((r) => r.data.text)
}

export function getModelStats(): Promise<ModelStats> {
  return api.get<ModelStats>('/stats').then((r) => r.data)
}

export async function streamNarrative(
  stayId: number,
  modelName: string,
  onChunk: (text: string) => void,
  onDone: () => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/narrative/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stay_id: stayId, model_name: modelName }),
    signal,
  })

  if (!response.ok) {
    throw new Error(`Stream failed: ${response.status} ${response.statusText}`)
  }

  const reader = response.body?.getReader()
  if (!reader) {
    onDone()
    return
  }

  const decoder = new TextDecoder()

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    const chunk = decoder.decode(value, { stream: true })
    if (chunk) onChunk(chunk)
  }

  onDone()
}
