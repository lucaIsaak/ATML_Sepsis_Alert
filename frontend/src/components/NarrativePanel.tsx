import { useState, useRef, useCallback, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Mic, MicOff, Send, Star, Loader2, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { getModels, streamNarrative, saveNarrativeFeedback, transcribeAudio, getWhisperStatus } from '@/api/client'
import type { PatientDetail } from '@/types'

interface NarrativePanelProps {
  stayId: number
  patientDetail: PatientDetail
}

export function NarrativePanel({ stayId, patientDetail }: NarrativePanelProps) {
  const [narrative, setNarrative] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [selectedModel, setSelectedModel] = useState('')
  const [rating, setRating] = useState(0)
  const [correctionNote, setCorrectionNote] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [feedbackSaved, setFeedbackSaved] = useState(false)
  const [recordingSeconds, setRecordingSeconds] = useState(0)

  const abortRef = useRef<AbortController | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Recording duration counter
  useEffect(() => {
    if (isRecording) {
      setRecordingSeconds(0)
      recordingTimerRef.current = setInterval(() => setRecordingSeconds((s) => s + 1), 1000)
    } else {
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current)
      setRecordingSeconds(0)
    }
    return () => { if (recordingTimerRef.current) clearInterval(recordingTimerRef.current) }
  }, [isRecording])

  const { data: models = [] } = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
    staleTime: 60_000,
  })

  const { data: whisperStatus } = useQuery({
    queryKey: ['whisper-status'],
    queryFn: getWhisperStatus,
    staleTime: 5 * 60_000,
  })
  const whisperAvailable = whisperStatus?.available ?? false

  const currentModel = selectedModel || models[0] || ''

  const handleGenerate = useCallback(async () => {
    if (!currentModel) return
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    setNarrative('')
    setIsStreaming(true)
    setFeedbackSaved(false)

    try {
      await streamNarrative(
        stayId,
        currentModel,
        (chunk) => setNarrative((prev) => prev + chunk),
        () => setIsStreaming(false),
        abortRef.current.signal,
      )
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setNarrative('[Error] Failed to connect to Ollama. Is it running?')
      }
      setIsStreaming(false)
    }
  }, [stayId, currentModel])

  const handleSubmitFeedback = useCallback(async () => {
    if (!narrative || rating === 0) return
    await saveNarrativeFeedback({
      stay_id: stayId,
      rating,
      correction_note: correctionNote,
      narrative_text: narrative,
      model_used: currentModel,
    })
    setFeedbackSaved(true)
  }, [stayId, rating, correctionNote, narrative, currentModel])

  const handleStartRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      chunksRef.current = []
      const mr = new MediaRecorder(stream)
      mr.ondataavailable = (e) => chunksRef.current.push(e.data)
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setIsTranscribing(true)
        try {
          const text = await transcribeAudio(blob)
          setCorrectionNote((prev) => (prev ? prev + ' ' + text : text))
        } catch {
          setCorrectionNote((prev) => prev + ' [Transcription failed]')
        } finally {
          setIsTranscribing(false)
        }
      }
      mr.start()
      mediaRecorderRef.current = mr
      setIsRecording(true)
    } catch {
      alert('Microphone access denied or not available.')
    }
  }, [])

  const handleStopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop()
    setIsRecording(false)
  }, [])

  return (
    <div className="space-y-4">
      {/* Model selector + generate */}
      <div className="flex gap-2 items-center">
        <select
          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          value={currentModel}
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          {models.length === 0 && (
            <option value="">— Ollama not available —</option>
          )}
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <Button
          onClick={handleGenerate}
          disabled={isStreaming || !currentModel}
          className="shrink-0"
        >
          {isStreaming ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Zap className="h-4 w-4" />
          )}
          {isStreaming ? 'Generating…' : 'Generate'}
        </Button>
      </div>

      {/* Narrative output */}
      {(narrative || isStreaming) && (
        <div className="rounded-md border bg-muted/30 p-4 text-sm leading-relaxed whitespace-pre-wrap min-h-[120px] font-mono">
          {narrative}
          {isStreaming && <span className="inline-block w-2 h-4 bg-primary animate-pulse ml-1" />}
        </div>
      )}

      {/* Feedback section */}
      {narrative && !isStreaming && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Rate this narrative</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {/* Star rating */}
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map((s) => (
                <button
                  key={s}
                  onClick={() => setRating(s)}
                  className="text-xl transition-transform hover:scale-110"
                  aria-label={`Rate ${s} stars`}
                >
                  <Star
                    className={`h-6 w-6 ${s <= rating ? 'fill-warning stroke-warning' : 'stroke-muted-foreground'}`}
                  />
                </button>
              ))}
              {rating > 0 && (
                <span className="ml-2 text-sm text-muted-foreground self-center">{rating}/5</span>
              )}
            </div>

            {/* Correction note + voice */}
            <div className="flex gap-2">
              <textarea
                className="flex min-h-[64px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring resize-none"
                placeholder="Optional: note corrections or issues…"
                value={correctionNote}
                onChange={(e) => setCorrectionNote(e.target.value)}
              />
              <Button
                variant="outline"
                size="icon"
                className={isRecording ? 'border-destructive text-destructive' : !whisperAvailable ? 'opacity-40 cursor-not-allowed' : ''}
                onClick={whisperAvailable ? (isRecording ? handleStopRecording : handleStartRecording) : undefined}
                title={
                  !whisperAvailable
                    ? (whisperStatus?.message ?? 'Whisper not available')
                    : isRecording ? 'Stop recording' : 'Record voice note'
                }
                disabled={!whisperAvailable && !isRecording}
              >
                {isTranscribing ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : isRecording ? (
                  <MicOff className="h-4 w-4" />
                ) : (
                  <Mic className="h-4 w-4" />
                )}
              </Button>
            </div>

            {isRecording && (
              <div className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full bg-destructive animate-pulse" />
                <p className="text-xs text-destructive font-mono">
                  {String(Math.floor(recordingSeconds / 60)).padStart(2, '0')}:{String(recordingSeconds % 60).padStart(2, '0')}
                  {' '}— click mic to stop
                  {recordingSeconds >= 50 && (
                    <span className="ml-2 font-semibold"> (max 60 s)</span>
                  )}
                </p>
              </div>
            )}

            <Button
              onClick={handleSubmitFeedback}
              disabled={rating === 0 || feedbackSaved}
              size="sm"
              variant={feedbackSaved ? 'outline' : 'default'}
            >
              <Send className="h-3.5 w-3.5" />
              {feedbackSaved ? 'Feedback saved' : 'Submit feedback'}
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
