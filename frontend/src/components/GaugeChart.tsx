interface GaugeChartProps {
  value: number   // 0-100
  label?: string
}

/**
 * Pure SVG half-circle gauge.
 * ViewBox: 0 0 200 120, center at (100, 100), radius 80.
 * Three colored background zones: green (0-40), yellow (40-60), blue (60-100).
 * Value arc shown in primary blue. Threshold line at 40% (alert threshold).
 */
export function GaugeChart({ value, label }: GaugeChartProps) {
  const cx = 100
  const cy = 100
  const r = 78
  const strokeWidth = 14

  // Half-circle arc: from 180° (left) to 0° (right)
  // For a value pct ∈ [0,1], the angle in [180°, 0°] → angle = 180 - pct*180
  function polarToCartesian(angleDeg: number) {
    const rad = (angleDeg * Math.PI) / 180
    return {
      x: cx + r * Math.cos(rad),
      y: cy - r * Math.sin(rad),  // SVG y-axis is inverted
    }
  }

  function arcPath(startPct: number, endPct: number) {
    // startPct and endPct in [0, 1]
    const startAngle = 180 - startPct * 180
    const endAngle   = 180 - endPct * 180
    const start = polarToCartesian(startAngle)
    const end   = polarToCartesian(endAngle)
    const largeArc = endPct - startPct > 0.5 ? 1 : 0
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`
  }

  const pct = Math.min(Math.max(value, 0), 100) / 100

  // Threshold line at 40% (F2-optimal alert threshold from config)
  const thresholdAngle = 180 - 0.4 * 180   // = 108°
  const thresholdOuter = polarToCartesian(thresholdAngle)
  const thresholdInner = {
    x: cx + (r - strokeWidth - 2) * Math.cos((thresholdAngle * Math.PI) / 180),
    y: cy - (r - strokeWidth - 2) * Math.sin((thresholdAngle * Math.PI) / 180),
  }

  // Needle tip at value pct
  const needleAngle = 180 - pct * 180
  const needleTip = polarToCartesian(needleAngle)

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 120" width="200" height="120" aria-label={`Risk gauge: ${value}%`}>
        {/* Background track */}
        <path
          d={arcPath(0, 1)}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Green zone 0–40% */}
        <path
          d={arcPath(0, 0.4)}
          fill="none"
          stroke="#22c55e"
          strokeWidth={strokeWidth}
          strokeOpacity={0.4}
          strokeLinecap="butt"
        />

        {/* Yellow zone 40–60% */}
        <path
          d={arcPath(0.4, 0.6)}
          fill="none"
          stroke="#f59e0b"
          strokeWidth={strokeWidth}
          strokeOpacity={0.4}
          strokeLinecap="butt"
        />

        {/* Blue zone 60–100% */}
        <path
          d={arcPath(0.6, 1)}
          fill="none"
          stroke="#0284c7"
          strokeWidth={strokeWidth}
          strokeOpacity={0.25}
          strokeLinecap="butt"
        />

        {/* Value arc */}
        {pct > 0 && (
          <path
            d={arcPath(0, pct)}
            fill="none"
            stroke="#0284c7"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
        )}

        {/* Threshold tick at 60% */}
        <line
          x1={thresholdInner.x}
          y1={thresholdInner.y}
          x2={thresholdOuter.x}
          y2={thresholdOuter.y}
          stroke="#ef4444"
          strokeWidth={2}
        />

        {/* Center value text */}
        <text
          x={cx}
          y={cy - 4}
          textAnchor="middle"
          dominantBaseline="auto"
          className="text-2xl font-bold fill-foreground"
          style={{ fontSize: '22px', fontWeight: 700, fill: '#0f172a', fontFamily: 'Inter, sans-serif' }}
        >
          {Math.floor(value)}%
        </text>

        {/* Min/max labels */}
        <text x="14" y="108" style={{ fontSize: '9px', fill: '#64748b', fontFamily: 'Inter, sans-serif' }}>0</text>
        <text x="178" y="108" style={{ fontSize: '9px', fill: '#64748b', fontFamily: 'Inter, sans-serif' }}>100</text>
      </svg>

      {label && (
        <p className="text-xs text-muted-foreground mt-1 text-center">{label}</p>
      )}
    </div>
  )
}
