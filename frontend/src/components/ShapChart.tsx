import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Tooltip,
  ResponsiveContainer,
  LabelList,
} from 'recharts'
import type { ShapFeature } from '@/types'

interface ShapChartProps {
  features: ShapFeature[]
  color: string
  title: string
}

export function ShapChart({ features, color }: ShapChartProps) {
  const data = features.map((f) => ({
    label: f.label.length > 20 ? f.label.slice(0, 20) + '…' : f.label,
    shap: parseFloat(f.shap.toFixed(4)),
    value: f.value,
    fullLabel: f.label,
  }))

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.shap)), 0.01)
  const domain: [number, number] = [-maxAbs * 1.2, maxAbs * 1.2]

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 8, right: 48, left: 8, bottom: 8 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          domain={domain}
          tickFormatter={(v) => v.toFixed(3)}
          tick={{ fontSize: 10 }}
        />
        <YAxis
          type="category"
          dataKey="label"
          width={160}
          tick={{ fontSize: 11 }}
        />
        <Tooltip
          formatter={(val: number, _name: string, props) => [
            `SHAP: ${val.toFixed(4)}  |  Value: ${props.payload.value ?? 'N/A'}`,
            props.payload.fullLabel,
          ]}
        />
        <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={1.5} />
        <Bar dataKey="shap" fill={color} radius={[0, 3, 3, 0]}>
          <LabelList
            dataKey="shap"
            position="right"
            formatter={(v: number) => v.toFixed(3)}
            style={{ fontSize: 10, fill: '#64748b' }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
