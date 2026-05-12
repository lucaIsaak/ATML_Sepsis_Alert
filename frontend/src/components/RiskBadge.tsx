import { Badge } from '@/components/ui/badge'
import type { Patient } from '@/types'

interface RiskBadgeProps {
  label: Patient['risk_label']
  className?: string
}

export function RiskBadge({ label, className }: RiskBadgeProps) {
  const variant =
    label === 'CRITICAL' || label === 'HIGH' ? 'destructive' : label === 'MODERATE' ? 'warning' : 'secondary'
  return (
    <Badge variant={variant} className={className}>
      {label}
    </Badge>
  )
}
