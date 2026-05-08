import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Activity, User, BarChart2, Heart, AlertTriangle, Users } from 'lucide-react'
import { getPatients } from '@/api/client'
import { LiveMonitor } from '@/pages/LiveMonitor'
import { PatientDetailPage } from '@/pages/PatientDetail'
import { ModelPerformance } from '@/pages/ModelPerformance'
import { cn } from '@/lib/utils'

function Sidebar() {
  const location = useLocation()
  const patientMatch = location.pathname.match(/^\/patient\/(\d+)/)
  const stayIdFromUrl = patientMatch ? patientMatch[1] : null

  const { data: patients = [] } = useQuery({
    queryKey: ['patients'],
    queryFn: getPatients,
    refetchInterval: 60_000,
  })

  const sorted = [...patients].sort((a, b) => b.risk_score - a.risk_score)
  const defaultStayId = sorted[0]?.stay_id?.toString() ?? null
  const patientLinkId = stayIdFromUrl ?? defaultStayId

  const highCount = patients.filter((p) => p.risk_label === 'HIGH').length
  const moderateCount = patients.filter((p) => p.risk_label === 'MODERATE').length
  const lowCount = patients.filter((p) => p.risk_label === 'LOW').length

  const navItems = [
    { to: '/', end: true, icon: Activity, label: 'Live Monitor' },
    {
      to: patientLinkId ? `/patient/${patientLinkId}` : '/',
      end: false,
      icon: User,
      label: 'Patient Detail',
      disabled: !patientLinkId,
    },
    { to: '/performance', end: false, icon: BarChart2, label: 'Model Performance' },
  ]

  return (
    <aside className="fixed left-0 top-0 h-full w-64 p-3 z-30 bg-muted/40">
      <div className="h-full rounded-2xl bg-white border border-border shadow-sm card-blue-shadow flex flex-col overflow-hidden">

        {/* Brand */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-border">
          <div className="h-9 w-9 rounded-lg bg-primary flex items-center justify-center shadow-sm flex-shrink-0">
            <Heart className="h-5 w-5 text-white fill-white" />
          </div>
          <div>
            <p className="font-bold text-foreground text-base leading-tight">SepsisAlert</p>
            <p className="text-[11px] text-muted-foreground leading-tight">ICU Early Warning</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="px-3 py-4 flex flex-col gap-0.5">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground px-2 pb-2">
            Navigation
          </p>
          {navItems.map(({ to, end, icon: Icon, label, disabled }) => (
            <NavLink
              key={label}
              to={to}
              end={end}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150',
                  isActive
                    ? 'bg-primary text-white shadow-sm'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground',
                  disabled && 'opacity-40 pointer-events-none',
                )
              }
            >
              <Icon className="h-4 w-4 flex-shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Stats */}
        <div className="px-4 py-4 border-t border-border flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <Users className="h-3.5 w-3.5 text-muted-foreground" />
            <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
              Patient Census
            </p>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Active patients</span>
            <span className="text-sm font-bold text-foreground">{patients.length}</span>
          </div>

          <div className="h-px bg-border" />

          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-xs">
                <span className="h-2 w-2 rounded-full bg-destructive inline-block" />
                <span className="text-destructive font-medium">High risk</span>
              </span>
              <span className="text-xs font-bold text-destructive">{highCount}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-xs">
                <span className="h-2 w-2 rounded-full bg-amber-400 inline-block" />
                <span className="text-amber-600 font-medium">Moderate</span>
              </span>
              <span className="text-xs font-bold text-amber-600">{moderateCount}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-xs">
                <span className="h-2 w-2 rounded-full bg-emerald-500 inline-block" />
                <span className="text-emerald-600 font-medium">Low risk</span>
              </span>
              <span className="text-xs font-bold text-emerald-600">{lowCount}</span>
            </div>
          </div>

          {patients.length > 0 && (
            <div className="flex rounded-full overflow-hidden h-1.5">
              <div className="bg-destructive transition-all" style={{ width: `${(highCount / patients.length) * 100}%` }} />
              <div className="bg-amber-400 transition-all" style={{ width: `${(moderateCount / patients.length) * 100}%` }} />
              <div className="bg-emerald-500 transition-all" style={{ width: `${(lowCount / patients.length) * 100}%` }} />
            </div>
          )}

          {highCount > 0 && (
            <div className="rounded-lg bg-destructive/10 border border-destructive/20 px-3 py-2 flex items-center gap-2">
              <AlertTriangle className="h-3.5 w-3.5 text-destructive flex-shrink-0" />
              <p className="text-xs text-destructive font-medium">
                {highCount} patient{highCount > 1 ? 's' : ''} need attention
              </p>
            </div>
          )}
        </div>

      </div>
    </aside>
  )
}

export default function App() {
  return (
    <div className="min-h-screen bg-muted/40">
      <Sidebar />
      <main className="ml-64 p-6 min-h-screen">
        <Routes>
          <Route path="/" element={<LiveMonitor />} />
          <Route path="/patient/:stayId" element={<PatientDetailPage />} />
          <Route path="/performance" element={<ModelPerformance />} />
        </Routes>
      </main>
    </div>
  )
}
