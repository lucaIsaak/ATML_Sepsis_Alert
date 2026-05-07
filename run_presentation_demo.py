"""
SepsisAlert — Presentation Demo
================================
Self-contained demonstration of the full alert pipeline.
No MIMIC-IV data required — uses five hand-crafted clinical scenarios
that span all escalation tiers so every part of the system is shown.

Run:
    python run_presentation_demo.py
    python run_presentation_demo.py --no-narrative   # skip Ollama (faster)
    python run_presentation_demo.py --delay 1.5      # add pause between patients

What the demo shows
-------------------
  Patient A  CRITICAL  — septic shock   (score ≥ 0.80)
  Patient B  DOCTOR    — early sepsis   (score 0.60–0.79)
  Patient C  NURSE     — at-risk        (score 0.40–0.59)
  Patient D  LOW       — stable         (score < 0.40)  → no alert
  Patient E  TREND     — deteriorating  (starts LOW, worsens each cycle)
             Demonstrates rapid-deterioration detection (agent escalates
             even before the score crosses the doctor threshold)

Output for each alert
---------------------
  • Risk score + escalation tier
  • Top SHAP feature drivers with values and units
  • OOD confidence flag (Layer 1 safety)
  • Nurse SBAR narrative (Layer 2 validated)
  • Doctor clinical summary (HIGH / CRITICAL only)
  • Audit log entry written to logs/audit.jsonl
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

from src.agent.monitor_agent import EscalationTier, PatientMonitorAgent
from src.data.patient_buffer import ALL_LAB_ITEMS, ALL_VITAL_ITEMS, Observation


# ------------------------------------------------------------------ #
# Clinical scenarios — feature values chosen to span all tiers        #
# Each scenario is a dict: {item_name: [value_h0, value_h4, value_h8]}
# Three observations per item spread over 12 simulated hours.         #
# ------------------------------------------------------------------ #

SCENARIOS: dict[str, dict] = {
    "P-CRITICAL": {
        "label": "Septic Shock",
        "description": "72yo M | MICU | Admitted from ED with fever and altered mentation",
        "expected_tier": "CRITICAL",
        # MAP: 48 mmHg (shock), HR: 135, RR: 30, Temp: 102.8°F, SpO2: 88%
        "heart_rate":    [125, 132, 138],
        "map":           [55,  50,  46],
        "resp_rate":     [26,  29,  32],
        "temperature_f": [101.5, 102.2, 102.9],
        "spo2":          [91,  89,  87],
        # Labs: lactate 6.8, creatinine 2.9, WBC 22
        "lactate":       [4.2, 5.9, 6.8],
        "wbc":           [16,  19,  22],
        "creatinine":    [1.8, 2.4, 2.9],
        "bilirubin":     [2.1, 2.8, 3.4],
        "platelets":     [140, 115, 88],
        "bicarbonate":   [20,  18,  15],
        "glucose":       [180, 210, 195],
    },
    "P-DOCTOR": {
        "label": "Early Sepsis",
        "description": "58yo F | Surgical ICU | Post-op day 2, rising inflammatory markers",
        "expected_tier": "DOCTOR",
        "heart_rate":    [105, 110, 118],
        "map":           [64,  62,  60],
        "resp_rate":     [22,  24,  26],
        "temperature_f": [101.0, 101.4, 101.8],
        "spo2":          [94,  93,  92],
        "lactate":       [2.1, 2.8, 3.2],
        "wbc":           [13,  15,  17],
        "creatinine":    [1.3, 1.5, 1.8],
        "bilirubin":     [1.0, 1.4, 1.8],
        "platelets":     [185, 160, 145],
        "bicarbonate":   [22,  21,  20],
        "glucose":       [145, 158, 162],
    },
    "P-NURSE": {
        "label": "At-Risk",
        "description": "65yo M | Medical ICU | Pneumonia, borderline vitals",
        "expected_tier": "NURSE",
        "heart_rate":    [98,  102, 105],
        "map":           [68,  66,  64],
        "resp_rate":     [20,  22,  23],
        "temperature_f": [100.2, 100.5, 100.8],
        "spo2":          [94,  94,  93],
        "lactate":       [1.8, 2.0, 2.2],
        "wbc":           [11,  12,  13],
        "creatinine":    [1.1, 1.2, 1.3],
        "bilirubin":     [0.8, 1.0, 1.1],
        "platelets":     [200, 190, 180],
        "bicarbonate":   [23,  22,  22],
        "glucose":       [130, 138, 142],
    },
    "P-STABLE": {
        "label": "Stable / Recovering",
        "description": "45yo F | Cardiac ICU | Post-CABG day 3, all markers improving",
        "expected_tier": "NONE",
        "heart_rate":    [75,  72,  70],
        "map":           [85,  88,  90],
        "resp_rate":     [14,  14,  13],
        "temperature_f": [98.6, 98.4, 98.2],
        "spo2":          [98,  99,  99],
        "lactate":       [1.0, 0.9, 0.8],
        "wbc":           [7,   6.5, 6],
        "creatinine":    [0.8, 0.8, 0.7],
        "bilirubin":     [0.5, 0.4, 0.4],
        "platelets":     [220, 230, 240],
        "bicarbonate":   [25,  25,  26],
        "glucose":       [100, 95,  92],
    },
    "P-TREND": {
        "label": "Deteriorating (Trend Detection)",
        "description": "68yo M | Medical ICU | Initially stable, rapid decline over 3 cycles",
        "expected_tier": "TREND",
        # Starts borderline, worsens each cycle — tests trend escalation logic
        "heart_rate":    [88,  102, 122],
        "map":           [75,  66,  58],
        "resp_rate":     [16,  22,  28],
        "temperature_f": [99.0, 100.6, 102.2],
        "spo2":          [96,  93,  90],
        "lactate":       [1.2, 2.5, 4.1],
        "wbc":           [8,   13,  18],
        "creatinine":    [0.9, 1.4, 2.1],
        "bilirubin":     [0.6, 1.1, 1.9],
        "platelets":     [210, 170, 130],
        "bicarbonate":   [24,  21,  17],
        "glucose":       [110, 140, 175],
    },
}

ITEM_IDS = {**ALL_VITAL_ITEMS, **ALL_LAB_ITEMS}
# Invert: name → first matching id
_NAME_TO_ID = {v: k for k, v in ITEM_IDS.items()}


def _build_observations(
    scenario: dict, base_time: datetime, cycle: int = 0
) -> list[Observation]:
    """
    Build Observation objects for one time step (cycle 0, 1, or 2).

    Each cycle advances 4 simulated hours so the agent sees trend data.
    """
    obs_time = base_time + timedelta(hours=cycle * 4)
    observations = []
    for name, values in scenario.items():
        if name in ("label", "description", "expected_tier"):
            continue
        if not isinstance(values, list) or cycle >= len(values):
            continue
        obs = Observation(
            timestamp=obs_time,
            item_name=name,
            value=float(values[cycle]),
            source="demo",
            tier=1,
        )
        obs.stay_id = None     # set by caller
        observations.append(obs)
    return observations


def _divider(char: str = "─", width: int = 58) -> str:
    return char * width


def _print_alert(alert, scenario_info: dict, delay: float) -> None:
    """Print a formatted alert block for the presentation."""
    tier = alert.tier
    tier_colors = {
        EscalationTier.CRITICAL: "🔴 CRITICAL",
        EscalationTier.DOCTOR:   "🟠 DOCTOR",
        EscalationTier.NURSE:    "🟡 NURSE",
    }
    label = tier_colors.get(tier, str(tier.name))

    print(f"\n{_divider('═')}")
    print(f"  {label} ALERT  |  Stay {alert.stay_id}")
    print(f"  {scenario_info['label']} — {scenario_info['description']}")
    print(_divider())
    print(f"  Risk Score : {alert.risk_score:.3f}")
    print(f"  Threshold  : 0.4 (nurse) / 0.6 (doctor) / 0.8 (critical)")

    if alert.top_features:
        print(f"\n  Top SHAP drivers:")
        for feat in alert.top_features[:5]:
            val  = f"{feat['value']:.1f} {feat['unit']}" if feat.get("value") else "N/A"
            arrow = "↑" if feat["direction"] == "increases_risk" else "↓"
            shap_sign = "+" if feat["shap"] > 0 else ""
            print(f"    {arrow} {feat['label']:35s} {val:12s}  "
                  f"[SHAP {shap_sign}{feat['shap']:.3f}]")

    if alert.nurse_narrative:
        print(f"\n  NURSE ALERT (SBAR):")
        for line in alert.nurse_narrative.strip().splitlines():
            print(f"    {line}")

    if alert.doctor_narrative:
        print(f"\n  PHYSICIAN SUMMARY:")
        for line in alert.doctor_narrative.strip().splitlines():
            print(f"    {line}")

    print(_divider())
    if delay > 0:
        time.sleep(delay)


def run_demo(no_narrative: bool = False, delay: float = 1.0) -> None:
    """Run the full presentation demo."""
    print(_divider("═"))
    print("  SepsisAlert — ICU Early Warning System")
    print("  Presentation Demo — 5 Clinical Scenarios")
    print(_divider("═"))
    print("  Loading model and agent...")

    agent = PatientMonitorAgent()

    # Disable LLM if requested (for fast offline demo)
    if no_narrative:
        agent.narrative = None
        print("  [Narrative generation disabled]")

    base_time = datetime(2024, 3, 15, 8, 0, 0)   # 08:00 ICU round
    total_alerts = 0

    print(f"\n  Model loaded. Running {len(SCENARIOS)} patient scenarios.\n")

    for stay_id, scenario in enumerate(SCENARIOS.values(), start=1001):
        sid = str(stay_id)
        print(f"\n{'─'*58}")
        print(f"  Processing: {scenario['label']} (Stay {sid})")
        print(f"  {scenario['description']}")

        # Push 3 observation cycles (0h, 4h, 8h) so the agent sees trends
        for cycle in range(3):
            obs_list = _build_observations(scenario, base_time, cycle)
            for obs in obs_list:
                obs.stay_id = sid
                agent.registry.push(sid, obs)

            cycle_time = base_time + timedelta(hours=cycle * 4)
            new_alerts = agent.run_cycle(timestamp=cycle_time)

            for alert in new_alerts:
                total_alerts += 1
                _print_alert(alert, scenario, delay)

        # If no alert fired (stable patient), print a confirmation
        mem = agent._memory.get(sid)  # noqa: SLF001 (private access for demo)
        last_score = mem.last_risk_score if mem else None
        if last_score is not None and all(
            a.stay_id != sid for a in agent.alert_log
        ):
            print(f"  ✓ No alert — Risk Score {last_score:.3f} (below threshold 0.4)")

    # Final dashboard-style summary
    print(f"\n{_divider('═')}")
    print("  DEMO COMPLETE — Agent Summary")
    print(_divider("═"))
    summary = agent.summary()
    for key, val in summary.items():
        print(f"  {key:25s}: {val}")

    tier_counts = {t.name: 0 for t in EscalationTier}
    for alert in agent.alert_log:
        tier_counts[alert.tier.name] += 1

    print("\n  Escalation breakdown:")
    for tier_name, count in tier_counts.items():
        if count > 0:
            print(f"    {tier_name:10s}: {count}")

    print(f"\n  Audit log: logs/audit.jsonl ({total_alerts} records written)")
    print(_divider("═"))  # pylint: disable=f-string-without-interpolation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SepsisAlert presentation demo")
    parser.add_argument("--no-narrative", action="store_true",
                        help="Skip LLM narrative generation (no Ollama required)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to pause between alert displays (default 0.5)")
    args = parser.parse_args()

    run_demo(no_narrative=args.no_narrative, delay=args.delay)
