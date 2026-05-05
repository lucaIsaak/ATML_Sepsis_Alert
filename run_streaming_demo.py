"""
SepsisAlert — Streaming Demo

Replays MIMIC-IV data in time order through the full agent pipeline.
Shows exactly how the system would behave in a live ICU deployment.

Usage:
    python run_streaming_demo.py
    python run_streaming_demo.py --patients 10 --hours 48
"""

import argparse
import time
from datetime import datetime

from src.data.streaming import MIMICStreamSimulator
from src.agent.monitor_agent import PatientMonitorAgent, EscalationTier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=int, default=20,
                        help="Number of ICU patients to simulate")
    parser.add_argument("--hours", type=int, default=48,
                        help="Hours of ICU time to replay")
    parser.add_argument("--no-narrative", action="store_true",
                        help="Skip LLM narrative generation (faster demo)")
    args = parser.parse_args()

    print("=" * 60)
    print("SepsisAlert — Streaming Demo")
    print(f"Patients: {args.patients} | Replay: {args.hours}h of ICU time")
    print("=" * 60)

    # Initialise agent and simulator
    agent = PatientMonitorAgent()
    sim = MIMICStreamSimulator(n_patients=args.patients)
    sim.load_events()

    # Limit to first N hours for demo
    all_alerts = []
    bucket_count = 0
    max_buckets = args.hours   # each bucket = 1 hour

    for timestamp, observations in sim.stream(batch_minutes=60):
        if bucket_count >= max_buckets:
            break

        # Push observations into patient buffers
        new_alerts = agent.process_streaming_batch(timestamp, observations)

        for alert in new_alerts:
            all_alerts.append(alert)
            tier_name = alert.tier.name
            print(f"\n{'='*50}")
            print(f"  [{tier_name}] Stay {alert.stay_id} @ {timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Risk Score: {alert.risk_score:.3f}")

            if alert.top_features:
                print("  Key drivers:")
                for feat in alert.top_features[:3]:
                    val = f"{feat['value']:.1f} {feat['unit']}" if feat.get("value") else "N/A"
                    print(f"    - {feat['label']}: {val}")

            if not args.no_narrative and alert.nurse_narrative:
                print(f"\n  NURSE ALERT:\n{alert.nurse_narrative}")

            if not args.no_narrative and alert.doctor_narrative:
                print(f"\n  DOCTOR SUMMARY:\n{alert.doctor_narrative}")

        bucket_count += 1

        # Progress every 6 hours of simulated time
        if bucket_count % 6 == 0:
            summary = agent.summary()
            print(
                f"\n[t={bucket_count}h] Active: {summary['active_patients']} patients | "
                f"Alerts: {summary['total_alerts']} total | "
                f"Critical: {summary['critical']} | "
                f"Doctor: {summary['doctor']} | "
                f"Nurse: {summary['nurse']}"
            )

    # Final summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE — Final Summary")
    print("=" * 60)
    summary = agent.summary()
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")

    print(f"\n  Tier breakdown:")
    for tier in EscalationTier:
        count = sum(1 for a in all_alerts if a.tier == tier)
        if count > 0:
            print(f"    {tier.name:10s}: {count} alerts")


if __name__ == "__main__":
    main()
