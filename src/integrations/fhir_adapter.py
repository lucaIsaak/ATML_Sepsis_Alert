"""
HL7 FHIR R4 Adapter — Epic / Oracle Health (Cerner) compatibility.

PRODUCTION STUB — not active in the demo.
This adapter is ready to connect to a hospital EHR once OAuth credentials
and a FHIR endpoint URL are provided by the hospital partner.
No changes to the rest of the codebase are required — plug in the base_url
and bearer token and call get_patient_features() from the monitoring loop.

Both Epic and Oracle Health expose patient data via HL7 FHIR R4 APIs.
This adapter translates FHIR Observation resources into the feature
format SepsisAlert's model expects.

FHIR endpoint pattern:
  Epic:   https://<hospital>/api/FHIR/R4/Observation?patient=<id>&category=vital-signs
  Cerner: https://<hospital>/r4/Observation?patient=<id>&code=<loinc>

LOINC code mapping (standard across both systems):
  https://loinc.org/
"""

from __future__ import annotations

from typing import Optional

import requests

from src.schemas import LabValues, VitalSigns


# LOINC codes for the values SepsisAlert uses
# These are universal — same in Epic, Cerner, and any FHIR-compliant system
LOINC_MAP = {
    # Vitals
    "8867-4":   "heart_rate",
    "8480-6":   "sbp",
    "8462-4":   "dbp",
    "8478-0":   "map",
    "9279-1":   "resp_rate",
    "8310-5":   "temperature_c",
    "2708-6":   "spo2",
    "9269-2":   "gcs_total",
    # Labs — infection
    "6690-2":   "wbc",
    "2524-7":   "lactate",
    # Organ function (SOFA)
    "2160-0":   "creatinine",
    "1975-2":   "bilirubin",
    "777-3":    "platelets",
    # BMP
    "2951-2":   "sodium",
    "2823-3":   "potassium",
    "1963-8":   "bicarbonate",
    "3094-0":   "bun",
    "2345-7":   "glucose",
    # CBC
    "718-7":    "hemoglobin",
}


class FHIRAdapter:
    """
    Fetches and transforms FHIR Observations into SepsisAlert input format.

    Works with any FHIR R4 compliant endpoint (Epic, Oracle/Cerner, Azure Health).

    Usage:
        adapter = FHIRAdapter(base_url="https://epic.hospital.de/api/FHIR/R4",
                              token="Bearer <oauth_token>")
        vitals, labs = adapter.get_patient_features(patient_id="12345",
                                                     encounter_id="67890")
    """

    def __init__(self, base_url: str, token: str):
        """Initialise the adapter with an EHR base URL and OAuth bearer token."""
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": token,
            "Accept": "application/fhir+json",
        }

    def get_observations(self, patient_id: str, loinc_codes: list[str]) -> list[dict]:
        """Fetch FHIR Observations for a patient filtered by LOINC codes."""
        code_param = ",".join(loinc_codes)
        url = f"{self.base_url}/Observation"
        params = {
            "patient": patient_id,
            "code": code_param,
            "_sort": "-date",
            "_count": 100,
        }
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        bundle = response.json()
        return bundle.get("entry", [])

    def get_patient_features(
        self, patient_id: str, encounter_id: Optional[str] = None
    ) -> tuple[VitalSigns, LabValues]:
        """
        Main entry point: fetch latest vitals + labs for a patient.

        Returns validated VitalSigns and LabValues pydantic objects
        ready to feed into the SepsisAlert feature pipeline.
        """
        entries = self.get_observations(patient_id, list(LOINC_MAP.keys()))

        # Optionally filter by encounter
        if encounter_id:
            entries = [
                e for e in entries
                if e.get("resource", {}).get("encounter", {}).get("reference", "")
                == f"Encounter/{encounter_id}"
            ]

        # Take the most recent value per LOINC code
        latest: dict[str, float] = {}
        for entry in entries:
            obs = entry.get("resource", {})
            codings = obs.get("code", {}).get("coding", [])
            for coding in codings:
                loinc = coding.get("code")
                if loinc in LOINC_MAP:
                    feat_name = LOINC_MAP[loinc]
                    value = obs.get("valueQuantity", {}).get("value")
                    if value is not None and feat_name not in latest:
                        latest[feat_name] = float(value)

        vitals = VitalSigns(
            heart_rate=latest.get("heart_rate"),
            sbp=latest.get("sbp"),
            dbp=latest.get("dbp"),
            map=latest.get("map"),
            resp_rate=latest.get("resp_rate"),
            temperature_c=latest.get("temperature_c"),
            spo2=latest.get("spo2"),
            gcs_total=latest.get("gcs_total"),
        )

        labs = LabValues(
            wbc=latest.get("wbc"),
            lactate=latest.get("lactate"),
            creatinine=latest.get("creatinine"),
            bilirubin=latest.get("bilirubin"),
            platelets=latest.get("platelets"),
            sodium=latest.get("sodium"),
            potassium=latest.get("potassium"),
            bicarbonate=latest.get("bicarbonate"),
            bun=latest.get("bun"),
            glucose=latest.get("glucose"),
            hemoglobin=latest.get("hemoglobin"),
        )

        return vitals, labs

    @staticmethod
    def vitals_and_labs_to_feature_dict(
        vitals: VitalSigns, labs: LabValues
    ) -> dict:
        """
        Convert validated pydantic objects to the flat feature dict
        the model's predict_patient() function expects.

        Note: the model expects aggregated stats (mean/min/max/last/delta).
        For real-time single-timepoint data from FHIR, we use the
        current value for all aggregates. Future versions will maintain
        a rolling buffer per patient.
        """
        features = {}

        # Vitals — replicate value across all aggregates for single-point inference
        vital_map = {
            "heart_rate": vitals.heart_rate,
            "map": vitals.map,
            "resp_rate": vitals.resp_rate,
            "temperature_f": (vitals.temperature_c * 9/5 + 32) if vitals.temperature_c else None,
            "spo2": vitals.spo2,
        }
        for name, val in vital_map.items():
            for suffix in ["mean", "min", "max", "last"]:
                features[f"{name}_{suffix}"] = val

        # Labs — last + mean = current value, delta = 0 (single point)
        lab_map = {
            "lactate": labs.lactate,
            "wbc": labs.wbc,
            "creatinine": labs.creatinine,
            "bilirubin": labs.bilirubin,
            "platelets": labs.platelets,
            "bicarbonate": labs.bicarbonate,
            "glucose": labs.glucose,
        }
        for name, val in lab_map.items():
            features[f"{name}_last"] = val
            features[f"{name}_mean"] = val
            features[f"{name}_delta"] = 0.0

        return features
