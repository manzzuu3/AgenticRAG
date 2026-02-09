import json
import os
from typing import Dict, Optional, List

# Path to the actual patient data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PATIENTS_FILE = os.path.join(PROJECT_ROOT, "data", "patients.json")

class PatientDataTool:
    def __init__(self):
        self.patients = self._load_patients()
    
    def _load_patients(self) -> Dict:
        if not os.path.exists(PATIENTS_FILE):
            print(f"Warning: Patient data file not found at {PATIENTS_FILE}")
            return {}
        
        with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
            patients_list = json.load(f)
        
        return {p["patient_id"]: p for p in patients_list}
    
    def get_patient_data(self, patient_id: str) -> Optional[Dict]:
        return self.patients.get(patient_id)
    
    def list_patients(self) -> List[str]:
        return list(self.patients.keys())

# Singleton instance
patient_tool = PatientDataTool()
