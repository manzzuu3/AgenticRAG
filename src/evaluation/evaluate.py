import asyncio
import json
import os

from src.agent.agent import run_chat
from src.tools.rag_search import rag_tool
from src.tools.patient_data import patient_tool

PATIENTS = patient_tool.patients

async def eval_patient(patient_id: str, description: str):
    """Evaluate a single patient case."""
    print(f"\n{'='*60}")
    print(f"Patient: {patient_id} - {description}")
    print('='*60)
    
    session_id = f"eval_{patient_id}"
    prompt = f"Assess patient {patient_id} for cancer referral"
    
    try:
        response = await run_chat(session_id, prompt)
        
        print(f"\nAssessment: {response.assessment}")
        print(f"Reasoning: {response.reasoning[:150]}...")
        print(f"Citations: {len(response.citations)} found")
        
        if response.citations:
            print("\nCitation samples:")
            for i, cite in enumerate(response.citations[:2]):
                print(f"  [{i+1}] {cite.source} p.{cite.page}: {cite.excerpt[:60]}...")
        
        return True
        
    except Exception as e:
        print(f"Evaluation failed for {patient_id}: {str(e)[:100]}")
        return False

async def main():
    print("="*60)
    print("NG12 Agent Evaluation Suite")
    print("="*60)
    
    # Verify pipeline is loaded
    if rag_tool.index is None:
        print("FAILED: FAISS index not loaded. Run 'python main.py' first.")
        return
    print("Pipeline loaded (FAISS index ready)")
    print(f"Patient database loaded ({len(PATIENTS)} patients)")
    
    test_cases = [
        ("PT-101", "Hemoptysis - high risk"),
        ("PT-108", "Breast lump - urgent referral"),
        ("PT-107", "Persistent hoarseness - smoker"),
        ("PT-102", "Low risk case (young, brief symptoms)")
    ]
    
    results = []
    for pid, desc in test_cases:
        if pid in PATIENTS:
            success = await eval_patient(pid, desc)
            results.append(success)
        else:
            print(f"\n✗ Patient {pid} not found in database")
            results.append(False)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    print('='*60)

if __name__ == "__main__":
    asyncio.run(main())
