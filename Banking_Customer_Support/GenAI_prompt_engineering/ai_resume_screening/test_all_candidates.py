

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix Unicode on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

from ai_resume_screening.main import run_screening_pipeline

load_dotenv()


def main():
    """Run screening for all 3 candidate types"""
    
    base_dir = Path(__file__).resolve().parent
    job_path = base_dir / "data" / "job_description.txt"
    
    candidates = [
        {"name": "STRONG - John Alexander", "file": base_dir / "data" / "resumes" / "sample_resume.txt"},
        {"name": "AVERAGE - Sarah Johnson", "file": base_dir / "data" / "resumes" / "average_resume.txt"},
        {"name": "WEAK - Alex Kumar", "file": base_dir / "data" / "resumes" / "weak_resume.txt"},
    ]
    
    all_results = []
    
    print("=" * 70)
    print("RESUME SCREENING - ALL CANDIDATES")
    print("=" * 70)
    
    # Check setup
    groq_key = os.getenv("GROQ_API_KEY")
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    
    if not groq_key:
        print("ERROR: GROQ_API_KEY missing in .env")
        return
    
    print(f"GROQ API Key: Found")
    print(f"LangSmith API Key: {'Found' if langsmith_key else 'Not found'}")
    print(f"Tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
    
    # Process each candidate
    for i, candidate in enumerate(candidates, 1):
        print("\n" + "=" * 70)
        print(f"CANDIDATE {i}: {candidate['name']}")
        print("=" * 70)
        
        try:
            if not Path(candidate['file']).exists():
                print(f"ERROR: File not found: {candidate['file']}")
                continue
            
            results = run_screening_pipeline(str(candidate['file']), str(job_path))
            all_results.append({"candidate": candidate['name'], "results": results})
            print(f"\nStatus: COMPLETE")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for candidate in all_results:
        name = candidate['candidate']
        score = candidate['results']['scores']['final_score']
        match = candidate['results']['matching_analysis']['match_percentage']
        print(f"{name}: {score}/100 | Match: {match}%")
    
    print(f"\nTotal Evaluated: {len(all_results)}")
    
    print("\n" + "=" * 70)
    print("SCREENING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
