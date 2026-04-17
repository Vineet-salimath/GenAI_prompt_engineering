"""
Resume Screening Pipeline - Main orchestrator
Runs the complete candidate evaluation workflow
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Fix Unicode on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

from chains.extraction_chain import create_extraction_chain, extract_skills
from chains.matching_chain import create_matching_chain, match_skills
from chains.scoring_chain import create_scoring_chain, score_candidate
from chains.explanation_chain import create_explanation_chain, generate_explanation

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")


def load_resume(file_path):
    """Read resume file"""
    try:
        if file_path.endswith('.pdf'):
            print("PDF support not implemented yet")
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise Exception(f"Resume file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Failed to read resume: {e}")


def load_job_description(file_path):
    """Read job description file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise Exception(f"Job description file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Failed to read job description: {e}")


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title):
    """Print formatted subsection header"""
    print("\n" + title)
    print("-" * 60)


def format_skills_list(skills_dict):
    """Format skills dictionary nicely"""
    output = []
    if isinstance(skills_dict, dict):
        for category, items in skills_dict.items():
            if items:
                if isinstance(items, list):
                    output.append(f"{category}: {', '.join(items)}")
                else:
                    output.append(f"{category}: {items}")
    return "\n".join(output)


def run_screening_pipeline(resume_path, job_description_path):
    """
    Run complete screening pipeline for a resume
    Returns dict with all evaluation results
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env")
    
    print_section("AI RESUME SCREENING SYSTEM")
    
    # Load resume and job description
    print("\nLoading files...")
    resume_text = load_resume(resume_path)
    job_description = load_job_description(job_description_path)
    
    if not resume_text or not job_description:
        raise Exception("Failed to load files")
    
    print(f"Resume loaded: {len(resume_text)} characters")
    print(f"Job description loaded: {len(job_description)} characters")
    
    # Step 1: Extract skills
    print("\nExtracting skills...")
    extraction_chain = create_extraction_chain(GROQ_API_KEY)
    candidate_skills = extract_skills(extraction_chain, resume_text)
    
    # Step 2: Match skills
    print("Analyzing job match...")
    matching_chain = create_matching_chain(GROQ_API_KEY)
    matching_analysis = match_skills(matching_chain, candidate_skills, job_description)
    
    # Step 3: Score
    print("Calculating score...")
    scoring_chain = create_scoring_chain(GROQ_API_KEY)
    scores = score_candidate(scoring_chain, matching_analysis, job_description, candidate_skills)
    
    # Step 4: Generate explanation
    print("Generating explanation...")
    explanation_chain = create_explanation_chain(GROQ_API_KEY)
    explanation = generate_explanation(
        explanation_chain,
        resume_path,
        job_description,
        scores,
        matching_analysis
    )
    
    # Display results in professional format
    print_section("SCREENING RESULTS")
    
    # Candidate Details
    print_subsection("CANDIDATE DETAILS")
    if isinstance(candidate_skills, dict):
        print(f"Name: {candidate_skills.get('name', 'Not specified')}")
        print(f"Email: {candidate_skills.get('email', 'Not specified')}")
        print(f"Experience: {candidate_skills.get('experience_years', 'Not specified')} years")
    
    # Extracted Information
    print_subsection("EXTRACTED INFORMATION")
    if isinstance(candidate_skills, dict):
        skills_formatted = format_skills_list(candidate_skills.get('skills', {}))
        print(skills_formatted)
        if candidate_skills.get('certifications'):
            print(f"\nCertifications: {', '.join(candidate_skills['certifications'])}")
    
    # Match Analysis
    print_subsection("MATCH ANALYSIS")
    if isinstance(matching_analysis, dict):
        if matching_analysis.get('matched'):
            print(f"Matched Skills: {len(matching_analysis['matched'])}")
            for skill in matching_analysis['matched'][:10]:
                print(f"  - {skill}")
            if len(matching_analysis['matched']) > 10:
                print(f"  ... and {len(matching_analysis['matched']) - 10} more")
        
        if matching_analysis.get('missing'):
            print(f"\nMissing Skills: {len(matching_analysis['missing'])}")
            for skill in matching_analysis['missing'][:10]:
                print(f"  - {skill}")
            if len(matching_analysis['missing']) > 10:
                print(f"  ... and {len(matching_analysis['missing']) - 10} more")
        
        print(f"\nMatch Percentage: {matching_analysis.get('match_percentage', 'N/A')}%")
    
    # Scoring Results
    print_subsection("SCORE")
    if isinstance(scores, dict):
        print(f"Final Score: {scores.get('final_score', 'N/A')}/100")
        print(f"Technical Score: {scores.get('technical_score', 'N/A')}/100")
        print(f"Experience Score: {scores.get('experience_score', 'N/A')}/100")
        print(f"Certification Score: {scores.get('certification_score', 'N/A')}/100")
        print(f"Soft Skills Score: {scores.get('soft_skills_score', 'N/A')}/100")
    
    # Explanation
    print_subsection("EXPLANATION")
    print(explanation)
    
    print_section("SCREENING COMPLETE")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "resume_file": resume_path,
        "job_description_file": job_description_path,
        "candidate_info": candidate_skills,
        "matching_analysis": matching_analysis,
        "scores": scores,
        "explanation": explanation
    }
    
    return results


def main():
    """Entry point for single resume screening"""
    
    resume_path = "data/resumes/sample_resume.txt"
    job_description_path = "data/job_description.txt"
    
    if not Path(resume_path).exists():
        print(f"Resume not found: {resume_path}")
        return
    
    if not Path(job_description_path).exists():
        print(f"Job description not found: {job_description_path}")
        return
    
    try:
        run_screening_pipeline(resume_path, job_description_path)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
