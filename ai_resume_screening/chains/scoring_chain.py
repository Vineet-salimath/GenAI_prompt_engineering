
import json
from langchain_groq import ChatGroq
from ai_resume_screening.prompts.scoring_prompt import scoring_prompt


def create_scoring_chain(api_key):
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=512
    )
    
    chain = scoring_prompt | llm
    return chain


def score_candidate(chain, matching_analysis, job_requirements, candidate_skills):
    try:
        # Convert dicts to strings if needed
        if isinstance(matching_analysis, dict):
            matching_analysis_str = json.dumps(matching_analysis, indent=2)
        else:
            matching_analysis_str = str(matching_analysis)
        
        if isinstance(candidate_skills, dict):
            candidate_skills_str = json.dumps(candidate_skills, indent=2)
        else:
            candidate_skills_str = str(candidate_skills)
        
        result = chain.invoke({
            "matching_analysis": matching_analysis_str,
            "job_requirements": job_requirements,
            "candidate_skills": candidate_skills_str
        })
        
        response_text = result.content
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        scores = json.loads(json_str)
        
        return scores
    except Exception as e:
        raise Exception(f"Scoring failed: {e}")
