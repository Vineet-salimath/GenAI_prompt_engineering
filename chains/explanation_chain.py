
"""Explanation Generation Chain"""

import json
from langchain_groq import ChatGroq
from prompts.explanation_prompt import explanation_prompt


def create_explanation_chain(api_key):
    """Create chain for generating detailed explanations"""
    
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1024
    )
    
    chain = explanation_prompt | llm
    return chain


def generate_explanation(chain, resume_path, job_requirements, scores, matching_analysis):
    """Generate human-readable explanation of fit"""
    
    try:
        # Convert dicts to strings if needed
        if isinstance(scores, dict):
            scores_str = json.dumps(scores, indent=2)
        else:
            scores_str = str(scores)
        
        if isinstance(matching_analysis, dict):
            matching_analysis_str = json.dumps(matching_analysis, indent=2)
        else:
            matching_analysis_str = str(matching_analysis)
        
        result = chain.invoke({
            "candidate_info": resume_path,
            "job_requirements": job_requirements,
            "scoring_results": scores_str,
            "matching_analysis": matching_analysis_str
        })
        return result.content
    except Exception as e:
        raise Exception(f"Explanation generation failed: {e}")
