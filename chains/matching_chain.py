
from langchain_groq import ChatGroq
from prompts.matching_prompt import matching_prompt
import json


def create_matching_chain(api_key):
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024
    )
    
    chain = matching_prompt | llm
    return chain


def match_skills(chain, candidate_skills, job_requirements):
    try:
        # Convert candidate_skills dict to string if needed
        if isinstance(candidate_skills, dict):
            candidate_skills_str = json.dumps(candidate_skills, indent=2)
        else:
            candidate_skills_str = str(candidate_skills)
        
        result = chain.invoke({
            "candidate_skills": candidate_skills_str,
            "job_requirements": job_requirements
        })
        
        content = result.content.strip()
        
        # Parse JSON from response
        if content.startswith('{'):
            match_data = json.loads(content)
        else:
            # Extract JSON if there's extra text
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                match_data = json.loads(content[json_start:json_end])
            else:
                raise ValueError("No valid JSON found in response")
        
        return match_data
    except Exception as e:
        raise Exception(f"Matching failed: {e}")
