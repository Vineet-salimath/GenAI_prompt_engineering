
from langchain_core.prompts import PromptTemplate

template = """
Score the candidate 0-100 based on job match.

Matching Analysis:
{matching_analysis}

Job Requirements:
{job_requirements}

Candidate Skills:
{candidate_skills}

Score components (weights):
- Technical Skills: 40%
- Experience: 30%
- Certifications: 15%
- Soft Skills: 15%

Return ONLY valid JSON, no other text:
{{
    "final_score": <number 0-100>,
    "technical_score": <number 0-100>,
    "experience_score": <number 0-100>,
    "certification_score": <number 0-100>,
    "soft_skills_score": <number 0-100>
}}
"""

scoring_prompt = PromptTemplate(
    input_variables=["matching_analysis", "job_requirements", "candidate_skills"],
    template=template
)
