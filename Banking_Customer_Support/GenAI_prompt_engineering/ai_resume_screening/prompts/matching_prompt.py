
from langchain_core.prompts import PromptTemplate

template = """Compare candidate skills with job requirements.

Candidate Skills:
{candidate_skills}

Job Requirements:
{job_requirements}

Return STRICT JSON:
{{
  "matched": [],
  "missing": [],
  "extra": [],
  "match_percentage": 0,
  "summary": ""
}}

Rules:
- Only include exact matches in matched array
- List all missing required skills
- Include bonus skills not required but present
- Calculate match percentage (matched/total required * 100)
- Keep output as valid JSON only
- No explanations or text outside JSON"""

matching_prompt = PromptTemplate(
    input_variables=["candidate_skills", "job_requirements"],
    template=template
)
