
from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import PromptTemplate

template = """You are an AI Resume Parser.

Extract ONLY information explicitly present in the resume.

Resume:
{resume_text}

Return STRICT JSON in this exact format:
{{
  "name": "",
  "email": "",
  "skills": [
    "Programming Languages: ",
    "Frameworks: ",
    "Tools & Platforms: ",
    "Databases: ",
    "DevOps & Cloud: ",
    "Soft Skills: "
  ],
  "certifications": [],
  "experience_years": "",
  "domain": ""
}}

Rules:
- Do NOT add extra fields
- Do NOT hallucinate
- Group skills into categories
- Keep output as valid JSON only
- No explanations or text outside JSON"""

extraction_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template=template
)
