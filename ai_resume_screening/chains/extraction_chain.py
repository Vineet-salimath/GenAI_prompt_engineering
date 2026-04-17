from langchain_groq import ChatGroq
from ai_resume_screening.prompts.extraction_prompt import extraction_prompt
import json


def create_extraction_chain(api_key):
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024
    )
    
    chain = extraction_prompt | llm
    return chain


def extract_skills(chain, resume_text):
    try:
        result = chain.invoke({"resume_text": resume_text})
        content = result.content.strip()
        
        # Parse JSON from response
        if content.startswith('{'):
            extracted_data = json.loads(content)
        else:
            # Extract JSON if there's extra text
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                extracted_data = json.loads(content[json_start:json_end])
            else:
                raise ValueError("No valid JSON found in response")
        
        return extracted_data
    except Exception as e:
        raise Exception(f"Skill extraction failed: {e}")
