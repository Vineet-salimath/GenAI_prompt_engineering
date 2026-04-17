from langchain_core.prompts import PromptTemplate

template = """
Write concise explanation of candidate fit.

Candidate:
{candidate_info}

Job Requirements:
{job_requirements}

Scores:
{scoring_results}

Analysis:
{matching_analysis}

Format output as:

EXPLANATION
-----------
Summary: [2-3 sentences about fit]

Strengths: [list 3-4 main strengths]

Gaps: [list 2-3 missing skills]

Recommendation: [Highly Suitable / Suitable / Consider / Not Suitable]

Keep it factual and concise.
"""

explanation_prompt = PromptTemplate(
    input_variables=["candidate_info", "job_requirements", "scoring_results", "matching_analysis"],
    template=template
)
