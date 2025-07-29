from pydantic import BaseModel, Field, field_validator
from typing import Optional
import uuid
from datetime import datetime

class InterviewDataRow(BaseModel):
    """Defines and validates the schema for a single row in our feedback CSV."""
    interview_id: uuid.UUID
    timestamp: datetime
    interview_type: str
    question_number: int
    question_text: str
    answer_text: str
    wpm: int
    star_score: Optional[int] = None
    star_reason: Optional[str] = None
    keywords_score: Optional[int] = None
    keywords_reason: Optional[str] = None
    professionalism_score: Optional[int] = None
    professionalism_reason: Optional[str] = None

    @field_validator('*', mode='before')
    def empty_str_to_none(cls, v):
        if isinstance(v, str) and v.strip() == '':
            return None
        return v