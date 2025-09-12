from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import date

class PersonalInfo(BaseModel):
    full_name: str
    email: EmailStr
    phone_number: Optional[str] = None
    password: str           #MFA will be provided by Auth0
    date_of_birth: date
    gender: Optional[str] = None

class LifestyleInfo(BaseModel):
    occupation: str
    work_hours: int
    work_mode: str  # Remote/In-person
    physical_activity: str
    diet_habits: str
    caffeine_intake: str
    smoking_status: bool
    sleep_hours: float
    water_intake: float

class MentalHealthHistory(BaseModel):
    prior_conditions: List[str]
    current_medications: Optional[List[str]] = None
    therapy_sessions: Optional[str] = None

class CurrentSymptoms(BaseModel):
    stress_level: int  # 1-10
    anxiety_frequency: str
    burnout_feelings: str
    trouble_concentrating: bool
    sleep_disturbances: List[str]
    physical_symptoms: List[str]

class WorkStudyContext(BaseModel):
    work_type: str
    work_pressure: str
    deadlines_frequency: str
    support_system: bool

class ConsentPrivacy(BaseModel):
    agree_terms: bool
    consent_sensitive_data: bool
    receive_notifications: bool

class AdvancedData(BaseModel):
    wearable_data: Optional[dict] = None
    cognitive_tests: Optional[dict] = None
    voice_analysis: Optional[dict] = None
    journaling_inputs: Optional[List[str]] = None

class User(BaseModel):
    personal_info: PersonalInfo
    lifestyle_info: LifestyleInfo
    mental_health_history: MentalHealthHistory
    current_symptoms: CurrentSymptoms
    work_study_context: WorkStudyContext
    consent_privacy: ConsentPrivacy
    advanced_data: Optional[AdvancedData] = None
