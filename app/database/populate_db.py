from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import date

uri = "mongodb+srv://admin:test1234@devhack.6gqwt4w.mongodb.net/?retryWrites=true&w=majority&appName=DevHack"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["pieceofmind"]

# Collections based on model.py
users = db["users"]

# Sample user document
sample_user = {
    "personal_info": {
        "full_name": "Jane Doe",
        "email": "jane.doe@example.com",
        "phone_number": "1234567890",
        "password": "hashedpassword",
        "date_of_birth": date(1995, 5, 20).isoformat(),
        "gender": "Female"
    },
    "lifestyle_info": {
        "occupation": "Student",
        "work_hours": 6,
        "work_mode": "Remote",
        "physical_activity": "Moderate",
        "diet_habits": "Vegetarian",
        "caffeine_intake": "Low",
        "smoking_status": False,
        "sleep_hours": 7.5,
        "water_intake": 2.0
    },
    "mental_health_history": {
        "prior_conditions": ["Anxiety"],
        "current_medications": ["None"],
        "therapy_sessions": "None"
    },
    "current_symptoms": {
        "stress_level": 5,
        "anxiety_frequency": "Occasional",
        "burnout_feelings": "Rare",
        "trouble_concentrating": False,
        "sleep_disturbances": ["None"],
        "physical_symptoms": ["Headache"]
    },
    "work_study_context": {
        "work_type": "Study",
        "work_pressure": "Medium",
        "deadlines_frequency": "Monthly",
        "support_system": True
    },
    "consent_privacy": {
        "agree_terms": True,
        "consent_sensitive_data": True,
        "receive_notifications": True
    },
    "advanced_data": {
        "wearable_data": {"steps": 5000},
        "cognitive_tests": {"memory_score": 85},
        "voice_analysis": {"mood": "Neutral"},
        "journaling_inputs": ["Feeling good today."]
    }
}

# Insert sample user
def insert_sample_user():
    result = users.insert_one(sample_user)
    print(f"Inserted user ID: {result.inserted_id}")

# Retrieve and print all users
def print_all_users():
    for user in users.find():
        print(user)

if __name__ == "__main__":
    insert_sample_user()
    print_all_users()
