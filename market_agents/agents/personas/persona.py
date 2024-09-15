from pydantic import BaseModel, Field
from typing import List, Optional
import yaml
import random
from pathlib import Path
import names

class Persona(BaseModel):
    name: str
    role: str
    persona: str
    objectives: List[str]

def generate_persona() -> Persona:
    gender = random.choice(["Male", "Female", "Non-binary"])
    name = names.get_full_name(gender=gender.lower())
    role = random.choice(["Buyer", "Seller"])
    age = random.randint(18, 100)
    occupation = random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist'])
    investment_experience = random.choice(['Novice', 'Intermediate', 'Expert'])
    risk_appetite = random.choice(['Conservative', 'Moderate', 'Aggressive'])
    
    demographic_characteristics = {
        "age": age,
        "gender": gender,
        "education_level": random.choice(["High School", "Bachelor's", "Master's", "PhD"]),
        "occupation": occupation,
        "income_bracket": random.choice(["Low", "Medium", "High"]),
        "geographic_location": random.choice(["Urban", "Suburban", "Rural"])
    }
    
    economic_attributes = {
        "spending_habits": random.choice(["Frugal", "Moderate", "Lavish"]),
        "saving_preferences": random.choice(["Low", "Medium", "High"]),
        "risk_tolerance": round(random.random(), 2),
        "investment_experience": investment_experience
    }
    
    personality_traits = {
        "decision_making_style": random.choice(["Rational", "Emotional", "Impulsive", "Collaborative"]),
        "openness": round(random.random(), 2),
        "conscientiousness": round(random.random(), 2),
        "extraversion": round(random.random(), 2),
        "agreeableness": round(random.random(), 2),
        "neuroticism": round(random.random(), 2)
    }
    
    hobbies_and_interests = [random.choice(["Reading", "Sports", "Cooking", "Travel", "Music"]) for _ in range(3)]
    
    dynamic_attributes = {
        "current_mood": random.choice(["Happy", "Sad", "Neutral", "Excited"]),
        "recent_life_events": ["Got a promotion", "Moved to a new city"]
    }
    
    financial_objectives = {
        "short_term_goals": ["Build emergency fund", "Pay off credit card debt"],
        "long_term_goals": ["Save for retirement", "Buy a house"],
        "risk_appetite": risk_appetite,
        "investment_preferences": ["Stocks", "Bonds", "Real Estate"]
    }
    
    with open('./market_agents/agents/personas/persona_template.yaml', 'r') as file:
        template = file.read()
    
    persona = template.format(
        name=name,
        **demographic_characteristics,
        **economic_attributes,
        **personality_traits,
        hobbies_and_interests=", ".join(hobbies_and_interests),
        **dynamic_attributes,
        **financial_objectives
    )
    
    objectives = [
        f"{'Purchase' if role == 'Buyer' else 'Sell'} goods at favorable prices",
        f"Your goal is to {'maximize utility' if role == 'Buyer' else 'maximize profits'}"
    ]

    return Persona(
        name=name,
        role=role,
        persona=persona,
        objectives=objectives
    )

def save_persona_to_file(persona: Persona, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{persona.name.replace(' ', '_')}.yaml", "w") as f:
        yaml.dump(persona.model_dump(), f)

def generate_and_save_personas(num_personas: int, output_dir: Path):
    for _ in range(num_personas):
        persona = generate_persona()
        save_persona_to_file(persona, output_dir)

if __name__ == "__main__":
    output_dir = Path("./market_agents/agents/personas/generated_personas")
    generate_and_save_personas(10, output_dir)
    print(f"Generated 10 personas in {output_dir}")
