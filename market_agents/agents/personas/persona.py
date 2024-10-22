from pydantic import BaseModel, Field
from typing import List, Dict, Any
import yaml
import random
from pathlib import Path
import names
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaSchema(BaseModel):
    demographic_characteristics: Dict[str, Any] = Field(..., description="Demographic information about the persona")
    economic_attributes: Dict[str, Any] = Field(..., description="Economic attributes of the persona")
    personality_traits: Dict[str, Any] = Field(..., description="Personality traits of the persona")
    hobbies_and_interests: List[str] = Field(..., description="Hobbies and interests of the persona")
    dynamic_attributes: Dict[str, Any] = Field(..., description="Dynamic attributes of the persona")
    financial_objectives: Dict[str, Any] = Field(..., description="Financial objectives of the persona")

class Persona(BaseModel):
    name: str
    role: str
    persona: str
    objectives: List[str]
    trader_type: List[str]
    schema: PersonaSchema

    def log_persona(self):
        logger.info(f"Persona created: {self.name}")
        logger.debug(f"Persona details: {self.model_dump_json(indent=2)}")

def generate_persona() -> Persona:
    gender = random.choice(["Male", "Female", "Non-binary"])
    name = names.get_full_name(gender=gender.lower())
    role = random.choice(["Buyer", "Seller"])
    
    occupation_data = {
        'Doctor': {
            'education_levels': ["Master's", "PhD"],
            'income_brackets': ["High"]
        },
        'Engineer': {
            'education_levels': ["Bachelor's", "Master's", "PhD"],
            'income_brackets': ["Medium", "High"]
        },
        'Teacher': {
            'education_levels': ["Bachelor's", "Master's"],
            'income_brackets': ["Low", "Medium"]
        },
        'Artist': {
            'education_levels': ["High School", "Bachelor's"],
            'income_brackets': ["Low", "Medium"]
        },
        'Mechanic': {
            'education_levels': ["High School", "Associate's"],
            'income_brackets': ["Low", "Medium"]
        },
        'Scientist': {
            'education_levels': ["Master's", "PhD"],
            'income_brackets': ["Medium", "High"]
        },
        'Nurse': {
            'education_levels': ["Associate's", "Bachelor's"],
            'income_brackets': ["Low", "Medium"]
        },
        'Lawyer': {
            'education_levels': ["Master's", "PhD"],
            'income_brackets': ["High"]
        },
        'Salesperson': {
            'education_levels': ["High School", "Associate's", "Bachelor's"],
            'income_brackets': ["Low", "Medium"]
        },
        'Entrepreneur': {
            'education_levels': ["High School", "Bachelor's", "Master's"],
            'income_brackets': ["Medium", "High"]
        }
    }
    
    occupation = random.choice(list(occupation_data.keys()))
    occupation_info = occupation_data[occupation]
    education_level = random.choice(occupation_info['education_levels'])
    income_bracket = random.choice(occupation_info['income_brackets'])
    
    def get_min_age_for_education(education_level):
        if education_level == "High School":
            return 18
        elif education_level == "Associate's":
            return 20
        elif education_level == "Bachelor's":
            return 22
        elif education_level == "Master's":
            return 24
        elif education_level == "PhD":
            return 27
        else:
            return 18

    min_age = get_min_age_for_education(education_level)
    age = random.randint(min_age, 100)
    
    investment_experience = random.choice(['Novice', 'Intermediate', 'Expert'])
    risk_appetite = random.choice(['Conservative', 'Moderate', 'Aggressive'])
    
    demographic_characteristics = {
        "age": age,
        "gender": gender,
        "education_level": education_level,
        "occupation": occupation,
        "income_bracket": income_bracket,
        "geographic_location": random.choice(["Urban", "Suburban", "Rural"])
    }
    
    economic_attributes = {
        "spending_habits": random.choice(["Frugal", "Moderate", "Lavish"]),
        "saving_preferences": random.choice(["Low", "Medium", "High"]),
        "risk_tolerance": round(random.uniform(0.0, 1.0), 2),
        "investment_experience": investment_experience
    }
    
    personality_traits = {
        "decision_making_style": random.choice(["Rational", "Emotional", "Impulsive", "Collaborative"]),
        "openness": round(random.uniform(0.0, 1.0), 2),
        "conscientiousness": round(random.uniform(0.0, 1.0), 2),
        "extraversion": round(random.uniform(0.0, 1.0), 2),
        "agreeableness": round(random.uniform(0.0, 1.0), 2),
        "neuroticism": round(random.uniform(0.0, 1.0), 2)
    }
    
    hobbies_list = ["Reading", "Sports", "Cooking", "Travel", "Music", "Art", "Gardening", "Photography", "Technology"]
    hobbies_and_interests = random.sample(hobbies_list, k=3)
    hobbies_and_interests_str = ", ".join(hobbies_and_interests)
    
    recent_life_events_list = random.sample(
        ["Got a promotion", "Moved to a new city", "Started a new hobby", "Graduated", "Retired"],
        k=2
    )
    dynamic_attributes = {
        "current_mood": random.choice(["Happy", "Sad", "Neutral", "Excited"]),
        "recent_life_events": recent_life_events_list
    }
    recent_life_events_str = ", ".join(dynamic_attributes["recent_life_events"])
    
    short_term_goals_list = random.sample(
        ["Build emergency fund", "Pay off credit card debt", "Save for vacation"],
        k=2
    )
    long_term_goals_list = random.sample(
        ["Save for retirement", "Buy a house", "Start a business"],
        k=2
    )
    investment_preferences_list = random.sample(
        ["Stocks", "Bonds", "Real Estate", "Cryptocurrency", "Commodities"],
        k=3
    )
    financial_objectives = {
        "short_term_goals": short_term_goals_list,
        "long_term_goals": long_term_goals_list,
        "risk_appetite": risk_appetite,
        "investment_preferences": investment_preferences_list
    }
    short_term_goals_str = ", ".join(financial_objectives["short_term_goals"])
    long_term_goals_str = ", ".join(financial_objectives["long_term_goals"])
    investment_preferences_str = ", ".join(financial_objectives["investment_preferences"])
    
    with open('./market_agents/agents/personas/persona_template.yaml', 'r') as file:
        template = file.read()
    
    persona_description = template.format(
        name=name,
        age=age,
        gender=gender,
        education_level=education_level,
        occupation=occupation,
        income_bracket=income_bracket,
        geographic_location=demographic_characteristics["geographic_location"],
        spending_habits=economic_attributes["spending_habits"],
        saving_preferences=economic_attributes["saving_preferences"],
        risk_tolerance=economic_attributes["risk_tolerance"],
        investment_experience=investment_experience,
        decision_making_style=personality_traits["decision_making_style"],
        openness=personality_traits["openness"],
        conscientiousness=personality_traits["conscientiousness"],
        extraversion=personality_traits["extraversion"],
        agreeableness=personality_traits["agreeableness"],
        neuroticism=personality_traits["neuroticism"],
        hobbies_and_interests=hobbies_and_interests_str,
        current_mood=dynamic_attributes["current_mood"],
        recent_life_events=recent_life_events_str,
        short_term_goals=short_term_goals_str,
        long_term_goals=long_term_goals_str,
        risk_appetite=risk_appetite,
        investment_preferences=investment_preferences_str
    )
    
    objectives = [
        f"{'Purchase' if role == 'Buyer' else 'Sell'} goods at favorable prices",
        f"Your goal is to {'maximize utility' if role == 'Buyer' else 'maximize profits'}"
    ]

    trader_type = [investment_experience, risk_appetite, personality_traits["decision_making_style"]]

    schema = PersonaSchema(
        demographic_characteristics=demographic_characteristics,
        economic_attributes=economic_attributes,
        personality_traits=personality_traits,
        hobbies_and_interests=hobbies_and_interests,
        dynamic_attributes=dynamic_attributes,
        financial_objectives=financial_objectives
    )

    persona = Persona(
        name=name,
        role=role,
        persona=persona_description,
        objectives=objectives,
        trader_type=trader_type,
        schema=schema
    )

    persona.log_persona()

    return persona

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