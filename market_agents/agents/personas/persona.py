from pydantic import BaseModel
from typing import List, Dict, Union
import yaml
import random
from pathlib import Path
import names

class Persona(BaseModel):
    name: str
    role: str
    persona: str
    objectives: List[str]
    trader_type: List[str]
    communication_style: str
    routines: List[str]
    skills: List[str]

def generate_persona() -> Persona:
    # Gender and Name
    gender = random.choice(["Male", "Female", "Non-binary"])
    name = names.get_full_name(gender=gender.lower())

    # Pronouns with verb forms
    if gender == "Male":
        pronouns = {
            "subject": "he",
            "object": "him",
            "possessive": "his",
            "be": "is",
            "has": "has"
        }
    elif gender == "Female":
        pronouns = {
            "subject": "she",
            "object": "her",
            "possessive": "her",
            "be": "is",
            "has": "has"
        }
    else:  # Non-binary
        pronouns = {
            "subject": "they",
            "object": "them",
            "possessive": "their",
            "be": "are",
            "has": "have"
        }

    # Capitalize pronouns for sentence beginnings
    pronouns_cap = {
        k: v.capitalize() if k not in ["be", "has"] else v
        for k, v in pronouns.items()
    }

    # Role
    role = random.choice(["Buyer", "Seller"])

    # Occupation Mapping with Routines and Skills
    occupation_data = {
        'Doctor': {
            'education_levels': ["Master's", "PhD"],
            'income_brackets': ["High"],
            'routines': ['Review patient charts', 'Perform surgeries', 'Attend medical conferences', 'Consult with patients', 'Supervise medical staff'],
            'skills': ['Medical diagnosis', 'Surgery', 'Patient care', 'Medical research', 'Team management']
        },
        'Engineer': {
            'education_levels': ["Bachelor's", "Master's", "PhD"],
            'income_brackets': ["Medium", "High"],
            'routines': ['Design systems', 'Write code', 'Attend team meetings', 'Debug software', 'Review code'],
            'skills': ['Programming', 'Systems design', 'Problem-solving', 'Software development', 'Code optimization']
        },
        'Teacher': {
            'education_levels': ["Bachelor's", "Master's"],
            'income_brackets': ["Low", "Medium"],
            'routines': ['Prepare lesson plans', 'Teach classes', 'Grade assignments', 'Meet with parents', 'Attend workshops'],
            'skills': ['Instruction', 'Curriculum development', 'Classroom management', 'Communication', 'Assessment']
        },
        'Artist': {
            'education_levels': ["High School", "Bachelor's"],
            'income_brackets': ["Low", "Medium"],
            'routines': ['Create artworks', 'Attend exhibitions', 'Market artwork', 'Collaborate with other artists', 'Research new techniques'],
            'skills': ['Creativity', 'Artistic skills', 'Marketing', 'Networking', 'Critical thinking']
        },
        'Mechanic': {
            'education_levels': ["High School", "Associate's"],
            'income_brackets': ["Low", "Medium"],
            'routines': ['Inspect vehicles', 'Repair engines', 'Order parts', 'Maintain equipment', 'Provide customer service'],
            'skills': ['Mechanical knowledge', 'Problem-solving', 'Technical skills', 'Customer service', 'Attention to detail']
        },
        'Scientist': {
            'education_levels': ["Master's", "PhD"],
            'income_brackets': ["Medium", "High"],
            'routines': ['Conduct experiments', 'Analyze data', 'Publish papers', 'Attend conferences', 'Collaborate with peers'],
            'skills': ['Research', 'Data analysis', 'Scientific writing', 'Critical thinking', 'Collaboration']
        },
        'Nurse': {
            'education_levels': ["Associate's", "Bachelor's"],
            'income_brackets': ["Low", "Medium"],
            'routines': ['Monitor patient health', 'Administer medications', 'Update records', 'Assist doctors', 'Educate patients'],
            'skills': ['Patient care', 'Medical knowledge', 'Compassion', 'Attention to detail', 'Communication']
        },
        'Lawyer': {
            'education_levels': ["Master's", "PhD"],
            'income_brackets': ["High"],
            'routines': ['Meet clients', 'Prepare legal documents', 'Represent clients in court', 'Research case law', 'Negotiate settlements'],
            'skills': ['Legal knowledge', 'Negotiation', 'Analytical thinking', 'Public speaking', 'Writing']
        },
        'Salesperson': {
            'education_levels': ["High School", "Associate's", "Bachelor's"],
            'income_brackets': ["Low", "Medium"],
            'routines': ['Contact potential clients', 'Present products', 'Negotiate deals', 'Follow up with customers', 'Meet sales targets'],
            'skills': ['Communication', 'Persuasion', 'Negotiation', 'Customer service', 'Time management']
        },
        'Entrepreneur': {
            'education_levels': ["High School", "Bachelor's", "Master's"],
            'income_brackets': ["Medium", "High"],
            'routines': ['Develop business strategies', 'Meet with investors', 'Manage team', 'Oversee operations', 'Analyze market trends'],
            'skills': ['Leadership', 'Strategic planning', 'Risk management', 'Networking', 'Financial literacy']
        }
    }

    # Select occupation and corresponding data
    occupation = random.choice(list(occupation_data.keys()))
    occupation_info = occupation_data[occupation]
    education_level = random.choice(occupation_info['education_levels'])
    income_bracket = random.choice(occupation_info['income_brackets'])

    # Determine minimum age based on education level
    def get_min_age_for_education(education_level):
        education_age = {
            "High School": 18,
            "Associate's": 20,
            "Bachelor's": 22,
            "Master's": 24,
            "PhD": 27
        }
        return education_age.get(education_level, 18)

    min_age = get_min_age_for_education(education_level)
    age = random.randint(min_age, 100)

    # Investment Experience and Risk Appetite
    investment_experience = random.choice(['Novice', 'Intermediate', 'Expert'])
    risk_appetite = random.choice(['Conservative', 'Moderate', 'Aggressive'])

    # Demographic Characteristics
    demographic_characteristics = {
        "age": age,
        "gender": gender,
        "education_level": education_level,
        "occupation": occupation,
        "income_bracket": income_bracket,
        "geographic_location": random.choice(["Urban", "Suburban", "Rural"])
    }

    # Economic Attributes
    economic_attributes = {
        "spending_habits": random.choice(["Frugal", "Moderate", "Lavish"]),
        "saving_preferences": random.choice(["Low", "Medium", "High"]),
        "risk_tolerance": round(random.uniform(0.0, 1.0), 2),
        "investment_experience": investment_experience
    }

    # Personality Traits
    personality_traits = {
        "decision_making_style": random.choice(["Rational", "Emotional", "Impulsive", "Collaborative"]),
        "openness": round(random.uniform(0.0, 1.0), 2),
        "conscientiousness": round(random.uniform(0.0, 1.0), 2),
        "extraversion": round(random.uniform(0.0, 1.0), 2),
        "agreeableness": round(random.uniform(0.0, 1.0), 2),
        "neuroticism": round(random.uniform(0.0, 1.0), 2)
    }

    # Hobbies and Interests
    hobbies_list = ["Reading", "Sports", "Cooking", "Travel", "Music", "Art", "Gardening", "Photography", "Technology"]
    hobbies_and_interests = random.sample(hobbies_list, k=3)
    hobbies_and_interests_str = ", ".join(hobbies_and_interests)

    # Dynamic Attributes
    recent_life_events_list = random.sample(
        ["Got a promotion", "Moved to a new city", "Started a new hobby", "Graduated", "Retired"],
        k=2
    )
    dynamic_attributes = {
        "current_mood": random.choice(["Happy", "Sad", "Neutral", "Excited"]),
        "recent_life_events": recent_life_events_list
    }
    recent_life_events_str = ", ".join(dynamic_attributes["recent_life_events"])

    # Financial Objectives
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

    # Routines and Skills
    routines_list = occupation_info['routines']
    skills_list = occupation_info['skills']

    routines = random.sample(routines_list, k=3) if len(routines_list) >= 3 else routines_list
    skills = random.sample(skills_list, k=3) if len(skills_list) >= 3 else skills_list

    routines_str = ", ".join(routines)
    skills_str = ", ".join(skills)

    # Communication Style
    decision_making_to_communication = {
        "Rational": ["Direct", "Formal"],
        "Emotional": ["Persuasive", "Friendly", "Informal"],
        "Impulsive": ["Informal", "Direct"],
        "Collaborative": ["Friendly", "Persuasive"]
    }

    communication_styles = ["Direct", "Persuasive", "Reserved", "Friendly", "Formal", "Informal"]
    decision_making_style = personality_traits["decision_making_style"]
    communication_style_options = decision_making_to_communication.get(decision_making_style, communication_styles)
    communication_style = random.choice(communication_style_options)

    # Read Persona Template as YAML and extract content under 'persona'
    with open('./market_agents/agents/personas/persona_template.yaml', 'r') as file:
        template_yaml = yaml.safe_load(file)
    template_content = template_yaml.get('persona', '')

    # Format Persona Description
    persona_description = template_content.format(
        name=name,
        age=age,
        gender=gender,
        pronoun_subject=pronouns["subject"],
        pronoun_object=pronouns["object"],
        pronoun_possessive=pronouns["possessive"],
        pronoun_be=pronouns["be"],
        has=pronouns["has"],
        pronoun_subject_cap=pronouns_cap["subject"],
        pronoun_object_cap=pronouns_cap["object"],
        pronoun_possessive_cap=pronouns_cap["possessive"],
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
        investment_preferences=investment_preferences_str,
        communication_style=communication_style,
        routines=routines_str,
        skills=skills_str
    )

    # Objectives
    objectives = [
        f"{'Purchase' if role == 'Buyer' else 'Sell'} goods at favorable prices",
        f"Your goal is to {'maximize utility' if role == 'Buyer' else 'maximize profits'}"
    ]

    # Trader Type
    trader_type = [investment_experience, risk_appetite, personality_traits["decision_making_style"]]

    return Persona(
        name=name,
        role=role,
        persona=persona_description,
        objectives=objectives,
        trader_type=trader_type,
        communication_style=communication_style,
        routines=routines,
        skills=skills
    )

def save_persona_to_file(persona: Persona, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the persona dictionary
    persona_dict = {
        'name': persona.name,
        'role': persona.role,
        'persona': persona.persona,
        'objectives': persona.objectives,
        'trader_type': persona.trader_type,
        'communication_style': persona.communication_style,
        'routines': persona.routines,
        'skills': persona.skills
    }

    # Custom YAML dumper to force literal block style for the persona field
    class LiteralDumper(yaml.SafeDumper):
        def represent_scalar(self, tag, value, style=None):
            if tag == 'tag:yaml.org,2002:str' and '\n' in value:
                style = '|'
            return super().represent_scalar(tag, value, style)

    with open(output_dir / f"{persona.name.replace(' ', '_')}.yaml", "w") as f:
        yaml.dump(
            persona_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=1000,
            indent=2,
            Dumper=LiteralDumper
        )

def generate_and_save_personas(num_personas: int, output_dir: Path):
    for _ in range(num_personas):
        persona = generate_persona()
        save_persona_to_file(persona, output_dir)

def load_persona_from_file(filepath: Union[str, Path]) -> Persona:
    """Load a persona from a YAML file."""
    with open(filepath, 'r') as file:
        persona_data = yaml.safe_load(file)
        return Persona(**persona_data)

if __name__ == "__main__":
    output_dir = Path("./market_agents/agents/personas/generated_personas")
    generate_and_save_personas(10, output_dir)
    print(f"Generated 10 personas in {output_dir}")
