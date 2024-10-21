import yaml
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import random
from pathlib import Path
import names

class Persona(BaseModel):
    name: str
    role: str
    persona: str
    objectives: List[str]

class AttributeOptions:
    def __init__(self, yaml_file: str):
        with open(yaml_file, 'r') as file:
            self.options = yaml.safe_load(file)

    def get_random_option(self, attribute: str, persona_data: Dict[str, Any]) -> Any:
        attr_config = self.options[attribute]
        if 'options' in attr_config:
            options = attr_config['options']
            valid_options = self.filter_options(attribute, options, persona_data)
            if not valid_options:
                # If no valid options, return a random option without filtering
                return random.choice(options)
            if isinstance(valid_options[0], dict):
                values = [opt['value'] for opt in valid_options]
                weights = [opt.get('distribution', 1) for opt in valid_options]
                return random.choices(values, weights=weights)[0]
            return random.choice(valid_options)
        elif 'range' in attr_config:
            min_val, max_val = map(float, attr_config['range'].split('-'))
            if attribute in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                return self.generate_personality_trait(min_val, max_val)
            return self.generate_ranged_value(attribute, min_val, max_val, persona_data)
        return None

    def filter_options(self, attribute: str, options: List[Any], persona_data: Dict[str, Any]) -> List[Any]:
        if attribute == 'occupation':
            valid_occupations = [opt for opt in options if self.is_valid_occupation(opt, persona_data)]
            if not valid_occupations:
                # If no valid occupations, return all options
                return options
            return valid_occupations
        return options

    def is_valid_occupation(self, occupation: Dict[str, Any], persona_data: Dict[str, Any]) -> bool:
        age = persona_data.get('age', 0)
        education = persona_data.get('education_level', '')
        income = persona_data.get('income', 0)

        age_valid = age >= occupation.get('min_age', 0)
        education_valid = education in occupation.get('valid_education', [])
        income_valid = occupation['valid_income_range'][0] <= income <= occupation['valid_income_range'][1]

        # Return True if at least two out of three conditions are met
        return sum([age_valid, education_valid, income_valid]) >= 2

    def generate_personality_trait(self, min_val: float, max_val: float) -> float:
        mean = (min_val + max_val) / 2
        std_dev = (max_val - min_val) / 6
        value = random.gauss(mean, std_dev)
        return round(max(min_val, min(max_val, value)), 2)

    def generate_ranged_value(self, attribute: str, min_val: float, max_val: float, persona_data: Dict[str, Any]) -> Union[int, float]:
        if attribute == 'age':
            return int(min_val + (max_val - min_val) * (random.random() ** 1.5))
        return random.randint(int(min_val), int(max_val))

class AttributeRelationships:
    def __init__(self, yaml_file: str):
        with open(yaml_file, 'r') as file:
            self.relationships = yaml.safe_load(file)

    def get_weighted_value(self, primary_attr: str, primary_value: Any, secondary_attr: str, persona_data: Dict[str, Any]) -> List[Union[str, float]]:
        if primary_attr in self.relationships and 'relationships' in self.relationships[primary_attr]:
            for relation in self.relationships[primary_attr]['relationships']:
                if relation['secondary_attribute'] == secondary_attr:
                    if self.check_conditions(relation.get('conditions', []), persona_data):
                        weight = relation['weight']
                        if 'value' in relation:
                            return [relation['value'], abs(weight)]
                        return [primary_value, weight]
        return [primary_value, 1.0]

    def check_conditions(self, conditions: List[str], persona_data: Dict[str, Any]) -> bool:
        return all(self.check_condition(cond, persona_data) for cond in conditions)

    def check_condition(self, condition: str, persona_data: Dict[str, Any]) -> bool:
        operators = ['>=', '<=', '>', '<', '==']
        for op in operators:
            if op in condition:
                attr, value = condition.split(op)
                attr, value = attr.strip(), value.strip()
                attr_value = persona_data.get(attr)
                if attr_value is None:
                    return False
                return eval(f"{float(attr_value)} {op} {float(value)}")
        return True

class PersonaGenerator:
    def __init__(self, relationships: AttributeRelationships, options: AttributeOptions):
        self.relationships = relationships
        self.options = options
        self.attributes = list(options.options.keys())

    def generate_persona(self) -> Persona:
        persona_data = {}
        
        # Generate attributes in a specific order to maintain consistency
        attribute_order = ['age', 'gender', 'education_level', 'occupation', 'income'] + [attr for attr in self.attributes if attr not in ['age', 'gender', 'education_level', 'occupation', 'income']]
        
        for attribute in attribute_order:
            if attribute in ['hobbies_and_interests', 'life_events', 'short_term_goals', 'long_term_goals', 'investment_preferences']:
                persona_data[attribute] = [self.generate_attribute(attribute, persona_data) for _ in range(min(3, len(self.options.options[attribute].get('options', []))))]
            else:
                persona_data[attribute] = self.generate_attribute(attribute, persona_data)

        name = names.get_full_name(gender=persona_data['gender'].lower())
        role = persona_data['role']

        persona = self.format_persona(persona_data, name)
        
        objectives = [
            f"{'Purchase' if role == 'Buyer' else 'Sell'} goods at favorable prices",
            f"Your goal is to {'maximize utility' if role == 'Buyer' else 'maximize profits'}"
        ]
        
        return Persona(name=name, role=role, persona=persona, objectives=objectives)

    def generate_attribute(self, attribute: str, persona_data: Dict[str, Any]) -> Any:
        value = self.options.get_random_option(attribute, persona_data)
        if value is None:
            return None  # Or some default value

        if attribute in self.relationships.relationships and 'relationships' in self.relationships.relationships[attribute]:
            for relation in self.relationships.relationships[attribute]['relationships']:
                secondary_attr = relation['secondary_attribute']
                weighted_value, weight = self.relationships.get_weighted_value(
                    attribute, value, secondary_attr, persona_data
                )
                if random.random() < abs(weight):
                    if isinstance(weighted_value, str):
                        value = weighted_value
                    elif weight < 0:
                        # Inverse relationship
                        options = self.options.options[attribute].get('options', [])
                        if options:
                            opposite_index = (options.index(weighted_value) + len(options) // 2) % len(options)
                            value = options[opposite_index]
                        elif isinstance(weighted_value, (int, float)):
                            attr_config = self.options.options[attribute]
                            if 'range' in attr_config:
                                min_val, max_val = map(float, attr_config['range'].split('-'))
                                value = min_val + max_val - weighted_value
                    else:
                        value = weighted_value

        # Ensure integer values for non-personality traits
        if isinstance(value, float) and attribute not in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            value = int(round(value))
        elif isinstance(value, float):
            value = round(value, 2)

        return value

    def format_persona(self, persona_data: Dict[str, Any], name: str) -> str:
        with open('config/03/persona_template.yaml', 'r', encoding='utf-8') as file:
            template = file.read()
        return template.format(name=name, **persona_data)

def str_presenter(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)

def save_persona_to_file(persona: Persona, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{persona.name.replace(' ', '_')}.yaml", "w", encoding='utf-8') as f:
        yaml.dump(
            persona.dict(),
            f,
            default_flow_style=False,
            allow_unicode=True
        )

def generate_and_save_personas(num_personas: int, output_dir: Path, generator: PersonaGenerator):
    for _ in range(num_personas):
        persona = generator.generate_persona()
        save_persona_to_file(persona, output_dir)

if __name__ == "__main__":
    relationships = AttributeRelationships('config/03/attribute_relationships.yaml')
    options = AttributeOptions('config/03/attribute_options.yaml')
    generator = PersonaGenerator(relationships, options)
    output_dir = Path("output")
    num_personas = 100
    generate_and_save_personas(num_personas, output_dir, generator)
    print(f"Generated {num_personas} personas in {output_dir}")