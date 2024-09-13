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

    def get_random_option(self, attribute: str) -> Any:
        attr_config = self.options[attribute]
        if 'options' in attr_config:
            options = attr_config['options']
            if isinstance(options[0], dict):
                values = [opt['value'] for opt in options]
                weights = [opt.get('distribution', 1) for opt in options]
                return random.choices(values, weights=weights)[0]
            return random.choice(options)
        elif 'range' in attr_config:
            min_val, max_val = map(float, attr_config['range'].split('-'))
            if attribute in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                return round(random.uniform(min_val, max_val), 2)
            return random.randint(int(min_val), int(max_val))
        return None

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
        
        # Ensure 'gender' is processed first
        if 'gender' in self.attributes:
            persona_data['gender'] = self.generate_attribute('gender', persona_data)
        
        for attribute in self.attributes:
            if attribute != 'gender':  # Skip 'gender' as it's already processed
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
        value = self.options.get_random_option(attribute)
        if attribute in self.relationships.relationships and 'relationships' in self.relationships.relationships[attribute]:
            for relation in self.relationships.relationships[attribute]['relationships']:
                secondary_attr = relation['secondary_attribute']
                weighted_value, weight = self.relationships.get_weighted_value(
                    attribute, value, secondary_attr, persona_data
                )
                if random.random() < weight:
                    if isinstance(weighted_value, str):
                        value = weighted_value
                    elif weight < 0.5:
                        # Inverse relationship: choose a value opposite to the weighted_value
                        options = self.options.options[attribute].get('options', [])
                        if options:
                            opposite_index = (options.index(weighted_value) + len(options) // 2) % len(options)
                            value = options[opposite_index]
                        elif isinstance(weighted_value, (int, float)):
                            attr_config = self.options.options[attribute]
                            if 'range' in attr_config:
                                min_val, max_val = map(float, attr_config['range'].split('-'))
                                value = min_val + max_val - weighted_value
                                # Cast value to int if the attribute expects an integer
                                if attribute not in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                                    value = int(round(value))
                    else:
                        value = weighted_value
                        # Cast value to int if the attribute expects an integer
                        if isinstance(value, float) and attribute not in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                            value = int(round(value))
        # Ensure float values are rounded to two decimal places
        if isinstance(value, float):
            value = round(value, 2)
        return value




    def format_persona(self, persona_data: Dict[str, Any], name: str) -> str:
        with open('config/01/persona_template.yaml', 'r', encoding='utf-8') as file:
            template = file.read()
        return template.format(name=name, **persona_data)
# Custom representer for multiline strings
def str_presenter(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)
# Update save_persona_to_file function
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
    relationships = AttributeRelationships('config/01/attribute_relationships.yaml')
    options = AttributeOptions('config/01/attribute_options.yaml')
    generator = PersonaGenerator(relationships, options)
    output_dir = Path("output")
    num_personas = 100
    generate_and_save_personas(num_personas, output_dir, generator)
    print(f"Generated {num_personas} personas in {output_dir}")