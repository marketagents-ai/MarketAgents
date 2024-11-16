import unittest
from unittest.mock import patch, mock_open
import random
from pathlib import Path
from market_agents.agents.personas.persona import generate_persona, save_persona_to_file, Persona

class TestPersona(unittest.TestCase):
    @patch('market_agents.agents.personas.persona.random.choice')
    @patch('market_agents.agents.personas.persona.random.randint')
    @patch('market_agents.agents.personas.persona.names.get_full_name')
    @patch('market_agents.agents.personas.persona.random.random')
    def test_generate_persona(self, mock_random, mock_get_full_name, mock_randint, mock_choice):
        # Mock random choices and name generation
        mock_get_full_name.return_value = "John Doe"
        mock_choice.side_effect = [
            "Male",              # gender
            "Buyer",             # role
            "Engineer",          # occupation
            "Bachelor's",        # education_level
            "High",              # income_bracket
            "Expert",            # investment_experience
            "Aggressive",        # risk_appetite
            "Urban",             # geographic_location
            "Moderate",          # spending_habits
            "High",              # saving_preferences
            "Rational",          # decision_making_style
            "Happy",            # current_mood
        ]
        mock_randint.return_value = 30
        mock_random.return_value = 0.5

        # Mock the template file
        mock_template = """
        Personal Background:
        {name} is a {gender} who participates in the market. They are {age} years old and work as a {occupation}. 
        Their investment experience is {investment_experience} and they have a {risk_appetite} risk appetite.
        """
        
        with patch("builtins.open", mock_open(read_data=mock_template)):
            persona = generate_persona()

        # Test assertions
        self.assertEqual(persona.name, "John Doe")
        self.assertEqual(persona.role, "Buyer")
        self.assertIn("John Doe is a Male who participates in the market.", persona.persona)
        self.assertIn("They are 30 years old and work as a Engineer.", persona.persona)
        self.assertIn("Their investment experience is Expert", persona.persona)
        self.assertIn("they have a Aggressive risk appetite.", persona.persona)
        self.assertIn("Purchase goods at favorable prices", persona.objectives)
        self.assertIn("Your goal is to maximize utility", persona.objectives)

    @patch('market_agents.agents.personas.persona.yaml.dump')
    def test_save_persona_to_file(self, mock_yaml_dump):
        persona = Persona(
            name="Test Person", 
            role="Buyer", 
            persona="Test persona description", 
            objectives=["Objective 1", "Objective 2"],
            trader_type=["test_trader"]  # Changed to a list containing the trader type
        )
        output_dir = Path("./test_output")
if __name__ == '__main__':
    unittest.main()
