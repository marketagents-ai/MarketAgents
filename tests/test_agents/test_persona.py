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
        mock_choice.side_effect = [
            "Male", "Buyer", "Engineer", "Expert", "Aggressive",
            "Bachelor's", "High", "Urban", "Moderate", "Medium",
            "Rational", "Reading", "Sports", "Cooking", "Happy",
            "Stocks", "Bonds", "Real Estate"
        ]
        mock_randint.return_value = 30
        mock_get_full_name.return_value = "John Doe"
        mock_random.return_value = 0.5

        # Mock the template file
        mock_template = """
        Personal Background:
        {name} is a {gender} who participates in the market. They are {age} years old and work as a {occupation}. Their investment experience is {investment_experience} and they have a {risk_appetite} risk appetite.
        """
        
        with patch("builtins.open", mock_open(read_data=mock_template)):
            persona = generate_persona()

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
        persona = Persona(name="Test Person", role="Buyer", persona="Test persona description", objectives=["Objective 1", "Objective 2"])
        output_dir = Path("./test_output")
        
        with patch('market_agents.agents.personas.persona.open', mock_open()) as mock_file:
            save_persona_to_file(persona, output_dir)

        mock_file.assert_called_once_with(output_dir / "Test_Person.yaml", "w")
        mock_yaml_dump.assert_called_once_with(persona.model_dump(), mock_file())

if __name__ == '__main__':
    unittest.main()
