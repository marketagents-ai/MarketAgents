import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from persona_weighted import PersonaGenerator, AttributeRelationships, AttributeOptions, Persona

class TestPersonaGenerator(unittest.TestCase):

    @patch('persona_weighted.AttributeRelationships')
    @patch('persona_weighted.AttributeOptions')
    def setUp(self, mock_options, mock_relationships):
        self.mock_options = mock_options.return_value
        self.mock_relationships = mock_relationships.return_value
        self.generator = PersonaGenerator(self.mock_relationships, self.mock_options)

    def test_generate_persona(self):
        # Mock the necessary methods and attributes
        self.mock_options.options = {
            'gender': {'options': ['Male', 'Female']},
            'role': {'options': ['Buyer', 'Seller']},
            'age': {'range': '18-80'},
            'hobbies_and_interests': {'options': ['Reading', 'Sports', 'Cooking']},
            'life_events': {'options': ['Graduation', 'Marriage', 'Career change']},
            'short_term_goals': {'options': ['Save money', 'Learn a skill', 'Travel']},
            'long_term_goals': {'options': ['Buy a house', 'Start a business', 'Retire early']},
            'investment_preferences': {'options': ['Stocks', 'Real estate', 'Bonds']}
        }
        self.mock_options.get_random_option.side_effect = [
            'Male',  # gender
            'Buyer',  # role
            30,  # age
            'Reading', 'Sports', 'Cooking',  # hobbies_and_interests (3 times)
            'Graduation', 'Marriage', 'Career change',  # life_events (3 times)
            'Save money', 'Learn a skill', 'Travel',  # short_term_goals (3 times)
            'Buy a house', 'Start a business', 'Retire early',  # long_term_goals (3 times)
            'Stocks', 'Real estate', 'Bonds'  # investment_preferences (3 times)
        ]
        self.mock_relationships.relationships = {}
        
        # Mock the attributes list to include 'gender'
        self.generator.attributes = ['gender'] + [attr for attr in self.mock_options.options.keys() if attr != 'gender']
        
        # Instead, mock the format_persona method to return a string
        self.generator.format_persona = MagicMock(return_value="Mocked formatted persona")
        
        with patch('persona_weighted.names.get_full_name', return_value='John Doe'):
            with patch('persona_weighted.open', MagicMock()):
                persona = self.generator.generate_persona()

        self.assertIsInstance(persona, Persona)
        self.assertEqual(persona.name, 'John Doe')
        self.assertEqual(persona.role, 'Buyer')
        self.assertEqual(persona.persona, "Mocked formatted persona")
        self.assertIn('Purchase goods at favorable prices', persona.objectives)
        self.assertIn('Your goal is to maximize utility', persona.objectives)

    def test_generate_attribute(self):
        self.mock_options.get_random_option.return_value = 'TestValue'
        self.mock_relationships.relationships = {}

        result = self.generator.generate_attribute('test_attribute', {})
        self.assertEqual(result, 'TestValue')

    @patch('persona_weighted.open')
    def test_format_persona(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = "Name: {name}, Age: {age}"
        persona_data = {'age': 30}
        result = self.generator.format_persona(persona_data, 'John Doe')
        self.assertEqual(result, "Name: John Doe, Age: 30")

if __name__ == '__main__':
    unittest.main()