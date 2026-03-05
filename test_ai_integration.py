"""Tests for AI Integration Features."""
import unittest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_engine.gemini_integration import GeminiNutritionAI
from ai_engine.meal_planner import MealPlanner
from ai_engine.comprehensive_chatbot import ComprehensiveChatbot


class TestGeminiIntegration(unittest.TestCase):
    """Test Gemini AI integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gemini_ai = GeminiNutritionAI()
    
    def test_gemini_initialization(self):
        """Test that Gemini AI initializes correctly."""
        self.assertIsNotNone(self.gemini_ai)
        # It's ok if AI is not available (no API key)
        self.assertIsInstance(self.gemini_ai.available, bool)
    
    def test_gemini_graceful_degradation(self):
        """Test that Gemini handles missing API key gracefully."""
        # If no API key, should still initialize but be unavailable
        if not self.gemini_ai.available:
            # Methods should return None when unavailable
            result = self.gemini_ai.generate_meal_plan_ai('North', 'veg', 1, 7, '3meals')
            self.assertIsNone(result)
            
            result = self.gemini_ai.enhance_chatbot_response('test question', {})
            self.assertIsNone(result)
            
            result = self.gemini_ai.calculate_advanced_nutrition([])
            self.assertIsNone(result)


class TestMealPlannerWithAI(unittest.TestCase):
    """Test meal planner AI integration."""
    
    def test_meal_planner_imports(self):
        """Test that meal planner imports successfully."""
        from ai_engine.meal_planner import MealPlanner
        self.assertIsNotNone(MealPlanner)
    
    def test_meal_planner_initialization(self):
        """Test that meal planner initializes with AI support."""
        # Mock database for testing
        class MockDB:
            pass
        
        planner = MealPlanner(MockDB())
        self.assertIsNotNone(planner)
        self.assertIsNotNone(planner.gemini_ai)
        self.assertIsNotNone(planner.unified_loader)


class TestChatbotWithAI(unittest.TestCase):
    """Test chatbot AI integration."""
    
    def test_chatbot_imports(self):
        """Test that chatbot imports successfully."""
        from ai_engine.comprehensive_chatbot import ComprehensiveChatbot
        self.assertIsNotNone(ComprehensiveChatbot)
    
    def test_chatbot_initialization(self):
        """Test that chatbot initializes with AI support."""
        chatbot = ComprehensiveChatbot()
        self.assertIsNotNone(chatbot)
        self.assertIsNotNone(chatbot.gemini_ai)
        self.assertIsInstance(chatbot.gemini_ai.available, bool)


class TestSeasonRemoval(unittest.TestCase):
    """Test that season parameter has been removed."""
    
    def test_meal_planner_no_season_param(self):
        """Test that meal planner generate_meal_plan doesn't have season parameter."""
        from ai_engine.meal_planner import MealPlanner
        import inspect
        
        sig = inspect.signature(MealPlanner.generate_meal_plan)
        params = list(sig.parameters.keys())
        
        # Ensure season is not in parameters
        self.assertNotIn('season', params)
        
        # Ensure required parameters are still there
        self.assertIn('user', params)
        self.assertIn('region', params)
        self.assertIn('diet_type', params)
        self.assertIn('trimester', params)
    
    def test_nutrition_tracking_enhanced(self):
        """Test that nutrition tracking includes 16+ nutrients."""
        from ai_engine.meal_planner import MealPlanner
        
        class MockDB:
            pass
        
        planner = MealPlanner(MockDB())
        
        # Check _calculate_day_nutrition returns comprehensive nutrients
        test_meals = {
            'breakfast': {
                'calories': 400,
                'protein': 15,
                'iron': 5,
                'calcium': 200
            }
        }
        
        nutrition = planner._calculate_day_nutrition(test_meals)
        
        # Check that it includes all 16 nutrients
        expected_nutrients = [
            'calories', 'protein', 'carbs', 'fat', 'fiber',
            'iron', 'calcium', 'folic_acid',
            'vitamin_a', 'vitamin_b6', 'vitamin_b12',
            'vitamin_c', 'vitamin_d', 'zinc', 'magnesium', 'omega3'
        ]
        
        for nutrient in expected_nutrients:
            self.assertIn(nutrient, nutrition)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
