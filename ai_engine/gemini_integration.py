"""Google Gemini AI Integration for iBaby Pregnancy Nutrition App."""
try:
    import google.generativeai as genai
except ImportError:
    genai = None
import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GeminiNutritionAI:
    """AI-powered nutrition and meal planning using Google Gemini (free tier)."""
    
    def __init__(self):
        """Initialize Gemini AI with API key from environment."""
        if genai is None:
            print("⚠️  WARNING: google-generativeai package is not installed")
            print("   Gemini features are disabled. Install with: pip install google-generativeai")
            self.model = None
            self.available = False
            return

        # Get API key from environment variable
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            print("⚠️  WARNING: No GEMINI_API_KEY or GOOGLE_API_KEY found in environment")
            print("   AI-powered features will be disabled. Set GEMINI_API_KEY to enable.")
            self.model = None
            self.available = False
            return
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.available = True
            print("✓ Gemini AI initialized successfully")
        except Exception as e:
            print(f"⚠️  Failed to initialize Gemini AI: {e}")
            self.model = None
            self.available = False
    
    def generate_meal_plan_ai(
        self, 
        region: str, 
        diet_type: str, 
        trimester: int, 
        days: int, 
        meal_frequency: str
    ) -> Optional[Dict]:
        """
        Generate AI-powered meal plan.
        
        Args:
            region: Regional cuisine preference (e.g., 'North', 'South')
            diet_type: Diet type ('veg', 'nonveg', 'vegan')
            trimester: Pregnancy trimester (1-3)
            days: Number of days for meal plan
            meal_frequency: '3meals' or '5meals'
            
        Returns:
            Dictionary with meal plan structure or None if unavailable
        """
        if not self.available or not self.model:
            return None
        
        try:
            # Determine meals per day
            meals_list = ['breakfast', 'lunch', 'dinner']
            if meal_frequency == '5meals':
                meals_list = ['breakfast', 'mid_morning_snack', 'lunch', 'evening_snack', 'dinner']
            
            prompt = f"""
Create a detailed {days}-day pregnancy meal plan with the following specifications:
- Region: {region} Indian cuisine
- Diet: {diet_type}
- Trimester: {trimester}
- Meals per day: {len(meals_list)} ({', '.join(meals_list)})

For each day, provide:
1. Specific dish names with portion sizes for each meal
2. Nutrition per meal (calories, protein, carbs, fat, iron, calcium, folic_acid)
3. Total daily nutrition summary

Requirements:
- Use traditional {region} Indian dishes appropriate for pregnancy
- Ensure meals are rich in nutrients needed during trimester {trimester}
- Include variety across days
- Consider {diet_type} dietary restrictions

Format as JSON with this exact structure:
{{
  "days": [
    {{
      "day": 1,
      "meals": {{
        "breakfast": {{
          "dish": "Dish name",
          "portion": "Portion size",
          "nutrition": {{"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "iron": 0, "calcium": 0, "folic_acid": 0}}
        }},
        "lunch": {{"dish": "", "portion": "", "nutrition": {{}}}},
        "dinner": {{"dish": "", "portion": "", "nutrition": {{}}}}
      }},
      "daily_total": {{"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "iron": 0, "calcium": 0, "folic_acid": 0}}
    }}
  ]
}}

Return ONLY valid JSON, no additional text.
"""
            
            response = self.model.generate_content(prompt)
            return self._parse_meal_plan(response.text)
            
        except Exception as e:
            print(f"Error generating AI meal plan: {e}")
            return None
    
    def enhance_chatbot_response(self, question: str, context: Dict) -> Optional[str]:
        """
        Generate accurate AI chatbot responses for pregnancy nutrition questions.
        
        Args:
            question: User's question
            context: Context dictionary with 'trimester', 'region', etc.
            
        Returns:
            AI-generated response or None if unavailable
        """
        if not self.available or not self.model:
            return None
        
        try:
            trimester = context.get('trimester', 'all trimesters')
            region = context.get('region', 'general')
            diet_type = context.get('diet_type', 'general')
            
            prompt = f"""
You are a pregnancy nutrition expert specializing in Indian cuisine. Answer this question accurately:

Question: {question}

Context:
- Trimester: {trimester}
- Region: {region} Indian cuisine
- Diet: {diet_type}

Provide:
1. Direct answer to the question
2. Nutritional benefits
3. Safety considerations during pregnancy
4. Specific recommendations for trimester {trimester}

Keep response under 300 words. Use bullet points for clarity.
Be evidence-based and culturally appropriate for Indian dietary practices.
"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"Error generating AI chatbot response: {e}")
            return None
    
    def calculate_advanced_nutrition(self, meals: List[Dict]) -> Optional[Dict]:
        """
        AI-powered nutrition calculation for complex meal combinations.
        
        Args:
            meals: List of meal dictionaries
            
        Returns:
            Dictionary with detailed nutrition breakdown or None
        """
        if not self.available or not self.model:
            return None
        
        try:
            meals_str = json.dumps(meals, indent=2)
            
            prompt = f"""
Calculate detailed nutrition for these pregnancy meals:
{meals_str}

Provide comprehensive breakdown:
- Macros: calories, protein, carbs, fat, fiber
- Minerals: iron, calcium, zinc, magnesium
- Vitamins: vitamin_a, vitamin_b6, vitamin_b12, vitamin_c, vitamin_d, folic_acid, omega3

Return as JSON:
{{
  "calories": 0,
  "protein": 0,
  "carbs": 0,
  "fat": 0,
  "fiber": 0,
  "iron": 0,
  "calcium": 0,
  "folic_acid": 0,
  "vitamin_a": 0,
  "vitamin_b6": 0,
  "vitamin_b12": 0,
  "vitamin_c": 0,
  "vitamin_d": 0,
  "zinc": 0,
  "magnesium": 0,
  "omega3": 0
}}

Return ONLY valid JSON.
"""
            
            response = self.model.generate_content(prompt)
            return self._parse_nutrition(response.text)
            
        except Exception as e:
            print(f"Error calculating AI nutrition: {e}")
            return None
    
    def _parse_meal_plan(self, response_text: str) -> Optional[Dict]:
        """Parse AI response into meal plan structure."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            text = response_text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            data = json.loads(text)
            return data
            
        except Exception as e:
            print(f"Error parsing AI meal plan response: {e}")
            return None
    
    def _parse_nutrition(self, response_text: str) -> Optional[Dict]:
        """Parse AI response into nutrition data."""
        try:
            # Extract JSON from response
            text = response_text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            data = json.loads(text)
            return data
            
        except Exception as e:
            print(f"Error parsing AI nutrition response: {e}")
            return None
