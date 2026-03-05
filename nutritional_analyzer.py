"""Nutritional analyzer for food recommendations."""

from ai_engine.nutrition_estimator import NutritionEstimator


class NutritionalAnalyzer:
    """Analyzes nutritional content and matches with user needs."""
    
    def __init__(self):
        """Initialize the nutritional analyzer."""
        self.trimester_requirements = {
            1: {
                'folic_acid': 600,  # mcg
                'vitamin_b6': 1.9,  # mg
                'iron': 27,  # mg
                'calcium': 1000,  # mg
                'protein': 60,  # g
                'calories': 1800  # kcal
            },
            2: {
                'calcium': 1000,  # mg
                'vitamin_d': 600,  # IU
                'omega3': 200,  # mg
                'protein': 70,  # g
                'iron': 27,  # mg
                'calories': 2200  # kcal
            },
            3: {
                'iron': 27,  # mg
                'protein': 75,  # g
                'vitamin_k': 90,  # mcg
                'fiber': 28,  # g
                'calcium': 1000,  # mg
                'calories': 2400  # kcal
            }
        }
    
    def calculate_nutritional_score(self, food_item, trimester, user_needs=None):
        """
        Calculate nutritional score for a food item based on trimester needs.
        
        Args:
            food_item: FoodItem object
            trimester: Current trimester (1, 2, or 3)
            user_needs: Optional dictionary of specific user nutritional needs
            
        Returns:
            float: Score between 0 and 1
        """
        nutrition = NutritionEstimator.get_nutrition_with_estimate(food_item)
        requirements = self.trimester_requirements.get(trimester, self.trimester_requirements[1])
        
        if not nutrition:
            return 0.5  # Default score for foods without nutritional data
        
        # Calculate how well the food matches trimester requirements
        score = 0.0
        matched_nutrients = 0
        
        for nutrient, required_amount in requirements.items():
            if nutrient in nutrition:
                food_nutrient_amount = nutrition[nutrient]
                
                # Normalize score (percentage of daily requirement per 100g serving)
                nutrient_score = min(food_nutrient_amount / required_amount, 1.0)
                score += nutrient_score
                matched_nutrients += 1
        
        # Average score
        if matched_nutrients > 0:
            score = score / matched_nutrients
        else:
            score = 0.5
        
        return min(score, 1.0)
    
    def check_safety(self, food_item, user_health_conditions):
        """
        Check if a food item is safe for the user based on health conditions.
        
        Args:
            food_item: FoodItem object
            user_health_conditions: Dictionary of health conditions
            
        Returns:
            tuple: (is_safe: bool, warnings: list)
        """
        warnings = []
        
        # Check for allergies
        allergies = user_health_conditions.get('allergies', [])
        food_name_lower = food_item.name_english.lower()
        
        for allergy in allergies:
            if allergy.lower() in food_name_lower:
                return False, [f"Contains allergen: {allergy}"]
        
        # Check for diabetes
        if user_health_conditions.get('diabetes', False):
            nutrition = food_item.get_nutritional_info()
            if nutrition.get('sugar', 0) > 15:  # High sugar content
                warnings.append("High sugar content - consume in moderation")
        
        # Check for high blood pressure
        if user_health_conditions.get('hypertension', False):
            nutrition = food_item.get_nutritional_info()
            if nutrition.get('sodium', 0) > 200:  # High sodium
                warnings.append("High sodium - limit intake")
        
        # Check precautions
        if food_item.precautions:
            warnings.append(food_item.precautions)
        
        return True, warnings
    
    def get_complementary_foods(self, food_item, all_foods):
        """
        Find foods that complement the given food nutritionally.
        
        Args:
            food_item: FoodItem object
            all_foods: List of all available FoodItem objects
            
        Returns:
            list: List of complementary food items
        """
        complementary = []
        food_nutrition = food_item.get_nutritional_info()
        
        # Find foods that provide nutrients this food lacks
        for other_food in all_foods:
            if other_food.id == food_item.id:
                continue
            
            other_nutrition = other_food.get_nutritional_info()
            
            # Simple complementarity: different categories are complementary
            if other_food.category != food_item.category:
                complementary.append(other_food)
        
        return complementary[:5]  # Return top 5
