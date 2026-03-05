"""Nutrition estimator for foods with missing nutritional data."""


class NutritionEstimator:
    """Estimate nutritional values for foods based on category and name."""
    
    # Estimated nutrition per 100g by food category (realistic for Indian pregnancy diet)
    CATEGORY_NUTRITION = {
        'vegetables': {
            'calories': 25,
            'protein': 2,
            'iron': 0.8,
            'calcium': 40,
            'fiber': 2.5,
            'folic_acid': 50
        },
        'fruits': {
            'calories': 50,
            'protein': 0.6,
            'iron': 0.3,
            'calcium': 30,
            'fiber': 2,
            'folic_acid': 20
        },
        'dairy': {
            'calories': 120,
            'protein': 3.5,
            'iron': 0.1,
            'calcium': 300,
            'fiber': 0,
            'folic_acid': 5
        },
        'grains': {
            'calories': 130,
            'protein': 3,
            'iron': 0.8,
            'calcium': 15,
            'fiber': 1.5,
            'folic_acid': 15
        },
        'lentils': {
            'calories': 116,
            'protein': 9,
            'iron': 3.3,
            'calcium': 40,
            'fiber': 7.9,
            'folic_acid': 181
        },
        'dry_fruits': {
            'calories': 400,
            'protein': 8,
            'iron': 2,
            'calcium': 150,
            'fiber': 6,
            'folic_acid': 50
        },
        'proteins': {
            'calories': 155,
            'protein': 13,
            'iron': 1.8,
            'calcium': 50,
            'fiber': 0,
            'folic_acid': 50
        },
        'beverages': {
            'calories': 50,
            'protein': 1,
            'iron': 0.1,
            'calcium': 100,
            'fiber': 0.5,
            'folic_acid': 10
        },
        'other': {
            'calories': 60,
            'protein': 2,
            'iron': 0.5,
            'calcium': 50,
            'fiber': 1,
            'folic_acid': 20
        }
    }
    
    # Food-specific overrides (more accurate values)
    FOOD_SPECIFIC_NUTRITION = {
        'spinach': {'calories': 23, 'protein': 2.9, 'iron': 2.7, 'calcium': 99, 'fiber': 2.2, 'folic_acid': 194},
        'milk': {'calories': 61, 'protein': 3.4, 'iron': 0.1, 'calcium': 120, 'fiber': 0, 'folic_acid': 5},
        'rice': {'calories': 130, 'protein': 2.7, 'iron': 0.2, 'calcium': 10, 'fiber': 0.4, 'folic_acid': 8},
        'papaya': {'calories': 43, 'protein': 0.5, 'iron': 0.3, 'calcium': 20, 'fiber': 1.7, 'folic_acid': 37},
        'almonds': {'calories': 579, 'protein': 21, 'iron': 3.7, 'calcium': 269, 'fiber': 12.5, 'folic_acid': 44},
        'lentils': {'calories': 116, 'protein': 9, 'iron': 3.3, 'calcium': 40, 'fiber': 7.9, 'folic_acid': 181},
        'yogurt': {'calories': 59, 'protein': 3.5, 'iron': 0.1, 'calcium': 121, 'fiber': 0, 'folic_acid': 7},
        'eggs': {'calories': 155, 'protein': 13, 'iron': 1.8, 'calcium': 50, 'fiber': 0, 'folic_acid': 47},
        'dates': {'calories': 277, 'protein': 1.7, 'iron': 0.9, 'calcium': 64, 'fiber': 6.7, 'folic_acid': 15},
        'ghee': {'calories': 112, 'protein': 0, 'iron': 0, 'calcium': 5, 'fiber': 0, 'folic_acid': 0},
        'banana': {'calories': 89, 'protein': 1.1, 'iron': 0.3, 'calcium': 5, 'fiber': 2.6, 'folic_acid': 20},
        'carrot': {'calories': 41, 'protein': 0.9, 'iron': 0.3, 'calcium': 33, 'fiber': 2.8, 'folic_acid': 19},
        'tomato': {'calories': 18, 'protein': 0.9, 'iron': 0.3, 'calcium': 12, 'fiber': 1.2, 'folic_acid': 15},
        'cucumber': {'calories': 16, 'protein': 0.7, 'iron': 0.3, 'calcium': 16, 'fiber': 0.5, 'folic_acid': 7},
        'beetroot': {'calories': 43, 'protein': 1.6, 'iron': 0.8, 'calcium': 16, 'fiber': 2.8, 'folic_acid': 109},
        'guava': {'calories': 68, 'protein': 2.6, 'iron': 0.3, 'calcium': 18, 'fiber': 5.4, 'folic_acid': 49},
        'orange': {'calories': 47, 'protein': 0.7, 'iron': 0.1, 'calcium': 40, 'fiber': 2.4, 'folic_acid': 30},
        'apple': {'calories': 52, 'protein': 0.3, 'iron': 0.1, 'calcium': 5, 'fiber': 2.4, 'folic_acid': 3},
        'mango': {'calories': 60, 'protein': 0.8, 'iron': 0.2, 'calcium': 10, 'fiber': 1.6, 'folic_acid': 43},
        'raisin': {'calories': 299, 'protein': 3.1, 'iron': 1.5, 'calcium': 50, 'fiber': 3.7, 'folic_acid': 5},
        'walnut': {'calories': 654, 'protein': 9.1, 'iron': 2.9, 'calcium': 98, 'fiber': 6.7, 'folic_acid': 77},
        'chickpea': {'calories': 364, 'protein': 19, 'iron': 4.3, 'calcium': 150, 'fiber': 10, 'folic_acid': 557},
        'moong': {'calories': 347, 'protein': 24, 'iron': 6.4, 'calcium': 190, 'fiber': 16, 'folic_acid': 625},
        'paneer': {'calories': 265, 'protein': 25, 'iron': 0.2, 'calcium': 400, 'fiber': 0, 'folic_acid': 10},
        'chicken': {'calories': 165, 'protein': 31, 'iron': 0.8, 'calcium': 11, 'fiber': 0, 'folic_acid': 7},
        'fish': {'calories': 100, 'protein': 20, 'iron': 0.7, 'calcium': 12, 'fiber': 0, 'folic_acid': 5},
        'honey': {'calories': 304, 'protein': 0.3, 'iron': 0.4, 'calcium': 6, 'fiber': 0, 'folic_acid': 2},
        'jaggery': {'calories': 383, 'protein': 0.4, 'iron': 1.7, 'calcium': 40, 'fiber': 0, 'folic_acid': 0},
    }
    
    @staticmethod
    def estimate_nutrition(food_item):
        """
        Estimate nutrition for a food item.
        
        Args:
            food_item: FoodItem object
            
        Returns:
            dict: Estimated nutrition values
        """
        food_name_lower = food_item.name_english.lower()
        
        # Check for food-specific values first
        for key, nutrition in NutritionEstimator.FOOD_SPECIFIC_NUTRITION.items():
            if key in food_name_lower:
                return nutrition.copy()
        
        # Fall back to category-based estimation
        category = food_item.category.lower() if food_item.category else 'other'
        if category not in NutritionEstimator.CATEGORY_NUTRITION:
            category = 'other'
        
        return NutritionEstimator.CATEGORY_NUTRITION[category].copy()
    
    @staticmethod
    def get_nutrition_with_estimate(food_item):
        """
        Get nutrition from food item, with estimation fallback.
        
        Args:
            food_item: FoodItem object
            
        Returns:
            dict: Nutrition values (estimated or from database)
        """
        nutrition = food_item.get_nutritional_info()
        
        # Check if nutrition has valid numeric values
        if nutrition and isinstance(nutrition, dict):
            # Try to validate numeric values
            has_valid_values = False
            for key in ['calories', 'protein', 'iron', 'calcium', 'fiber', 'folic_acid']:
                try:
                    val = nutrition.get(key)
                    if val and float(val) > 0:
                        has_valid_values = True
                        break
                except (ValueError, TypeError):
                    pass
            
            if has_valid_values:
                # Merge with defaults for missing values
                estimated = NutritionEstimator.estimate_nutrition(food_item)
                for key in estimated:
                    if key not in nutrition or not nutrition[key]:
                        nutrition[key] = estimated[key]
                return nutrition
        
        # Return estimate if no valid nutrition data
        return NutritionEstimator.estimate_nutrition(food_item)
