"""Food recommender for personalized nutrition suggestions."""

from typing import List, Dict, Optional
from models.food import FoodItem
from ai_engine.nutritional_analyzer import NutritionalAnalyzer


class FoodRecommender:
    """Recommends foods based on user preferences and nutritional needs."""
    
    def __init__(self, db):
        """
        Initialize the food recommender.
        
        Args:
            db: Database instance
        """
        self.db = db
        self.analyzer = NutritionalAnalyzer()
    
    def get_recommendations(
        self,
        user,
        max_items: int = 10,
        category: Optional[str] = None,
        exclude_foods: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Get food recommendations for a user.
        
        Args:
            user: User object
            max_items: Maximum number of recommendations
            category: Optional food category to filter by
            exclude_foods: Optional list of food IDs to exclude
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Get all foods
            query = FoodItem.query
            
            if category:
                query = query.filter_by(category=category)
            
            foods = query.all()
            
            if not foods:
                return []
            
            # Filter out excluded foods
            if exclude_foods:
                foods = [f for f in foods if f.id not in exclude_foods]
            
            # Score each food
            recommendations = []
            for food in foods:
                score = self._calculate_recommendation_score(food, user)
                recommendations.append({
                    'food': food,
                    'score': score,
                    'nutrition_score': self.analyzer.calculate_nutritional_score(
                        food, user.current_trimester
                    ),
                    'trimester_score': self._calculate_trimester_score(food, user),
                    'preference_score': self._calculate_preference_score(food, user),
                    'safety_score': self._calculate_safety_score(food, user)
                })
            
            # Sort by score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return recommendations[:max_items]
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def _calculate_recommendation_score(self, food: FoodItem, user) -> float:
        """
        Calculate overall recommendation score for a food.
        
        Args:
            food: FoodItem object
            user: User object
            
        Returns:
            float: Score between 0 and 1
        """
        try:
            # Calculate component scores
            nutrition_score = self.analyzer.calculate_nutritional_score(
                food, user.current_trimester
            )
            trimester_score = self._calculate_trimester_score(food, user)
            preference_score = self._calculate_preference_score(food, user)
            safety_score = self._calculate_safety_score(food, user)
            
            # Weighted average
            weights = {
                'nutrition': 0.4,
                'trimester': 0.25,
                'preference': 0.2,
                'safety': 0.15
            }
            
            overall_score = (
                nutrition_score * weights['nutrition'] +
                trimester_score * weights['trimester'] +
                preference_score * weights['preference'] +
                safety_score * weights['safety']
            )
            
            return min(1.0, max(0.0, overall_score))
            
        except Exception as e:
            print(f"Error calculating recommendation score: {e}")
            return 0.0
    
    def _calculate_trimester_score(self, food: FoodItem, user) -> float:
        """
        Calculate trimester appropriateness score.
        
        Args:
            food: FoodItem object
            user: User object
            
        Returns:
            float: Score between 0 and 1
        """
        try:
            # Base score
            score = 0.5
            
            # Get trimester-specific recommendations
            trimester = user.current_trimester
            trimester_foods = getattr(food, f'recommended_trimester_{trimester}', False)
            
            if trimester_foods:
                score = 0.9
            elif hasattr(food, 'unsafe_during_pregnancy') and food.unsafe_during_pregnancy:
                score = 0.1
            else:
                score = 0.6
            
            return min(1.0, max(0.0, score))
            
        except Exception:
            return 0.5
    
    def _calculate_preference_score(self, food: FoodItem, user) -> float:
        """
        Calculate preference score based on user diet type.
        
        Args:
            food: FoodItem object
            user: User object
            
        Returns:
            float: Score between 0 and 1
        """
        try:
            # Check if food matches user's diet preference
            diet_type = getattr(user, 'diet_type', 'vegetarian')
            
            if diet_type == 'vegetarian':
                if hasattr(food, 'is_vegetarian') and food.is_vegetarian:
                    return 0.9
                elif hasattr(food, 'is_non_vegetarian') and food.is_non_vegetarian:
                    return 0.2
            elif diet_type == 'non-vegetarian':
                if hasattr(food, 'is_non_vegetarian') and food.is_non_vegetarian:
                    return 0.9
                else:
                    return 0.7
            elif diet_type == 'vegan':
                if hasattr(food, 'is_vegan') and food.is_vegan:
                    return 0.9
                elif hasattr(food, 'is_vegetarian') and food.is_vegetarian:
                    return 0.7
                else:
                    return 0.2
            
            return 0.6
            
        except Exception:
            return 0.5
    
    def _calculate_safety_score(self, food: FoodItem, user) -> float:
        """
        Calculate safety score for a food during pregnancy.
        
        Args:
            food: FoodItem object
            user: User object
            
        Returns:
            float: Score between 0 and 1
        """
        try:
            # Check if food has safety warnings
            if hasattr(food, 'unsafe_during_pregnancy') and food.unsafe_during_pregnancy:
                return 0.1
            elif hasattr(food, 'precautions') and food.precautions:
                return 0.6
            else:
                return 0.9
                
        except Exception:
            return 0.7
    
    def get_category_foods(self, category: str, limit: int = 20) -> List[FoodItem]:
        """
        Get foods by category.
        
        Args:
            category: Food category
            limit: Maximum number of foods to return
            
        Returns:
            List of FoodItem objects
        """
        try:
            foods = FoodItem.query.filter_by(category=category).limit(limit).all()
            return foods if foods else []
        except Exception:
            return []
    
    def get_all_categories(self) -> List[str]:
        """
        Get all available food categories.
        
        Returns:
            List of category names
        """
        try:
            categories = FoodItem.query.distinct(FoodItem.category).all()
            return [c.category for c in categories if c.category]
        except Exception:
            return []
