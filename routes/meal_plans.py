"""Meal plan routes for generating personalized meal plans with comprehensive user preferences."""
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime
from models import db
from models.interaction import UserInteraction
from ai_engine.meal_planner import MealPlanner
from ai_engine.unified_dataset_loader import UnifiedDatasetLoader

meal_plans_bp = Blueprint('meal_plans', __name__)

# Initialize unified dataset loader
unified_loader = UnifiedDatasetLoader()


@meal_plans_bp.route('/')
@login_required
def meal_plans_page():
    """Render meal plans generator page."""
    return render_template('dashboard/meal_plans.html')


@meal_plans_bp.route('/api/preferences/validate', methods=['POST'])
@login_required
def validate_preferences():
    """
    Validate user preferences are complete before meal generation.
    
    Expects JSON:
        {
            "region_preference": "North",
            "seasonal_preference": "summer",
            "dietary_preferences": "vegetarian",
            "current_trimester": 2,
            "special_conditions": ["diabetes"]  # optional
        }
    
    Returns:
        {
            "success": true/false,
            "is_valid": true/false,
            "missing_fields": [...],
            "available_meals": 123  # if valid
        }
    """
    try:
        data = request.get_json()
        
        # Update user preferences
        if data.get('region_preference'):
            current_user.region_preference = data['region_preference']
        if data.get('dietary_preferences'):
            current_user.dietary_preferences = data['dietary_preferences']
        if data.get('current_trimester'):
            current_user.current_trimester = int(data['current_trimester'])
        if data.get('meal_frequency_preference'):
            current_user.meal_frequency_preference = data['meal_frequency_preference']
        
        # Handle special conditions
        if 'special_conditions' in data:
            current_user.set_special_conditions(data['special_conditions'])
            current_user.is_diabetic = 'diabetes' in data.get('special_conditions', [])
            current_user.is_gestational_diabetic = 'gestational_diabetes' in data.get('special_conditions', [])
        
        # Validate preferences
        is_valid, missing = current_user.validate_preferences()
        
        if is_valid:
            current_user.preferences_updated_at = datetime.utcnow()
            
            # Count available meals for these preferences
            available_meals = unified_loader.get_meals_by_preference(
                region=current_user.region_preference,
                diet_type=current_user.dietary_preferences,
                trimester=current_user.current_trimester,
                condition=current_user.get_special_conditions()[0] if current_user.get_special_conditions() else None
            )
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'is_valid': True,
                'message': 'All preferences validated successfully',
                'available_meals': len(available_meals),
                'user_preferences': current_user.to_dict()
            })
        else:
            return jsonify({
                'success': True,
                'is_valid': False,
                'message': 'Some required preferences are missing',
                'missing_fields': missing,
                'user_preferences': current_user.to_dict()
            }), 400
    
    except Exception as e:
        print(f"Error validating preferences: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@meal_plans_bp.route('/api/preferences/available', methods=['GET'])
def get_available_preferences():
    """
    Get all available preference options for the user to select from.
    This endpoint does NOT require authentication since it returns static data about available options.
    
    Returns:
        {
            "success": true,
            "regions": [...],
            "diets": [...],
            "seasons": [...],
            "conditions": [...],
            "trimesters": [1, 2, 3]
        }
    """
    try:
        options = unified_loader.get_available_options()
        
        return jsonify({
            'success': True,
            'regions': options['regions'] or ['North', 'South'],
            'diets': options['diets'] or ['veg', 'nonveg', 'vegan'],
            'seasons': options['seasons'] or ['summer', 'winter', 'monsoon'],
            'conditions': options['conditions'] or ['diabetes', 'gestational_diabetes'],
            'trimesters': [1, 2, 3],
            'categories': options['categories'],
            'stats': unified_loader.get_statistics()
        })
        
    except Exception as e:
        print(f"Error getting available preferences: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@meal_plans_bp.route('/api/generate', methods=['POST'])
@login_required
def generate_meal_plan():
    """
    Generate a personalized meal plan based on user preferences.
    
    Supports all datasets: regional, trimester-wise, seasonal, diabetes-specific, postpartum
    
    Expects JSON:
        {
            "days": 7,
            "region": "North",
            "diet_type": "veg",
            "season": "summer",  # Optional
            "meal_frequency": "3meals"  # "3meals" or "5meals"
        }
    
    Returns:
        {
            "success": true,
            "meal_plan": [...],
            "nutrition_summary": {...},
            "table_format": [...],
            "preferences_used": {...}
        }
    """
    try:
        data = request.get_json()
        
        # Update user preferences from request
        if data.get('region'):
            current_user.region_preference = data['region']
        if data.get('diet_type'):
            current_user.dietary_preferences = data['diet_type']
        
        # Ensure trimester is set
        if not current_user.current_trimester or current_user.current_trimester <= 0:
            current_user.current_trimester = 1
        
        # CRITICAL: Validate user has completed required preferences
        is_valid, missing = current_user.validate_preferences()
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Please complete your preferences first. Missing: {", ".join(missing)}',
                'missing_fields': missing
            }), 400
        
        # Save preferences to database
        current_user.preferences_updated_at = datetime.utcnow()
        db.session.commit()
        
        # Validate input
        if not data:
            return jsonify({'error': 'Request data is required'}), 400
        
        # Get parameters
        days = data.get('days', 7)
        meal_frequency = data.get('meal_frequency', '3meals')
        
        # Validate days
        try:
            days = int(days)
            if days < 1 or days > 30:
                return jsonify({'error': 'Days must be between 1 and 30'}), 400
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid days value'}), 400
        
        # Create meal planner with unified dataset loader
        meal_planner = MealPlanner(db, unified_loader)
        
        # Generate meal plan using all datasets and user preferences
        result = meal_planner.generate_meal_plan(
            user=current_user,
            days=days,
            region=current_user.region_preference,
            diet_type=current_user.dietary_preferences,
            trimester=current_user.current_trimester,
            special_conditions=current_user.get_special_conditions(),
            meal_frequency=meal_frequency
        )
        
        # Check for errors
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        # Log interaction
        interaction = UserInteraction(
            user_id=current_user.id,
            interaction_type='meal_plan_generation'
        )
        interaction.set_details({
            'days': days,
            'region_preference': current_user.region_preference,
            'diet_type': current_user.dietary_preferences,
            'seasonal_preference': current_user.seasonal_preference,
            'trimester': current_user.current_trimester,
            'special_conditions': current_user.get_special_conditions(),
            'meal_frequency': meal_frequency,
            'total_meals': len(result.get('meal_plan', [])),
            'data_sources_used': result.get('data_sources_used', [])
        })
        
        db.session.add(interaction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'meal_plan': result['meal_plan'],
            'nutrition_summary': result['nutrition_summary'],
            'table_format': result['table_format'],
            'preferences_used': {
                'region': current_user.region_preference,
                'diet': current_user.dietary_preferences,
                'season': current_user.seasonal_preference,
                'trimester': current_user.current_trimester,
                'special_conditions': current_user.get_special_conditions(),
                'meal_frequency': meal_frequency
            },
            'data_sources': result.get('data_sources_used', [])
        })
        
    except Exception as e:
        print(f"Error generating meal plan: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An error occurred generating the meal plan. Please try again.'
        }), 500


@meal_plans_bp.route('/api/guidance', methods=['GET'])
@login_required
def get_guidance():
    """
    Get relevant guidance (dos/donts, foods to avoid) based on user preferences.
    
    Returns:
        {
            "success": true,
            "dos": [...],
            "donts": [...],
            "foods_to_avoid": [...]
        }
    """
    try:
        special_conditions = current_user.get_special_conditions()
        trimester = current_user.current_trimester
        
        # Get guidance for user's specific situation
        dos_donts = unified_loader.get_guidance('dos_donts')
        avoid_foods = unified_loader.get_guidance('avoid_foods')
        
        # Filter by condition if applicable
        relevant_avoid = avoid_foods
        if special_conditions:
            condition = special_conditions[0]
            relevant_avoid = [f for f in avoid_foods 
                            if condition.lower() in str(f).lower()]
        
        return jsonify({
            'success': True,
            'dos': [d for d in dos_donts if 'do' in str(d).lower()],
            'donts': [d for d in dos_donts if 'dont' in str(d).lower() or 'avoid' in str(d).lower()],
            'foods_to_avoid': relevant_avoid,
            'for_trimester': trimester,
            'for_conditions': special_conditions
        })
        
    except Exception as e:
        print(f"Error getting guidance: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

