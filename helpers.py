"""Helper functions."""
from datetime import date, timedelta


def calculate_trimester_from_due_date(due_date):
    """
    Calculate current trimester based on due date.
    
    Pregnancy timeline:
    - 1st trimester: Weeks 1-12
    - 2nd trimester: Weeks 13-27
    - 3rd trimester: Weeks 28-40
    """
    if not due_date:
        return 1
    
    today = date.today()
    days_until_due = (due_date - today).days
    
    # Convert to weeks pregnant (40 weeks total pregnancy)
    weeks_pregnant = 40 - (days_until_due / 7)
    
    if weeks_pregnant <= 12:
        return 1
    elif weeks_pregnant <= 27:
        return 2
    else:
        return 3


def calculate_weeks_pregnant(due_date):
    """Calculate how many weeks pregnant based on due date."""
    if not due_date:
        return 0
    
    today = date.today()
    days_until_due = (due_date - today).days
    weeks_pregnant = 40 - (days_until_due / 7)
    
    return max(0, min(40, weeks_pregnant))


def get_trimester_nutritional_needs(trimester):
    """
    Get key nutritional needs for each trimester.
    Returns a dictionary of nutrients and their importance.
    """
    needs = {
        1: {
            'folic_acid': 'critical',
            'vitamin_b6': 'high',
            'iron': 'high',
            'calcium': 'moderate',
            'protein': 'moderate'
        },
        2: {
            'calcium': 'critical',
            'vitamin_d': 'critical',
            'omega3': 'high',
            'protein': 'high',
            'iron': 'high',
            'folic_acid': 'moderate'
        },
        3: {
            'iron': 'critical',
            'protein': 'critical',
            'vitamin_k': 'high',
            'fiber': 'high',
            'calcium': 'high',
            'omega3': 'moderate'
        }
    }
    return needs.get(trimester, needs[1])


def format_indian_food_name(english_name, hindi_name=None):
    """Format food name with both English and Hindi."""
    if hindi_name:
        return f"{english_name} ({hindi_name})"
    return english_name


def get_meal_time_recommendation():
    """Get current meal time category based on time of day."""
    from datetime import datetime
    current_hour = datetime.now().hour
    
    if 6 <= current_hour < 10:
        return 'breakfast'
    elif 10 <= current_hour < 12:
        return 'mid_morning_snack'
    elif 12 <= current_hour < 15:
        return 'lunch'
    elif 15 <= current_hour < 17:
        return 'evening_snack'
    elif 17 <= current_hour < 21:
        return 'dinner'
    else:
        return 'night_snack'


def sanitize_search_query(query):
    """Sanitize search query to prevent SQL injection."""
    if not query:
        return ""
    
    # Remove special characters except spaces, hyphens, and common punctuation
    import re
    sanitized = re.sub(r'[^\w\s\-.,()]', '', query)
    return sanitized.strip()
