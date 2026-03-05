"""Form validators."""
import re
from datetime import datetime, date


def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """
    Validate password strength.
    Requirements: at least 8 characters, one uppercase, one lowercase, one number.
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
    return True, "Password is valid"


def validate_username(username):
    """
    Validate username.
    Requirements: 3-80 characters, alphanumeric and underscores only.
    """
    if len(username) < 3 or len(username) > 80:
        return False, "Username must be between 3 and 80 characters"
    
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    
    return True, "Username is valid"


def validate_due_date(due_date_str):
    """
    Validate and parse due date.
    Returns (is_valid, parsed_date_or_error_message).
    """
    try:
        due_date = datetime.strptime(due_date_str, '%Y-%m-%d').date()
        
        # Check if date is in the future
        today = date.today()
        if due_date < today:
            return False, "Due date must be in the future"
        
        # Check if date is not too far in the future (more than 280 days)
        days_diff = (due_date - today).days
        if days_diff > 280:
            return False, "Due date seems too far in the future"
        
        return True, due_date
    except ValueError:
        return False, "Invalid date format. Use YYYY-MM-DD"


def validate_trimester(trimester):
    """Validate trimester value."""
    try:
        trimester = int(trimester)
        if trimester in [1, 2, 3]:
            return True, trimester
        else:
            return False, "Trimester must be 1, 2, or 3"
    except (ValueError, TypeError):
        return False, "Invalid trimester value"
