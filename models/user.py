"""User model."""
from datetime import datetime
from typing import Tuple, List
from flask_login import UserMixin
from models import db
import json


class User(UserMixin, db.Model):
    """User model for authentication and profile management."""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=True, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120), nullable=True)
    due_date = db.Column(db.Date, nullable=True)
    current_trimester = db.Column(db.Integer, default=1)
    health_conditions = db.Column(db.Text, default='{}')  # JSON string
    dietary_preferences = db.Column(db.String(50), default='vegetarian')
    language = db.Column(db.String(20), default='english')  # Selected language
    
    # Comprehensive meal planning preferences (NO DEFAULTS - ALL USER SELECTED)
    region_preference = db.Column(db.String(50), nullable=True)  # 'North' or 'South' - REQUIRED
    seasonal_preference = db.Column(db.String(50), nullable=True)  # 'summer', 'winter', 'monsoon' - OPTIONAL
    special_conditions = db.Column(db.Text, default='[]')  # JSON array of selected conditions
    meal_frequency_preference = db.Column(db.String(50), nullable=True)  # '3meals', '5meals', 'custom'
    is_diabetic = db.Column(db.Boolean, default=False)
    is_gestational_diabetic = db.Column(db.Boolean, default=False)
    postpartum_phase = db.Column(db.String(50), nullable=True)  # 'postnatal', 'postpartum', None
    preferences_completed = db.Column(db.Boolean, default=False)  # User has selected all required preferences
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    preferences_updated_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def get_health_conditions(self):
        """Get health conditions as a dictionary."""
        try:
            return json.loads(self.health_conditions) if self.health_conditions else {}
        except:
            return {}
    
    def set_health_conditions(self, conditions_dict):
        """Set health conditions from a dictionary."""
        self.health_conditions = json.dumps(conditions_dict)
    
    def get_special_conditions(self):
        """Get special conditions as a list."""
        try:
            conditions = json.loads(self.special_conditions) if self.special_conditions else []
            # Add derived conditions
            if self.is_diabetic and 'diabetes' not in conditions:
                conditions.append('diabetes')
            if self.is_gestational_diabetic and 'gestational_diabetes' not in conditions:
                conditions.append('gestational_diabetes')
            return list(set(conditions))  # Remove duplicates
        except:
            return []
    
    def set_special_conditions(self, conditions_list):
        """Set special conditions from a list."""
        self.special_conditions = json.dumps(conditions_list if conditions_list else [])
    
    def add_special_condition(self, condition: str):
        """Add a special condition."""
        conditions = self.get_special_conditions()
        if condition not in conditions:
            conditions.append(condition)
        self.set_special_conditions(conditions)
    
    def remove_special_condition(self, condition: str):
        """Remove a special condition."""
        conditions = self.get_special_conditions()
        if condition in conditions:
            conditions.remove(condition)
        self.set_special_conditions(conditions)
    
    def validate_preferences(self) -> Tuple[bool, List[str]]:
        """Validate that all required preferences are selected.
        
        Returns:
            (is_valid, list_of_missing_preferences)
        """
        missing = []
        
        if not self.region_preference:
            missing.append('region_preference')
        if not self.dietary_preferences:
            missing.append('dietary_preferences')
        if self.current_trimester is None or self.current_trimester <= 0:
            missing.append('current_trimester')
        
        is_valid = len(missing) == 0
        self.preferences_completed = is_valid
        
        return is_valid, missing
    
    def to_dict(self):
        """Convert user to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'current_trimester': self.current_trimester,
            'health_conditions': self.get_health_conditions(),
            'dietary_preferences': self.dietary_preferences,
            'language': self.language,
            'region_preference': self.region_preference,
            'seasonal_preference': self.seasonal_preference,
            'special_conditions': self.get_special_conditions(),
            'meal_frequency_preference': self.meal_frequency_preference,
            'is_diabetic': self.is_diabetic,
            'is_gestational_diabetic': self.is_gestational_diabetic,
            'postpartum_phase': self.postpartum_phase,
            'preferences_completed': self.preferences_completed,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
