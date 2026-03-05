"""Food item model."""
from models import db
import json


class FoodItem(db.Model):
    """Food item model for storing Indian food information."""
    
    __tablename__ = 'food_items'
    
    id = db.Column(db.Integer, primary_key=True)
    name_english = db.Column(db.String(100), nullable=False, index=True)
    name_hindi = db.Column(db.String(100), nullable=True)
    name_telugu = db.Column(db.String(100), nullable=True)
    name_kannada = db.Column(db.String(100), nullable=True)
    name_malayalam = db.Column(db.String(100), nullable=True)
    name_tamil = db.Column(db.String(100), nullable=True)
    category = db.Column(db.String(50), nullable=False, index=True)
    nutritional_info = db.Column(db.Text, default='{}')  # JSON string
    trimester_suitability = db.Column(db.Text, default='{}')  # JSON string
    regional_origin = db.Column(db.String(50), nullable=True)
    preparation_tips = db.Column(db.Text, nullable=True)
    benefits = db.Column(db.Text, nullable=True)
    precautions = db.Column(db.Text, nullable=True)
    seasonal_availability = db.Column(db.String(50), nullable=True)  # Summer, Winter, Monsoon, All
    
    # Relationships
    interactions = db.relationship('UserInteraction', backref='food_item', lazy='dynamic')
    
    def __repr__(self):
        return f'<FoodItem {self.name_english}>'
    
    def get_nutritional_info(self):
        """Get nutritional info as a dictionary."""
        try:
            return json.loads(self.nutritional_info) if self.nutritional_info else {}
        except:
            return {}
    
    def set_nutritional_info(self, nutrition_dict):
        """Set nutritional info from a dictionary."""
        self.nutritional_info = json.dumps(nutrition_dict)
    
    def get_trimester_suitability(self):
        """Get trimester suitability as a dictionary."""
        try:
            return json.loads(self.trimester_suitability) if self.trimester_suitability else {}
        except:
            return {}
    
    def set_trimester_suitability(self, trimester_dict):
        """Set trimester suitability from a dictionary."""
        self.trimester_suitability = json.dumps(trimester_dict)
    
    def to_dict(self):
        """Convert food item to dictionary."""
        return {
            'id': self.id,
            'name_english': self.name_english,
            'name_hindi': self.name_hindi,
               'name_telugu': self.name_telugu,
               'name_kannada': self.name_kannada,
               'name_malayalam': self.name_malayalam,
               'name_tamil': self.name_tamil,
            'category': self.category,
            'nutritional_info': self.get_nutritional_info(),
            'trimester_suitability': self.get_trimester_suitability(),
            'regional_origin': self.regional_origin,
            'preparation_tips': self.preparation_tips,
            'benefits': self.benefits,
            'precautions': self.precautions,
            'seasonal_availability': self.seasonal_availability
        }

        def get_name(self, language='english'):
            """Get food name in specified language."""
            language_map = {
                'english': self.name_english,
                'hindi': self.name_hindi,
                'telugu': self.name_telugu,
                'kannada': self.name_kannada,
                'malayalam': self.name_malayalam,
                'tamil': self.name_tamil
            }
            return language_map.get(language, self.name_english)
