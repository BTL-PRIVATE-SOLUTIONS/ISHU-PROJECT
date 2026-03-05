"""Recommendation model."""
from datetime import datetime
from models import db
import json


class Recommendation(db.Model):
    """Recommendation model for storing meal and food recommendations."""
    
    __tablename__ = 'recommendations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    recommendation_type = db.Column(db.String(50), nullable=False)  # 'meal_plan', 'food', 'nutrition'
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    recommendations = db.Column(db.Text, default='{}')  # JSON string
    nutrition_summary = db.Column(db.Text, default='{}')  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    interactions = db.relationship('UserInteraction', backref='recommendation', lazy='dynamic')
    
    def __repr__(self):
        return f'<Recommendation {self.recommendation_type} - {self.title}>'
    
    def get_recommendations(self):
        """Get recommendations as a dictionary."""
        try:
            return json.loads(self.recommendations) if self.recommendations else {}
        except:
            return {}
    
    def set_recommendations(self, rec_dict):
        """Set recommendations from a dictionary."""
        self.recommendations = json.dumps(rec_dict)
    
    def get_nutrition_summary(self):
        """Get nutrition summary as a dictionary."""
        try:
            return json.loads(self.nutrition_summary) if self.nutrition_summary else {}
        except:
            return {}
    
    def set_nutrition_summary(self, nutrition_dict):
        """Set nutrition summary from a dictionary."""
        self.nutrition_summary = json.dumps(nutrition_dict)
    
    def to_dict(self):
        """Convert recommendation to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'recommendation_type': self.recommendation_type,
            'title': self.title,
            'description': self.description,
            'recommendations': self.get_recommendations(),
            'nutrition_summary': self.get_nutrition_summary(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
