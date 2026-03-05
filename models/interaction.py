"""User interaction model."""
from datetime import datetime
from models import db
import json


class UserInteraction(db.Model):
    """User interaction model for tracking user behavior."""
    
    __tablename__ = 'user_interactions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    interaction_type = db.Column(db.String(50), nullable=False, index=True)
    food_item_id = db.Column(db.Integer, db.ForeignKey('food_items.id'), nullable=True, index=True)
    recommendation_id = db.Column(db.Integer, db.ForeignKey('recommendations.id'), nullable=True)
    details = db.Column(db.Text, default='{}')  # JSON string
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<UserInteraction {self.interaction_type} by User {self.user_id}>'
    
    def get_details(self):
        """Get details as a dictionary."""
        try:
            return json.loads(self.details) if self.details else {}
        except:
            return {}
    
    def set_details(self, details_dict):
        """Set details from a dictionary."""
        self.details = json.dumps(details_dict)
    
    def to_dict(self):
        """Convert interaction to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'interaction_type': self.interaction_type,
            'food_item_id': self.food_item_id,
            'recommendation_id': self.recommendation_id,
            'details': self.get_details(),
            'timestamp': self.timestamp.isoformat()
        }
