"""User interaction routes."""
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime, timedelta
from models import db
from models.interaction import UserInteraction
from models.food import FoodItem
from sqlalchemy import func

interactions_bp = Blueprint('interactions', __name__)


@interactions_bp.route('/api/interactions', methods=['POST'])
@login_required
def log_interaction():
    """Log a user interaction."""
    data = request.get_json()
    
    interaction_type = data.get('interaction_type')
    food_item_id = data.get('food_item_id')
    recommendation_id = data.get('recommendation_id')
    details = data.get('details', {})
    
    if not interaction_type:
        return jsonify({'error': 'interaction_type is required'}), 400
    
    interaction = UserInteraction(
        user_id=current_user.id,
        interaction_type=interaction_type,
        food_item_id=food_item_id,
        recommendation_id=recommendation_id
    )
    interaction.set_details(details)
    
    db.session.add(interaction)
    db.session.commit()
    
    return jsonify(interaction.to_dict()), 201


@interactions_bp.route('/api/interactions', methods=['GET'])
@login_required
def get_interactions():
    """Get user interaction history."""
    interaction_type = request.args.get('type')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    # Build query
    query = UserInteraction.query.filter_by(user_id=current_user.id)
    
    if interaction_type:
        query = query.filter_by(interaction_type=interaction_type)
    
    if start_date:
        start = datetime.fromisoformat(start_date)
        query = query.filter(UserInteraction.timestamp >= start)
    
    if end_date:
        end = datetime.fromisoformat(end_date)
        query = query.filter(UserInteraction.timestamp <= end)
    
    # Order by timestamp descending
    query = query.order_by(UserInteraction.timestamp.desc())
    
    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    # Format interactions with food details
    interactions = []
    for interaction in pagination.items:
        interaction_dict = interaction.to_dict()
        
        # Add food details if available
        if interaction.food_item_id:
            food = FoodItem.query.get(interaction.food_item_id)
            if food:
                interaction_dict['food'] = food.to_dict()
        
        interactions.append(interaction_dict)
    
    return jsonify({
        'interactions': interactions,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })


@interactions_bp.route('/api/interactions/analytics', methods=['GET'])
@login_required
def get_interaction_analytics():
    """Get analytics on user interactions."""
    days = request.args.get('days', 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get interaction counts by type
    interaction_counts = db.session.query(
        UserInteraction.interaction_type,
        func.count(UserInteraction.id).label('count')
    ).filter(
        UserInteraction.user_id == current_user.id,
        UserInteraction.timestamp >= start_date
    ).group_by(UserInteraction.interaction_type).all()
    
    # Get most viewed foods
    most_viewed = db.session.query(
        FoodItem,
        func.count(UserInteraction.id).label('view_count')
    ).join(
        UserInteraction, FoodItem.id == UserInteraction.food_item_id
    ).filter(
        UserInteraction.user_id == current_user.id,
        UserInteraction.interaction_type == 'view',
        UserInteraction.timestamp >= start_date
    ).group_by(FoodItem.id).order_by(
        func.count(UserInteraction.id).desc()
    ).limit(10).all()
    
    # Get liked foods
    liked_foods = db.session.query(FoodItem).join(
        UserInteraction, FoodItem.id == UserInteraction.food_item_id
    ).filter(
        UserInteraction.user_id == current_user.id,
        UserInteraction.interaction_type == 'like'
    ).all()
    
    # Get bookmarked foods
    bookmarked_foods = db.session.query(FoodItem).join(
        UserInteraction, FoodItem.id == UserInteraction.food_item_id
    ).filter(
        UserInteraction.user_id == current_user.id,
        UserInteraction.interaction_type == 'bookmark'
    ).all()
    
    # Get category distribution
    category_counts = db.session.query(
        FoodItem.category,
        func.count(UserInteraction.id).label('count')
    ).join(
        UserInteraction, FoodItem.id == UserInteraction.food_item_id
    ).filter(
        UserInteraction.user_id == current_user.id,
        UserInteraction.timestamp >= start_date
    ).group_by(FoodItem.category).all()
    
    return jsonify({
        'interaction_counts': {item[0]: item[1] for item in interaction_counts},
        'most_viewed': [
            {'food': food.to_dict(), 'view_count': count}
            for food, count in most_viewed
        ],
        'liked_foods': [food.to_dict() for food in liked_foods],
        'bookmarked_foods': [food.to_dict() for food in bookmarked_foods],
        'category_distribution': {item[0]: item[1] for item in category_counts},
        'total_interactions': sum(item[1] for item in interaction_counts),
        'period_days': days
    })


@interactions_bp.route('/history')
@login_required
def history_page():
    """Interaction history page."""
    return render_template('dashboard/history.html')
