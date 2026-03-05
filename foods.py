"""Food routes."""
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from models import db
from models.food import FoodItem
from models.interaction import UserInteraction
from utils.helpers import sanitize_search_query

foods_bp = Blueprint('foods', __name__)


@foods_bp.route('/api/foods', methods=['GET'])
@login_required
def get_foods():
    """Get all foods with optional filters."""
    # Get query parameters
    category = request.args.get('category')
    region = request.args.get('region')
    trimester = request.args.get('trimester', type=int)
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Build query
    query = FoodItem.query
    
    if category:
        query = query.filter_by(category=category)
    
    if region:
        query = query.filter_by(regional_origin=region)
    
    # Pagination
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    foods = [food.to_dict() for food in pagination.items]
    
    return jsonify({
        'foods': foods,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })


@foods_bp.route('/api/foods/<int:food_id>', methods=['GET'])
@login_required
def get_food(food_id):
    """Get specific food details."""
    food = FoodItem.query.get_or_404(food_id)
    
    # Log view interaction
    interaction = UserInteraction(
        user_id=current_user.id,
        interaction_type='view',
        food_item_id=food.id
    )
    db.session.add(interaction)
    db.session.commit()
    
    return jsonify(food.to_dict())


@foods_bp.route('/api/foods/search', methods=['GET'])
@login_required
def search_foods():
    """Search foods by name or description."""
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    if not query:
        return jsonify({'foods': [], 'total': 0})
    
    # Sanitize query
    query = sanitize_search_query(query)
    
    # Search in both English and Hindi names
    search_pattern = f'%{query}%'
    results = FoodItem.query.filter(
        (FoodItem.name_english.ilike(search_pattern)) |
        (FoodItem.name_hindi.ilike(search_pattern)) |
        (FoodItem.benefits.ilike(search_pattern))
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    # Log search interaction
    interaction = UserInteraction(
        user_id=current_user.id,
        interaction_type='search'
    )
    interaction.set_details({'query': query})
    db.session.add(interaction)
    db.session.commit()
    
    foods = [food.to_dict() for food in results.items]
    
    return jsonify({
        'foods': foods,
        'total': results.total,
        'pages': results.pages,
        'current_page': page,
        'query': query
    })


@foods_bp.route('/foods')
@login_required
def browse_foods():
    """Browse foods page."""
    categories = db.session.query(FoodItem.category).distinct().all()
    categories = [c[0] for c in categories]
    
    regions = db.session.query(FoodItem.regional_origin).distinct().all()
    regions = [r[0] for r in regions if r[0]]
    
    return render_template('dashboard/foods.html', categories=categories, regions=regions)
