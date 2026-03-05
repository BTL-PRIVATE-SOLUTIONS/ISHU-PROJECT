"""Route for the enhanced Do's and Don'ts Chatbot."""
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from models import db
from models.interaction import UserInteraction
from datetime import datetime

chatbot_dos_donts_bp = Blueprint('chatbot_dos_donts', __name__)

# Initialize chatbot globally with lazy loading
_chatbot = None


def get_chatbot():
    """Get the comprehensive chatbot with all datasets (lazy loading)."""
    global _chatbot
    if _chatbot is None:
        from ai_engine.comprehensive_chatbot import get_comprehensive_chatbot
        _chatbot = get_comprehensive_chatbot()
    return _chatbot


@chatbot_dos_donts_bp.route('/ask', methods=['POST'])
@login_required
def ask_question():
    """
    Ask chatbot a question and get FAST Do's and Don'Ts answer.
    Response time: < 3 seconds (uses dataset cache + external AI fallback)
    
    Expected JSON:
    {
        "question": "Can I eat raw papaya during pregnancy?",
        "trimester": 2  (optional)
    }
    """
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        trimester = data.get('trimester')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        if len(question) < 3:
            return jsonify({'error': 'Question too short'}), 400
        
        if len(question) > 500:
            return jsonify({'error': 'Question too long (max 500 chars)'}), 400
        
        # Get trimester from user if not provided
        if trimester is None and hasattr(current_user, 'current_trimester'):
            trimester = current_user.current_trimester
        
        # Get comprehensive chatbot
        chatbot = get_chatbot()
        
        # Get detailed answer from all datasets with Do's and Don'Ts structure
        import time
        start_time = time.time()
        
        response_data = chatbot.answer_question_structured(
            question=question,
            trimester=trimester
        )
        
        response_time = time.time() - start_time
        
        # Log interaction to database
        try:
            interaction = UserInteraction(
                user_id=current_user.id,
                interaction_type='chatbot_question'
            )
            interaction.set_details({
                'question': question,
                'trimester': trimester,
                'source': response_data['source'],
                'response_time': response_time,
                'dos_count': len(response_data.get('dos', [])),
                'donts_count': len(response_data.get('donts', []))
            })
            
            db.session.add(interaction)
            db.session.commit()
        except Exception as e:
            print(f"⚠️ Could not log interaction: {e}")
        
        return jsonify({
            'success': True,
            'query_reflection': response_data.get('query_reflection', ''),
            'question': question,
            'answer': response_data.get('answer', ''),
            'dos': response_data.get('dos', []),
            'donts': response_data.get('donts', []),
            'source': response_data.get('source', 'comprehensive_dataset'),
            'response_time': round(response_time, 2),
            'trimester': trimester,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@chatbot_dos_donts_bp.route('/get-dos-donts', methods=['POST'])
@login_required
def get_dos_donts():
    """
    Get specific Do's and Don'Ts for a topic.
    
    Expected JSON:
    {
        "topic": "papaya",
        "trimester": 2
    }
    """
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        trimester = data.get('trimester')
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        chatbot = get_chatbot()
        
        # Get trimester from user if not provided
        if trimester is None and hasattr(current_user, 'current_trimester') and current_user.current_trimester:
            trimester = current_user.current_trimester
        
        # Get Do's and Don'Ts for the topic
        response = chatbot.get_dos_donts_answer(topic, trimester)
        
        return jsonify({
            'success': True,
            'topic': topic,
            'dos': response.get('dos', []),
            'donts': response.get('donts', []),
            'found': response.get('found', False),
            'source': response.get('source', 'database'),
            'trimester': trimester,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@chatbot_dos_donts_bp.route('/get-trimester-recommendations', methods=['GET'])
@login_required
def get_trimester_recommendations():
    """Get all Do's and Don'Ts for a specific trimester."""
    try:
        trimester = request.args.get('trimester', type=int)
        
        if trimester is None and hasattr(current_user, 'current_trimester'):
            trimester = current_user.current_trimester
        
        if trimester not in [1, 2, 3]:
            return jsonify({'error': 'Invalid trimester. Must be 1, 2, or 3'}), 400
        
        chatbot = get_chatbot()
        
        # Get trimester-specific recommendations
        trimester_foods = chatbot.knowledge_base.get('trimester_specific', {}).get(trimester, [])
        
        dos = []
        donts = []
        
        # Process trimester-specific data
        for item in trimester_foods:
            # Try to classify as do or dont
            if isinstance(item, dict):
                item_str = str(item).lower()
                if 'eat' in item_str or 'good' in item_str:
                    dos.append(item)
                else:
                    donts.append(item)
        
        # Fallback to general dos/donts if no trimester-specific items
        if not dos and not donts:
            dos = chatbot.knowledge_base.get('dos_donts', {}).get('dos', [])[:5]
            donts = chatbot.knowledge_base.get('dos_donts', {}).get('donts', [])[:5]
        
        return jsonify({
            'success': True,
            'trimester': trimester,
            'dos': dos,
            'donts': donts,
            'total_recommendations': len(dos) + len(donts),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@chatbot_dos_donts_bp.route('/search-items', methods=['GET'])
def search_items():
    """Search for food items in Do's and Don'Ts database."""
    try:
        query = request.args.get('q', '').lower().strip()
        
        if not query or len(query) < 2:
            return jsonify({
                'success': False,
                'error': 'Query must be at least 2 characters'
            }), 400
        
        chatbot = get_chatbot()
        
        results = {
            'dos': [],
            'donts': []
        }
        
        # Search in foods to eat (do's)
        for food_name, food_info in chatbot.knowledge_base.get('foods_to_eat', {}).items():
            if query in food_name.lower():
                results['dos'].append({
                    'item': food_name.title(),
                    'description': food_info.get('benefit', food_info.get('food_group', 'Food item')),
                    'category': food_info.get('food_group', 'General')
                })
        
        # Search in foods to avoid (don't's)
        for food_name, food_info in chatbot.knowledge_base.get('foods_to_avoid', {}).items():
            if query in food_name.lower():
                results['donts'].append({
                    'item': food_name.title(),
                    'description': food_info.get('risk', food_info.get('category', 'Food to avoid')),
                    'category': food_info.get('category', 'General')
                })
        
        # Search in do's and don'ts dataset
        for do_item in chatbot.knowledge_base.get('dos_donts', {}).get('dos', []):
            if query in str(do_item.get('item', '')).lower() or query in str(do_item.get('description', '')).lower():
                results['dos'].append({
                    'item': do_item.get('item', 'Food').title(),
                    'description': do_item.get('description', 'Recommended food'),
                    'category': do_item.get('category', 'General')
                })
        
        for dont_item in chatbot.knowledge_base.get('dos_donts', {}).get('donts', []):
            if query in str(dont_item.get('item', '')).lower() or query in str(dont_item.get('description', '')).lower():
                results['donts'].append({
                    'item': dont_item.get('item', 'Food').title(),
                    'description': dont_item.get('description', 'Food to avoid'),
                    'category': dont_item.get('category', 'General')
                })
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'total_results': len(results['dos']) + len(results['donts']),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
