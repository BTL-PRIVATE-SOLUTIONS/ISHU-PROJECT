"""Chatbot routes for AI-powered food recommendations with external API fallback."""
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime
from models import db
from models.interaction import UserInteraction

# Lazy load chatbots
_comprehensive_chatbot = None


def get_comprehensive_chatbot():
    """Get comprehensive chatbot with all datasets (lazy loading)."""
    global _comprehensive_chatbot
    if _comprehensive_chatbot is None:
        from ai_engine.comprehensive_chatbot import get_comprehensive_chatbot as load_chatbot
        _comprehensive_chatbot = load_chatbot()
    return _comprehensive_chatbot


chatbot_bp = Blueprint('chatbot', __name__)


@chatbot_bp.route('/')
@login_required
def chatbot_page():
    """Render chatbot interface page."""
    return render_template('dashboard/chatbot.html')


@chatbot_bp.route('/api/ask', methods=['POST'])
@login_required
def ask_question():
    """
    Answer user questions with FAST responses (< 3 seconds).
    Uses dataset + Gemini AI fallback.
    
    Expects JSON:
        {
            "question": "Can I eat eggs during pregnancy?",
            "trimester": 2 (optional)
        }
    
    Automatically returns Do's and Don'Ts format with fast response times.
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'success': False, 'error': 'Question is required'}), 400
        
        question = data['question'].strip()
        
        if not question or len(question) < 3:
            return jsonify({'success': False, 'error': 'Question too short (min 3 chars)'}), 400
        
        if len(question) > 500:
            return jsonify({'success': False, 'error': 'Question too long (max 500 chars)'}), 400
        
        # Get user context
        trimester = data.get('trimester')
        if trimester is None and hasattr(current_user, 'current_trimester'):
            trimester = current_user.current_trimester
        
        region = data.get('region')
        season = data.get('season')
        
        # Get comprehensive chatbot with all datasets + Gemini AI fallback
        try:
            chatbot = get_comprehensive_chatbot()
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': 'Chatbot initialization failed',
                'answer': 'Sorry, the chatbot is temporarily unavailable. Please try again later.'
            }), 500

        # Use structured answer to include dos/donts and intent metadata
        try:
            result = chatbot.answer_question_structured(
                question=question,
                trimester=trimester
            )
        except Exception as e:
            print(f"❌ Error generating answer for question '{question}': {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': 'Error generating answer',
                'answer': 'Sorry, I encountered an error processing your question. Please try rephrasing it or try again later.'
            }), 500
        
        # Determine AI backend used based on answer content
        ai_backend = 'rule_based'
        answer_text = result.get('answer', '')
        if 'BERT+Flan-T5' in answer_text:
            ai_backend = 'bert_flan_t5'
        elif 'AI-Powered Answer' in answer_text:
            ai_backend = 'ai_model'  # Gemini or LangChain
        elif result.get('source') == 'database_cache':
            ai_backend = 'database'
        
        # Inject region/season for logging context
        result['region'] = region
        result['season'] = season
        result['ai_backend'] = ai_backend
        
        # Log interaction
        try:
            interaction = UserInteraction(
                user_id=current_user.id,
                interaction_type='chatbot_query'
            )
            interaction.set_details({
                'question': question,
                'trimester': trimester,
                'source': result.get('source'),
                'response_time': result.get('response_time'),
                'answer_length': len(result.get('answer', '')),
                'keywords': result.get('keywords', []),
                'intent': result.get('intent')
            })
            
            db.session.add(interaction)
            db.session.commit()
        except Exception as e:
            print(f"⚠️ Could not log interaction: {e}")
            db.session.rollback()
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': result.get('answer', ''),
            'dos': result.get('dos', []),
            'donts': result.get('donts', []),
            'query_reflection': result.get('query_reflection', ''),
            'keywords': result.get('keywords', []),
            'intent': result.get('intent'),
            'source': result.get('source'),  # 'dataset', 'ai_model', 'fallback'
            'ai_backend': result.get('ai_backend', 'rule_based'),  # 'bert_flan_t5', 'ai_model', 'database', 'rule_based'
            'response_time': round(result.get('response_time', 0), 2),
            'trimester': trimester,
            'region': region,
            'season': season,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        import traceback
        print(f"❌ Error in chatbot ask: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Error processing question',
            'answer': 'Sorry, I encountered an error. Please try again.'
        }), 500


@chatbot_bp.route('/api/suggestions', methods=['GET'])
@login_required
def get_suggestions():
    """
    Get trimester-specific and contextual suggested questions.
    
    Returns:
        {
            "suggestions": ["Question 1", "Question 2", ...]
        }
    """
    try:
        trimester = current_user.current_trimester if hasattr(current_user, 'current_trimester') and current_user.current_trimester else 2
        region = current_user.region_preference if hasattr(current_user, 'region_preference') else None
        
        # Trimester-specific questions that are answerable by the chatbot
        trimester_questions = {
            1: [
                "What foods help with morning sickness?",
                "What should I eat in first trimester?",
                "Can I eat eggs during pregnancy?",
                "Which fruits are best for first trimester?",
                "What foods should I avoid in early pregnancy?",
                "Is fish safe during pregnancy?",
                "What are good sources of folic acid?",
                "Can I drink milk during pregnancy?"
            ],
            2: [
                f"What should I eat in trimester {trimester}?",
                "What foods should I avoid during pregnancy?",
                "Can I eat eggs during pregnancy?",
                "Is fish safe during pregnancy?",
                "What are good sources of iron?",
                "Which fruits are best for pregnancy?",
                "What foods help prevent anemia?",
                "Can I eat seafood during pregnancy?"
            ],
            3: [
                "What should I eat in third trimester?",
                "What foods should I avoid in late pregnancy?",
                "What foods help with energy in third trimester?",
                "Can I eat spicy food in third trimester?",
                "What are good sources of calcium?",
                "Which foods help prepare for labor?",
                "Is it safe to eat dates in third trimester?",
                "What foods prevent swelling during pregnancy?"
            ]
        }
        
        # Get trimester-specific questions
        base_suggestions = trimester_questions.get(trimester, trimester_questions[2])
        
        # Add region-specific question if region is set
        if region:
            base_suggestions.insert(0, f"What are good {region} Indian foods for pregnancy?")
        
        # Limit to 8 suggestions
        suggestions = base_suggestions[:8]
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'trimester': trimester,
            'region': region
        })
        
    except Exception as e:
        print(f"Error getting suggestions: {e}")
        return jsonify({
            'error': 'Could not load suggestions',
            'suggestions': [
                "What should I eat during pregnancy?",
                "What foods should I avoid?",
                "Can I eat eggs during pregnancy?",
                "Is fish safe during pregnancy?"
            ]
        }), 200  # Return 200 with default suggestions instead of error


@chatbot_bp.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """
    Get user's chat history.
    
    Query params:
        limit: Number of recent interactions (default: 20)
    
    Returns:
        {
            "history": [
                {
                    "id": 1,
                    "question": "...",
                    "timestamp": "2024-01-01T12:00:00",
                    "source": "dataset"
                },
                ...
            ]
        }
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        limit = min(limit, 100)  # Max 100 items
        
        # Get user's chatbot interactions
        interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='chatbot_query'
        ).order_by(
            UserInteraction.timestamp.desc()
        ).limit(limit).all()
        
        history = []
        for interaction in interactions:
            details = interaction.get_details()
            history.append({
                'id': interaction.id,
                'question': details.get('question', ''),
                'intent': details.get('intent', ''),
                'foods_mentioned': details.get('foods_mentioned', []),
                'timestamp': interaction.timestamp.isoformat()
            })
        
        return jsonify({
            'success': True,
            'history': history,
            'total': len(history)
        })
        
    except Exception as e:
        print(f"Error getting history: {e}")
        return jsonify({
            'error': 'Could not load chat history',
            'history': []
        }), 500
