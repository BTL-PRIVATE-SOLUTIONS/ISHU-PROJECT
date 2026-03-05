"""Chatbot routes for AI-powered food recommendations with external API fallback."""
from flask import Blueprint, render_template, request, jsonify, session as flask_session
from flask_login import login_required, current_user
from datetime import datetime
from models import db
from models.interaction import UserInteraction

# Lazy-loaded ChatbotEngine singleton
_chatbot_engine = None


def get_chatbot_engine():
    """Get the ChatbotEngine singleton (lazy loading)."""
    global _chatbot_engine
    if _chatbot_engine is None:
        from ai_engine.chatbot_engine import get_chatbot_engine as load_engine
        _chatbot_engine = load_engine()
    return _chatbot_engine


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
    Answer user questions using the ChatbotEngine (dataset + AI fallback).

    Expects JSON:
        {
            "question": "Can I eat eggs during pregnancy?",
            "trimester": 2 (optional),
            "language": "en" (optional, "en" or "te")
        }

    Returns Do's and Don'Ts format, domain restriction, diabetes follow-up,
    and a question_id for feedback/regeneration.
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
        language = data.get('language', 'en')

        # Use a stable session ID keyed to the current Flask session
        session_id = flask_session.get('chatbot_session_id')
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
            flask_session['chatbot_session_id'] = session_id

        # Initialise ChatbotEngine
        try:
            engine = get_chatbot_engine()
        except Exception as e:
            print(f"❌ Failed to initialize chatbot engine: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': 'Chatbot initialization failed',
                'answer': 'Sorry, the chatbot is temporarily unavailable. Please try again later.'
            }), 500

        # Generate answer
        try:
            result = engine.ask(
                question=question,
                trimester=trimester,
                region=region,
                season=season,
                session_id=session_id,
                language=language,
            )
        except Exception as e:
            print(f"❌ Error generating answer for '{question}': {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': 'Error generating answer',
                'answer': 'Sorry, I encountered an error processing your question. Please try rephrasing it or try again later.'
            }), 500

        # Log interaction (non-blocking)
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
                'intent': result.get('intent'),
                'question_id': result.get('question_id'),
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
            'intent': result.get('intent', ''),
            'source': result.get('source', ''),
            'question_id': result.get('question_id', ''),
            'session_id': session_id,
            'awaiting_followup': result.get('awaiting_followup', False),
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


@chatbot_bp.route('/api/feedback', methods=['POST'])
@login_required
def submit_feedback():
    """
    Record user feedback for a chatbot response.

    Expects JSON:
        {
            "question_id": "uuid-string",
            "rating": 4  (1-5, where 5 is most helpful)
        }
    """
    try:
        data = request.get_json()
        if not data or 'question_id' not in data or 'rating' not in data:
            return jsonify({'success': False, 'error': 'question_id and rating are required'}), 400

        rating = int(data['rating'])
        if rating < 1 or rating > 5:
            return jsonify({'success': False, 'error': 'rating must be between 1 and 5'}), 400

        try:
            engine = get_chatbot_engine()
        except Exception as e:
            return jsonify({'success': False, 'error': 'Chatbot not available'}), 500

        recorded = engine.record_feedback(data['question_id'], rating)
        if not recorded:
            return jsonify({'success': False, 'error': 'question_id not found'}), 404

        return jsonify({'success': True, 'message': 'Feedback recorded. Thank you!'}), 200

    except Exception as e:
        print(f"❌ Error recording feedback: {e}")
        return jsonify({'success': False, 'error': 'Error recording feedback'}), 500


@chatbot_bp.route('/api/regenerate', methods=['POST'])
@login_required
def regenerate_answer():
    """
    Regenerate a chatbot response using a different source tier.

    Expects JSON:
        {
            "question_id": "uuid-string"
        }
    """
    try:
        data = request.get_json()
        if not data or 'question_id' not in data:
            return jsonify({'success': False, 'error': 'question_id is required'}), 400

        try:
            engine = get_chatbot_engine()
        except Exception as e:
            return jsonify({'success': False, 'error': 'Chatbot not available'}), 500

        result = engine.regenerate(data['question_id'])
        if result is None:
            return jsonify({'success': False, 'error': 'Original question not found for regeneration'}), 404

        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 500

        return jsonify({
            'success': True,
            'question': result.get('question', ''),
            'answer': result.get('answer', ''),
            'dos': result.get('dos', []),
            'donts': result.get('donts', []),
            'query_reflection': result.get('query_reflection', ''),
            'keywords': result.get('keywords', []),
            'intent': result.get('intent', ''),
            'source': result.get('source', ''),
            'question_id': result.get('question_id', ''),
            'awaiting_followup': result.get('awaiting_followup', False),
            'response_time': round(result.get('response_time', 0), 2),
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error regenerating answer: {e}")
        return jsonify({'success': False, 'error': 'Error regenerating answer'}), 500


@chatbot_bp.route('/api/suggestions', methods=['GET'])
@login_required
def get_suggestions():
    """
    Get trimester-specific and contextual suggested questions covering all query formats.

    Returns:
        {
            "suggestions": ["Question 1", "Question 2", ...]
        }
    """
    try:
        trimester = (
            current_user.current_trimester
            if hasattr(current_user, 'current_trimester') and current_user.current_trimester
            else 2
        )
        region = getattr(current_user, 'region_preference', None)

        # Comprehensive trimester-specific questions in various formats
        trimester_questions = {
            1: [
                "What foods help with morning sickness?",
                "What should I eat in first trimester?",
                "Which fruits are best for first trimester?",
                "What are good sources of folic acid?",
                "What foods should I avoid in early pregnancy?",
                "Is fish safe during pregnancy?",
                "How much water should I drink during pregnancy?",
                "What snacks are healthy for early pregnancy?",
            ],
            2: [
                f"What should I eat in trimester {trimester}?",
                "What foods help prevent anemia during pregnancy?",
                "Which vegetables are best during second trimester?",
                "What are good sources of iron for pregnancy?",
                "How much protein do I need during pregnancy?",
                "What foods should I avoid during pregnancy?",
                "How do I manage heartburn during pregnancy?",
                "What foods help with pregnancy swelling?",
            ],
            3: [
                "What should I eat in third trimester?",
                "Which foods help prepare for labor?",
                "Is it safe to eat dates in third trimester?",
                "What foods give energy in late pregnancy?",
                "What are good sources of calcium for third trimester?",
                "What foods prevent swelling during pregnancy?",
                "What should I eat in the last month of pregnancy?",
                "How do I manage constipation during pregnancy?",
            ],
        }

        base_suggestions = list(trimester_questions.get(trimester, trimester_questions[2]))

        # Prepend region-specific suggestion if available
        if region:
            base_suggestions.insert(0, f"What are good {region} Indian foods for pregnancy?")

        return jsonify({
            'success': True,
            'suggestions': base_suggestions[:8],
            'trimester': trimester,
            'region': region,
        })

    except Exception as e:
        print(f"Error getting suggestions: {e}")
        return jsonify({
            'error': 'Could not load suggestions',
            'suggestions': [
                "What should I eat during pregnancy?",
                "What foods should I avoid?",
                "What are good sources of iron?",
                "How much water should I drink?",
            ],
        }), 200  # Return 200 with default suggestions on error


@chatbot_bp.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """
    Get user's chat history.

    Query params:
        limit: Number of recent interactions (default: 20)

    Returns:
        {
            "history": [{"id": 1, "question": "...", "timestamp": "...", "source": "dataset"}, ...]
        }
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        limit = min(limit, 100)  # Max 100 items

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
                'source': details.get('source', ''),
                'question_id': details.get('question_id', ''),
                'timestamp': interaction.timestamp.isoformat(),
            })

        return jsonify({
            'success': True,
            'history': history,
            'total': len(history),
        })

    except Exception as e:
        print(f"Error getting history: {e}")
        return jsonify({
            'error': 'Could not load chat history',
            'history': [],
        }), 500
