"""Language switching routes."""
from flask import Blueprint, session, redirect, request, url_for
from flask_login import current_user
from models import db

language_bp = Blueprint('language', __name__)

# Language mapping
LANGUAGES = {
    'en': 'English',
    'te': 'తెలుగు',  # Telugu
}


@language_bp.route('/set/<lang_code>')
def set_language(lang_code):
    """Set the user's preferred language."""
    if lang_code in LANGUAGES:
        # Store in session
        session['language'] = lang_code
        
        # Also update user's language preference in database if logged in
        if current_user and current_user.is_authenticated:
            current_user.language = lang_code
            db.session.commit()
    
    # Redirect back to the referring page or home
    return redirect(request.referrer or url_for('index'))
