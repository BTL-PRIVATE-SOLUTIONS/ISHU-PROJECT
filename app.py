"""Main Flask application entry point."""
import os
from dotenv import load_dotenv
from flask import Flask, render_template, redirect, url_for, request, session
from flask_login import LoginManager, current_user
from flask_bcrypt import Bcrypt
from flask_babel import Babel
from config import config
from models import db
from models.user import User
from models.food import FoodItem
from models.interaction import UserInteraction
from models.recommendation import Recommendation

# Import blueprints
from routes.auth import auth_bp, bcrypt as auth_bcrypt
from routes.foods import foods_bp
from routes.interactions import interactions_bp


def create_app(config_name=None):
    """Create and configure the Flask application."""
    load_dotenv()
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development').strip()
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    auth_bcrypt.init_app(app)
    
    # Initialize Flask-Babel for multi-language support
    babel = Babel(app)
    
    def get_locale():
        """Determine the best language match from user preference or browser."""
        # First check if language is set in session
        if 'language' in session:
            return session['language']
        # Check if user is logged in and has language preference
        if current_user and current_user.is_authenticated and hasattr(current_user, 'language') and current_user.language:
            return current_user.language
        # Otherwise, try to guess from browser accept languages
        return request.accept_languages.best_match(['en', 'te'])
    
    babel.init_app(app, locale_selector=get_locale)
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))
    
    # Register existing blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(foods_bp, url_prefix='/foods')
    app.register_blueprint(interactions_bp, url_prefix='/interactions')
    
    # Register language blueprint
    try:
        from routes.language import language_bp
        app.register_blueprint(language_bp, url_prefix='/language')
    except ImportError:
        pass
    
    # Import and register new blueprints (lazy import to avoid circular dependencies)
    try:
        from routes.chatbot import chatbot_bp
        app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
    except ImportError:
        pass  # Chatbot blueprint not yet created
    
    try:
        from routes.chatbot_dos_donts import chatbot_dos_donts_bp
        app.register_blueprint(chatbot_dos_donts_bp, url_prefix='/chatbot-dos-donts')
    except ImportError as e:
        print(f"Warning: Could not load Do's and Don'Ts chatbot: {e}")
    
    try:
        from routes.meal_plans import meal_plans_bp
        app.register_blueprint(meal_plans_bp, url_prefix='/meal-plans')
    except ImportError as e:
        print(f"Warning: Could not load meal plans: {e}")
    
    # Add translation utility to template context
    from utils.translations import get_translation, get_language_name, LANGUAGES
    
    @app.context_processor
    def inject_translations():
        """Make translation functions available in all templates."""
        current_lang = session.get('language', 'en')
        return dict(
            get_translation=get_translation,
            get_language_name=get_language_name,
            supported_languages=LANGUAGES,
            current_language=current_lang,
            LANGUAGES=LANGUAGES
        )
    
    # Main routes
    @app.route('/')
    def index():
        """Landing page."""
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        return render_template('index.html')
    
    @app.route('/dashboard')
    def dashboard():
        """Main dashboard page."""
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        return render_template('dashboard/index.html')
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app


if __name__ == '__main__':
    app = create_app()
    port = app.config['PORT']
    # Flask will display startup info automatically
    # Disable debug mode to avoid reload issues
    debug_mode = False
    app.run(host='127.0.0.1', port=port, debug=debug_mode)
