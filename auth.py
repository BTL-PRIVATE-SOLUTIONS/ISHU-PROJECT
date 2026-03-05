"""Authentication routes."""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from datetime import datetime
from models import db
from models.user import User
from utils.validators import validate_email, validate_password, validate_username, validate_due_date, validate_trimester
from utils.helpers import calculate_trimester_from_due_date

auth_bp = Blueprint('auth', __name__)
bcrypt = Bcrypt()


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Get form data
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validate inputs
        errors = []
        
        # Username validation
        if not username:
            errors.append("Username is required")
        else:
            is_valid, msg = validate_username(username)
            if not is_valid:
                errors.append(msg)
            elif User.query.filter_by(username=username).first():
                errors.append("Username already taken")
        
        # Email validation
        if not validate_email(email):
            errors.append("Invalid email format")
        elif User.query.filter_by(email=email).first():
            errors.append("Email already registered")
        
        # Password validation
        if password != confirm_password:
            errors.append("Passwords do not match")
        else:
            is_valid, msg = validate_password(password)
            if not is_valid:
                errors.append(msg)
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('auth/register.html')
        
        # Create user
        user = User(
            username=username,
            email=email,
            full_name=username,  # Use username as default display name (can be updated in profile)
            password_hash=bcrypt.generate_password_hash(password).decode('utf-8')
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username_or_email = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)
        
        # Find user by username or email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            # Update last login
            user.last_login = datetime.utcnow()
            
            # Update trimester based on due date
            if user.due_date:
                user.current_trimester = calculate_trimester_from_due_date(user.due_date)
            
            db.session.commit()
            
            login_user(user, remember=remember)
            flash(f'Welcome back, {user.full_name}!', 'success')
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username/email or password', 'error')
    
    return render_template('auth/login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """User logout."""
    logout_user()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('index'))


@auth_bp.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    """Get user profile API."""
    return jsonify(current_user.to_dict())


@auth_bp.route('/api/profile', methods=['PUT'])
@login_required
def update_profile():
    """Update user profile API."""
    data = request.get_json()
    
    errors = []
    
    # Update full name
    if 'full_name' in data:
        full_name = data['full_name'].strip()
        if len(full_name) >= 2:
            current_user.full_name = full_name
        else:
            errors.append("Full name must be at least 2 characters")
    
    # Update due date
    if 'due_date' in data:
        is_valid, result = validate_due_date(data['due_date'])
        if is_valid:
            current_user.due_date = result
            current_user.current_trimester = calculate_trimester_from_due_date(result)
        else:
            errors.append(result)
    
    # Update dietary preferences
    if 'dietary_preferences' in data:
        if data['dietary_preferences'] in ['vegetarian', 'vegan', 'non-vegetarian']:
            current_user.dietary_preferences = data['dietary_preferences']
        else:
            errors.append("Invalid dietary preference")
    
    # Update health conditions
    if 'health_conditions' in data:
        current_user.set_health_conditions(data['health_conditions'])
    
    if errors:
        return jsonify({'error': ', '.join(errors)}), 400
    
    db.session.commit()
    return jsonify(current_user.to_dict())
