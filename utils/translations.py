"""Translation utilities and language configurations - Complete multilingual support."""

# Language mapping with native names
LANGUAGES = {
    'en': {
        'name': 'English',
        'native': 'English',
        'flag': '🇬🇧'
    },
    'te': {
        'name': 'Telugu',
        'native': 'తెలుగు',
        'flag': '🇮🇳'
    }
}

# Comprehensive UI translations for the entire application
QUICK_TRANSLATIONS = {
    'en': {
        # Welcome & Greetings
        'welcome': 'Welcome',
        'welcome_back': 'Welcome back',
        'welcome_back_user': 'Welcome back, {}!',
        'hello': 'Hello',
        
        # Navigation & Main Menu
        'dashboard': 'Dashboard',
        'chatbot': 'Chatbot',
        'meal_plans': 'Meal Plans',
        'recommendations': 'Recommendations',
        'login': 'Login',
        'register': 'Register',
        'logout': 'Logout',
        'home': 'Home',
        'profile': 'Profile',
        'settings': 'Settings',
        
        # Dashboard Content
        'foods_available': 'Foods Available',
        'safe_foods': 'Safe Foods',
        'your_trimester': 'Your Trimester',
        'meal_plans_label': 'Meal Plans',
        'meal_plans_range': '1-30 Days',
        'quick_actions': 'Quick Actions',
        'choose_action': 'Choose what you\'d like to do today',
        
        # Chatbot Section
        'ai_nutrition_chatbot': 'AI Nutrition Chatbot',
        'ai_chatbot_description': 'Ask questions about food safety, benefits, and nutrition for your pregnancy',
        'ask_question': 'Ask a question...',
        'type_message': 'Type your message...',
        'send': 'Send',
        'ask_about_pregnancy': 'Ask questions about pregnancy nutrition',
        
        # Meal Planner Section
        'meal_planner': 'Meal Planner',
        'meal_planner_description': 'Generate personalized meal plans (1-30 days) tailored to your trimester',
        'generate_meal_plan': 'Generate Meal Plan',
        'meal_plan_settings': 'Plan Settings',
        'meal_number_of_days': 'Number of Days',
        'meal_days_range': '1-30 days',
        'meal_regional_preference': 'Regional Preference',
        'meal_all_regions': 'All Regions',
        'meal_diet_type': 'Diet Type',
        'meal_all_types': 'All Types',
        'meal_generate_plan': 'Generate Meal Plan',
        'meal_your_information': 'Your Information',
        'meal_customized_for_trimester': 'Meal plans are customized for your trimester',
        'meal_generating_plan': 'Generating your personalized meal plan...',
        'meal_no_plan_yet': 'No Meal Plan Yet',
        'meal_configure_to_start': 'Configure your preferences and click "Generate Meal Plan" to get started',
        'meal_your_plan': 'Your Meal Plan',
        'meal_day': 'Day',
        'meal_date': 'Date',
        'meal_breakfast': 'Breakfast',
        'meal_mid_morning': 'Mid-Morning',
        'meal_lunch': 'Lunch',
        'meal_evening': 'Evening',
        'meal_dinner': 'Dinner',
        'meal_calories': 'Calories',
        'meal_avg_calories': 'Avg Calories',
        'meal_avg_protein': 'Avg Protein',
        'meal_avg_carbs': 'Avg Carbs',
        'meal_avg_fat': 'Avg Fat',
        'print': 'Print',
        'loading': 'Loading',
        
        # Nutrition Tips
        'essential_nutrition_tips': 'Essential Nutrition Tips for Trimester {}',
        'folic_acid_tip': 'Focus on folic acid (600 mcg daily) - found in spinach, lentils, and fortified grains',
        'hydration_tip': 'Stay hydrated and eat small, frequent meals to manage nausea',
        'vitamin_b6_tip': 'Include foods rich in vitamin B6 to help with morning sickness',
        'foods_avoid_tip': 'Avoid raw or undercooked foods, unpasteurized dairy, and high-mercury fish',
        
        # Help & Support
        'need_help': 'Need Help?',
        'help_description': 'Have questions about using the app or pregnancy nutrition?',
        'contact_support': 'Contact Support',
        'view_help': 'View Help',
        
        # Safety Section
        'safety_first': 'Safety First',
        'safety_description': 'All recommendations are based on medical research and traditional wisdom.',
        'consult_doctor': 'Always consult your healthcare provider for medical advice',
        
        # Footer
        'app_name': 'Maternal Food AI',
        'app_description': 'AI-powered nutrition guidance for a healthy pregnancy journey. Combining modern science with traditional wisdom.',
        'quick_links': 'Quick Links',
        'important_note': 'Important Note',
        'disclaimer': 'This tool provides general nutritional information. Always consult your healthcare provider.',
        'copyright': '© 2026 Maternal Food Recommendation AI. Made with ❤️ for healthy pregnancies.',
        
        # Trimester Labels
        'trimester': 'Trimester',
        'first_trimester': 'First Trimester',
        'second_trimester': 'Second Trimester',
        'third_trimester': 'Third Trimester',
        'trimester_1': 'Trimester 1',
        'trimester_2': 'Trimester 2',
        'trimester_3': 'Trimester 3',
        
        # Meal Types
        'breakfast': 'Breakfast',
        'lunch': 'Lunch',
        'dinner': 'Dinner',
        'snacks': 'Snacks',
        'meals': 'Meals',
        
        # Nutrition Info
        'nutrients': 'Nutrients',
        'protein': 'Protein',
        'carbohydrates': 'Carbohydrates',
        'fats': 'Fats',
        'vitamins': 'Vitamins',
        'minerals': 'Minerals',
        'calories': 'Calories',
        'benefits': 'Benefits',
        'precautions': 'Precautions',
        'dietary_preferences': 'Dietary Preferences',
        'vegetarian': 'Vegetarian',
        'non_vegetarian': 'Non-Vegetarian',
        
        # User Info
        'full_name': 'Full Name',
        'email': 'Email',
        'password': 'Password',
        'confirm_password': 'Confirm Password',
        'age': 'Age',
        'height': 'Height',
        'weight': 'Weight',
        'current_trimester': 'Current Trimester',
        
        # Actions & Buttons
        'submit': 'Submit',
        'cancel': 'Cancel',
        'save': 'Save',
        'delete': 'Delete',
        'edit': 'Edit',
        'yes': 'Yes',
        'no': 'No',
        'ok': 'OK',
        'back': 'Back',
        'next': 'Next',
        'previous': 'Previous',
        'start': 'Start',
        'continue': 'Continue',
        'finish': 'Finish',
        'search': 'Search',
        'filter': 'Filter',
        'sort': 'Sort',
        
        # Messages
        'success': 'Success',
        'error': 'Error',
        'warning': 'Warning',
        'info': 'Info',
        'no_data': 'No data available',
        'loading': 'Loading...',
        'please_wait': 'Please wait...',
        'something_wrong': 'Something went wrong',
        'try_again': 'Try Again',
        
        # Chatbot Page
        'chatbot_title': 'AI Nutrition Chatbot',
        'chatbot_subtitle': 'Ask questions about pregnancy nutrition',
        'chatbot_greeting': 'Hello {name}! I\'m your pregnancy nutrition assistant.',
        'chatbot_can_help': 'I can help you with:',
        'chatbot_food_safety': '🛡️ Food safety during pregnancy',
        'chatbot_health_benefits': '💪 Health benefits of specific foods',
        'chatbot_nutrition_info': '🔬 Nutritional information',
        'chatbot_cooking_tips': '👨‍🍳 Preparation and cooking tips',
        'chatbot_portion_sizes': '📏 Portion sizes and quantities',
        'chatbot_precautions': '⚠️ Precautions and warnings',
        'chatbot_manage_conditions': '🏥 Managing pregnancy conditions',
        'chatbot_placeholder': 'Ask me about any food or pregnancy nutrition topic...',
        'chatbot_hint': 'Try: "Can I eat papaya?" or "Benefits of milk?" or "What helps with morning sickness?"',
        'chatbot_popular_questions': '🪄 Popular Questions',
        'chatbot_loading_suggestions': 'Loading suggestions...',
        'chatbot_no_suggestions': 'No suggestions available',
        'chatbot_recent_questions': '📜 Your Recent Questions',
        'chatbot_no_recent': 'Your questions will appear here',
        'chatbot_analyzing': 'Analyzing your question',
        'chatbot_could_not_get_answer': 'Could not get answer. Please try again.',
        'chatbot_error_occurred': 'Error: Sorry, something went wrong. Please try again.',
        'chatbot_you': '👤 You',
        'chatbot_bot': 'Chatbot',
    },
    'te': {
        # Welcome & Greetings
        'welcome': 'స్వాగతం',
        'welcome_back': 'తిరిగి స్వాగతం',
        'welcome_back_user': 'తిరిగి స్వాగతం, {}!',
        'hello': 'హలో',
        
        # Navigation & Main Menu
        'dashboard': 'డాష్‌బోర్డ్',
        'chatbot': 'చాట్‌బాట్',
        'meal_plans': 'భోజన ప్రణాళికలు',
        'recommendations': 'సిఫార్సులు',
        'login': 'లాగిన్',
        'register': 'నమోదు',
        'logout': 'లాగౌట్',
        'home': 'హోమ్',
        'profile': 'ప్రొఫైల్',
        'settings': 'సెట్టింగ్‌లు',
        
        # Dashboard Content
        'foods_available': 'ఉపలब్ధ ఆహారాలు',
        'safe_foods': 'సురక్షిత ఆహారాలు',
        'your_trimester': 'మీ త్రైమాసికం',
        'meal_plans_label': 'భోజన ప్రణాళికలు',
        'meal_plans_range': '1-30 రోజులు',
        'quick_actions': 'త్వరిత చర్యలు',
        'choose_action': 'ఈ రోజు మీరు ఏమి చేయాలనుకుంటున్నారో ఎంచుకోండి',
        
        # Chatbot Section
        'ai_nutrition_chatbot': 'AI పోషకాహార చాట్‌బాట్',
        'ai_chatbot_description': 'ఆహార సురక్ష, ప్రయోజనాలు మరియు గర్భధారణ పోషణ గురించి ప్రశ్నలు అడగండి',
        'ask_question': 'ప్రశ్న అడగండి...',
        'type_message': 'మీ సందేశాన్ని టైప్ చేయండి...',
        'send': 'పంపు',
        'ask_about_pregnancy': 'గర్భధారణ పోషణ గురించి ప్రశ్నలు అడగండి',
        
        # Meal Planner Section
        'meal_planner': 'భోజన ప్రణాళికకర్త',
        'meal_planner_description': 'మీ త్రైమాసికానికి맞춤 వ్యక్తిగతకృత భోజన ప్రణాళికలను (1-30 రోజులు) రూపొందించండి',
        'generate_meal_plan': 'భోజన ప్రణాళిక రూపొందించండి',
        'meal_plan_settings': 'ప్రణాళిక సెట్టింగ్‌లు',
        'meal_number_of_days': 'రోజుల సంఖ్య',
        'meal_days_range': '1-30 రోజులు',
        'meal_regional_preference': 'ప్రాంతీయ ప్రాధాన్యత',
        'meal_all_regions': 'అన్ని ప్రాంతాలు',
        'meal_diet_type': 'ఆహార రకం',
        'meal_all_types': 'అన్ని రకాలు',
        'meal_generate_plan': 'భోజన ప్రణాళిక రూపొందించండి',
        'meal_your_information': 'మీ సమాచారం',
        'meal_customized_for_trimester': 'భోజన ప్రణాళికలు మీ త్రైమాసికం కోసం కస్టమ్‌ఐజ్ చేయబడతాయి',
        'meal_generating_plan': 'మీ వ్యక్తిగతకృత భోజన ప్రణాళిక రూపొందించబడుతోంది...',
        'meal_no_plan_yet': 'ఇంకా ఎటువంటి భోజన ప్రణాళిక లేదు',
        'meal_configure_to_start': 'మీ ప్రాధాన్యతలను కాన్ఫిగర్ చేసి \"భోజన ప్రణాళిక రూపొందించండి\"ని క్లిక్ చేసి ప్రారంభించండి',
        'meal_your_plan': 'మీ భోజన ప్రణాళిక',
        'meal_day': 'రోజు',
        'meal_date': 'తేదీ',
        'meal_breakfast': 'ఉదయం భోజనం',
        'meal_mid_morning': 'మధ్యాహ్నం ఉదయం',
        'meal_lunch': 'భోజనం',
        'meal_evening': 'సాయంకాలం',
        'meal_dinner': 'రాత్రి భోజనం',
        'meal_calories': 'కేలరీలు',
        'meal_avg_calories': 'సగటు కేలరీలు',
        'meal_avg_protein': 'సగటు ప్రోటీన్',
        'meal_avg_carbs': 'సగటు కార్బోహైడ్రేట్‌లు',
        'meal_avg_fat': 'సగటు కొవ్వు',
        'print': 'ముద్రణ',
        'loading': 'లోడ్ చేస్తోంది',
        
        # Nutrition Tips
        'essential_nutrition_tips': 'త్రైమాసికం {} కోసం ముఖ్యమైన పోషణ చిట్కాలు',
        'folic_acid_tip': 'ఫోలిక్ ఆమ్లం (నిత్యం 600 mcg) పై దృష్టి సారించండి - పालక, దాలు మరియు చేతిలో తయారు చేసిన ధాన్యాలలో కనిపిస్తుంది',
        'hydration_tip': 'చికూ నిర్వహించడానికి తేమ ఉంచండి మరియు చిన్న, నిరంతర భోజనాలు తినండి',
        'vitamin_b6_tip': 'ఉదయం అనారోగ్య పని సహాయానికి విటమిన్ B6 సమృద్ధ ఆహారాలను చేర్చండి',
        'foods_avoid_tip': 'ముడి లేదా తక్కువ వంచిన ఆహారాలు, పాస్చర్రైజ్ చేయని జడ్జ మరియు అధిక-మెర్క్యూరీ చేపలను నివారించండి',
        
        # Help & Support
        'need_help': 'సహాయం కావాలా?',
        'help_description': 'అనువర్తనం ఉపయోగం లేదా గర్భధారణ పోషణ గురించి ప్రశ్నలు ఉన్నాయా?',
        'contact_support': 'సపోర్టుకు సంబంధం',
        'view_help': 'సహాయం వీక్షించండి',
        
        # Safety Section
        'safety_first': 'ముందు సురక్ష',
        'safety_description': 'సమస్త సిఫారసులు వైద్య పరిశోధన మరియు సాంప్రదాయ జ్ఞానం ఆధారంగా ఉన్నాయి.',
        'consult_doctor': 'విధానపరమైన సలహా కోసం ఎల్లప్పుడు మీ ఆరోగ్య సేవా ప్రదాతను సంప్రదించండి',
        
        # Footer
        'app_name': 'ప్రసూతి ఆహార AI',
        'app_description': 'ఆరోగ్యకరమైన గర్భధారణ ప్రయాణం కోసం AI-శక్తితో కూడిన పోషణ సలహా. ఆధునిక విజ్ఞానం మరియు సాంప్రదాయ జ్ఞానం కలయిక.',
        'quick_links': 'త్వరిత లింకులు',
        'important_note': 'ముఖ్యమైన గమనిక',
        'disclaimer': 'ఈ సాధనం సాధారణ పోషణ సమాచారాన్ని అందిస్తుంది. ఎల్లప్పుడు మీ ఆరోగ్య సేవా ప్రదాతను సంప్రదించండి.',
        'copyright': '© 2026 ప్రసూతి ఆహార సిఫారసు AI. ఆరోగ్యకరమైన గర్భధారణల కోసం ❤️తో తయారు చేయబడింది.',
        
        # Trimester Labels
        'trimester': 'త్రైమాసికం',
        'first_trimester': 'మొదటి త్రైమాసికం',
        'second_trimester': 'రెండవ త్రైమాసికం',
        'third_trimester': 'మూడవ త్రైమాసికం',
        'trimester_1': 'త్రైమాసికం 1',
        'trimester_2': 'త్రైమాసికం 2',
        'trimester_3': 'త్రైమాసికం 3',
        
        # Meal Types
        'breakfast': 'అల్పాహారం',
        'lunch': 'మధ్యాహ్న భోజనం',
        'dinner': 'రాత్రి భోజనం',
        'snacks': 'చిరుతిండి',
        'meals': 'భోజనాలు',
        
        # Nutrition Info
        'nutrients': 'పోషక తత్వాలు',
        'protein': 'ప్రోటీన్',
        'carbohydrates': 'కార్బోహైడ్రేట్‌లు',
        'fats': 'కొవ్వులు',
        'vitamins': 'విటమిన్‌లు',
        'minerals': 'ఖనిజాలు',
        'calories': 'కేలరీలు',
        'benefits': 'ప్రయోజనాలు',
        'precautions': 'జాగ్రత్తలు',
        'dietary_preferences': 'ఆహార ప్రాధాన్యతలు',
        'vegetarian': 'శాకాహారం',
        'non_vegetarian': 'మాంసాహారం',
        
        # User Info
        'full_name': 'పూర్తి పేరు',
        'email': 'ఇమెయిల్',
        'password': 'పాస్‌వర్డ్',
        'confirm_password': 'పాస్‌వర్డ్ నిర్ధారించండి',
        'age': 'వయస్సు',
        'height': 'ఎత్తు',
        'weight': 'బరువు',
        'current_trimester': 'ప్రస్తుత త్రైమాసికం',
        
        # Actions & Buttons
        'submit': 'సమర్పించు',
        'cancel': 'రద్దు చేయు',
        'save': 'సేవ్ చేయు',
        'delete': 'తొలగించు',
        'edit': 'సవరించు',
        'yes': 'అవును',
        'no': 'లేదు',
        'ok': 'సరి',
        'back': 'తిరిగి',
        'next': 'తరువాత',
        'previous': 'మునుపటి',
        'start': 'ప్రారంభించండి',
        'continue': 'కొనసాగించండి',
        'finish': 'ముగించండి',
        'search': 'వెతకండి',
        'filter': 'ఫిల్టర్',
        'sort': 'క్రమబద్ధ చేయు',
        
        # Messages
        'success': 'విజయం',
        'error': 'లోపం',
        'warning': 'హెచ్చరిక',
        'info': 'సమాచారం',
        'no_data': 'డేటా అందుబాటులో లేదు',
        'loading': 'లోడ్ అవుతోంది...',
        'please_wait': 'దయచేసి వేచి ఉండండి...',
        'something_wrong': 'ఏదో తప్పు జరిగింది',
        'try_again': 'మళ్లీ ప్రయత్నించండి',        
        # Chatbot Page
        'chatbot_title': 'AI పోషకాహార చాట్‌బాట్',
        'chatbot_subtitle': 'గర్భధారణ పోషణ గురించి ప్రశ్నలు అడగండి',
        'chatbot_greeting': 'హలో {name}! నేను మీ గర్భధారణ పోషణ సహాయకుడిని.',
        'chatbot_can_help': 'నేను మీకు సహాయం చేయగలను:',
        'chatbot_food_safety': '🛡️ గర్భధారణ సమయంలో ఆహార సురక్ష',
        'chatbot_health_benefits': '💪 నిర్దిష్ట ఆహారాల ఆరోగ్య ప్రయోజనాలు',
        'chatbot_nutrition_info': '🔬 పోషణ సమాచారం',
        'chatbot_cooking_tips': '👨‍🍳 ఉపయోగం మరియు వంట చిట్కాలు',
        'chatbot_portion_sizes': '📏 భోజన పరిమాణాలు మరియు మొత్తాలు',
        'chatbot_precautions': '⚠️ జాగ్రత్తలు మరియు హెచ్చరికలు',
        'chatbot_manage_conditions': '🏥 గర్భధారణ పరిస్థితులను నిర్వహించడం',
        'chatbot_placeholder': 'ఏ ఆహారం లేదా గర్భధారణ పోషణ విషయం గురించైనా నన్ను అడగండి...',
        'chatbot_hint': 'ప్రయత్నించండి: "నేను అప్పటిని తినవచ్చా?" లేదా "పాలు యొక్క ప్రయోజనాలు?" లేదా "ఉదయం అనారోగ్య పనిలో సహాయపడేది ఏమిటి?"',
        'chatbot_popular_questions': '🪄 ప్రసిద్ధ ప్రశ్నలు',
        'chatbot_loading_suggestions': 'సూచనలు లోడ్ అవుతోంది...',
        'chatbot_no_suggestions': 'సూచనలు అందుబాటులో లేవు',
        'chatbot_recent_questions': '📜 మీ ఇటీవలి ప్రశ్నలు',
        'chatbot_no_recent': 'మీ ప్రశ్నలు ఇక్కడ కనిపిస్తాయి',
        'chatbot_analyzing': 'మీ ప్రశ్నను విశ్లేషించుకుంటోంది',
        'chatbot_could_not_get_answer': 'సమాధానం పొందలేము. దయచేసి మళ్లీ ప్రయత్నించండి.',
        'chatbot_error_occurred': 'లోపం: క్షమించండి, ఏదో తప్పు జరిగింది. దయచేసి మళ్లీ ప్రయత్నించండి.',
        'chatbot_you': '👤 మీరు',
        'chatbot_bot': 'చాట్‌బాట్',    }
}


def get_translation(key, lang='en', **kwargs):
    """Get a translation for a given key and language.
    
    Args:
        key: Translation key
        lang: Language code ('en' or 'te')
        **kwargs: Format arguments for string formatting
    
    Returns:
        Translated string
    """
    if lang in QUICK_TRANSLATIONS and key in QUICK_TRANSLATIONS[lang]:
        translated = QUICK_TRANSLATIONS[lang][key]
        if kwargs:
            try:
                return translated.format(**kwargs)
            except (KeyError, ValueError):
                return translated
        return translated
    # Fallback to English
    translated = QUICK_TRANSLATIONS['en'].get(key, key)
    if kwargs:
        try:
            return translated.format(**kwargs)
        except (KeyError, ValueError):
            return translated
    return translated


def get_language_name(lang_code):
    """Get the native name of a language by its code."""
    if lang_code in LANGUAGES:
        return LANGUAGES[lang_code]['native']
    return 'English'
