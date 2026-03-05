"""Language translation and localization utilities."""
import json
from enum import Enum


class Language(Enum):
    """Supported languages."""
    ENGLISH = 'english'
    HINDI = 'hindi'
    TELUGU = 'telugu'
    KANNADA = 'kannada'
    MALAYALAM = 'malayalam'
    TAMIL = 'tamil'


class LanguageManager:
    """Manages language translations and localization."""
    
    SUPPORTED_LANGUAGES = [lang.value for lang in Language]
    
    # UI Strings translations
    TRANSLATIONS = {
        'english': {
            # Navigation & Headers
            'dashboard': 'Dashboard',
            'recommendations': 'Recommendations',
            'meal_plans': 'Meal Plans',
            'chatbot': 'Chatbot',
            'logout': 'Logout',
            'welcome': 'Welcome',
            'profile': 'Profile',
            
            # Recommendation Page
            'personalized_recommendations': 'Personalized Food Recommendations',
            'based_on_trimester': 'Based on your trimester and preferences',
            'meal_type': 'Meal Type',
            'select_meal_type': 'Select meal type',
            'breakfast': 'Breakfast',
            'lunch': 'Lunch',
            'dinner': 'Dinner',
            'snacks': 'Snacks',
            'all_meals': 'All Meals',
            'get_recommendations': 'Get Recommendations',
            'loading': 'Loading...',
            'no_recommendations': 'No recommendations available',
            'current_trimester': 'Current Trimester',
            
            # Food Card Information
            'food_name': 'Food Name',
            'category': 'Category',
            'benefits': 'Benefits',
            'preparation_tips': 'Preparation Tips',
            'precautions': 'Precautions',
            'nutritional_info': 'Nutritional Information',
            'protein': 'Protein',
            'calories': 'Calories',
            'carbs': 'Carbohydrates',
            'fiber': 'Fiber',
            'iron': 'Iron',
            'calcium': 'Calcium',
            'folic_acid': 'Folic Acid',
            
            # Ratings
            'excellent': 'Excellent',
            'very_good': 'Very Good',
            'good': 'Good',
            'fair': 'Fair',
            'recommendation_score': 'Recommendation Score',
            'suitability_score': 'Suitability Score',
            
            # Feedback
            'helpful': 'Helpful',
            'not_helpful': 'Not Helpful',
            'feedback_submitted': 'Thank you for your feedback!',
            
            # Warnings
            'warnings': 'Warnings',
            'precaution': 'Caution',
            'consult_doctor': 'Please consult your doctor before consuming',
            
            # Messages
            'success': 'Success',
            'error': 'Error',
            'loading_error': 'Failed to load recommendations',
            'dietary_preference': 'Dietary Preference',
        },
        'hindi': {
            # Navigation & Headers
            'dashboard': 'डैशबोर्ड',
            'recommendations': 'सिफारिशें',
            'meal_plans': 'भोजन योजनाएं',
            'chatbot': 'चैटबॉट',
            'logout': 'लॉगआउट',
            'welcome': 'स्वागत है',
            'profile': 'प्रोफाइल',
            
            # Recommendation Page
            'personalized_recommendations': 'व्यक्तिगत खाद्य सिफारिशें',
            'based_on_trimester': 'आपकी तिमाही और प्राथमिकताओं के आधार पर',
            'meal_type': 'भोजन का प्रकार',
            'select_meal_type': 'भोजन का प्रकार चुनें',
            'breakfast': 'नाश्ता',
            'lunch': 'दोपहर का भोजन',
            'dinner': 'रात का खाना',
            'snacks': 'नाश्ते',
            'all_meals': 'सभी भोजन',
            'get_recommendations': 'सिफारिशें प्राप्त करें',
            'loading': 'लोड हो रहा है...',
            'no_recommendations': 'कोई सिफारिशें उपलब्ध नहीं',
            'current_trimester': 'वर्तमान तिमाही',
            
            # Food Card Information
            'food_name': 'खाद्य नाम',
            'category': 'श्रेणी',
            'benefits': 'लाभ',
            'preparation_tips': 'तैयारी के टिप्स',
            'precautions': 'सावधानियां',
            'nutritional_info': 'पोषण संबंधी जानकारी',
            'protein': 'प्रोटीन',
            'calories': 'कैलोरी',
            'carbs': 'कार्बोहाइड्रेट',
            'fiber': 'फाइबर',
            'iron': 'लोहा',
            'calcium': 'कैल्शियम',
            'folic_acid': 'फोलिक एसिड',
            
            # Ratings
            'excellent': 'उत्कृष्ट',
            'very_good': 'बहुत अच्छा',
            'good': 'अच्छा',
            'fair': 'उचित',
            'recommendation_score': 'सिफारिश स्कोर',
            'suitability_score': 'उपयुक्तता स्कोर',
            
            # Feedback
            'helpful': 'मददगार',
            'not_helpful': 'मददगार नहीं',
            'feedback_submitted': 'आपकी प्रतिक्रिया के लिए धन्यवाद!',
            
            # Warnings
            'warnings': 'चेतावनियाँ',
            'precaution': 'सावधान',
            'consult_doctor': 'सेवन करने से पहले अपने डॉक्टर से परामर्श लें',
            
            # Messages
            'success': 'सफल',
            'error': 'त्रुटि',
            'loading_error': 'सिफारिशें लोड करने में विफल',
            'dietary_preference': 'आहार संबंधी प्राथमिकता',
        },
        'telugu': {
            # Navigation & Headers
            'dashboard': 'డ్యాష్‌బోర్డ్',
            'recommendations': 'సిఫారిషులు',
            'meal_plans': 'భోజన ప్రణాలికలు',
            'chatbot': 'చాట్‌బాట్',
            'logout': 'లాగআఉట్',
            'welcome': 'స్వాగతం',
            'profile': 'ప్రొఫైల్',
            
            # Recommendation Page
            'personalized_recommendations': 'వ్యక్తిగత ఆహార సిఫారిషులు',
            'based_on_trimester': 'మీ త్రైమాసికం మరియు ప్రాధాన్యతల ఆధారంగా',
            'meal_type': 'భోజన రకం',
            'select_meal_type': 'భోజన రకం ఎంచుకోండి',
            'breakfast': ' breakfast',
            'lunch': 'మధ్యాహ్న భోజనం',
            'dinner': 'రాత్రి భోజనం',
            'snacks': 'చిరుతిండులు',
            'all_meals': 'అన్ని భోజనాలు',
            'get_recommendations': 'సిఫారిషులను పొందండి',
            'loading': 'లోడ్ చేస్తున్నది...',
            'no_recommendations': 'సిఫారిషులు అందుబాటులో లేవు',
            'current_trimester': 'ప్రస్తుత త్రైమాసికం',
            
            # Food Card Information
            'food_name': 'ఆహార పేరు',
            'category': 'వర్గం',
            'benefits': 'ప్రయోజనాలు',
            'preparation_tips': 'తయారీ చిట్కాలు',
            'precautions': 'జాగ్రత్తలు',
            'nutritional_info': 'పోషక సమాచారం',
            'protein': 'ప్రోటీన్',
            'calories': 'కేలరీలు',
            'carbs': 'కార్బోహైడ్రేట్‌లు',
            'fiber': 'ఫైబర్',
            'iron': 'ఇనుము',
            'calcium': 'కాల్షియం',
            'folic_acid': 'ఫోలిక్ ఆసిడ్',
            
            # Ratings
            'excellent': 'అద్భుతమైన',
            'very_good': 'చాలా మంచిది',
            'good': 'మంచిది',
            'fair': 'సరిపోతుంది',
            'recommendation_score': 'సిఫారిష్ స్కోర్',
            'suitability_score': 'ఉపయుక్తతా స్కోర్',
            
            # Feedback
            'helpful': 'సహాయకమైన',
            'not_helpful': 'సహాయకం కానిది',
            'feedback_submitted': 'మీ ఫీడ్‌బ్యాక్ కోసం ధన్యవాదాలు!',
            
            # Warnings
            'warnings': 'హెచ్చరికలు',
            'precaution': 'జాగ్రత్త',
            'consult_doctor': 'తీసుకోకముందు మీ డాక్టర్‌ను సంప్రదించండి',
            
            # Messages
            'success': 'విజయం',
            'error': 'లోపం',
            'loading_error': 'సిఫారిషులను లోడ్ చేయడం విఫలమైంది',
            'dietary_preference': 'ఆహార ప్రాధాన్యత',
        },
        'kannada': {
            # Navigation & Headers
            'dashboard': 'ಡ್ಯಾಶ್‌ಬೋರ್ಡ್',
            'recommendations': 'ಸಿಫಾರಿಸುಗಳು',
            'meal_plans': 'ಊಟದ ಯೋಜನೆಗಳು',
            'chatbot': 'ಚಾಟ್‌ಬಾಟ್',
            'logout': 'ಲಾಗ್‌ಔಟ್',
            'welcome': 'ಸ್ವಾಗತ',
            'profile': 'ಪ್ರೊಫೈಲ್',
            
            # Recommendation Page
            'personalized_recommendations': 'ವ್ಯಕ್ತಿಗತ ಆಹಾರ ಸಿಫಾರಿಸುಗಳು',
            'based_on_trimester': 'ನಿಮ್ಮ ತ್ರೈಮಾಸಿಕ ಮತ್ತು ಆದ್ಯತೆಗಳ ಆಧಾರದ ಮೇಲೆ',
            'meal_type': 'ಊಟದ ಪ್ರಕಾರ',
            'select_meal_type': 'ಊಟದ ಪ್ರಕಾರ ಆಯ್ಕೆ ಮಾಡಿ',
            'breakfast': 'ಬೆಳಗಿನ ತಿಂಡಿ',
            'lunch': 'ಮಧ್ಯಾಹ್ನ ಊಟ',
            'dinner': 'ರಾತ್ರಿ ಊಟ',
            'snacks': 'ತಿಂಡಿ',
            'all_meals': 'ಎಲ್ಲಾ ಊಟಗಳು',
            'get_recommendations': 'ಸಿಫಾರಿಸುಗಳನ್ನು ಪಡೆಯಿರಿ',
            'loading': 'ಲೋಡ್ ಆಗುತ್ತಿದೆ...',
            'no_recommendations': 'ಯಾವುದೇ ಸಿಫಾರಿಸುಗಳು ಲಭ್ಯವಿಲ್ಲ',
            'current_trimester': 'ಪ್ರಸ್ತುತ ತ್ರೈಮಾಸಿಕ',
            
            # Food Card Information
            'food_name': 'ಆಹಾರ ಹೆಸರು',
            'category': 'ವರ್ಗ',
            'benefits': 'ಪ್ರಯೋಜನಗಳು',
            'preparation_tips': 'ತಯಾರಿ ಸಲಹೆಗಳು',
            'precautions': 'ಎಚ್ಚರಿಕೆಗಳು',
            'nutritional_info': 'ಪೌಷ್ಟಿಕ ಮಾಹಿತಿ',
            'protein': 'ಪ್ರೋಟೀನ್',
            'calories': 'ಕ್ಯಾಲೋರಿಗಳು',
            'carbs': 'ಕಾರ್ಬೋಹೈಡ್ರೇಟ್‌ಗಳು',
            'fiber': 'ಫೈಬರ್',
            'iron': 'ಕಬ್ಬಿಣ',
            'calcium': 'ಕ್ಯಾಲ್ಸಿಯಮ್',
            'folic_acid': 'ಫೋಲಿಕ್ ಆಮ್ಲ',
            
            # Ratings
            'excellent': 'ಶ್ರೇಷ್ಠ',
            'very_good': 'ಬಹಳ ಚೆನ್ನಾಗಿದೆ',
            'good': 'ಚೆನ್ನಾಗಿದೆ',
            'fair': 'ಯುಕ್ತವಾಗಿದೆ',
            'recommendation_score': 'ಸಿಫಾರಿಸು ಸ್ಕೋರ್',
            'suitability_score': 'ಸೂಕ್ತತೆಯ ಸ್ಕೋರ್',
            
            # Feedback
            'helpful': 'ಸಹಾಯಕ',
            'not_helpful': 'ಸಹಾಯಕ ಅಲ್ಲ',
            'feedback_submitted': 'ನಿಮ್ಮ ಪ್ರತಿಕ್ರಿಯೆಗೆ ಧನ್ಯವಾದಗಳು!',
            
            # Warnings
            'warnings': 'ಎಚ್ಚರಿಕೆಗಳು',
            'precaution': 'ಎಚ್ಚರಿಕೆ',
            'consult_doctor': 'ಸೇವಿಸುವ ಮೊದಲು ನಿಮ್ಮ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ',
            
            # Messages
            'success': 'ಯಶಸ್ವಿ',
            'error': 'ದೋಷ',
            'loading_error': 'ಸಿಫಾರಿಸುಗಳನ್ನು ಲೋಡ್ ಮಾಡಲು ವಿಫಲವಾಗಿದೆ',
            'dietary_preference': 'ಆಹಾರ ಆದ್ಯತೆ',
        },
        'malayalam': {
            # Navigation & Headers
            'dashboard': 'ഡാഷ്ബോർഡ്',
            'recommendations': 'ശുപാർശകൾ',
            'meal_plans': 'ഭക്ഷണ പദ്ധതികൾ',
            'chatbot': 'ചാറ്റ്ബോട്ട്',
            'logout': 'ലോഗ്ഔട്ട്',
            'welcome': 'സ്വാഗതം',
            'profile': 'പ്രൊഫൈൽ',
            
            # Recommendation Page
            'personalized_recommendations': 'വ്യക്തിഗത ഭക്ഷണ ശുപാർശകൾ',
            'based_on_trimester': 'നിങ്ങളുടെ ത്രൈമാസികത്തിന്റെയും മുൻഗണനകളുടെയും അടിസ്ഥാനത്തിൽ',
            'meal_type': 'ഭക്ഷണ തരം',
            'select_meal_type': 'ഭക്ഷണ തരം തിരഞ്ഞെടുക്കുക',
            'breakfast': 'പ്രഭാതഭക്ഷണം',
            'lunch': 'ഉച്ചഭക്ഷണം',
            'dinner': 'രാത്രിഭക്ഷണം',
            'snacks': 'പലഹാരം',
            'all_meals': 'എല്ലാ ഭക്ഷണങ്ങളും',
            'get_recommendations': 'ശുപാർശകൾ നേടുക',
            'loading': 'ലോഡ് ചെയ്യുന്നു...',
            'no_recommendations': 'ശുപാർശകൾ ലഭ്യമല്ല',
            'current_trimester': 'നിലവിലെ ത്രൈമാസികം',
            
            # Food Card Information
            'food_name': 'ഭക്ഷണനാമം',
            'category': 'വിഭാഗം',
            'benefits': 'ഗുണങ്ങൾ',
            'preparation_tips': 'തയാറാക്കൽ സൂചനകൾ',
            'precautions': 'മുന്നറിപ്പുകൾ',
            'nutritional_info': 'പോഷകാഹാര വിവരങ്ങൾ',
            'protein': 'പ്രോട്ടീൻ',
            'calories': 'കലോരി',
            'carbs': 'കാർബോഹൈഡ്രേറ്റ്',
            'fiber': 'ഫൈബർ',
            'iron': 'ഇരുമ്പ്',
            'calcium': 'കാൽസിയം',
            'folic_acid': 'ഫോലിക് ആസിഡ്',
            
            # Ratings
            'excellent': 'അത്യുത്തമം',
            'very_good': 'വളരെ നല്ലത്',
            'good': 'നല്ലത്',
            'fair': 'സാധാരണ',
            'recommendation_score': 'ശുപാർശ സ്കോർ',
            'suitability_score': 'അനുയോജ്യതാ സ്കോർ',
            
            # Feedback
            'helpful': 'സഹായകമായ',
            'not_helpful': 'സഹായകമല്ലാത്ത',
            'feedback_submitted': 'നിങ്ങളുടെ പ്രതികരണത്തിന് ധന്യവാദങ്ങൾ!',
            
            # Warnings
            'warnings': 'മുന്നറിപ്പുകൾ',
            'precaution': 'മുന്നറിപ്പ്',
            'consult_doctor': 'കഴിക്കുന്നതിന് മുമ്പ് നിങ്ങളുടെ ഡോക്ടറെ സമീപിക്കുക',
            
            # Messages
            'success': 'വിജയം',
            'error': 'പിശക്',
            'loading_error': 'ശുപാർശകൾ ലോഡ് ചെയ്യുന്നത് ব്യর്ഥമായി',
            'dietary_preference': 'ഭക്ഷണ മുൻഗണന',
        },
        'tamil': {
            # Navigation & Headers
            'dashboard': 'டேஷ்போர்டு',
            'recommendations': 'பரிந்துரைகள்',
            'meal_plans': 'உணவு திட்டங்கள்',
            'chatbot': 'சாட்பாட்',
            'logout': 'வெளியேறு',
            'welcome': 'வரவேற்கிறோம்',
            'profile': 'சுயவிவரம்',
            
            # Recommendation Page
            'personalized_recommendations': 'ব্যক্তিগতকৃত உணவு பரிந்துரைகள்',
            'based_on_trimester': 'உங்கள் ত்रैমাসिक மற्र निष््यन्दोও आधार पर',
            'meal_type': 'உணவு வகை',
            'select_meal_type': 'உணவு வகை தேர்ந்தெடுக்கவும்',
            'breakfast': 'காலை உணவு',
            'lunch': 'மதிய உணவு',
            'dinner': 'இரவு உணவு',
            'snacks': 'சிறு உணவு',
            'all_meals': 'அனைத்து உணவுகள்',
            'get_recommendations': 'பரிந்துரைகளைப் பெறவும்',
            'loading': 'ஏற்றுதல் நடந்துகொண்டிருக்கிறது...',
            'no_recommendations': 'பரிந்துரைகள் கிடைக்கவில்லை',
            'current_trimester': 'தற்போதைய ત्रैमाসिक',
            
            # Food Card Information
            'food_name': 'உணவின் பெயர்',
            'category': 'வகை',
            'benefits': 'நன்மைகள்',
            'preparation_tips': 'தயாரிப்பு குறிப்புகள்',
            'precautions': 'எச்சரிக்கைகள்',
            'nutritional_info': 'ஊட்டச்சத்து தகவல்',
            'protein': 'புரதம்',
            'calories': 'கலோரி',
            'carbs': 'கார்போஹைட்ரேட்',
            'fiber': 'நார்',
            'iron': 'இரும்பு',
            'calcium': 'கால்சியம்',
            'folic_acid': 'ஃபோலிக் அமிலம்',
            
            # Ratings
            'excellent': 'சிறந்த',
            'very_good': 'மிக நன்று',
            'good': 'நன்று',
            'fair': 'நியாயமான',
            'recommendation_score': 'பரிந்துரை மதிப்பெண்',
            'suitability_score': 'பொருத்தத்தன்மை மதிப்பெண்',
            
            # Feedback
            'helpful': 'உதவியாக',
            'not_helpful': 'உதவியாக இல்லை',
            'feedback_submitted': 'உங்கள் கருத்திற்கு நன்றி!',
            
            # Warnings
            'warnings': 'எச்சரிக்கைகள்',
            'precaution': 'கவனம்',
            'consult_doctor': 'நுகர்வதற்கு முன் உங்கள் மருத்துவரைப் பரிசீலிக்கவும்',
            
            # Messages
            'success': 'வெற்றி',
            'error': 'பிழை',
            'loading_error': 'பரிந்துரைகளை ஏற்றுவது தோல்வியடைந்தது',
            'dietary_preference': 'உணவு முன்னுரிமை',
        }
    }
    
    @staticmethod
    def get_text(key, language='english'):
        """Get translated text for a key.
        
        Args:
            key: Translation key
            language: Language code (default: english)
            
        Returns:
            Translated text or key if not found
        """
        if language not in LanguageManager.TRANSLATIONS:
            language = 'english'
        
        return LanguageManager.TRANSLATIONS[language].get(key, key)
    
    @staticmethod
    def get_language_display_name(language):
        """Get display name for a language."""
        display_names = {
            'english': 'English',
            'hindi': 'हिंदी',
            'telugu': 'తెలుగు',
            'kannada': 'ಕನ್ನಡ',
            'malayalam': 'മലയാളം',
            'tamil': 'தமிழ்'
        }
        return display_names.get(language, language)
    
    @staticmethod
    def translate_food_item(food_dict, language='english'):
        """Translate a food item dictionary.
        
        Args:
            food_dict: Food item dictionary
            language: Target language
            
        Returns:
            Translated food item dictionary
        """
        if language == 'english':
            return food_dict
        
        # Food names in different languages would be stored in database
        translated = food_dict.copy()
        
        # Map language to database column
        language_column_map = {
            'hindi': 'name_hindi',
            'telugu': 'name_telugu',
            'kannada': 'name_kannada',
            'malayalam': 'name_malayalam',
            'tamil': 'name_tamil'
        }
        
        if language in language_column_map:
            column = language_column_map[language]
            if column in food_dict:
                translated['food_name'] = food_dict[column]
        
        return translated
