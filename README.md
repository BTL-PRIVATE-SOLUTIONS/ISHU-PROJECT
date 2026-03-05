# iBaby - AI-Powered Pregnancy Nutrition App

An intelligent pregnancy nutrition application that provides personalized meal planning, nutritional guidance, and chatbot support for pregnant women, powered by advanced AI technology.

## 🌟 Features

### 1. AI-Powered Meal Planning
- **Personalized meal plans** based on trimester, region, and dietary preferences
- **Hybrid approach**: Dataset-based meals enhanced with AI nutrition validation
- **16+ nutrient tracking** including vitamins, minerals, and omega-3
- **Visual nutrition dashboard** with color-coded progress indicators
- Supports both 3-meal and 5-meal daily plans

### 2. Comprehensive Nutrition Tracking
Track 16 essential nutrients:
- **Macros**: Calories, Protein, Carbs, Fat, Fiber
- **Minerals**: Iron, Calcium, Zinc, Magnesium
- **Vitamins**: A, B6, B12, C, D, Folic Acid
- **Omega-3** fatty acids

Each nutrient shows:
- Current daily average
- Recommended target
- Compliance percentage
- Color-coded status (✅ Good, ⚠️ Needs Attention, ❌ Deficient)

### 3. AI-Enhanced Chatbot
Multi-tier intelligent chatbot with advanced fallback logic:
1. **Dataset Search** - Answers from comprehensive pregnancy nutrition datasets
2. **Google Gemini AI** - Google's Gemini Pro for contextual responses (if configured)
3. **LangChain + HuggingFace** - Alternative AI models with multiple options (if configured)
4. **Rule-based Fallback** - Comprehensive hardcoded responses for common questions

Features:
- Context-aware responses (trimester, region, diet type)
- Safety checks for food consumption
- Nutritional benefits information
- Pregnancy do's and don'ts
- **Works perfectly without any API keys** - uses dataset + rule-based responses

### 4. Regional & Dietary Support
- **Regions**: North Indian, South Indian cuisines
- **Diets**: Vegetarian, Non-Vegetarian, Vegan
- **Trimesters**: Customized nutrition for each stage
- **Special Conditions**: Diabetes, gestational diabetes support

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tejaslakshmangith/ibaby.git
   cd ibaby
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (Optional for AI features)
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the app**
   Open your browser and navigate to `http://localhost:5000`

## 🔑 Environment Variables

Create a `.env` file in the root directory with the following optional configurations:

### AI Features (Optional)
```env
# Google Gemini API (Free Tier Available)
# Get your key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your-gemini-api-key-here

# Alternative: Use GOOGLE_API_KEY instead
GOOGLE_API_KEY=your-google-api-key-here
```

### Flask Configuration
```env
FLASK_ENV=development
SECRET_KEY=your-secret-key-change-this-in-production
DATABASE_URL=sqlite:///instance/database.db
```

### AI Configuration (Advanced)
```env
# AI request timeout in seconds (default: 2.5)
AI_TIMEOUT_SECONDS=2.5

# Chatbot rate limit per minute (default: 20)
CHATBOT_RATE_LIMIT_PER_MIN=20

# Response cache TTL in seconds (default: 3600)
CACHE_TTL_SECONDS=3600
```

## 📊 Usage Examples

### Meal Planning
1. Navigate to the Meal Plans page
2. Select your preferences:
   - Number of days (1-30)
   - Region (North/South Indian)
   - Diet type (Vegetarian/Non-Vegetarian/Vegan)
   - Meals per day (3 or 5 meals)
3. Click "Generate Meal Plan"
4. View your personalized plan with comprehensive nutrition tracking

### Chatbot Queries
Ask questions like:
- "Is it safe to eat papaya during pregnancy?"
- "What foods are high in iron for pregnancy?"
- "What should I eat in the second trimester?"
- "Can I have coffee while pregnant?"

The chatbot provides:
- Safety information
- Nutritional benefits
- Trimester-specific guidance
- Medical disclaimers

## 🤖 AI Integration

The chatbot uses a **multi-tier AI system** with comprehensive fallbacks:

### 1. Google Gemini (Optional - Free Tier Available)
1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to `.env`: `GEMINI_API_KEY=your-key-here`
3. Restart the application

### 2. LangChain + HuggingFace (Optional - Free Tier Available)
1. Get a free API key from [HuggingFace](https://huggingface.co/settings/tokens)
2. Add to `.env`: `HUGGINGFACE_API_KEY=your-key-here`
3. Restart the application

### 3. Rule-based Fallback (Always Available)
- Works without any API keys
- Provides comprehensive answers for common pregnancy nutrition questions
- Fast response times (< 1 second)

**Features enabled with AI:**
- AI-powered contextual responses
- Advanced nutrition calculations
- Enhanced meal recommendations
- Better understanding of complex questions

**Free Tier Limits:**
- Gemini: 60 requests per minute
- HuggingFace: Varies by model
- Rate limiting built-in
- Graceful degradation if unavailable

**✨ New: Works perfectly without API keys!** The chatbot now provides excellent responses using dataset search and rule-based fallbacks even when no AI APIs are configured.

See [AI Integration Documentation](docs/AI_INTEGRATION.md) for detailed information.

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/ -v

# Run AI integration tests
python tests/test_ai_integration.py

# Run with coverage
python -m pytest tests/ --cov=ai_engine --cov-report=html
```

**Test Coverage:**
- ✅ Gemini AI initialization and graceful degradation
- ✅ Meal planner AI integration
- ✅ Chatbot AI fallback logic
- ✅ Season parameter removal verification
- ✅ Enhanced nutrition tracking (16 nutrients)

## 🔒 Security

- **CodeQL Scanning**: Automated security analysis (0 vulnerabilities)
- **Graceful Degradation**: Works without AI keys
- **Rate Limiting**: Built-in protection against API abuse
- **Input Validation**: All user inputs sanitized
- **Medical Disclaimers**: Clear notices on AI-generated advice

## 📁 Project Structure

```
ibaby/
├── ai_engine/
│   ├── gemini_integration.py      # Google Gemini AI wrapper
│   ├── langchain_ai.py            # LangChain + HuggingFace integration
│   ├── meal_planner.py            # Meal plan generation
│   ├── comprehensive_chatbot.py   # Multi-tier chatbot
│   └── unified_dataset_loader.py  # Dataset management
├── routes/
│   ├── meal_plans.py              # Meal planning API
│   ├── chatbot.py                 # Chatbot API
│   └── ...
├── templates/
│   └── dashboard/
│       ├── meal_plans.html        # Meal planning UI
│       └── chatbot.html           # Chatbot UI
├── data/                          # Nutrition datasets
├── docs/
│   └── AI_INTEGRATION.md          # AI integration documentation
├── tests/
│   └── test_ai_integration.py     # AI feature tests
├── .env.example                   # Environment variables template
├── requirements.txt               # Python dependencies
└── app.py                         # Flask application entry point
```

## 🔄 Recent Updates

### v2.1.0 - Enhanced AI Integration (Current)
- ✅ Added LangChain + HuggingFace AI integration
- ✅ Multi-tier AI fallback system (Gemini → LangChain → Rule-based)
- ✅ Chatbot works perfectly without API keys
- ✅ Comprehensive rule-based fallback responses
- ✅ Improved answer quality detection
- ✅ Support for multiple AI providers
- ✅ Updated dependencies for compatibility
- ✅ Added AI integration documentation

### v2.0.0 - AI-Powered Enhancement
- ✅ Removed season dependency from meal planning
- ✅ Integrated Google Gemini AI (free tier)
- ✅ Expanded nutrition tracking to 16+ nutrients
- ✅ Visual progress bars with color-coded status
- ✅ AI-enhanced chatbot with multi-tier fallback
- ✅ Hybrid meal generation (Dataset + AI)
- ✅ Comprehensive test suite (8 tests, all passing)
- ✅ Zero security vulnerabilities (CodeQL verified)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Tejas Lakshman ([@tejaslakshmangith](https://github.com/tejaslakshmangith))

## 🙏 Acknowledgments

- Google Gemini AI for free-tier AI capabilities
- Pregnancy nutrition datasets from various open sources
- Flask framework and community

## 📧 Support

For issues, questions, or suggestions:
- 📫 Open an issue on GitHub
- 💬 Contact: [GitHub Issues](https://github.com/tejaslakshmangith/ibaby/issues)

## ⚕️ Medical Disclaimer

**Important**: This application provides nutritional information and AI-generated advice for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

Always consult with:
- Your doctor or healthcare provider
- A registered dietitian
- A qualified medical professional

Before making any dietary changes during pregnancy.

---

**Made with ❤️ for expecting mothers**
