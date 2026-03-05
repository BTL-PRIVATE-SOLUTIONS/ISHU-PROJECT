"""LangChain + HuggingFace AI Integration for Pregnancy Nutrition Chatbot.

This module provides AI-powered responses using:
1. LangChain for AI orchestration and prompt management
2. HuggingFace models for local/cloud-based inference
3. Graceful fallbacks when external APIs are unavailable
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LangChainNutritionAI:
    """AI-powered nutrition chatbot using LangChain + HuggingFace."""
    
    def __init__(self):
        """Initialize LangChain AI with HuggingFace models."""
        self.available = False
        self.llm = None
        self.chain = None
        
        # Try to initialize LangChain with available models
        self._initialize_ai()
    
    def _initialize_ai(self):
        """Initialize AI models with graceful fallback."""
        try:
            # Try importing LangChain components (modern version)
            from langchain_core.prompts import PromptTemplate
            
            # Try multiple AI providers in order of preference
            llm_initialized = False
            
            # 1. Try Google Gemini (via LangChain)
            if not llm_initialized:
                llm_initialized = self._try_gemini_langchain()
            
            # 2. Try HuggingFace Inference API (free tier)
            if not llm_initialized:
                llm_initialized = self._try_huggingface_api()
            
            # 3. Try local HuggingFace model (offline mode)
            if not llm_initialized:
                llm_initialized = self._try_local_huggingface()
            
            if llm_initialized and self.llm:
                # Create prompt template for pregnancy nutrition
                prompt_template = """You are an expert pregnancy nutrition advisor specializing in Indian cuisine.

Question: {question}

Context:
- Trimester: {trimester}
- Region: {region}
- Diet Type: {diet_type}

Provide a helpful, accurate, and concise answer (under 200 words) covering:
1. Direct answer to the question
2. Safety considerations during pregnancy
3. Nutritional benefits
4. Specific recommendations

Answer:"""
                
                prompt = PromptTemplate(
                    input_variables=["question", "trimester", "region", "diet_type"],
                    template=prompt_template
                )
                
                # Create chain using modern LCEL (LangChain Expression Language)
                self.chain = prompt | self.llm
                self.available = True
                print("✓ LangChain AI initialized successfully")
            else:
                print("⚠️  LangChain AI not available - will use fallback responses")
                
        except ImportError as e:
            print(f"⚠️  LangChain not available: {e}")
            print("   Install with: pip install langchain langchain-community langchain-huggingface")
        except Exception as e:
            print(f"⚠️  Failed to initialize LangChain AI: {e}")
    
    def _try_gemini_langchain(self) -> bool:
        """Try to initialize Google Gemini via LangChain."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                return False
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=api_key,
                temperature=0.7,
                max_output_tokens=500
            )
            print("✓ Using Google Gemini via LangChain")
            return True
            
        except ImportError:
            print("⚠️  langchain-google-genai not available (install with: pip install langchain-google-genai)")
            return False
        except Exception as e:
            print(f"⚠️  Failed to initialize Gemini: {e}")
            return False
    
    def _try_huggingface_api(self) -> bool:
        """Try to initialize HuggingFace Inference API."""
        try:
            from langchain_huggingface import HuggingFaceEndpoint
            
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not api_key:
                print("⚠️  No HUGGINGFACE_API_KEY found - skipping HF Inference API")
                return False
            
            # Use a good free model for text generation
            model_id = os.getenv('HUGGINGFACE_MODEL', 'google/flan-t5-base')
            
            self.llm = HuggingFaceEndpoint(
                repo_id=model_id,
                huggingfacehub_api_token=api_key,
                temperature=0.7,
                max_new_tokens=300
            )
            print(f"✓ Using HuggingFace Inference API: {model_id}")
            return True
            
        except ImportError:
            print("⚠️  langchain-huggingface not available")
            return False
        except Exception as e:
            print(f"⚠️  Failed to initialize HuggingFace API: {e}")
            return False
    
    def _try_local_huggingface(self) -> bool:
        """Try to initialize local HuggingFace model (offline mode)."""
        try:
            from langchain_huggingface import HuggingFacePipeline
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            
            print("⚠️  Attempting to load local HuggingFace model (this may take time)...")
            
            # Use a small, efficient model that can run locally
            model_name = "google/flan-t5-small"  # ~80MB, good for basic text generation
            
            try:
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                # Create text generation pipeline
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=200,
                    temperature=0.7
                )
                
                self.llm = HuggingFacePipeline(pipeline=pipe)
                print(f"✓ Using local HuggingFace model: {model_name}")
                return True
                
            except Exception as e:
                print(f"⚠️  Failed to load local model: {e}")
                return False
            
        except ImportError:
            print("⚠️  transformers not available for local model")
            return False
        except Exception as e:
            print(f"⚠️  Failed to initialize local HuggingFace model: {e}")
            return False
    
    def generate_response(
        self,
        question: str,
        context: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Generate AI response using LangChain + HuggingFace.
        
        Args:
            question: User's question
            context: Optional context dictionary with trimester, region, diet_type
            
        Returns:
            AI-generated response or None if unavailable
        """
        if not self.available or not self.chain:
            return None
        
        try:
            # Extract context or use defaults
            context = context or {}
            trimester = context.get('trimester', 'all trimesters')
            region = context.get('region', 'general Indian')
            diet_type = context.get('diet_type', 'general')
            
            # Generate response using LangChain (modern LCEL approach)
            response = self.chain.invoke({
                "question": question,
                "trimester": trimester,
                "region": region,
                "diet_type": diet_type
            })
            
            # Handle different response types (string or AIMessage)
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()
            
        except Exception as e:
            print(f"Error generating LangChain response: {e}")
            return None
    
    def get_fallback_response(self, question: str) -> str:
        """
        Provide rule-based fallback response when AI is unavailable.
        
        Args:
            question: User's question
            
        Returns:
            Fallback response based on question keywords
        """
        question_lower = question.lower()
        
        # Safety check questions
        if any(word in question_lower for word in ['can i eat', 'is it safe', 'can i have']):
            # Extract food items
            foods = self._extract_food_items(question_lower)
            
            # Common safe foods
            safe_foods = {
                'mutton': '✅ **Mutton is generally safe during pregnancy** when properly cooked.\n\n**Benefits:**\n- High-quality protein for baby\'s growth\n- Rich in iron (prevents anemia)\n- Contains B vitamins\n- Good source of zinc\n\n**Important:**\n- Cook thoroughly (no pink meat)\n- Avoid liver (high vitamin A)\n- Limit to 2-3 times per week\n- Choose fresh, quality meat\n\n💡 Always consult your doctor for personalized advice.',
                
                'eggs': '✅ **Eggs are excellent during pregnancy** when properly cooked.\n\n**Benefits:**\n- Complete protein source\n- Rich in choline (brain development)\n- Contains vitamin D\n- Good for both mother and baby\n\n**Important:**\n- Cook eggs fully (no runny yolk)\n- Avoid raw eggs\n- 1-2 eggs per day is ideal\n\n💡 Always consult your doctor for personalized advice.',
                
                'fish': '✅ **Fish is generally safe and beneficial** during pregnancy with precautions.\n\n**Benefits:**\n- Omega-3 for baby\'s brain\n- High-quality protein\n- Vitamin D and B12\n\n**Important:**\n- Avoid high-mercury fish (shark, swordfish)\n- Choose low-mercury options (salmon, sardines)\n- Cook thoroughly\n- Limit to 2-3 servings per week\n\n💡 Always consult your doctor for personalized advice.',
            }
            
            for food, response in safe_foods.items():
                if food in question_lower:
                    return response
        
        # Foods to avoid questions
        if 'avoid' in question_lower or 'not eat' in question_lower:
            return """⚠️ **Foods to Avoid During Pregnancy:**

**High Risk Foods:**
❌ Raw/undercooked meat, eggs, fish
❌ Unpasteurized dairy products
❌ Raw sprouts
❌ High-mercury fish (shark, swordfish)
❌ Alcohol
❌ Unwashed fruits/vegetables

**Limit These:**
⚠️ Caffeine (< 200mg/day)
⚠️ Processed foods
⚠️ High-sugar items
⚠️ Papaya (unripe)
⚠️ Pineapple (excessive amounts)

💡 **General Rule:** Eat well-cooked, fresh foods. Always consult your doctor for personalized advice."""
        
        # Trimester-specific questions
        if 'trimester' in question_lower or 'first' in question_lower or 'second' in question_lower or 'third' in question_lower:
            return """🤰 **Trimester-Specific Nutrition:**

**First Trimester (Weeks 1-12):**
- Focus on folic acid (leafy greens, lentils)
- Small, frequent meals for nausea
- Stay hydrated
- Vitamin B6 for morning sickness

**Second Trimester (Weeks 13-26):**
- Increase protein and calcium
- Iron-rich foods (prevent anemia)
- Healthy fats for brain development
- Balanced meals

**Third Trimester (Weeks 27-40):**
- Extra calories (300-500/day)
- Calcium for bone development
- Iron and protein
- Fiber to prevent constipation

💡 Always consult your doctor for personalized meal plans."""
        
        # Default response
        return """🤖 **Pregnancy Nutrition Guidance:**

I can help you with:
- Food safety during pregnancy
- Nutritional benefits of specific foods
- Trimester-specific recommendations
- Foods to avoid
- Meal planning advice

Please ask a specific question like:
- "Can I eat [food name] during pregnancy?"
- "What foods should I avoid in [trimester]?"
- "What are good sources of [nutrient]?"

💡 For personalized advice, always consult your doctor or dietitian."""
    
    def _extract_food_items(self, text: str) -> List[str]:
        """Extract food items from question text."""
        # Common food keywords
        foods = [
            'mutton', 'chicken', 'fish', 'eggs', 'milk', 'cheese',
            'papaya', 'pineapple', 'mango', 'apple', 'banana',
            'rice', 'wheat', 'lentils', 'dal', 'beans',
            'spinach', 'carrot', 'tomato'
        ]
        
        found_foods = [food for food in foods if food in text]
        return found_foods


# Singleton instance
_langchain_ai_instance = None


def get_langchain_ai() -> LangChainNutritionAI:
    """Get or create LangChain AI singleton instance."""
    global _langchain_ai_instance
    if _langchain_ai_instance is None:
        _langchain_ai_instance = LangChainNutritionAI()
    return _langchain_ai_instance
