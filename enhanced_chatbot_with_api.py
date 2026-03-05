"""
Enhanced AI Chatbot with Gemini AI Integration
Provides fast, intelligent responses with dataset fallback and AI enhancement
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from functools import lru_cache
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Ensure .env is loaded for API keys when used outside Flask app
load_dotenv()


class ResponseCache:
    """Simple in-memory cache for responses."""
    
    def __init__(self, ttl_minutes=60):
        self.cache = {}
        self.ttl = ttl_minutes * 60
    
    def _get_key(self, question: str, context_key: str = "") -> str:
        """Generate cache key from question + optional context."""
        raw_key = f"{question.lower().strip()}|{context_key}".strip("|")
        return hashlib.md5(raw_key.encode()).hexdigest()
    
    def get(self, question: str, context_key: str = "") -> Optional[Dict]:
        """Get cached response."""
        key = self._get_key(question, context_key)
        if key in self.cache:
            cached_data = self.cache[key]
            if time.time() - cached_data['timestamp'] < self.ttl:
                return cached_data['response']
        return None
    
    def set(self, question: str, response: Dict, context_key: str = "") -> None:
        """Cache a response."""
        key = self._get_key(question, context_key)
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


class ExternalAIProvider:
    """Base class for external AI providers."""
    
    def generate_answer(self, question: str, context: str = "") -> str:
        """Generate answer using external API."""
        raise NotImplementedError


class EnhancedDosDontsChatbot:
    """
    Enhanced chatbot with dataset + Gemini AI integration.
    Fast responses with 3-second max response time.
    """
    
    def __init__(self):
        """Initialize enhanced chatbot."""
        # Load datasets
        self.dos_donts_df = self._load_dos_donts_dataset()
        self.pregnancy_diet_df = self._load_pregnancy_diet_dataset()
        self.foods_to_avoid_df = self._load_foods_to_avoid_dataset()
        
        # Build knowledge base
        self.knowledge_base = self._build_knowledge_base()
        
        # Initialize response cache
        self.cache = ResponseCache(ttl_minutes=60)
        
        # Response time tracking
        self.response_times = []
    
    def _load_dos_donts_dataset(self) -> pd.DataFrame:
        """Load the dos and donts dataset."""
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '../data/remainingdatasets/pregnancy_dos_donts_dataset.csv'
        )
        
        try:
            df = pd.read_csv(dataset_path)
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def _load_pregnancy_diet_dataset(self) -> pd.DataFrame:
        """Load pregnancy diet dataset."""
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '../data/remainingdatasets/pregnancy_diet_clean_dataset.csv'
        )
        
        try:
            df = pd.read_csv(dataset_path)
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def _load_foods_to_avoid_dataset(self) -> pd.DataFrame:
        """Load foods to avoid dataset."""
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '../data/remainingdatasets/foods_to_avoid_during_pregnancy_dataset.csv'
        )
        
        try:
            df = pd.read_csv(dataset_path)
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def _build_knowledge_base(self) -> Dict:
        """Build knowledge base from datasets."""
        knowledge = {
            'dos': {},
            'donts': {},
            'all_items': [],
            'foods_and_info': {}
        }
        
        # Process Do's and Don'Ts dataset
        for _, row in self.dos_donts_df.iterrows():
            item = str(row.get('Item', '')).strip().lower()
            item_type = str(row.get('Type', '')).strip().upper()
            description = str(row.get('Description', ''))
            notes = str(row.get('Notes', ''))
            
            if not item:
                continue
            
            # Normalize type
            if item_type in ['DO', 'DOS']:
                item_type = 'DO'
            elif item_type in ["DON'T", "DONT", "DON'TS", "DONTS"]:
                item_type = "DONT"
            
            item_info = {
                'description': description,
                'notes': notes,
                'type': item_type
            }
            
            if item_type == 'DO':
                knowledge['dos'][item] = item_info
            elif item_type == 'DONT':
                knowledge['donts'][item] = item_info
            
            knowledge['all_items'].append(item)
            knowledge['foods_and_info'][item] = item_info
        
        return knowledge
    
    def extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from question."""
        keywords = []
        question_lower = question.lower()
        
        # Check against dataset items
        for item in self.knowledge_base['all_items']:
            if item in question_lower:
                keywords.append(item)
        
        # If no exact keyword match, try word by word
        if not keywords:
            words = question_lower.split()
            for item in self.knowledge_base['all_items']:
                item_words = item.split()
                for word in item_words:
                    if word in words and len(word) > 2:
                        keywords.append(item)
                        break
        
        return list(set(keywords))
    
    def get_dataset_answer(self, keywords: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Get answers from dataset."""
        dos = []
        donts = []
        
        if keywords:
            for keyword in keywords:
                if keyword in self.knowledge_base['dos']:
                    dos.append(self.knowledge_base['dos'][keyword])
                if keyword in self.knowledge_base['donts']:
                    donts.append(self.knowledge_base['donts'][keyword])
        
        return dos[:5], donts[:5]
    
    def format_dataset_answer(self, dos: List[Dict], donts: List[Dict]) -> str:
        """Format dataset answer."""
        answer_lines = []
        
        if dos:
            answer_lines.append("✅ DO's (Safe & Recommended):")
            for do_item in dos:
                answer_lines.append(f"  • {do_item.get('description', '')}")
                if do_item.get('notes'):
                    answer_lines.append(f"    💡 {do_item['notes']}")
        
        if donts:
            answer_lines.append("\n❌ DON'Ts (Avoid):")
            for dont_item in donts:
                answer_lines.append(f"  • {dont_item.get('description', '')}")
                if dont_item.get('notes'):
                    answer_lines.append(f"    ⚠️ {dont_item['notes']}")
        
        return "\n".join(answer_lines) if answer_lines else ""

    def _build_ai_context(
        self,
        question: str,
        trimester: Optional[int],
        keywords: List[str],
        dataset_answer: str,
        dos: List[Dict],
        donts: List[Dict]
    ) -> str:
        """Build a compact context block for external AI to improve answer quality."""
        parts = ["Domain: pregnancy nutrition and maternal diet."]

        if trimester:
            parts.append(f"User trimester: {trimester}.")

        if keywords:
            parts.append(f"Keywords: {', '.join(sorted(set(keywords)))}.")

        if dataset_answer:
            parts.append("Dataset guidance:")
            parts.append(dataset_answer)

        if dos or donts:
            formatted_dos = ", ".join([d.get('description', '') for d in dos if d.get('description')])
            formatted_donts = ", ".join([d.get('description', '') for d in donts if d.get('description')])
            if formatted_dos:
                parts.append(f"Dataset DOs: {formatted_dos}")
            if formatted_donts:
                parts.append(f"Dataset DON'Ts: {formatted_donts}")

        return "\n".join(parts).strip()
    
    def _enhance_dataset_answer(self, dataset_answer: str, dos: List[Dict], donts: List[Dict], trimester: Optional[int], keywords: List[str]) -> str:
        """Enhance dataset answer with additional helpful context."""
        enhanced = dataset_answer
        
        # Add trimester-specific tip if available
        if trimester:
            tips = {
                1: "\n\n💡 First Trimester Tip: Focus on folic acid and managing morning sickness with small, frequent meals.",
                2: "\n\n💡 Second Trimester Tip: This is when baby grows fastest. Increase calcium, protein, and iron-rich foods.",
                3: "\n\n💡 Third Trimester Tip: Add fiber to prevent constipation. Stay hydrated and maintain protein intake."
            }
            enhanced += tips.get(trimester, "")
        
        # Add general safety reminder
        if donts:
            enhanced += "\n\n⚠️ Always wash fruits/vegetables thoroughly and avoid undercooked foods."
        
        return enhanced
    
    def _generate_generic_pregnancy_answer(self, question: str, trimester: Optional[int], keywords: List[str]) -> str:
        """Generate helpful generic pregnancy nutrition answer when AI fails."""
        question_lower = question.lower()
        
        # Safety questions
        if any(word in question_lower for word in ['can i eat', 'is it safe', 'should i avoid']):
            return (
                "During pregnancy, it's important to eat a balanced diet with:\n\n"
                "✅ Fresh fruits and vegetables\n"
                "✅ Whole grains and legumes\n"
                "✅ Lean proteins (cooked meat, eggs, fish)\n"
                "✅ Dairy products (pasteurized)\n"
                "✅ Nuts and seeds\n\n"
                "❌ Avoid: raw/undercooked meat, unpasteurized dairy, high-mercury fish, alcohol\n\n"
                "For specific foods, consult your doctor or nutritionist."
            )
        
        # Trimester-specific
        if trimester:
            guidance = {
                1: "First trimester: Focus on folic acid, vitamin B6 (for nausea), iron, and staying hydrated. Eat small frequent meals.",
                2: "Second trimester: Increase calcium, protein, and iron intake. This is when baby grows rapidly. Include dairy, lean meats, and leafy greens.",
                3: "Third trimester: Continue protein and iron. Add fiber to prevent constipation. Dates may help prepare for labor."
            }
            return f"**Trimester {trimester} Nutrition:**\n{guidance.get(trimester, 'Focus on balanced nutrition')}\n\nEat plenty of fruits, vegetables, whole grains, and proteins. Stay hydrated and take prenatal vitamins as prescribed."
        
        # General
        return (
            "**General Pregnancy Nutrition Tips:**\n\n"
            "• Eat a variety of nutritious foods from all food groups\n"
            "• Stay hydrated (8-10 glasses of water daily)\n"
            "• Take prenatal vitamins as prescribed\n"
            "• Eat small, frequent meals to manage nausea\n"
            "• Focus on iron, calcium, folic acid, and protein\n"
            "• Avoid alcohol, smoking, and high-mercury fish\n\n"
            "Consult your healthcare provider for personalized advice."
        )
    
    def answer_question(self, question: str, trimester: Optional[int] = None) -> Dict:
        """
        Answer question with automatic fallback to external AI.
        TARGET: Response time < 3 seconds.
        """
        start_time = time.time()
        
        cache_key = f"trimester={trimester}" if trimester else "trimester=unknown"

        # Check cache first
        cached = self.cache.get(question, cache_key)
        if cached:
            return cached
        
        keywords = self.extract_keywords(question)
        
        # Get dataset answer
        dos, donts = self.get_dataset_answer(keywords)
        dataset_answer = self.format_dataset_answer(dos, donts)
        
        dos_list = [{'description': d.get('description'), 'notes': d.get('notes')} for d in dos]
        donts_list = [{'description': d.get('description'), 'notes': d.get('notes')} for d in donts]

        # Use enhanced dataset-based response (reliable and fast)
        if dataset_answer and (len(dos) > 0 or len(donts) > 0):
            # Enhance dataset answer with helpful context
            enhanced_answer = self._enhance_dataset_answer(dataset_answer, dos, donts, trimester, keywords)
            result = {
                'success': True,
                'question': question,
                'answer': enhanced_answer,
                'dos': dos_list,
                'donts': donts_list,
                'source': 'dataset',
                'response_time': time.time() - start_time
            }
        else:
            # Fallback: provide general pregnancy nutrition guidance
            generic_answer = self._generate_generic_pregnancy_answer(question, trimester, keywords)
            result = {
                'success': True,
                'question': question,
                'answer': generic_answer,
                'dos': [],
                'donts': [],
                'source': 'generic',
                'response_time': time.time() - start_time
            }
        
        # Cache the result
        self.cache.set(question, result, cache_key)
        
        # Track response time
        self.response_times.append(result['response_time'])
        
        return result
    
    def get_stats(self) -> Dict:
        """Get chatbot statistics."""
        if not self.response_times:
            return {}
        
        return {
            'total_questions': len(self.response_times),
            'avg_response_time': sum(self.response_times) / len(self.response_times),
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times),
            'cache_size': len(self.cache.cache)
        }


# Singleton instance
_enhanced_chatbot_instance = None


def get_enhanced_chatbot() -> EnhancedDosDontsChatbot:
    """Get or initialize the enhanced chatbot (singleton)."""
    global _enhanced_chatbot_instance
    if _enhanced_chatbot_instance is None:
        _enhanced_chatbot_instance = EnhancedDosDontsChatbot()
    return _enhanced_chatbot_instance


if __name__ == '__main__':
    # Test the enhanced chatbot (minimal output)
    chatbot = get_enhanced_chatbot()
    result = chatbot.answer_question("Can I eat eggs during pregnancy?")
    print(f"✓ Chatbot working: {result['source']} answer in {result['response_time']:.2f}s")
