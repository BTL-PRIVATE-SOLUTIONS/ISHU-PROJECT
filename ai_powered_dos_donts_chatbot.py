"""AI-Powered Chatbot with Do's and Don'ts using Transformers and RAG."""
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
import os
import warnings
warnings.filterwarnings('ignore')


class AIPoweredDosDontsChatbot:
    """
    AI-powered chatbot that answers pregnancy nutrition questions in Do's and Don'Ts format.
    Uses transformers for semantic search and answer generation.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the AI-powered chatbot.
        
        Args:
            use_gpu: Whether to use GPU for inference (if available)
        """
        self.use_gpu = use_gpu
        self.device = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.flan_tokenizer = None
        self.flan_model = None
        
        # Load datasets
        self.dos_donts_df = self._load_dos_donts_dataset()
        self.pregnancy_diet_df = self._load_pregnancy_diet_dataset()
        self.foods_to_avoid_df = self._load_foods_to_avoid_dataset()
        self.all_datasets = self._load_all_datasets()
        
        # Build knowledge base
        self.knowledge_base = self._build_knowledge_base()
        
        # Initialize embeddings for semantic search
        self._load_semantic_search_model()
        
        print("✓ AI-Powered Do's and Don'ts Chatbot initialized successfully")
        print(f"✓ Loaded {len(self.dos_donts_df)} do's and don'ts recommendations")
        print(f"✓ Semantic search ready for question-answer matching")
    
    def _load_dos_donts_dataset(self) -> pd.DataFrame:
        """Load the dos and donts dataset."""
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '../data/remainingdatasets/pregnancy_dos_donts_dataset.csv'
        )
        
        try:
            df = pd.read_csv(dataset_path)
            print(f"✓ Loaded Do's and Don'Ts dataset ({len(df)} items)")
            return df
        except Exception as e:
            print(f"⚠ Could not load Do's and Don'Ts dataset: {e}")
            return pd.DataFrame()
    
    def _load_pregnancy_diet_dataset(self) -> pd.DataFrame:
        """Load pregnancy diet dataset."""
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '../data/remainingdatasets/pregnancy_diet_clean_dataset.csv'
        )
        
        try:
            df = pd.read_csv(dataset_path)
            print(f"✓ Loaded Pregnancy Diet dataset ({len(df)} items)")
            return df
        except Exception as e:
            print(f"⚠ Could not load Pregnancy Diet dataset: {e}")
            return pd.DataFrame()
    
    def _load_foods_to_avoid_dataset(self) -> pd.DataFrame:
        """Load foods to avoid dataset."""
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '../data/remainingdatasets/foods_to_avoid_during_pregnancy_dataset.csv'
        )
        
        try:
            df = pd.read_csv(dataset_path)
            print(f"✓ Loaded Foods to Avoid dataset ({len(df)} items)")
            return df
        except Exception as e:
            print(f"⚠ Could not load Foods to Avoid dataset: {e}")
            return pd.DataFrame()
    
    def _load_all_datasets(self) -> Dict:
        """Load all available datasets."""
        base_path = os.path.join(os.path.dirname(__file__), '../data')
        datasets = {}
        
        # Regional meals
        regional_files = {
            'northveg': 'data_1/northveg_cleaned.csv',
            'northnonveg': 'data_1/northnonveg_cleaned.csv',
            'southveg': 'data_1/southveg_cleaned.csv',
            'southnonveg': 'data_1/southnonveg_cleaned.csv'
        }
        
        for name, file in regional_files.items():
            try:
                df = pd.read_csv(os.path.join(base_path, file))
                datasets[name] = df
            except Exception as e:
                print(f"⚠ Could not load {name} dataset: {e}")
        
        return datasets
    
    def _load_semantic_search_model(self):
        """Load semantic search model for embeddings."""
        try:
            print("Loading semantic search model...")
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.semantic_model.max_seq_length = 512
            print("✓ Semantic search model loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load semantic search model: {e}")
            self.semantic_model = None
    
    def _build_knowledge_base(self) -> Dict:
        """Build comprehensive knowledge base from datasets."""
        knowledge = {
            'dos': {},
            'donts': {},
            'by_category': {},
            'foods_and_info': {},
            'all_items': []
        }
        
        # Process Do's and Don'Ts dataset
        for _, row in self.dos_donts_df.iterrows():
            item = str(row.get('Item', '')).strip().lower()
            item_type = str(row.get('Type', '')).strip().upper()
            description = str(row.get('Description', ''))
            category = str(row.get('Category', '')).lower()
            notes = str(row.get('Notes', ''))
            
            if not item:
                continue
            
            # Normalize type
            if item_type in ['DO', 'DOS']:
                item_type = 'DO'
            elif item_type in ["DON'T", "DONT", "DON'TS", "DONTS"]:
                item_type = "DONT"
            
            # Store item info
            item_info = {
                'description': description,
                'notes': notes,
                'category': category,
                'type': item_type
            }
            
            if item_type == 'DO':
                knowledge['dos'][item] = item_info
            elif item_type == 'DONT':
                knowledge['donts'][item] = item_info
            
            # Store by category
            if category not in knowledge['by_category']:
                knowledge['by_category'][category] = {'do': [], 'dont': []}
            
            type_key = 'do' if item_type == 'DO' else 'dont'
            if item_info not in knowledge['by_category'][category][type_key]:
                knowledge['by_category'][category][type_key].append(item_info)
            
            knowledge['all_items'].append(item)
            knowledge['foods_and_info'][item] = item_info
        
        return knowledge
    
    def classify_intent(self, question: str) -> str:
        """Classify user intent from question."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['should i', 'can i', 'safe', 'avoid', 'bad']):
            return 'dos_donts'
        elif any(word in question_lower for word in ['benefit', 'good', 'help', 'nutrient']):
            return 'benefits'
        elif any(word in question_lower for word in ['trimester', 'first', 'second', 'third', '1st', '2nd', '3rd']):
            return 'trimester'
        elif any(word in question_lower for word in ['meal plan', 'diet', 'eat', 'recipe']):
            return 'meal_plan'
        else:
            return 'general'
    
    def extract_keywords(self, question: str) -> List[str]:
        """Extract relevant keywords from question."""
        keywords = []
        question_lower = question.lower()
        
        # Check against dataset items
        for item in self.knowledge_base['all_items']:
            if item in question_lower:
                keywords.append(item)
        
        # Common pregnancy nutrition terms
        common_terms = [
            'papaya', 'pineapple', 'water', 'exercise', 'dairy', 'fish',
            'meat', 'chicken', 'eggs', 'vegetables', 'fruits', 'grains',
            'rice', 'wheat', 'lentil', 'dal', 'milk', 'yogurt', 'cheese',
            'nuts', 'almonds', 'protein', 'iron', 'calcium', 'vitamin'
        ]
        
        for term in common_terms:
            if term in question_lower and term not in keywords:
                keywords.append(term)
        
        return list(set(keywords))
    
    def get_relevant_recommendations(self, keywords: List[str], question: str) -> Tuple[List[Dict], List[Dict]]:
        """Get relevant do's and don'ts for given keywords."""
        dos = []
        donts = []
        
        if not keywords:
            # If no specific keywords, return all category-based recommendations
            for category, items in self.knowledge_base['by_category'].items():
                dos.extend(items['do'][:2])
                donts.extend(items['dont'][:2])
        else:
            # Search for matching items
            for keyword in keywords:
                # Check do's
                if keyword in self.knowledge_base['dos']:
                    dos.append(self.knowledge_base['dos'][keyword])
                
                # Check don'ts
                if keyword in self.knowledge_base['donts']:
                    donts.append(self.knowledge_base['donts'][keyword])
                
                # Search in descriptions
                for item, info in self.knowledge_base['foods_and_info'].items():
                    if keyword in info['description'].lower():
                        if info['type'] == 'DO' and info not in dos:
                            dos.append(info)
                        elif info['type'] == 'DONT' and info not in donts:
                            donts.append(info)
        
        return dos[:5], donts[:5]  # Limit to top 5
    
    def load_generation_model(self, model_name: str = 'google/flan-t5-base'):
        """Load FLAN-T5 model for answer generation."""
        try:
            print(f"Loading {model_name} for answer generation...")
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            self.device = torch.device('cuda' if (torch.cuda.is_available() and self.use_gpu) else 'cpu')
            self.flan_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            
            print(f"✓ Generation model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"⚠ Could not load generation model: {e}")
            return False
    
    def generate_dos_donts_answer(self, question: str, dos: List[Dict], donts: List[Dict]) -> str:
        """Generate a formatted do's and don'ts answer."""
        answer_lines = []
        
        # Add dos
        if dos:
            answer_lines.append("✅ DO's (Safe & Recommended):")
            for do_item in dos:
                item_name = do_item.get('description', 'Unknown')
                notes = do_item.get('notes', '')
                answer_lines.append(f"  • {item_name}")
                if notes and notes != 'nan':
                    answer_lines.append(f"    💡 Note: {notes}")
        
        # Add donts
        if donts:
            answer_lines.append("\n❌ DON'Ts (Avoid):")
            for dont_item in donts:
                item_name = dont_item.get('description', 'Unknown')
                notes = dont_item.get('notes', '')
                answer_lines.append(f"  • {item_name}")
                if notes and notes != 'nan':
                    answer_lines.append(f"    ⚠️ Reason: {notes}")
        
        if not answer_lines:
            answer_lines.append("No specific do's and don'ts found for your question.")
        
        return "\n".join(answer_lines)
    
    def generate_ai_answer(self, question: str, dos: List[Dict], donts: List[Dict]) -> str:
        """Generate AI-powered answer using FLAN-T5."""
        if self.flan_model is None:
            return self.generate_dos_donts_answer(question, dos, donts)
        
        try:
            # Build context from do's and don'ts
            dos_context = "\n".join([f"✓ {d.get('description', '')}" for d in dos[:3]])
            donts_context = "\n".join([f"✗ {d.get('description', '')}" for d in donts[:3]])
            
            prompt = f"""You are a pregnancy nutrition expert. Answer this question with clear DO's and DON'Ts:

Question: {question}

DO's (Safe recommendations):
{dos_context if dos_context else "Not applicable"}

DON'Ts (Things to avoid):
{donts_context if donts_context else "Not applicable"}

Provide a helpful, clear answer that explains what the user should and shouldn't do, with brief explanations. Format with bullet points and use emojis for clarity. Keep it under 300 words."""
            
            import torch
            inputs = self.flan_tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.flan_model.generate(
                    inputs,
                    max_length=350,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    early_stopping=True
                )
            
            answer = self.flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            print(f"⚠ Generation error: {e}")
            return self.generate_dos_donts_answer(question, dos, donts)
    
    def answer_question(self, question: str, trimester: Optional[int] = None, 
                       use_ai_generation: bool = True) -> Dict:
        """
        Answer user question with do's and don'ts format.
        
        Args:
            question: User's question
            trimester: Current trimester (optional)
            use_ai_generation: Whether to use AI for enhanced answer generation
            
        Returns:
            Dictionary with answer, do's, and don'ts
        """
        # Classify intent and extract keywords
        intent = self.classify_intent(question)
        keywords = self.extract_keywords(question)
        
        # Get relevant recommendations
        dos, donts = self.get_relevant_recommendations(keywords, question)
        
        # Generate answer
        if use_ai_generation and self.flan_model is not None:
            answer_text = self.generate_ai_answer(question, dos, donts)
        else:
            answer_text = self.generate_dos_donts_answer(question, dos, donts)
        
        return {
            'success': True,
            'question': question,
            'answer': answer_text,
            'dos': [{'description': d.get('description', ''), 'notes': d.get('notes', '')} for d in dos],
            'donts': [{'description': d.get('description', ''), 'notes': d.get('notes', '')} for d in donts],
            'intent': intent,
            'keywords': keywords,
            'trimester': trimester
        }
    
    def answer_question_simple(self, question: str) -> str:
        """Simple wrapper for just getting answer text."""
        result = self.answer_question(question, use_ai_generation=True)
        return result['answer']


# Singleton instance
_ai_chatbot_instance = None


def get_ai_powered_chatbot() -> AIPoweredDosDontsChatbot:
    """Get or initialize the AI-powered chatbot (singleton)."""
    global _ai_chatbot_instance
    if _ai_chatbot_instance is None:
        _ai_chatbot_instance = AIPoweredDosDontsChatbot()
    return _ai_chatbot_instance


if __name__ == '__main__':
    # Test the chatbot
    chatbot = get_ai_powered_chatbot()
    
    # Load generation model for better answers
    chatbot.load_generation_model('google/flan-t5-small')
    
    test_questions = [
        "Can I eat papaya during pregnancy?",
        "Is it safe to exercise in first trimester?",
        "What should I avoid during pregnancy?",
        "Is raw fish safe for pregnant women?",
        "Can I drink coffee during pregnancy?"
    ]
    
    print("\n" + "="*80)
    print("TESTING AI-POWERED DO'S AND DON'Ts CHATBOT")
    print("="*80)
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        result = chatbot.answer_question(question, trimester=2)
        print(f"\n💬 Answer:\n{result['answer']}")
        print("-" * 80)
