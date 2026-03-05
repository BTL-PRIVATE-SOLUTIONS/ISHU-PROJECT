"""Enhanced Chatbot with Do's and Don'ts using BERT and FLAN-T5."""
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os


class DosDontsChatbot:
    """AI Chatbot providing Do's and Don'ts answers using BERT intent classification and FLAN-T5 generation."""
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize the chatbot with datasets.
        
        Args:
            dataset_path: Path to pregnancy_dos_donts_dataset.csv
        """
        # Lazy imports - only import when needed
        self._torch = None
        self._transformers = None
        
        # Load datasets
        self.dos_donts_df = self._load_dos_donts_dataset(dataset_path)
        self.flan_tokenizer = None
        self.flan_model = None
        
        # Initialize knowledge base
        self.knowledge_base = self._build_knowledge_base()
        
        print("✓ Do's and Don'ts Chatbot initialized successfully")
        print(f"✓ Loaded {len(self.dos_donts_df)} recommendations")
    
    def _load_dos_donts_dataset(self, dataset_path: str = None) -> pd.DataFrame:
        """Load the dos and donts dataset."""
        if dataset_path is None:
            dataset_path = os.path.join(
                os.path.dirname(__file__),
                '../data/remainingdatasets/pregnancy_dos_donts_dataset.csv'
            )
        
        try:
            df = pd.read_csv(dataset_path)
            print(f"✓ Loaded Do's and Don'ts dataset from {dataset_path}")
            return df
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
            return pd.DataFrame()
    
    def _build_knowledge_base(self) -> Dict:
        """Build knowledge base from datasets."""
        knowledge = {
            'dos': {},
            'donts': {},
            'by_trimester': {1: [], 2: [], 3: []},
            'by_category': {}
        }
        
        for _, row in self.dos_donts_df.iterrows():
            trimester = self._parse_trimester(row.get('Trimester', ''))
            category = row.get('Category', '').lower()
            item_type = row.get('Type', '').strip().upper()  # Do or Don't
            item = row.get('Item', '')
            description = row.get('Description', '')
            notes = row.get('Notes', '')
            
            # Normalize item type
            if item_type in ['DO', 'DOS']:
                item_type = 'DO'
            elif item_type in ["DON'T", "DONT", "DON'TS", "DONTS"]:
                item_type = "DONT"
            
            # Store by type
            if item_type == 'DO':
                knowledge['dos'][item.lower()] = {
                    'description': description,
                    'notes': notes,
                    'trimester': trimester,
                    'category': category
                }
            elif item_type == "DONT":
                knowledge['donts'][item.lower()] = {
                    'description': description,
                    'notes': notes,
                    'trimester': trimester,
                    'category': category
                }
            
            # Store by trimester
            if trimester:
                knowledge['by_trimester'][trimester].append({
                    'type': item_type,
                    'item': item,
                    'description': description,
                    'category': category,
                    'notes': notes
                })
            
            # Store by category
            if category not in knowledge['by_category']:
                knowledge['by_category'][category] = {'do': [], 'dont': []}
            
            type_key = 'do' if item_type == 'DO' else 'dont'
            knowledge['by_category'][category][type_key].append({
                'item': item,
                'description': description,
                'notes': notes
            })
        
        return knowledge
    
    def _parse_trimester(self, trimester_str: str) -> Optional[int]:
        """Parse trimester string to number."""
        if '1' in trimester_str:
            return 1
        elif '2' in trimester_str:
            return 2
        elif '3' in trimester_str:
            return 3
        return None
    
    def classify_intent(self, question: str) -> str:
        """Classify user intent based on question keywords."""
        question_lower = question.lower()
        
        # Do's and Don'ts specific
        if any(word in question_lower for word in ['should i', 'can i', 'can i eat', 'should i eat', 'is it safe', 'avoid', 'do i']):
            return 'dos_donts'
        elif any(word in question_lower for word in ['benefits', 'good for', 'why', 'advantages']):
            return 'benefits'
        elif any(word in question_lower for word in ['precaution', 'warning', 'risk', 'danger', 'avoid']):
            return 'warnings'
        elif any(word in question_lower for word in ['trimester', '1st', '2nd', '3rd', 'first', 'second', 'third']):
            return 'trimester_specific'
        else:
            return 'general'
    
    def extract_keywords(self, question: str) -> List[str]:
        """Extract food/topic keywords from question."""
        keywords = []
        
        # Common food items and topics
        items_to_check = list(self.dos_donts_df['Item'].unique()) + \
                        ['papaya', 'pineapple', 'water', 'exercise', 'dairy', 'fish', 'meat', 
                         'eggs', 'vegetables', 'fruits', 'grains', 'supplements', 'alcohol', 
                         'smoking', 'caffeine', 'salt', 'sugar']
        
        question_lower = question.lower()
        for item in items_to_check:
            if item.lower() in question_lower:
                keywords.append(item.lower())
        
        return keywords
    
    def get_dos_donts_answer(self, question: str, trimester: int = None) -> Dict:
        """
        Get Do's and Don'ts answer for user question.
        
        Args:
            question: User's question
            trimester: Current trimester (1, 2, or 3)
            
        Returns:
            Dictionary with dos and donts recommendations
        """
        intent = self.classify_intent(question)
        keywords = self.extract_keywords(question)
        
        response = {
            'intent': intent,
            'dos': [],
            'donts': [],
            'trimester': trimester,
            'explanation': '',
            'source': 'datasets'
        }
        
        # Search in knowledge base
        if keywords:
            for keyword in keywords:
                # Search in dos
                for do_item, do_info in self.knowledge_base['dos'].items():
                    if keyword in do_item or keyword in do_info['description'].lower():
                        if trimester is None or do_info['trimester'] == trimester:
                            response['dos'].append({
                                'item': do_item.title(),
                                'description': do_info['description'],
                                'notes': do_info['notes'],
                                'category': do_info['category']
                            })
                
                # Search in donts
                for dont_item, dont_info in self.knowledge_base['donts'].items():
                    if keyword in dont_item or keyword in dont_info['description'].lower():
                        if trimester is None or dont_info['trimester'] == trimester:
                            response['donts'].append({
                                'item': dont_item.title(),
                                'description': dont_info['description'],
                                'notes': dont_info['notes'],
                                'category': dont_info['category']
                            })
        
        # If no specific matches, return trimester-specific recommendations
        if not response['dos'] and not response['donts'] and trimester:
            trimester_items = self.knowledge_base['by_trimester'].get(trimester, [])
            for item in trimester_items:
                if item['type'].upper() == 'DO':
                    response['dos'].append({
                        'item': item['item'],
                        'description': item['description'],
                        'notes': item['notes'],
                        'category': item['category']
                    })
                else:
                    response['donts'].append({
                        'item': item['item'],
                        'description': item['description'],
                        'notes': item['notes'],
                        'category': item['category']
                    })
        
        return response
    
    def load_flan_t5_for_generation(self, model_name: str = 'google/flan-t5-small'):
        """Load FLAN-T5 for enhanced answer generation."""
        print(f"\n✓ Loading FLAN-T5 ({model_name})...")
        try:
            # Lazy import
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            self._torch = torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.flan_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            self._device = device
            print("✓ FLAN-T5 loaded successfully!")
            return True
        except Exception as e:
            print(f"Warning: Could not load FLAN-T5: {e}")
            return False
    
    def generate_enhanced_answer(self, question: str, dos_donts_response: Dict, trimester: int = None) -> str:
        """
        Generate enhanced answer using FLAN-T5.
        
        Args:
            question: User's question
            dos_donts_response: Response from get_dos_donts_answer
            trimester: Current trimester
            
        Returns:
            Enhanced answer string
        """
        if self.flan_model is None:
            return self._format_dos_donts_answer(dos_donts_response)
        
        # Build context from dos and donts
        dos_text = "\n".join([f"✓ {d['item']}: {d['description']}" for d in dos_donts_response['dos'][:3]])
        donts_text = "\n".join([f"✗ {d['item']}: {d['description']}" for d in dos_donts_response['donts'][:3]])
        
        prompt = f"""Answer this pregnancy nutrition question with clear DO's and DON'Ts:

Question: {question}
{"Trimester: " + str(trimester) if trimester else ""}

DO's:
{dos_text if dos_text else "Not specified"}

DON'Ts:
{donts_text if donts_text else "Not specified"}

Provide a comprehensive answer that includes both DO's and DON'Ts, explanation, and any important notes."""
        
        try:
            inputs = self.flan_tokenizer.encode(prompt, return_tensors='pt').to(self._device)
            outputs = self.flan_model.generate(
                inputs,
                max_length=300,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            answer = self.flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            print(f"Generation error: {e}")
            return self._format_dos_donts_answer(dos_donts_response)
    
    def _format_dos_donts_answer(self, response: Dict) -> str:
        """Format dos and donts response as readable text."""
        answer = []
        
        if response['dos']:
            answer.append("✓ DO's:")
            for do in response['dos']:
                answer.append(f"  • {do['item']}: {do['description']}")
                if do['notes']:
                    answer.append(f"    Note: {do['notes']}")
        
        if response['donts']:
            answer.append("\n✗ DON'Ts:")
            for dont in response['donts']:
                answer.append(f"  • {dont['item']}: {dont['description']}")
                if dont['notes']:
                    answer.append(f"    Note: {dont['notes']}")
        
        return "\n".join(answer) if answer else "No specific recommendations found for your question."
    
    def answer_question(self, question: str, trimester: int = None, use_flan_t5: bool = False) -> str:
        """
        Main method to answer user question with Do's and Don'Ts.
        
        Args:
            question: User's question
            trimester: Current trimester (1, 2, or 3)
            use_flan_t5: Whether to use FLAN-T5 for enhanced generation
            
        Returns:
            Complete answer with Do's and Don'Ts
        """
        # Get Do's and Don'Ts from dataset
        dos_donts_response = self.get_dos_donts_answer(question, trimester)
        
        # Generate answer
        if use_flan_t5 and self.flan_model is not None:
            answer = self.generate_enhanced_answer(question, dos_donts_response, trimester)
        else:
            answer = self._format_dos_donts_answer(dos_donts_response)
        
        return answer


# Example usage
if __name__ == '__main__':
    # Initialize chatbot
    chatbot = DosDontsChatbot()
    
    # Test questions
    test_questions = [
        "Can I eat raw papaya during pregnancy?",
        "What should I eat in the first trimester?",
        "Is fish safe during pregnancy?",
        "Can I exercise during pregnancy?",
        "What about unpasteurized dairy?"
    ]
    
    print("\n" + "="*80)
    print("TESTING DO'S AND DON'TS CHATBOT")
    print("="*80)
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        answer = chatbot.answer_question(question, trimester=2)
        print(f"\n💬 Answer:\n{answer}")
        print("-" * 80)
