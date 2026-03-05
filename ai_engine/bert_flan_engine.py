"""BERT + Flan-T5 Engine for AI-powered chatbot responses."""
import threading
import time
import os
import warnings
warnings.filterwarnings('ignore')


class BertFlanEngine:
    """
    Background-loading AI engine using BERT for semantic search 
    and Flan-T5 for answer generation.
    """
    
    def __init__(self):
        self._bert_tokenizer = None
        self._bert_model = None
        self._flan_tokenizer = None
        self._flan_model = None
        self._models_loaded = False
        self._loading = False
        self._load_error = None
        self._knowledge_embeddings = None
        self._knowledge_texts = []
        # Start background loading
        self._start_background_load()
    
    def _start_background_load(self):
        """Load models in a background thread so app starts fast."""
        if self._loading or self._models_loaded:
            return
        self._loading = True
        thread = threading.Thread(target=self._load_models_background, daemon=True)
        thread.start()
    
    def _load_models_background(self):
        """Actually load BERT and Flan-T5 models."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load BERT for semantic similarity/search
            print("🔄 Loading BERT model for semantic search...")
            self._bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self._bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
            self._bert_model.eval()
            print("✓ BERT model loaded")
            
            # Load Flan-T5 for answer generation
            print("🔄 Loading Flan-T5 model for answer generation...")
            self._flan_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
            self._flan_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small').to(device)
            self._flan_model.eval()
            print("✓ Flan-T5 model loaded")
            
            self._device = device
            self._models_loaded = True
            self._loading = False
            print("✅ All AI models loaded successfully!")
            
        except ImportError as e:
            print(f"⚠ AI models not available (missing dependency): {e}")
            self._loading = False
            self._load_error = str(e)
        except Exception as e:
            print(f"⚠ Error loading AI models: {e}")
            self._loading = False
            self._load_error = str(e)
    
    @property
    def is_ready(self):
        return self._models_loaded
    
    @property
    def is_loading(self):
        return self._loading
    
    def get_bert_embedding(self, text: str):
        """Get BERT embedding for a text string."""
        if not self._models_loaded:
            return None
        import torch
        tokens = self._bert_tokenizer(text, return_tensors='pt', truncation=True, 
                                       max_length=512, padding=True).to(self._device)
        with torch.no_grad():
            output = self._bert_model(**tokens)
        # Mean pooling
        embeddings = output.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().flatten()
    
    def semantic_search(self, query: str, knowledge_texts: list, top_k: int = 5):
        """Use BERT to find the most relevant knowledge texts for a query."""
        if not self._models_loaded:
            return knowledge_texts[:top_k]  # Fallback: return first items
        
        import numpy as np
        query_emb = self.get_bert_embedding(query)
        if query_emb is None:
            return knowledge_texts[:top_k]
        
        similarities = []
        for text in knowledge_texts:
            text_emb = self.get_bert_embedding(text)
            if text_emb is not None:
                sim = np.dot(query_emb, text_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(text_emb) + 1e-8
                )
                similarities.append((text, sim))
            else:
                similarities.append((text, 0.0))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in similarities[:top_k]]
    
    def generate_answer(self, question: str, context: str, max_length: int = 256) -> str:
        """Use Flan-T5 to generate an answer given question and context."""
        if not self._models_loaded:
            return ""
        
        prompt = (
            f"Answer the following question about pregnancy nutrition based on the context.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Provide a helpful, detailed answer:"
        )
        
        inputs = self._flan_tokenizer(prompt, return_tensors='pt', truncation=True, 
                                       max_length=512).to(self._device)
        
        with __import__('torch').no_grad():
            outputs = self._flan_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,
                early_stopping=True,
                temperature=0.7,
                do_sample=False
            )
        
        answer = self._flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


# Singleton instance - loads models once on import
_engine_instance = None


def get_engine() -> BertFlanEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = BertFlanEngine()
    return _engine_instance
