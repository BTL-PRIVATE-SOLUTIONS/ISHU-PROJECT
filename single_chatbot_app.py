"""
Single-file chatbot app (Flask) with multi-tier AI fallback, session tracking,
feedback mechanism, domain restriction, and multilingual support.

Run:
    python single_chatbot_app.py

Env vars:
    HOST (default 0.0.0.0)
    PORT (default 5000)
    GEMINI_API_KEY (optional - enables Google Gemini AI responses)
    HUGGINGFACE_API_KEY (optional - for HuggingFace cloud inference)

Datasets are read from the existing data/ folder next to this file.
"""

import difflib
import os
import time
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

# ---------------------------------------------------------------------------
# Optional AI dependency imports (graceful degradation when not installed)
# ---------------------------------------------------------------------------
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pregnancy domain keywords used for domain restriction
# ---------------------------------------------------------------------------
PREGNANCY_KEYWORDS = [
    "pregnan", "maternal", "maternity", "baby", "infant", "fetus", "foetus",
    "trimester", "birth", "delivery", "labor", "labour", "breast", "breastfeed",
    "lactation", "postpartum", "postnatal", "prenatal", "antenatal",
    "morning sickness", "nausea", "vitamin", "iron", "calcium", "folic",
    "protein", "nutrient", "diet", "food", "eat", "meal", "nutrition",
    "fruit", "vegetable", "vegetarian", "vegan", "weight", "exercise",
    "yoga", "walking", "swimming", "safe", "avoid", "unsafe",
    "diabetes", "diabetic", "glucose", "sugar", "blood sugar", "gestational",
    "monsoon", "summer", "winter", "seasonal", "north", "south",
    "swelling", "edema", "back pain", "heartburn", "constipation",
    "supplement", "medicine", "medication", "paracetamol", "iron tablet",
    "anxiety", "stress", "mood", "depress", "mental", "emotion", "worry",
    "week", "month", "development", "growth", "scan", "ultrasound",
    "doctor", "midwife", "obstetrician", "gynecologist", "hospital",
    "allergy", "gluten", "dairy", "milk", "egg", "fish", "meat",
    "recipe", "cook", "prepare", "snack", "beverage", "water", "hydration",
]

# ---------------------------------------------------------------------------
# Diabetes detection keywords
# ---------------------------------------------------------------------------
DIABETES_KEYWORDS = ["diabetes", "diabetic", "sugar", "blood sugar", "glucose"]

# ---------------------------------------------------------------------------
# Answer store: maps question_id -> full answer payload (for feedback regeneration)
# ---------------------------------------------------------------------------
_answer_store: Dict[str, Dict] = {}

DATA_BASE = os.path.join(os.path.dirname(__file__), "data")


class ResponseCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self.store: Dict[str, Dict] = {}

    def _key(self, question: str, context: str = "") -> str:
        raw = f"{question.strip().lower()}|{context.strip().lower()}".strip("|")
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, question: str, context: str = "") -> Optional[Dict]:
        key = self._key(question, context)
        item = self.store.get(key)
        if not item:
            return None
        if time.time() - item["ts"] > self.ttl:
            self.store.pop(key, None)
            return None
        return item["val"]

    def set(self, question: str, value: Dict, context: str = "") -> None:
        key = self._key(question, context)
        self.store[key] = {"val": value, "ts": time.time()}

    def clear(self) -> None:
        self.store.clear()


class SessionManager:
    """Manages per-user conversation history and state."""

    SESSION_TTL = 1800  # 30 minutes of inactivity expires the session

    def __init__(self):
        self._sessions: Dict[str, Dict] = {}

    def _make_session(self) -> Dict:
        return {
            "history": [],          # list of {"role": "user"|"bot", "text": str}
            "state": None,          # conversation sub-state (e.g. "awaiting_diabetes_type")
            "state_data": {},       # extra data for the current state
            "last_active": time.time(),
        }

    def get_or_create(self, session_id: str) -> Dict:
        """Return existing session or create a fresh one."""
        session = self._sessions.get(session_id)
        if session is None or time.time() - session["last_active"] > self.SESSION_TTL:
            session = self._make_session()
            self._sessions[session_id] = session
        session["last_active"] = time.time()
        return session

    def add_turn(self, session_id: str, user_text: str, bot_text: str) -> None:
        session = self.get_or_create(session_id)
        session["history"].append({"role": "user", "text": user_text})
        session["history"].append({"role": "bot", "text": bot_text})
        # Keep last 10 turns (20 messages) to limit memory
        session["history"] = session["history"][-20:]

    def set_state(self, session_id: str, state: Optional[str], state_data: Optional[Dict] = None) -> None:
        session = self.get_or_create(session_id)
        session["state"] = state
        session["state_data"] = state_data or {}

    def get_state(self, session_id: str) -> Tuple[Optional[str], Dict]:
        session = self.get_or_create(session_id)
        return session["state"], session["state_data"]

    def get_history_text(self, session_id: str, last_n: int = 4) -> str:
        """Return last N turns as a formatted string for AI context."""
        session = self.get_or_create(session_id)
        recent = session["history"][-last_n * 2:]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['text']}")
        return "\n".join(lines)


class GeminiAIProvider:
    """Google Gemini AI provider for generating context-aware responses."""

    SYSTEM_PROMPT = (
        "You are a helpful pregnancy nutrition and health assistant specializing in "
        "Indian maternal health and diet. Provide accurate, compassionate, and "
        "culturally relevant advice. Always add a brief medical disclaimer when giving "
        "health-related information. Keep answers concise (2-4 sentences unless more detail is requested)."
    )

    def __init__(self):
        self._model = None
        api_key = os.getenv("GEMINI_API_KEY", "")
        if _GEMINI_AVAILABLE and api_key and api_key not in ("your-gemini-api-key-here", ""):
            try:
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception:
                self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def generate_answer(self, question: str, context: str = "", language: str = "en") -> str:
        if not self.available:
            return ""
        try:
            prompt_parts = [self.SYSTEM_PROMPT]
            if context:
                prompt_parts.append(f"Conversation context:\n{context}")
            prompt_parts.append(f"User question: {question}")
            if language == "te":
                prompt_parts.append("Please respond in Telugu (తెలుగు).")
            full_prompt = "\n\n".join(prompt_parts)
            resp = self._model.generate_content(full_prompt)
            text = resp.text.strip() if resp.text else ""
            if text and language == "en" and "disclaimer" not in text.lower():
                text += (
                    "\n\n\u26a0\ufe0f *Disclaimer: This is informational only. "
                    "Please consult your healthcare provider for personalised medical advice.*"
                )
            return text
        except Exception:
            return ""

    def translate_to_telugu(self, text: str) -> str:
        if not self.available:
            return text
        try:
            prompt = (
                "Translate the following text to Telugu (\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41). "
                "Preserve any bullet points or line breaks:\n\n" + text
            )
            resp = self._model.generate_content(prompt)
            return resp.text.strip() if resp.text else text
        except Exception:
            return text


class BERTSimilarityProvider:
    """Sentence-BERT based semantic similarity for dataset matching and intent classification."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None
        self._corpus_texts: List[str] = []
        self._corpus_embeddings = None
        if _BERT_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.MODEL_NAME)
            except Exception:
                self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def build_corpus(self, texts: List[str]) -> None:
        if not self.available or not texts:
            return
        self._corpus_texts = texts
        try:
            self._corpus_embeddings = self._model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
        except Exception:
            self._corpus_embeddings = None

    def find_best_match(self, query: str, threshold: float = 0.4) -> Tuple[Optional[str], float]:
        """Return (best_matching_text, similarity_score) or (None, 0.0) if below threshold."""
        if not self.available or self._corpus_embeddings is None or not self._corpus_texts:
            return None, 0.0
        try:
            q_emb = self._model.encode([query], convert_to_numpy=True)
            norms_corpus = np.linalg.norm(self._corpus_embeddings, axis=1, keepdims=True)
            norm_q = np.linalg.norm(q_emb)
            if norm_q == 0:
                return None, 0.0
            sims = (self._corpus_embeddings @ q_emb.T).flatten() / (
                norms_corpus.flatten() * norm_q + 1e-9
            )
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= threshold:
                return self._corpus_texts[best_idx], best_score
        except Exception:
            pass
        return None, 0.0

    def is_related_to_domain(self, query: str, domain_texts: List[str], threshold: float = 0.3) -> bool:
        """Check if a query is semantically related to any domain description."""
        if not self.available:
            return False
        try:
            q_emb = self._model.encode([query], convert_to_numpy=True)
            d_embs = self._model.encode(domain_texts, convert_to_numpy=True)
            norms_d = np.linalg.norm(d_embs, axis=1, keepdims=True)
            norm_q = np.linalg.norm(q_emb)
            if norm_q == 0:
                return False
            sims = (d_embs @ q_emb.T).flatten() / (norms_d.flatten() * norm_q + 1e-9)
            return float(np.max(sims)) >= threshold
        except Exception:
            return False


class FeedbackStore:
    """In-memory store for user feedback analytics."""

    def __init__(self):
        self._store: Dict[str, Dict] = {}

    def record(self, question_id: str, question: str, answer: str, satisfied: bool, source: str) -> None:
        self._store[question_id] = {
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "satisfied": satisfied,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get(self, question_id: str) -> Optional[Dict]:
        return self._store.get(question_id)

    def all_records(self) -> List[Dict]:
        return list(self._store.values())

    def summary(self) -> Dict:
        total = len(self._store)
        satisfied = sum(1 for r in self._store.values() if r.get("satisfied"))
        return {
            "total_feedback": total,
            "satisfied": satisfied,
            "unsatisfied": total - satisfied,
            "satisfaction_rate": round(satisfied / total, 2) if total else 0.0,
        }


class UnifiedDatasetLoaderLite:
    """Lightweight loader that keeps the essential lookup behavior in one file."""

    def __init__(self, base_dir: str = DATA_BASE):
        self.base_dir = base_dir
        self.meals: List[Dict] = []
        self.guidance: List[Dict] = []
        self.food_index: Dict[str, Dict] = {}
        self.keyword_index: Dict[str, List[Dict]] = {}
        self.dos_donts_index: Dict[str, Dict] = {}
        self.dataset_configs = {
            "data_1": {
                "files": {
                    "northveg_cleaned.csv": {"region": "North", "diet": "veg", "category": "regional"},
                    "northnonveg_cleaned.csv": {"region": "North", "diet": "nonveg", "category": "regional"},
                    "northnonveg_cleaned (1).csv": {"region": "North", "diet": "nonveg", "category": "regional"},
                    "southveg_cleaned.csv": {"region": "South", "diet": "veg", "category": "regional"},
                    "southnonveg_cleaned.csv": {"region": "South", "diet": "nonveg", "category": "regional"},
                }
            },
            "data_2": {
                "files": {
                    "Trimester_Wise_Diet_Plan.csv": {"category": "trimester"},
                    "pregnancy_diet_1st_2nd_3rd_trimester.xlsx.csv": {"category": "trimester"},
                }
            },
            "data_3": {
                "files": {
                    "monsoon_diet_pregnant_women.csv": {"category": "seasonal", "season": "monsoon"},
                    "summer_pregnancy_diet.csv": {"category": "seasonal", "season": "summer"},
                    "Winter_Pregnancy_Diet.csv": {"category": "seasonal", "season": "winter"},
                }
            },
            "diabetiesdatasets": {
                "files": {
                    "diabetes_pregnancy_indian_foods.csv": {"category": "special_condition", "condition": "diabetes"},
                    "gestational_diabetes_indian_diet_dataset.csv": {"category": "special_condition", "condition": "gestational_diabetes"},
                    "Indian_Diabetes_Diet (1).csv": {"category": "special_condition", "condition": "diabetes"},
                }
            },
            "remainingdatasets": {
                "files": {
                    "foods_to_avoid_during_pregnancy_dataset.csv": {"category": "guidance", "type": "avoid"},
                    "indian_diet_diabetes_pregnancy_dataset.csv": {"category": "special_condition", "condition": "diabetes"},
                    "postnatal_diet_india_dataset.csv": {"category": "postpartum"},
                    "postpartum_diet7_structured_dataset.csv": {"category": "postpartum"},
                    "pregnancy_diet_clean_dataset.csv": {"category": "general"},
                    "pregnancy_dos_donts_dataset.csv": {"category": "guidance", "type": "dos_donts"},
                }
            },
        }
        self._load_all()
        self._build_indexes()

    def _load_csv(self, path: str) -> Optional[pd.DataFrame]:
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "ascii"]
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                df = df.dropna(how="all")
                return df if len(df) else None
            except Exception:
                continue
        return None

    def _load_all(self) -> None:
        for folder, cfg in self.dataset_configs.items():
            folder_path = os.path.join(self.base_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for filename, meta in cfg["files"].items():
                file_path = os.path.join(folder_path, filename)
                if not os.path.exists(file_path):
                    continue
                df = self._load_csv(file_path)
                if df is None:
                    continue
                meta_lower = {k: (v.lower() if isinstance(v, str) else v) for k, v in meta.items()}
                for col, val in meta_lower.items():
                    df[f"source_{col}"] = val
                records = df.to_dict("records")
                if meta_lower.get("category") == "guidance" or meta_lower.get("type") in {"dos_donts", "avoid"}:
                    self.guidance.extend(records)
                else:
                    self.meals.extend(records)

    def _build_indexes(self) -> None:
        for meal in self.meals:
            for col in ["food", "food_item", "meal", "dish", "item", "recipe", "dish_name", "meal_name"]:
                if meal.get(col):
                    name = str(meal[col]).strip().lower()
                    if name:
                        self.food_index[name] = meal
                        for word in name.split():
                            if len(word) > 2:
                                self.keyword_index.setdefault(word, []).append(meal)
        for g in self.guidance:
            for col in ["item", "food", "food_item", "do", "dont", "description"]:
                if g.get(col):
                    name = str(g[col]).strip().lower()
                    if name:
                        self.dos_donts_index[name] = g

    def quick_lookup(self, query: str) -> Dict:
        q = query.strip().lower()

        # 1. Exact match
        if q in self.food_index:
            return {"found": True, "data": self.food_index[q], "type": "food"}
        if q in self.dos_donts_index:
            return {"found": True, "data": self.dos_donts_index[q], "type": "dos_donts"}

        # 2. Keyword match with relevance ranking (count keyword hits per candidate)
        keyword_scores: Dict[str, int] = {}
        for word in q.split():
            if word in self.keyword_index:
                for candidate in self.keyword_index[word]:
                    for col in ["food", "food_item", "meal", "dish", "item", "recipe"]:
                        if candidate.get(col):
                            key = str(candidate[col]).strip().lower()
                            keyword_scores[key] = keyword_scores.get(key, 0) + 1
                            break
        if keyword_scores:
            best_key = max(keyword_scores, key=lambda k: keyword_scores[k])
            if best_key in self.food_index:
                return {"found": True, "data": self.food_index[best_key], "type": "food"}

        # 3. Fuzzy match on food index keys
        all_food_keys = list(self.food_index.keys())
        if all_food_keys:
            matches = difflib.get_close_matches(q, all_food_keys, n=1, cutoff=0.6)
            if matches:
                return {"found": True, "data": self.food_index[matches[0]], "type": "food", "fuzzy": True}

        # 4. Fuzzy match on dos_donts index
        all_guidance_keys = list(self.dos_donts_index.keys())
        if all_guidance_keys:
            matches = difflib.get_close_matches(q, all_guidance_keys, n=1, cutoff=0.6)
            if matches:
                return {
                    "found": True,
                    "data": self.dos_donts_index[matches[0]],
                    "type": "dos_donts",
                    "fuzzy": True,
                }

        return {"found": False, "data": None, "type": None}

    def get_meals_by_preference(
        self,
        region: Optional[str] = None,
        diet_type: Optional[str] = None,
        trimester: Optional[int] = None,
        season: Optional[str] = None,
        condition: Optional[str] = None,
        meal_type: Optional[str] = None,
    ) -> List[Dict]:
        results = []
        rnorm = region.lower() if region else None
        dnorm = diet_type.lower() if diet_type else None
        snorm = season.lower() if season else None
        cnorm = condition.lower() if condition else None
        mnorm = meal_type.lower() if meal_type else None
        for meal in self.meals:
            ok = True
            # region/diet filters
            s_region = meal.get("source_region")
            if rnorm and s_region and s_region not in {"all", rnorm}:
                ok = False
            s_diet = meal.get("source_diet")
            if dnorm and s_diet and s_diet not in {"all", dnorm}:
                ok = False
            s_season = meal.get("source_season")
            if snorm and s_season and s_season not in {"all", snorm}:
                ok = False
            s_cond = meal.get("source_condition")
            if cnorm and s_cond and s_cond != cnorm:
                ok = False
            s_meal_type = meal.get("meal_type") or meal.get("meal") or meal.get("mealname")
            if mnorm and s_meal_type and mnorm not in str(s_meal_type).lower():
                ok = False
            if trimester:
                tri_col = meal.get("trimester") or meal.get("source_trimester")
                if tri_col:
                    tri_str = str(tri_col).lower()
                    if str(trimester) not in tri_str and "all" not in tri_str:
                        ok = False
            if ok:
                results.append(meal)
        return results


class SingleChatbot:
    """All-in-one chatbot with multi-tier AI fallback, session tracking, and feedback support."""

    _DOMAIN_DESCRIPTIONS = [
        "pregnancy diet and nutrition",
        "maternal health and wellness",
        "food to eat during pregnancy",
        "morning sickness nausea pregnancy",
        "baby development during pregnancy",
        "trimester specific diet",
        "postpartum breastfeeding nutrition",
        "gestational diabetes diet",
        "pregnancy exercise yoga walking",
        "prenatal vitamins supplements",
    ]

    def __init__(self):
        self.loader = UnifiedDatasetLoaderLite()
        self.cache = ResponseCache(ttl_seconds=3600)
        self.sessions = SessionManager()
        self.feedback = FeedbackStore()
        self.gemini = GeminiAIProvider()
        self.bert = BERTSimilarityProvider()
        # Build BERT corpus from food index keys for semantic search
        if self.bert.available:
            corpus = list(self.loader.food_index.keys()) + list(self.loader.dos_donts_index.keys())
            self.bert.build_corpus(corpus[:5000])  # limit to avoid memory issues

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_intent(self, question: str) -> str:
        q = question.lower()
        # Check specific intents FIRST before generic safety/benefits
        if any(w in q for w in ["morning sickness", "nausea", "vomit", "back pain", "swelling", "edema",
                                  "heartburn", "constipation", "cramp", "headache", "fatigue", "tired"]):
            return "symptoms"
        if any(w in q for w in ["meal plan", "diet plan", "what to eat", "menu", "breakfast", "lunch", "dinner"]):
            return "meal_plan"
        if any(w in q for w in ["exercise", "yoga", "walk", "swim", "gym", "workout", "physical activity"]):
            return "exercise"
        if any(w in q for w in ["medicine", "medication", "paracetamol", "tablet", "drug", "pill", "supplement"]):
            return "medication"
        if any(w in q for w in ["stress", "anxiety", "mood", "depress", "mental", "emotion", "worry", "fear"]):
            return "mental_health"
        if any(w in q for w in ["monsoon", "summer", "winter", "rainy", "season", "weather"]):
            return "seasonal"
        if any(w in q for w in ["after delivery", "postpartum", "postnatal", "breastfeed", "lactation", "after birth"]):
            return "postpartum"
        if any(w in q for w in DIABETES_KEYWORDS):
            return "diabetes"
        if any(w in q for w in ["trimester", "1st", "2nd", "3rd", "first trimester", "second trimester", "third trimester"]):
            return "trimester"
        if any(w in q for w in ["can i", "safe", "avoid", "dangerous", "should i", "is it okay", "allowed"]):
            return "safety"
        if any(w in q for w in ["benefit", "good for", "nutrient", "vitamin", "protein", "iron", "calcium", "folic"]):
            return "benefits"
        return "general"

    def _is_pregnancy_related(self, question: str) -> bool:
        """Return True if the question appears to be pregnancy/maternal health related."""
        q_lower = question.lower()
        for kw in PREGNANCY_KEYWORDS:
            if kw in q_lower:
                return True
        # BERT semantic check as secondary (only when keyword check fails)
        if self.bert.available:
            return self.bert.is_related_to_domain(question, self._DOMAIN_DESCRIPTIONS, threshold=0.25)
        return False

    def _has_diabetes_keyword(self, question: str) -> bool:
        q_lower = question.lower()
        return any(kw in q_lower for kw in DIABETES_KEYWORDS)

    def extract_keywords(self, question: str) -> List[str]:
        q = question.lower()
        tokens = [t.strip("?,.! ") for t in q.split() if len(t) > 2]
        uniq = []
        for t in tokens:
            if t not in uniq:
                uniq.append(t)
        return uniq[:5]

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_food_answer(self, meal: Dict, trimester: Optional[int]) -> str:
        name = (
            meal.get("food") or meal.get("food_item") or meal.get("meal")
            or meal.get("dish") or meal.get("item") or "This food"
        )
        region = meal.get("source_region", "All")
        diet = meal.get("source_diet", "all")
        source_cat = meal.get("source_category", meal.get("category", "dataset"))
        basics = [f"Name: {name}", f"Region: {region}", f"Diet: {diet}", f"Source: {source_cat}"]
        if trimester:
            basics.append(f"Trimester focus: {trimester}")
        extras = []
        for key in ["benefits", "notes", "remarks", "health_benefit", "description"]:
            if meal.get(key):
                extras.append(f"Notes: {meal[key]}")
                break
        return "\n".join(basics + extras)

    def _format_dos_donts(self, entry: Dict) -> Tuple[List[str], List[str]]:
        dos = []
        donts = []
        text = entry.get("description") or entry.get("notes") or entry.get("item") or ""
        if entry.get("type") in {"DO", "DOS"}:
            dos.append(text)
        elif entry.get("type") in {"DON'T", "DONT", "DONTS", "DONT'S"}:
            donts.append(text)
        else:
            if "avoid" in text.lower() or "not" in text.lower():
                donts.append(text)
            else:
                dos.append(text)
        return dos, donts

    def _rule_based_fallback(
        self,
        question: str,
        intent: str,
        trimester: Optional[int],
        region: Optional[str],
        season: Optional[str],
    ) -> str:
        """Comprehensive rule-based fallback that always returns useful information."""
        q = question.lower()
        disclaimer = (
            "\n\n\u26a0\ufe0f *Disclaimer: This is informational only. "
            "Please consult your healthcare provider for personalised medical advice.*"
        )
        if intent == "exercise":
            return (
                "During pregnancy, gentle exercise like walking (30 minutes daily), prenatal yoga, "
                "and swimming are generally safe and beneficial. Avoid high-impact activities, contact "
                "sports, and exercises lying flat on your back after the first trimester. Always consult "
                "your doctor before starting any new exercise routine during pregnancy." + disclaimer
            )
        if intent == "symptoms":
            if "morning sickness" in q or "nausea" in q:
                return (
                    "For morning sickness, try eating small, frequent meals (every 2 hours), "
                    "ginger tea or ginger biscuits, dry crackers before getting out of bed, "
                    "and staying hydrated with sips of water. Avoid spicy or fatty foods. "
                    "Most morning sickness improves after the first trimester." + disclaimer
                )
            return (
                "Pregnancy symptoms vary by trimester. Staying well-hydrated, eating balanced meals, "
                "getting adequate rest, and gentle exercise can help manage many common discomforts. "
                "Report any severe or unusual symptoms to your doctor promptly." + disclaimer
            )
        if intent == "medication":
            return (
                "During pregnancy, always consult your doctor before taking any medication, "
                "including over-the-counter drugs. Paracetamol (acetaminophen) is generally "
                "considered safe in recommended doses, but avoid NSAIDs like ibuprofen. "
                "Prenatal vitamins with folic acid, iron, and calcium are usually recommended." + disclaimer
            )
        if intent == "mental_health":
            return (
                "Mental health is important during pregnancy. Stress and anxiety are common but can "
                "be managed through prenatal yoga, meditation, talking to loved ones, and professional "
                "counselling. Share how you're feeling with your healthcare provider — they can refer "
                "you to appropriate support." + disclaimer
            )
        if intent == "seasonal":
            season_tips = {
                "monsoon": (
                    "During monsoon, focus on warm, freshly cooked foods. Avoid raw salads and street food. "
                    "Include ginger, turmeric, and warm soups. Stay hydrated with boiled or purified water."
                ),
                "summer": (
                    "In summer, stay hydrated with coconut water, buttermilk (lassi), and fresh fruits like "
                    "watermelon. Eat light, easily digestible meals. Avoid heavy fried foods."
                ),
                "winter": (
                    "In winter, warm nutritious foods like soups, dals, and dry fruits are excellent. "
                    "Include seasonal vegetables like carrots, spinach, and fenugreek. "
                    "Ensure adequate Vitamin D from sunlight."
                ),
            }
            for s, tip in season_tips.items():
                if s in q:
                    return tip + disclaimer
            return (
                "Seasonal eating during pregnancy means choosing fresh, locally available produce. "
                "Adjust meal temperatures (warm in winter, cool in summer) and always ensure food "
                "safety during monsoon season." + disclaimer
            )
        if intent == "postpartum":
            return (
                "After delivery, focus on nutrient-dense foods to support recovery and breastfeeding. "
                "Include iron-rich foods (leafy greens, lentils), calcium (dairy, sesame), and protein. "
                "Traditional Indian postpartum foods like methi ladoo, dink ladoo, and warm soups are "
                "excellent. Stay well-hydrated to support milk production." + disclaimer
            )
        if intent == "trimester":
            tri_tips = {
                "1": (
                    "In the 1st trimester, focus on folic acid (leafy greens, fortified cereals), "
                    "small frequent meals to manage nausea, and staying hydrated. "
                    "Avoid alcohol and limit caffeine."
                ),
                "2": (
                    "In the 2nd trimester, increase calories slightly (~340 extra/day). "
                    "Focus on calcium, iron, and omega-3s. Include dairy, leafy greens, "
                    "fish (low-mercury), and legumes."
                ),
                "3": (
                    "In the 3rd trimester, focus on easy-to-digest foods, fibre to prevent "
                    "constipation, and calcium for baby's bone development. "
                    "Eat smaller meals as the baby takes up space."
                ),
            }
            for t, tip in tri_tips.items():
                if t in q or f"{t}st" in q or f"{t}nd" in q or f"{t}rd" in q:
                    return tip + disclaimer
        # General pregnancy nutrition fallback
        return (
            "A balanced pregnancy diet should include: whole grains, lean protein (lentils, legumes, "
            "eggs, fish), dairy for calcium, colourful fruits and vegetables, and adequate water "
            "(8-10 glasses/day). Key nutrients are folic acid, iron, calcium, and omega-3 fatty acids. "
            "Avoid raw/undercooked foods, high-mercury fish, and unpasteurised products." + disclaimer
        )

    # ------------------------------------------------------------------
    # Core answer method
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        trimester: Optional[int] = None,
        region: Optional[str] = None,
        season: Optional[str] = None,
        session_id: Optional[str] = None,
        language: str = "en",
        skip_sources: Optional[List[str]] = None,
    ) -> Dict:
        question = question.strip()
        if not question:
            return {"error": "Question is required"}

        skip_sources = skip_sources or []
        if session_id is None:
            session_id = str(uuid.uuid4())

        # ------------------------------------------------------------------
        # Step 0: Check session state FIRST (e.g., awaiting diabetes follow-up)
        # User's follow-up response might not contain pregnancy keywords.
        # ------------------------------------------------------------------
        state, state_data = self.sessions.get_state(session_id)
        if state == "awaiting_diabetes_type":
            return self._handle_diabetes_followup(
                question, session_id, trimester, region, season, language, state_data
            )

        # ------------------------------------------------------------------
        # Step 1: Domain restriction check
        # ------------------------------------------------------------------
        if not self._is_pregnancy_related(question):
            payload: Dict = {
                "question": question,
                "answer": (
                    "I'm sorry, I can only assist with pregnancy-related queries. "
                    "Please ask me about pregnancy nutrition, diet, symptoms, or maternal health."
                ),
                "dos": [],
                "donts": [],
                "source": "domain_restriction",
                "question_id": str(uuid.uuid4()),
                "session_id": session_id,
                "trimester": trimester,
                "region": region,
                "season": season,
            }
            self.sessions.add_turn(session_id, question, payload["answer"])
            return payload

        # ------------------------------------------------------------------
        # Step 2: Diabetes detection — ask clarifying follow-up
        # ------------------------------------------------------------------
        if self._has_diabetes_keyword(question) and "dataset" not in skip_sources:
            followup = (
                "Are you asking about pre-existing diabetes or gestational diabetes "
                "(diabetes that develops during pregnancy)?"
            )
            self.sessions.set_state(
                session_id, "awaiting_diabetes_type", {"original_question": question}
            )
            self.sessions.add_turn(session_id, question, followup)
            return {
                "question": question,
                "answer": followup,
                "awaiting_followup": True,
                "dos": [],
                "donts": [],
                "source": "diabetes_followup",
                "question_id": str(uuid.uuid4()),
                "session_id": session_id,
                "trimester": trimester,
                "region": region,
                "season": season,
            }

        # Cache lookup (skip when regenerating with skip_sources)
        if not skip_sources:
            cached = self.cache.get(question)
            if cached:
                return {**cached, "source": cached.get("source", "cache"), "cached": True}

        start = time.time()
        intent = self.classify_intent(question)
        dos_list: List[str] = []
        donts_list: List[str] = []
        answer_text = ""
        source = "dataset"

        # ------------------------------------------------------------------
        # Tier 1: Dataset search
        # ------------------------------------------------------------------
        if "dataset" not in skip_sources:
            quick = self.loader.quick_lookup(question)
            if quick["found"]:
                if quick["type"] == "food":
                    answer_text = self._format_food_answer(quick["data"], trimester)
                else:
                    d, n = self._format_dos_donts(quick["data"])
                    dos_list.extend(d)
                    donts_list.extend(n)
                    answer_text = quick["data"].get("description", "Guidance available.")
                source = "dataset"
            elif intent == "meal_plan":
                meals = self.loader.get_meals_by_preference(
                    region=region, diet_type=None, trimester=trimester, season=season
                )
                if meals:
                    names = []
                    for meal in meals[:5]:
                        for col in ["food", "food_item", "meal", "dish", "item", "recipe"]:
                            if meal.get(col):
                                names.append(str(meal[col]))
                                break
                    answer_text = "Sample meals: " + ", ".join(names)
                    source = "dataset"

        # ------------------------------------------------------------------
        # Tier 2: BERT semantic similarity
        # ------------------------------------------------------------------
        if not answer_text and "bert" not in skip_sources and self.bert.available:
            best_text, score = self.bert.find_best_match(question, threshold=0.45)
            if best_text and score > 0:
                if best_text in self.loader.food_index:
                    meal_data = self.loader.food_index[best_text]
                    answer_text = self._format_food_answer(meal_data, trimester)
                    source = "bert_similarity"
                elif best_text in self.loader.dos_donts_index:
                    entry = self.loader.dos_donts_index[best_text]
                    d, n = self._format_dos_donts(entry)
                    dos_list.extend(d)
                    donts_list.extend(n)
                    answer_text = entry.get("description", best_text)
                    source = "bert_similarity"

        # ------------------------------------------------------------------
        # Tier 3: Google Gemini AI
        # ------------------------------------------------------------------
        if not answer_text and "gemini" not in skip_sources and self.gemini.available:
            history_context = self.sessions.get_history_text(session_id) if session_id else ""
            ai_answer = self.gemini.generate_answer(
                question, context=history_context, language=language
            )
            if ai_answer:
                answer_text = ai_answer
                source = "gemini"

        # ------------------------------------------------------------------
        # Tier 4: Rule-based comprehensive fallback
        # ------------------------------------------------------------------
        if not answer_text:
            answer_text = self._rule_based_fallback(question, intent, trimester, region, season)
            source = "rule_based"

        # Translate to Telugu if requested and not already in Telugu from Gemini
        if language == "te" and source != "gemini" and self.gemini.available:
            translated = self.gemini.translate_to_telugu(answer_text)
            if translated:
                answer_text = translated

        duration = time.time() - start
        question_id = str(uuid.uuid4())
        payload = {
            "question": question,
            "answer": answer_text,
            "dos": dos_list,
            "donts": donts_list,
            "source": source,
            "response_time": round(duration, 3),
            "trimester": trimester,
            "region": region,
            "season": season,
            "question_id": question_id,
            "session_id": session_id,
            "intent": intent,
        }

        # Cache only non-AI, non-regenerated responses
        if not skip_sources and source in {"dataset", "rule_based", "bert_similarity"}:
            self.cache.set(question, payload)

        # Store for feedback regeneration
        _answer_store[question_id] = {
            "question": question,
            "answer": answer_text,
            "source": source,
            "trimester": trimester,
            "region": region,
            "season": season,
            "session_id": session_id,
            "language": language,
        }

        self.sessions.add_turn(session_id, question, answer_text)
        return payload

    def _handle_diabetes_followup(
        self,
        response: str,
        session_id: str,
        trimester: Optional[int],
        region: Optional[str],
        season: Optional[str],
        language: str,
        state_data: Dict,
    ) -> Dict:
        """Handle the user's response to the diabetes type follow-up question."""
        self.sessions.set_state(session_id, None)
        r = response.lower()
        original_question = state_data.get("original_question", "diabetes diet")

        if any(w in r for w in ["gestational", "pregnancy diabetes", "during pregnancy", "pregnant"]):
            condition = "gestational_diabetes"
            condition_label = "gestational diabetes"
        else:
            condition = "diabetes"
            condition_label = "pre-existing diabetes"

        meals = self.loader.get_meals_by_preference(
            condition=condition, trimester=trimester, region=region, season=season
        )
        disclaimer = (
            "\n\n\u26a0\ufe0f *Disclaimer: This is informational only. "
            "Please consult your healthcare provider for personalised medical advice.*"
        )
        if meals:
            names = []
            for meal in meals[:5]:
                for col in ["food", "food_item", "meal", "dish", "item", "recipe"]:
                    if meal.get(col):
                        names.append(str(meal[col]))
                        break
            answer_text = (
                f"For {condition_label} during pregnancy, here are some recommended foods: "
                + ", ".join(names)
                + ". Focus on low-glycaemic index foods, high fibre, and balanced meals."
                + disclaimer
            )
            source = "dataset"
        else:
            answer_text = self._rule_based_fallback(
                original_question, "diabetes", trimester, region, season
            )
            source = "rule_based"

        if language == "te" and self.gemini.available:
            translated = self.gemini.translate_to_telugu(answer_text)
            if translated:
                answer_text = translated

        question_id = str(uuid.uuid4())
        payload = {
            "question": response,
            "answer": answer_text,
            "dos": [],
            "donts": [],
            "source": source,
            "question_id": question_id,
            "session_id": session_id,
            "trimester": trimester,
            "region": region,
            "season": season,
            "intent": "diabetes",
        }
        _answer_store[question_id] = {
            "question": response,
            "answer": answer_text,
            "source": source,
            "trimester": trimester,
            "region": region,
            "season": season,
            "session_id": session_id,
            "language": language,
        }
        self.sessions.add_turn(session_id, response, answer_text)
        return payload

    def answer_structured(
        self,
        question: str,
        trimester: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        result = self.answer(question, trimester=trimester, session_id=session_id)
        if result.get("dos") or result.get("donts"):
            return result
        text = result.get("answer", "")
        if "avoid" in text.lower():
            result.setdefault("donts", []).append(text)
        else:
            result.setdefault("dos", []).append(text)
        return result

    def meal_plan_preview(
        self,
        region: Optional[str],
        diet_type: Optional[str],
        trimester: Optional[int],
        season: Optional[str],
        condition: Optional[str] = None,
        limit: int = 5,
    ) -> Dict:
        meals = self.loader.get_meals_by_preference(
            region=region,
            diet_type=diet_type,
            trimester=trimester,
            season=season,
            condition=condition,
        )
        preview = []
        for meal in meals[:limit]:
            name = None
            for col in ["food", "food_item", "meal", "dish", "item", "recipe"]:
                if meal.get(col):
                    name = str(meal[col])
                    break
            preview.append({
                "name": name or "Meal",
                "region": meal.get("source_region"),
                "diet": meal.get("source_diet"),
                "season": meal.get("source_season"),
                "trimester": meal.get("trimester") or meal.get("source_trimester"),
                "condition": meal.get("source_condition"),
                "notes": meal.get("description") or meal.get("remarks") or meal.get("health_benefit"),
            })
        return {
            "count": len(preview),
            "meals": preview,
            "filters_applied": {
                "region": region,
                "diet_type": diet_type,
                "trimester": trimester,
                "season": season,
                "condition": condition,
            },
        }

    def regenerate_answer(self, question_id: str) -> Optional[Dict]:
        """Regenerate an answer using a different source in the fallback chain."""
        stored = _answer_store.get(question_id)
        if not stored:
            return None
        current_source = stored.get("source", "")
        # Determine which sources to skip to force a different tier
        if current_source == "dataset":
            skip_sources: List[str] = ["dataset"]
        elif current_source == "bert_similarity":
            skip_sources = ["dataset", "bert"]
        elif current_source == "gemini":
            skip_sources = ["dataset", "bert", "gemini"]
        else:
            skip_sources = ["dataset", "bert", "gemini"]

        return self.answer(
            question=stored["question"],
            trimester=stored.get("trimester"),
            region=stored.get("region"),
            season=stored.get("season"),
            session_id=stored.get("session_id"),
            language=stored.get("language", "en"),
            skip_sources=skip_sources,
        )


# Flask wiring in the same file

def create_app() -> Flask:
    app = Flask(__name__)
    bot = SingleChatbot()

    @app.route("/")
    def health() -> Tuple[str, int]:
        return "Chatbot service is running", 200

    @app.route("/chatbot/ask", methods=["POST"])
    def ask():
        data = request.get_json(force=True, silent=True) or {}
        question = str(data.get("question", "")).strip()
        trimester = data.get("trimester")
        region = data.get("region")
        season = data.get("season")
        session_id = data.get("session_id") or str(uuid.uuid4())
        language = str(data.get("language", "en")).lower()
        result = bot.answer(
            question=question,
            trimester=trimester,
            region=region,
            season=season,
            session_id=session_id,
            language=language,
        )
        status = 200 if not result.get("error") else 400
        return jsonify(result), status

    @app.route("/chatbot/feedback", methods=["POST"])
    def feedback():
        data = request.get_json(force=True, silent=True) or {}
        question_id = str(data.get("question_id", "")).strip()
        satisfied = data.get("satisfied")
        if not question_id:
            return jsonify({"error": "question_id is required"}), 400
        if satisfied is None:
            return jsonify({"error": "satisfied (true/false) is required"}), 400

        stored = _answer_store.get(question_id)
        if not stored:
            return jsonify({"error": "question_id not found"}), 404

        bot.feedback.record(
            question_id=question_id,
            question=stored.get("question", ""),
            answer=stored.get("answer", ""),
            satisfied=bool(satisfied),
            source=stored.get("source", ""),
        )

        result: Dict = {"recorded": True, "question_id": question_id, "satisfied": bool(satisfied)}

        # If unsatisfied, regenerate using a different source
        if not satisfied:
            regenerated = bot.regenerate_answer(question_id)
            if regenerated:
                result["regenerated_answer"] = regenerated

        return jsonify(result), 200

    @app.route("/chatbot/feedback/summary", methods=["GET"])
    def feedback_summary():
        return jsonify(bot.feedback.summary()), 200

    @app.route("/chatbot/dos-donts", methods=["POST"])
    def dos_donts():
        data = request.get_json(force=True, silent=True) or {}
        question = str(data.get("question", "")).strip()
        trimester = data.get("trimester")
        session_id = data.get("session_id")
        result = bot.answer_structured(question=question, trimester=trimester, session_id=session_id)
        status = 200 if not result.get("error") else 400
        return jsonify(result), status

    @app.route("/chatbot/mealplan", methods=["POST"])
    def mealplan():
        data = request.get_json(force=True, silent=True) or {}
        region = data.get("region")
        diet_type = data.get("diet_type")
        trimester = data.get("trimester")
        season = data.get("season")
        condition = data.get("condition")
        limit = int(data.get("limit", 5))
        preview = bot.meal_plan_preview(
            region=region,
            diet_type=diet_type,
            trimester=trimester,
            season=season,
            condition=condition,
            limit=limit,
        )
        return jsonify({
            "success": True,
            "preview": preview,
            "region": region,
            "diet_type": diet_type,
            "trimester": trimester,
            "season": season,
            "condition": condition,
        })

    @app.route("/chatbot/suggestions", methods=["GET"])
    def suggestions():
        trimester = int(request.args.get("trimester", 2)) if request.args.get("trimester") else 2
        return jsonify({
            "suggestions": [
                f"What should I eat in trimester {trimester}?",
                "What foods should I avoid during pregnancy?",
                "Can I eat eggs during pregnancy?",
                "Is fish safe during pregnancy?",
                "What are good sources of iron?",
                "Which fruits are best for pregnancy?",
                "What is a good meal plan for today?",
                "What foods help with morning sickness?",
                "Is walking safe during pregnancy?",
                "What is a monsoon diet for pregnant women?",
                "Can I do yoga during pregnancy?",
                "What vitamins do I need during pregnancy?",
            ],
            "trimester": trimester,
        })

    @app.route("/chatbot/all", methods=["POST"])
    def all_in_one():
        data = request.get_json(force=True, silent=True) or {}
        question = str(data.get("question", "")).strip()
        trimester = data.get("trimester")
        region = data.get("region")
        diet_type = data.get("diet_type")
        season = data.get("season")
        session_id = data.get("session_id") or str(uuid.uuid4())
        language = str(data.get("language", "en")).lower()

        answer_result = bot.answer(
            question=question,
            trimester=trimester,
            region=region,
            season=season,
            session_id=session_id,
            language=language,
        )
        dos_donts_result = bot.answer_structured(
            question=question, trimester=trimester, session_id=session_id
        )
        meal_preview = bot.meal_plan_preview(
            region=region, diet_type=diet_type, trimester=trimester, season=season
        )

        return jsonify({
            "question": question,
            "trimester": trimester,
            "region": region,
            "diet_type": diet_type,
            "season": season,
            "session_id": session_id,
            "answer": answer_result,
            "dos_donts": {
                "dos": dos_donts_result.get("dos", []),
                "donts": dos_donts_result.get("donts", []),
                "source": dos_donts_result.get("source"),
            },
            "meal_plan_preview": meal_preview,
            "suggestions": suggestions().get_json().get("suggestions", []),
        })

    @app.route("/ui")
    def ui():
        # Simple inline HTML UI for quick access to all features
        return (
            "<html><head><title>Chatbot UI</title></head><body>"
            "<h2>Pregnancy Nutrition Chatbot</h2>"
            "<form id='askForm'>"
            "Question: <input name='question' size='60' value='What should I eat in trimester 2?'><br>"
            "Trimester: <input name='trimester' value='2' size='3'>"
            "Region: <input name='region' value='North' size='8'>"
            "Diet: <input name='diet_type' value='veg' size='6'>"
            "Season: <input name='season' value='' size='8'>"
            "Language: <input name='language' value='en' size='4'>"
            "Session: <input name='session_id' value='' size='36' placeholder='leave blank for new session'><br>"
            "<button type='submit'>Ask</button>"
            "</form>"
            "<pre id='output'></pre>"
            "<script>"
            "document.getElementById('askForm').onsubmit = async (e) => {e.preventDefault();"
            "const fd=new FormData(e.target);const payload={};fd.forEach((v,k)=>{if(v)payload[k]=v});"
            "const res=await fetch('/chatbot/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});"
            "const json=await res.json();document.getElementById('output').textContent=JSON.stringify(json,null,2);"
            "if(json.session_id)document.querySelector('[name=session_id]').value=json.session_id;};"
            "</script>"
            "</body></html>"
        )

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_ENV", "development").lower() == "development"
    app.run(host=host, port=port, debug=debug)
