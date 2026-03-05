"""
Single-file chatbot app (Flask) with dataset-only answers (no external AI).

Run:
    python single_chatbot_app.py

Env vars:
    HOST (default 0.0.0.0)
    PORT (default 5000)

Datasets are read from the existing data/ folder next to this file.
"""

import os
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

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


class ExternalAIProvider:
    """Base class placeholder (kept for future extension)."""

    def generate_answer(self, question: str, context: str = "") -> str:
        return ""


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
        if q in self.food_index:
            return {"found": True, "data": self.food_index[q], "type": "food"}
        if q in self.dos_donts_index:
            return {"found": True, "data": self.dos_donts_index[q], "type": "dos_donts"}
        # fuzzy by keywords
        for word in q.split():
            if word in self.keyword_index:
                candidate = self.keyword_index[word][0]
                return {"found": True, "data": candidate, "type": "food"}
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
    """All-in-one chatbot that uses the lightweight loader and optional external AI."""

    def __init__(self):
        self.loader = UnifiedDatasetLoaderLite()
        self.cache = ResponseCache(ttl_seconds=3600)
        # External AI removed for simplicity and reliability (dataset-only answers).

    def classify_intent(self, question: str) -> str:
        q = question.lower()
        if any(w in q for w in ["meal plan", "diet plan", "what to eat", "menu", "breakfast", "lunch", "dinner"]):
            return "meal_plan"
        if any(w in q for w in ["can i", "safe", "avoid", "dangerous", "should i"]):
            return "safety"
        if any(w in q for w in ["benefit", "good for", "nutrient", "vitamin", "protein", "iron", "calcium"]):
            return "benefits"
        if any(w in q for w in ["trimester", "1st", "2nd", "3rd"]):
            return "trimester"
        return "general"

    def extract_keywords(self, question: str) -> List[str]:
        q = question.lower()
        tokens = [t.strip("?,.! ") for t in q.split() if len(t) > 2]
        uniq = []
        for t in tokens:
            if t not in uniq:
                uniq.append(t)
        return uniq[:5]

    def _format_food_answer(self, meal: Dict, trimester: Optional[int]) -> str:
        name = meal.get("food") or meal.get("food_item") or meal.get("meal") or meal.get("dish") or meal.get("item") or "This food"
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
            # fallback guess
            if "avoid" in text.lower() or "not" in text.lower():
                donts.append(text)
            else:
                dos.append(text)
        return dos, donts

    def answer(self, question: str, trimester: Optional[int] = None, region: Optional[str] = None, season: Optional[str] = None) -> Dict:
        question = question.strip()
        if not question:
            return {"error": "Question is required"}

        cached = self.cache.get(question)
        if cached:
            return {**cached, "source": cached.get("source", "cache"), "cached": True}

        start = time.time()
        intent = self.classify_intent(question)
        keywords = self.extract_keywords(question)

        quick = self.loader.quick_lookup(question)
        dos_list: List[str] = []
        donts_list: List[str] = []
        answer_text = ""
        source = "dataset"

        if quick["found"]:
            if quick["type"] == "food":
                answer_text = self._format_food_answer(quick["data"], trimester)
            else:
                d, n = self._format_dos_donts(quick["data"])
                dos_list.extend(d)
                donts_list.extend(n)
                answer_text = quick["data"].get("description", "Guidance available.")
        elif intent == "meal_plan":
            meals = self.loader.get_meals_by_preference(region=region, diet_type=None, trimester=trimester, season=season)
            if meals:
                preview = meals[:3]
                names = []
                for meal in preview:
                    for col in ["food", "food_item", "meal", "dish", "item", "recipe"]:
                        if meal.get(col):
                            names.append(str(meal[col]))
                            break
                answer_text = "Sample meals: " + ", ".join(names)
            else:
                answer_text = "No meals matched your preferences."
        else:
            answer_text = (
                "I could not find an exact match in the datasets. "
                "Focus on a balanced pregnancy diet: plenty of fruits and veggies, whole grains, lean protein, "
                "and safe hydration. Avoid raw/undercooked foods and high-mercury fish."
            )
            source = "fallback"

        duration = time.time() - start
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
        }
        self.cache.set(question, payload)
        return payload

    def answer_structured(self, question: str, trimester: Optional[int] = None) -> Dict:
        result = self.answer(question, trimester=trimester)
        if result.get("dos") or result.get("donts"):
            return result
        # derive simple dos/donts from free-text fallback when guidance is missing
        text = result.get("answer", "")
        if "avoid" in text.lower():
            result.setdefault("donts", []).append(text)
        else:
            result.setdefault("dos", []).append(text)
        return result

    def meal_plan_preview(self, region: Optional[str], diet_type: Optional[str], trimester: Optional[int], season: Optional[str], limit: int = 5) -> Dict:
        meals = self.loader.get_meals_by_preference(region=region, diet_type=diet_type, trimester=trimester, season=season)
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
                "notes": meal.get("description") or meal.get("remarks") or meal.get("health_benefit"),
            })
        return {
            "count": len(preview),
            "meals": preview,
        }


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
        result = bot.answer(question=question, trimester=trimester, region=region, season=season)
        status = 200 if not result.get("error") else 400
        return jsonify(result), status

    @app.route("/chatbot/dos-donts", methods=["POST"])
    def dos_donts():
        data = request.get_json(force=True, silent=True) or {}
        question = str(data.get("question", "")).strip()
        trimester = data.get("trimester")
        result = bot.answer_structured(question=question, trimester=trimester)
        status = 200 if not result.get("error") else 400
        return jsonify(result), status

    @app.route("/chatbot/mealplan", methods=["POST"])
    def mealplan():
        data = request.get_json(force=True, silent=True) or {}
        region = data.get("region")
        diet_type = data.get("diet_type")
        trimester = data.get("trimester")
        season = data.get("season")
        preview = bot.meal_plan_preview(region=region, diet_type=diet_type, trimester=trimester, season=season)
        return jsonify({
            "success": True,
            "preview": preview,
            "region": region,
            "diet_type": diet_type,
            "trimester": trimester,
            "season": season,
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

        answer = bot.answer(question=question, trimester=trimester, region=region, season=season)
        dos_donts = bot.answer_structured(question=question, trimester=trimester)
        meal_preview = bot.meal_plan_preview(region=region, diet_type=diet_type, trimester=trimester, season=season)

        return jsonify({
            "question": question,
            "trimester": trimester,
            "region": region,
            "diet_type": diet_type,
            "season": season,
            "answer": answer,
            "dos_donts": {
                "dos": dos_donts.get("dos", []),
                "donts": dos_donts.get("donts", []),
                "source": dos_donts.get("source"),
            },
            "meal_plan_preview": meal_preview,
            "suggestions": suggestions().get_json().get("suggestions", []),
        })

    @app.route("/ui")
    def ui():
        # Simple inline HTML UI for quick access to all features
        return (
            "<html><head><title>Chatbot UI</title></head><body>"
            "<h2>Dataset Chatbot</h2>"
            "<form id='askForm'>"
            "Question: <input name='question' size='60' value='What should I eat in trimester 2?'><br>"
            "Trimester: <input name='trimester' value='2' size='3'>"
            "Region: <input name='region' value='North' size='8'>"
            "Diet: <input name='diet_type' value='veg' size='6'>"
            "Season: <input name='season' value='' size='8'>"
            "<button type='submit'>Ask</button>"
            "</form>"
            "<pre id='output'></pre>"
            "<script>"
            "document.getElementById('askForm').onsubmit = async (e) => {e.preventDefault();"
            "const fd=new FormData(e.target);const payload={};fd.forEach((v,k)=>{payload[k]=v});"
            "const res=await fetch('/chatbot/all',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});"
            "const json=await res.json();document.getElementById('output').textContent=JSON.stringify(json,null,2);};"
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
