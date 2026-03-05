"""ChatbotEngine: Enhanced chatbot wrapper with session management, domain restriction,
diabetes follow-up, user feedback, and response regeneration.

Wraps ComprehensiveChatbot and adds the UX/logic layer from single_chatbot_app.py.
"""
import time
import uuid
from typing import Dict, List, Optional, Tuple

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

# Diabetes-specific keywords that trigger the follow-up question
DIABETES_KEYWORDS = ["diabetes", "diabetic", "blood sugar", "glucose", "gestational diabetes"]

# Session time-to-live: 30 minutes of inactivity
SESSION_TTL = 1800


class SessionManager:
    """Per-user conversation history and multi-turn state machine."""

    def __init__(self):
        self._sessions: Dict[str, Dict] = {}

    def _make_session(self) -> Dict:
        return {
            "history": [],
            "state": None,          # e.g. "awaiting_diabetes_type"
            "state_data": {},
            "last_active": time.time(),
        }

    def get_or_create(self, session_id: str) -> Dict:
        session = self._sessions.get(session_id)
        if session is None or time.time() - session["last_active"] > SESSION_TTL:
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

    def clear_session(self, session_id: str) -> None:
        """Clear session history and state."""
        self._sessions.pop(session_id, None)


class FeedbackStore:
    """In-memory store for user feedback analytics."""

    def __init__(self):
        self._records: Dict[str, Dict] = {}

    def record(self, question_id: str, question: str, answer: str, rating: int, source: str) -> None:
        """Record user feedback for an answer.

        Args:
            question_id: Unique ID of the question/answer pair
            question: Original question text
            answer: Answer that was rated
            rating: User rating (1-5, where 5 is best; or 0/1 for thumbs down/up)
            source: Source that generated the answer
        """
        self._records[question_id] = {
            "question": question,
            "answer": answer,
            "rating": rating,
            "source": source,
            "timestamp": time.time(),
        }

    def get(self, question_id: str) -> Optional[Dict]:
        return self._records.get(question_id)

    def summary(self) -> Dict:
        """Return aggregated feedback statistics."""
        records = list(self._records.values())
        if not records:
            return {"total_feedback": 0, "avg_rating": 0.0, "sources": {}}
        ratings = [r["rating"] for r in records]
        avg = sum(ratings) / len(ratings)
        sources: Dict[str, int] = {}
        for r in records:
            sources[r.get("source", "unknown")] = sources.get(r.get("source", "unknown"), 0) + 1
        return {"total_feedback": len(records), "avg_rating": round(avg, 2), "sources": sources}


class ChatbotEngine:
    """
    Enhanced chatbot engine that wraps ComprehensiveChatbot and adds:
    - Domain restriction (pregnancy-only responses)
    - Diabetes follow-up question logic
    - Session-based conversation context
    - User feedback and response regeneration
    """

    def __init__(self):
        from ai_engine.comprehensive_chatbot import get_comprehensive_chatbot
        self._chatbot = get_comprehensive_chatbot()
        self.sessions = SessionManager()
        self.feedback = FeedbackStore()
        # Map question_id -> stored answer payload for regeneration
        self._answer_store: Dict[str, Dict] = {}

    def _is_pregnancy_related(self, question: str) -> bool:
        """Check if the question is pregnancy-related."""
        q_lower = question.lower()
        return any(kw in q_lower for kw in PREGNANCY_KEYWORDS)

    def _has_diabetes_keyword(self, question: str) -> bool:
        """Check if the question mentions diabetes."""
        q_lower = question.lower()
        return any(kw in q_lower for kw in DIABETES_KEYWORDS)

    def ask(
        self,
        question: str,
        trimester: Optional[int] = None,
        region: Optional[str] = None,
        season: Optional[str] = None,
        session_id: Optional[str] = None,
        language: str = "en",
        skip_regenerate: bool = False,
    ) -> Dict:
        """
        Process a user question with full multi-turn, domain-restricted, AI-powered logic.

        Args:
            question: User's question text
            trimester: Current pregnancy trimester (1-3)
            region: Regional preference
            season: Current season
            session_id: Session identifier for conversation tracking
            language: Response language ('en' or 'te')
            skip_regenerate: If True, skip sources already used (for regeneration)

        Returns:
            Dict with keys: question, answer, dos, donts, source, question_id,
                            session_id, trimester, awaiting_followup, ...
        """
        question = question.strip()
        if not question:
            return {"error": "Question is required"}

        if session_id is None:
            session_id = str(uuid.uuid4())

        # ------------------------------------------------------------------
        # Step 0: Handle session state (e.g. awaiting_diabetes_type)
        # ------------------------------------------------------------------
        state, state_data = self.sessions.get_state(session_id)
        if state == "awaiting_diabetes_type":
            return self._handle_diabetes_followup(
                response=question,
                session_id=session_id,
                trimester=trimester,
                region=region,
                season=season,
                language=language,
                state_data=state_data,
            )

        # ------------------------------------------------------------------
        # Step 1: Domain restriction
        # ------------------------------------------------------------------
        if not self._is_pregnancy_related(question):
            answer_text = (
                "I'm sorry, I can only assist with pregnancy-related queries. "
                "Please ask me about pregnancy nutrition, diet, symptoms, or maternal health."
            )
            question_id = str(uuid.uuid4())
            payload = self._build_payload(
                question=question,
                answer=answer_text,
                source="domain_restriction",
                question_id=question_id,
                session_id=session_id,
                trimester=trimester,
                region=region,
                season=season,
            )
            self.sessions.add_turn(session_id, question, answer_text)
            return payload

        # ------------------------------------------------------------------
        # Step 2: Diabetes detection – ask clarifying follow-up
        # ------------------------------------------------------------------
        if self._has_diabetes_keyword(question) and not skip_regenerate:
            followup = (
                "Are you asking about pre-existing diabetes or gestational diabetes "
                "(diabetes that develops during pregnancy)?"
            )
            self.sessions.set_state(
                session_id, "awaiting_diabetes_type", {"original_question": question}
            )
            self.sessions.add_turn(session_id, question, followup)
            question_id = str(uuid.uuid4())
            return self._build_payload(
                question=question,
                answer=followup,
                source="diabetes_followup",
                question_id=question_id,
                session_id=session_id,
                trimester=trimester,
                region=region,
                season=season,
                awaiting_followup=True,
            )

        # ------------------------------------------------------------------
        # Step 3: Get answer from ComprehensiveChatbot
        # ------------------------------------------------------------------
        try:
            result = self._chatbot.answer_question_structured(
                question=question,
                trimester=trimester,
            )
        except Exception as exc:
            print(f"⚠️ ComprehensiveChatbot error: {exc}")
            result = {
                "answer": "I'm sorry, I encountered an error. Please try rephrasing your question.",
                "dos": [],
                "donts": [],
                "source": "error",
                "response_time": 0,
                "keywords": [],
                "intent": "unknown",
                "query_reflection": "",
            }

        answer_text = result.get("answer", "")
        question_id = str(uuid.uuid4())

        # Store for potential regeneration
        self._answer_store[question_id] = {
            "question": question,
            "answer": answer_text,
            "source": result.get("source", ""),
            "trimester": trimester,
            "region": region,
            "season": season,
            "session_id": session_id,
            "language": language,
        }

        self.sessions.add_turn(session_id, question, answer_text)

        payload = self._build_payload(
            question=question,
            answer=answer_text,
            source=result.get("source", "ai_model"),
            question_id=question_id,
            session_id=session_id,
            trimester=trimester,
            region=region,
            season=season,
        )
        # Carry over additional fields from the chatbot result
        payload["dos"] = result.get("dos", [])
        payload["donts"] = result.get("donts", [])
        payload["query_reflection"] = result.get("query_reflection", "")
        payload["keywords"] = result.get("keywords", [])
        payload["intent"] = result.get("intent", "")
        payload["response_time"] = result.get("response_time", 0)

        return payload

    def regenerate(self, question_id: str) -> Optional[Dict]:
        """Regenerate an answer for an existing question_id using a different source tier."""
        stored = self._answer_store.get(question_id)
        if not stored:
            return None
        return self.ask(
            question=stored["question"],
            trimester=stored.get("trimester"),
            region=stored.get("region"),
            season=stored.get("season"),
            session_id=stored.get("session_id"),
            language=stored.get("language", "en"),
            skip_regenerate=True,
        )

    def record_feedback(self, question_id: str, rating: int) -> bool:
        """Record user feedback for a given answer.

        Args:
            question_id: The ID of the question/answer pair
            rating: User's rating (1-5)

        Returns:
            True if feedback was recorded, False if question_id not found
        """
        stored = self._answer_store.get(question_id)
        if not stored:
            return False
        self.feedback.record(
            question_id=question_id,
            question=stored["question"],
            answer=stored["answer"],
            rating=rating,
            source=stored.get("source", ""),
        )
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        """Handle the user's reply to the diabetes type follow-up."""
        self.sessions.set_state(session_id, None)
        r = response.lower()
        original_question = state_data.get("original_question", "diabetes diet")

        if any(w in r for w in ["gestational", "pregnancy diabetes", "during pregnancy", "pregnant", "yes"]):
            condition_label = "gestational diabetes"
            augmented_question = f"gestational diabetes diet during pregnancy trimester {trimester or 2}"
        else:
            condition_label = "pre-existing diabetes"
            augmented_question = f"pre-existing diabetes diet during pregnancy trimester {trimester or 2}"

        disclaimer = (
            "\n\n⚠️ *Disclaimer: This is informational only. "
            "Please consult your healthcare provider for personalised medical advice.*"
        )

        try:
            result = self._chatbot.answer_question_structured(
                question=augmented_question,
                trimester=trimester,
            )
            answer_text = result.get("answer", "")
        except Exception:
            answer_text = ""

        if not answer_text or len(answer_text) < 50:
            answer_text = (
                f"For {condition_label} during pregnancy, focus on:\n"
                "• Low glycaemic index (GI) foods such as oats, barley, and legumes\n"
                "• High-fibre vegetables like spinach, brinjal, and bitter gourd\n"
                "• Small, frequent meals to maintain stable blood sugar\n"
                "• Avoiding refined carbohydrates, sweets, and sugary drinks\n"
                "• Including good protein sources like dal, paneer, and eggs (if non-vegetarian)"
            )
            result = {"source": "rule_based", "dos": [], "donts": [], "keywords": [], "intent": "diabetes"}

        answer_text += disclaimer
        question_id = str(uuid.uuid4())
        self._answer_store[question_id] = {
            "question": response,
            "answer": answer_text,
            "source": result.get("source", "dataset"),
            "trimester": trimester,
            "region": region,
            "season": season,
            "session_id": session_id,
            "language": language,
        }
        self.sessions.add_turn(session_id, response, answer_text)
        payload = self._build_payload(
            question=response,
            answer=answer_text,
            source=result.get("source", "dataset"),
            question_id=question_id,
            session_id=session_id,
            trimester=trimester,
            region=region,
            season=season,
        )
        payload["dos"] = result.get("dos", [])
        payload["donts"] = result.get("donts", [])
        payload["keywords"] = result.get("keywords", [])
        payload["intent"] = "diabetes"
        return payload

    @staticmethod
    def _build_payload(
        question: str,
        answer: str,
        source: str,
        question_id: str,
        session_id: str,
        trimester: Optional[int],
        region: Optional[str],
        season: Optional[str],
        awaiting_followup: bool = False,
    ) -> Dict:
        return {
            "question": question,
            "answer": answer,
            "dos": [],
            "donts": [],
            "source": source,
            "question_id": question_id,
            "session_id": session_id,
            "trimester": trimester,
            "region": region,
            "season": season,
            "awaiting_followup": awaiting_followup,
            "query_reflection": "",
            "keywords": [],
            "intent": "",
            "response_time": 0,
        }


# Singleton instance
_engine_instance: Optional[ChatbotEngine] = None


def get_chatbot_engine() -> ChatbotEngine:
    """Get or create the singleton ChatbotEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ChatbotEngine()
    return _engine_instance
