"""Tests for ChatbotEngine: domain restriction, diabetes follow-up, feedback, and regeneration."""
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_engine.chatbot_engine import (
    SessionManager,
    FeedbackStore,
    PREGNANCY_KEYWORDS,
    DIABETES_KEYWORDS,
)


class TestSessionManager(unittest.TestCase):
    """Unit tests for SessionManager (no AI required)."""

    def setUp(self):
        self.sm = SessionManager()

    def test_create_new_session(self):
        """A new session is created on first access."""
        session = self.sm.get_or_create('sess1')
        self.assertIsNotNone(session)
        self.assertEqual(session['state'], None)
        self.assertEqual(session['history'], [])

    def test_add_turn(self):
        """add_turn appends user and bot messages."""
        self.sm.add_turn('s1', 'Hello', 'Hi there!')
        session = self.sm.get_or_create('s1')
        self.assertEqual(len(session['history']), 2)
        self.assertEqual(session['history'][0]['role'], 'user')
        self.assertEqual(session['history'][1]['role'], 'bot')

    def test_set_and_get_state(self):
        """State can be set and retrieved."""
        self.sm.set_state('s2', 'awaiting_diabetes_type', {'original_question': 'q'})
        state, data = self.sm.get_state('s2')
        self.assertEqual(state, 'awaiting_diabetes_type')
        self.assertEqual(data['original_question'], 'q')

    def test_clear_state(self):
        """State can be cleared to None."""
        self.sm.set_state('s3', 'awaiting_diabetes_type')
        self.sm.set_state('s3', None)
        state, _ = self.sm.get_state('s3')
        self.assertIsNone(state)

    def test_history_truncation(self):
        """History is truncated to 20 most recent messages."""
        for i in range(15):
            self.sm.add_turn('s4', f'user {i}', f'bot {i}')
        session = self.sm.get_or_create('s4')
        self.assertLessEqual(len(session['history']), 20)

    def test_get_history_text(self):
        """get_history_text returns formatted history."""
        self.sm.add_turn('s5', 'Question?', 'Answer.')
        text = self.sm.get_history_text('s5', last_n=1)
        self.assertIn('User:', text)
        self.assertIn('Assistant:', text)

    def test_clear_session(self):
        """clear_session removes the session so a fresh one is created next access."""
        self.sm.add_turn('s6', 'hi', 'hello')
        self.sm.clear_session('s6')
        session = self.sm.get_or_create('s6')
        self.assertEqual(session['history'], [])

    def test_same_session_id_returns_same_session(self):
        """The same session_id always returns the same (continuing) session."""
        self.sm.add_turn('s7', 'turn1', 'reply1')
        session_a = self.sm.get_or_create('s7')
        session_b = self.sm.get_or_create('s7')
        self.assertIs(session_a, session_b)
        self.assertEqual(len(session_a['history']), 2)


class TestFeedbackStore(unittest.TestCase):
    """Unit tests for FeedbackStore (no AI required)."""

    def setUp(self):
        self.fs = FeedbackStore()

    def test_record_and_retrieve(self):
        """Recorded feedback can be retrieved by question_id."""
        self.fs.record('qid1', 'question?', 'answer.', rating=4, source='dataset')
        rec = self.fs.get('qid1')
        self.assertIsNotNone(rec)
        self.assertEqual(rec['rating'], 4)
        self.assertEqual(rec['source'], 'dataset')

    def test_missing_id_returns_none(self):
        """Fetching a non-existent ID returns None."""
        self.assertIsNone(self.fs.get('nonexistent'))

    def test_summary_empty(self):
        """Summary on empty store returns zero totals."""
        s = self.fs.summary()
        self.assertEqual(s['total_feedback'], 0)
        self.assertEqual(s['avg_rating'], 0.0)

    def test_summary_with_records(self):
        """Summary correctly aggregates ratings."""
        self.fs.record('q1', 'q1', 'a1', rating=5, source='gemini')
        self.fs.record('q2', 'q2', 'a2', rating=3, source='dataset')
        s = self.fs.summary()
        self.assertEqual(s['total_feedback'], 2)
        self.assertAlmostEqual(s['avg_rating'], 4.0)
        self.assertIn('gemini', s['sources'])
        self.assertIn('dataset', s['sources'])

    def test_overwrite_feedback(self):
        """Recording feedback for the same ID overwrites the previous entry."""
        self.fs.record('q3', 'q3', 'a3', rating=2, source='rule_based')
        self.fs.record('q3', 'q3', 'a3', rating=5, source='rule_based')
        self.assertEqual(self.fs.get('q3')['rating'], 5)


class TestDomainKeywords(unittest.TestCase):
    """Unit tests for pregnancy domain and diabetes keyword detection (no AI)."""

    def _is_pregnancy_related(self, q):
        q = q.lower()
        return any(kw in q for kw in PREGNANCY_KEYWORDS)

    def _has_diabetes_keyword(self, q):
        q = q.lower()
        return any(kw in q for kw in DIABETES_KEYWORDS)

    def test_pregnancy_keywords_cover_common_topics(self):
        """Common pregnancy topics are correctly identified as pregnancy-related."""
        self.assertTrue(self._is_pregnancy_related("what should I eat during pregnancy"))
        self.assertTrue(self._is_pregnancy_related("morning sickness remedies"))
        self.assertTrue(self._is_pregnancy_related("iron supplements for prenatal care"))
        self.assertTrue(self._is_pregnancy_related("foods to avoid while breastfeeding"))
        self.assertTrue(self._is_pregnancy_related("diet for gestational diabetes"))
        self.assertTrue(self._is_pregnancy_related("what meal plan for second trimester"))
        self.assertTrue(self._is_pregnancy_related("calcium and folic acid in pregnancy"))

    def test_non_pregnancy_topics_not_flagged(self):
        """Non-pregnancy topics are not flagged as pregnancy-related."""
        self.assertFalse(self._is_pregnancy_related("what is the capital of France"))
        self.assertFalse(self._is_pregnancy_related("how do I fix a car engine"))
        self.assertFalse(self._is_pregnancy_related("latest news headlines"))
        self.assertFalse(self._is_pregnancy_related("stock market analysis"))

    def test_diabetes_keywords_detected(self):
        """Diabetes-related questions are correctly flagged."""
        self.assertTrue(self._has_diabetes_keyword("diabetes diet during pregnancy"))
        self.assertTrue(self._has_diabetes_keyword("gestational diabetes food plan"))
        self.assertTrue(self._has_diabetes_keyword("blood sugar control in pregnancy"))
        self.assertTrue(self._has_diabetes_keyword("I am diabetic and pregnant"))

    def test_non_diabetes_questions_not_flagged(self):
        """Non-diabetes questions do not trigger the diabetes keyword flag."""
        self.assertFalse(self._has_diabetes_keyword("best iron-rich foods"))
        self.assertFalse(self._has_diabetes_keyword("morning sickness help"))
        self.assertFalse(self._has_diabetes_keyword("can I eat mango during pregnancy"))


class TestChatbotEngineImport(unittest.TestCase):
    """Smoke tests: verify ChatbotEngine can be imported and instantiated."""

    def test_chatbot_engine_import(self):
        """ChatbotEngine class is importable."""
        from ai_engine.chatbot_engine import ChatbotEngine
        self.assertIsNotNone(ChatbotEngine)

    def test_chatbot_engine_initialization(self):
        """ChatbotEngine initialises with required attributes."""
        from ai_engine.chatbot_engine import ChatbotEngine
        engine = ChatbotEngine()
        self.assertIsNotNone(engine)
        self.assertIsInstance(engine.sessions, SessionManager)
        self.assertIsInstance(engine.feedback, FeedbackStore)

    def test_singleton_factory(self):
        """get_chatbot_engine returns the same instance every call."""
        from ai_engine.chatbot_engine import get_chatbot_engine
        engine1 = get_chatbot_engine()
        engine2 = get_chatbot_engine()
        self.assertIs(engine1, engine2)

    def test_engine_domain_restriction_fast_path(self):
        """Non-pregnancy questions are rejected immediately (no AI call)."""
        from ai_engine.chatbot_engine import ChatbotEngine
        engine = ChatbotEngine()
        result = engine.ask('What is the capital of France?', session_id='dr1')
        self.assertEqual(result['source'], 'domain_restriction')
        self.assertIn('pregnancy-related', result['answer'].lower())
        self.assertIn('question_id', result)
        self.assertFalse(result.get('awaiting_followup', False))

    def test_engine_diabetes_followup_fast_path(self):
        """Diabetes questions trigger a follow-up immediately (no AI call)."""
        from ai_engine.chatbot_engine import ChatbotEngine
        engine = ChatbotEngine()
        result = engine.ask(
            'What can I eat if I have diabetes during pregnancy?',
            trimester=2,
            session_id='diab1',
        )
        self.assertTrue(result.get('awaiting_followup'), msg="Expected awaiting_followup=True")
        self.assertEqual(result['source'], 'diabetes_followup')
        self.assertIn('gestational', result['answer'].lower())

    def test_engine_feedback_invalid_id(self):
        """Recording feedback for an unknown question_id returns False."""
        from ai_engine.chatbot_engine import ChatbotEngine
        engine = ChatbotEngine()
        recorded = engine.record_feedback('nonexistent-id', rating=5)
        self.assertFalse(recorded)

    def test_engine_regenerate_unknown_id(self):
        """Regenerating an unknown question_id returns None."""
        from ai_engine.chatbot_engine import ChatbotEngine
        engine = ChatbotEngine()
        result = engine.regenerate('unknown-id-xyz')
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()


