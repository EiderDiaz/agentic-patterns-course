import unittest
from unittest.mock import patch

from agentic_patterns.reflection_pattern.reflection_agent import ReflectionAgent


class ReflectionAgentWaitTests(unittest.TestCase):
    def test_run_waits_one_minute_after_generating_feedback(self):
        agent = ReflectionAgent.__new__(ReflectionAgent)
        agent.generate = lambda history, verbose=0: "generated"
        agent.reflect = lambda history, verbose=0: "needs improvement"

        with patch(
            "agentic_patterns.reflection_pattern.reflection_agent.time.sleep"
        ) as mock_sleep:
            with patch(
                "agentic_patterns.reflection_pattern.reflection_agent.update_chat_history"
            ):
                output = agent.run("hello user", n_steps=1)

        self.assertEqual(output, "generated")
        mock_sleep.assert_called_once_with(60)


if __name__ == "__main__":
    unittest.main()
