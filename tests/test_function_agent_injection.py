"""Offline regression coverage for indirect-content injection setup."""

import unittest

from agent.env import BankingEnvironment
from agent.function_agent import AgentConfig, FunctionCallingAgent


class FunctionAgentInjectionTest(unittest.TestCase):
    def test_injection_is_available_before_first_api_call(self):
        environment = BankingEnvironment()
        agent = FunctionCallingAgent(AgentConfig(api_key=""), environment)

        agent.inject_content("email", {
            "sender": "external@example.test",
            "subject": "Test instruction",
            "body": "Untrusted external content",
        })

        self.assertEqual(environment.emails[0].sender, "external@example.test")
        self.assertEqual(agent._injected_content["email"]["subject"], "Test instruction")


if __name__ == "__main__":
    unittest.main()
