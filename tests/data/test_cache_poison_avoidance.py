import pytest
import unittest
import asyncio
from typing import Any
from unittest.mock import Mock
from edsl.agents.Invigilator import InvigilatorAI

from edsl.agents.Agent import Agent
from edsl.language_models import LanguageModel
from edsl.enums import InferenceServiceType


from edsl.data.Cache import Cache
from edsl.prompts.Prompt import Prompt


c = Cache()
a = Agent()

from edsl.questions import QuestionNumerical

q = QuestionNumerical.example()


class InvigilatorTest(InvigilatorAI):
    def get_prompts(self):
        return {
            "user_prompt": Prompt("XX1XX"),
            "system_prompt": Prompt("XX1XX"),
        }


def test_good_answer_cached():
    cache = Cache()
    from edsl.language_models.model import Model

    m = Model("test", canned_response=1)
    results = q.by(m).run(cache=cache)
    results.select("answer.*").print()
    try:
        assert cache.data != {}
    except AssertionError:
        raise Exception("Cache data is empty but should not be!")
