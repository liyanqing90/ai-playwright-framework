from pathlib import Path

import pytest

from ai_playwright.ai_generation.project_context import ProjectContext
from ai_playwright.ai_runtime.agent_case_executor import (
    AgentCaseExecutor,
    _first_entry_module_name,
    _module_name_from_entry_step,
)


def test_explicit_module_token_prefers_exact_name_over_prefix_match():
    modules = {
        "autohome_open_search": [],
        "autohome_open_search_no_modal": [],
    }

    assert _module_name_from_entry_step(
        "use project module autohome_open_search_no_modal",
        modules=modules,
    ) == ("autohome_open_search_no_modal", True)


def test_explicit_module_token_does_not_match_prefix_module():
    modules = {
        "autohome_open_search": [],
    }

    assert _module_name_from_entry_step(
        "use project module autohome_open_search_no_modal",
        modules=modules,
    ) == (None, True)


def test_first_entry_module_uses_exact_step_module_when_intent_has_url():
    modules = {
        "autohome_open_search": [],
        "autohome_open_search_no_modal": [],
    }
    steps = [
        "use project module autohome_open_search_no_modal",
        "verify search result title contains bmw x3",
    ]

    assert _first_entry_module_name(
        case_data={"steps": steps},
        intent="visit https://www.autohome.example/",
        steps=steps,
        modules=modules,
    ) == ("autohome_open_search_no_modal", True)


def test_missing_exact_entry_module_fails_before_fallback():
    context = ProjectContext(
        project="demo",
        test_dir=Path("."),
        base_url="https://example.test/",
        elements={},
        modules={"autohome_open_search": []},
        variables={},
        test_cases=[],
        test_data={},
    )
    executor = AgentCaseExecutor(
        page=object(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    steps = ["use project module autohome_open_search_no_modal"]

    with pytest.raises(AssertionError, match="显式项目模块未精确匹配"):
        executor._first_entry_module_decision(
            case_name="missing_exact_module",
            spec={"inputs": {}},
            case_data={"steps": steps},
            intent="",
            steps=steps,
        )
