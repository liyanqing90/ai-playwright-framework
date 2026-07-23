from ai_playwright.ai_runtime.agent_case_executor import _unmet_completion_criteria


def test_title_checkpoint_ignores_dom_text_match():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ['after search succeeds, page title contains "bmw x3"'],
            "final": [],
        },
        history=[],
        current_url="https://www.example.test/",
        dom_context={
            "meta": {"title": "Home"},
            "interactive_elements": [
                {
                    "text": "bmw x3",
                    "selector": "#search",
                    "visible": True,
                }
            ],
            "assertion_candidates": [],
        },
    )

    assert unmet == ['after search succeeds, page title contains "bmw x3"']


def test_title_checkpoint_ignores_non_title_assertion_history():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ['after search succeeds, page title contains "bmw x3"'],
            "final": [],
        },
        history=[
            {
                "step": {
                    "action": "assert_text_contains",
                    "selector": "#result",
                    "value": "bmw x3",
                },
                "result": "passed",
            }
        ],
        current_url="https://www.example.test/",
        dom_context={
            "meta": {"title": "Home"},
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == ['after search succeeds, page title contains "bmw x3"']


def test_title_checkpoint_accepts_title_assertion_history():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ['after search succeeds, page title contains "bmw x3"'],
            "final": [],
        },
        history=[
            {
                "step": {"action": "assert_title_contains", "value": "bmw x3"},
                "result": "passed",
            }
        ],
        current_url="https://www.example.test/",
        dom_context={
            "meta": {"title": "Home"},
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == []


def test_url_checkpoint_accepts_percent_encoded_current_url():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ['search succeeds and final URL contains "keyword=奥迪A6L"'],
            "final": [],
        },
        history=[],
        current_url=(
            "https://mall.example.test/search-result?"
            "keyword=%E5%A5%A5%E8%BF%AAA6L&tab=newCar"
        ),
        dom_context={
            "meta": {"title": "Search"},
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == []


def test_final_criteria_override_intermediate_checkpoints_for_completion():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["after login, page title is Products"],
            "final": ["after logout, Login button is visible"],
        },
        history=[],
        current_url="https://www.saucedemo.com/",
        dom_context={
            "meta": {"title": "Swag Labs"},
            "interactive_elements": [
                {
                    "text": "Login",
                    "selector": "input[data-test='login-button']",
                    "visible": True,
                }
            ],
            "assertion_candidates": [],
        },
    )

    assert unmet == []


def test_checkpoints_remain_completion_criteria_when_final_is_absent():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["after logout, Login button is visible"],
            "final": [],
        },
        history=[],
        current_url="https://www.saucedemo.com/",
        dom_context={
            "meta": {"title": "Swag Labs"},
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == ["after logout, Login button is visible"]


def test_url_checkpoint_rejects_only_dom_text_match():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ['search succeeds and final URL contains "keyword=奥迪A6L"'],
            "final": [],
        },
        history=[],
        current_url="https://mall.example.test/search-result?keyword=default",
        dom_context={
            "meta": {"title": "Search"},
            "interactive_elements": [{"text": "keyword=奥迪A6L", "visible": True}],
            "assertion_candidates": [],
        },
    )

    assert unmet == ['search succeeds and final URL contains "keyword=奥迪A6L"']
