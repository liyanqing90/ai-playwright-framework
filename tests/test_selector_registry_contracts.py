from pathlib import Path

from ai_playwright.ai_runtime.selector_registry import SelectorRegistry


def test_selector_registry_persists_ai_decision_metadata(tmp_path: Path):
    registry = SelectorRegistry(tmp_path / "selectors.db")
    try:
        record = registry.save(
            project="demo",
            env="test",
            page_key="/login",
            action="click",
            target="submit",
            selector="#submit",
            source="ai_selector",
            confidence=0.73,
            prompt_version="selector-v1",
            schema_version="schema-v1",
            model="test-model",
            candidate_hash="abc123",
            candidate_count=7,
        )

        found = registry.find(
            project="demo",
            env="test",
            page_key="/login",
            action="click",
            target="submit",
        )
        assert found is not None
        assert found.id == record.id
        assert found.prompt_version == "selector-v1"
        assert found.schema_version == "schema-v1"
        assert found.model == "test-model"
        assert found.candidate_hash == "abc123"
        assert found.candidate_count == 7

        registry.mark_failed(
            found.id,
            unstable_threshold=1,
            last_error="verification failed",
        )
        failed = registry.find(
            project="demo",
            env="test",
            page_key="/login",
            action="click",
            target="submit",
        )
        assert failed is not None
        assert failed.status == "unstable"
        assert failed.last_error == "verification failed"
    finally:
        registry.close()


def test_selector_registry_filters_low_score_and_deprecates_after_failures(
    tmp_path: Path,
):
    registry = SelectorRegistry(tmp_path / "selectors.db")
    try:
        record = registry.save(
            project="demo",
            env="test",
            page_key="/login",
            action="click",
            target="submit",
            selector="#weak",
            source="ai_selector",
            confidence=0.2,
        )

        assert (
            registry.find(
                project="demo",
                env="test",
                page_key="/login",
                action="click",
                target="submit",
                min_score=0.75,
            )
            is None
        )

        registry.mark_failed(
            record.id,
            unstable_threshold=1,
            deprecated_after_failures=2,
            last_error="first failure",
        )
        unstable = registry.find(
            project="demo",
            env="test",
            page_key="/login",
            action="click",
            target="submit",
        )
        assert unstable is not None
        assert unstable.status == "unstable"

        registry.mark_failed(
            record.id,
            unstable_threshold=1,
            deprecated_after_failures=2,
            last_error="second failure",
        )
        assert (
            registry.find(
                project="demo",
                env="test",
                page_key="/login",
                action="click",
                target="submit",
            )
            is None
        )
    finally:
        registry.close()
