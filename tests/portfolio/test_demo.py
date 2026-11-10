"""Exercise actual Streamlit widgets against a real trained service."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from flightbridge.training import train

ROOT = Path(__file__).resolve().parents[2]
MESSAGE = "My Aegean flight was moved by 6 hours. I want a refund. No alternative flight was offered."


@pytest.fixture(scope="module")
def model_directory(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ui-artifacts")
    train(ROOT / "schedule_change_dataset.csv", directory)
    return directory


@pytest.fixture
def app(monkeypatch, model_directory):
    monkeypatch.setenv("FLIGHTBRIDGE_ARTIFACTS", str(model_directory))
    page = AppTest.from_file(str(ROOT / "demo.py"), default_timeout=30).run()
    assert not page.exception
    return page


def analyze(app, message=MESSAGE):
    app.text_area(key="fb_message").set_value(message).run()
    app.button(key="fb_analyze").click().run()
    assert not app.exception
    return app


def confirm(app):
    app.checkbox(key="fb_confirmed").check().run()
    app.button(key="fb_prepare").click().run()
    assert not app.exception
    return app.session_state["fb_handoff"]


def test_confirmation_gate_then_actual_model_handoff(app):
    analyze(app)
    app.button(key="fb_prepare").click().run()
    assert "fb_handoff" not in app.session_state
    assert any("Confirm the case details" in e.value for e in app.error)
    result = confirm(app)
    assert len(result["prediction"]["predictions"]) == 2
    assert result["actions_executed"] == []
    assert result["confirmed_facts"]["delay_minutes"] == 360
    assert any("signal-card" in m.value for m in app.markdown)
    assert app.expander[-2].label == "Technical response"


def test_editing_facts_invalidates_handoff_and_confirmation(app):
    analyze(app)
    confirm(app)
    app.number_input(key="fb_minutes").set_value(180.0).run()
    assert "fb_handoff" not in app.session_state
    assert not app.checkbox(key="fb_confirmed").value
    result = confirm(app)
    assert result["confirmed_facts"]["delay_minutes"] == 180


def test_editing_message_and_blank_analysis_remove_previous_result(app):
    analyze(app)
    confirm(app)
    app.text_area(key="fb_message").set_value(" ").run()
    assert "fb_handoff" not in app.session_state and "fb_analysis" not in app.session_state
    app.button(key="fb_analyze").click().run()
    assert len(app.error) == 1 and not app.exception


def test_greek_cancellation_has_no_fake_zero_duration(app):
    app.button(key="fb_example_el").click().run()
    app.button(key="fb_analyze").click().run()
    assert not app.number_input
    result = confirm(app)
    assert result["confirmed_facts"]["delay_minutes"] is None
    assert result["prediction"]["sc_type"] == "CR SC"
    assert any("Not applicable" in item.value for item in app.markdown)


def test_unknown_airline_shows_review_reasons_without_predictions(app):
    analyze(app)
    app.text_input(key="fb_airline").set_value("UNKNOWN TEST CARRIER").run()
    result = confirm(app)
    assert result["prediction"]["predictions"] == {}
    assert result["prediction"]["reasons"] == ["airline_not_seen_in_training"]
    assert any("not represented" in warning.value for warning in app.warning)
    assert any("No model signal" in item.value for item in app.markdown)


def test_missing_fields_remain_unknown_instead_of_getting_defaults(app):
    analyze(app, "My Aegean flight changed. I want a refund.")
    assert app.selectbox(key="fb_alternative").value is None
    assert app.number_input(key="fb_minutes").value is None
    app.checkbox(key="fb_confirmed").check().run()
    app.button(key="fb_prepare").click().run()
    assert "fb_handoff" not in app.session_state
    assert any("Complete these fields" in error.value for error in app.error)


def test_unchecking_confirmation_removes_prepared_handoff(app):
    analyze(app)
    confirm(app)
    app.checkbox(key="fb_confirmed").uncheck().run()
    assert "fb_handoff" not in app.session_state


def test_specialist_route_and_new_case_clear_previous_work(app):
    analyze(app)
    confirm(app)
    analyze(app, "My baggage was lost. I need help with my luggage.")
    assert "fb_handoff" not in app.session_state
    assert any("specialist support" in title.value for title in app.subheader)
    assert not any(button.key == "fb_prepare" for button in app.button)
    app.button(key="fb_new_case").click().run()
    assert app.text_area(key="fb_message").value == ""
    assert "fb_analysis" not in app.session_state and not app.exception


def test_no_artifacts_keeps_intake_available_and_disables_model_handoff(monkeypatch, tmp_path):
    monkeypatch.setenv("FLIGHTBRIDGE_ARTIFACTS", str(tmp_path))
    page = AppTest.from_file(str(ROOT / "demo.py"), default_timeout=30).run()
    analyze(page)
    assert page.button(key="fb_prepare").disabled
    assert "fb_handoff" not in page.session_state and not page.exception
