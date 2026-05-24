"""Headless render smoke test for the Streamlit app.

Uses Streamlit's in-process ``AppTest`` (no browser) to confirm the full app script --
preset seeding, every sidebar widget, the engines and all three charts -- executes
without raising, and re-runs after switching presets.
"""

from __future__ import annotations

import pathlib

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP_PATH = str(pathlib.Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_default_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert at.exception == []
    labels = [m.label for m in at.metric]
    for expected in ("Total CapEx", "NPV", "Radiator Length", "IRR"):
        assert expected in labels


@pytest.mark.parametrize("preset", ["Starcloud", "First-Principles Rack-Sat"])
def test_app_switches_presets_without_exception(preset):
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.selectbox(key="preset_select").select(preset).run()
    assert at.exception == []


def test_app_switches_to_eur_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.selectbox(key="display_currency").select("EUR").run()
    assert at.exception == []
    # KPI labels are currency-independent; values change but labels stay.
    labels = [m.label for m in at.metric]
    assert "Total CapEx" in labels and "NPV" in labels
