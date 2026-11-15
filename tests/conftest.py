"""Tests that read the gitignored processed data are skipped when it has not been built.

The county weather, satellite and NASS files under data/processed/ are not part of the repository
(see README "Data"). Without them, a test that fails because such a file is missing is reported
as skipped instead of failed; with the data present every test runs normally.
"""

import re
from pathlib import Path

import pytest

DATA_MARKER = Path(__file__).resolve().parents[1] / "data/processed/weather/power_county_daily.json"
# These fail without the data through a symptom other than a missing-file error.
INDIRECT = {
    "tests/test_agent_outlook.py::test_failure_never_breaks_brief",
    "tests/test_app_advisor.py::test_page_questionnaire_submit",
    "tests/test_app_advisor.py::test_page_shows_field_errors",
}
REASON = "needs data/processed/ (not in the repository; build it with the fieldshift.data modules)"


def _missing_data_error(excinfo):
    exc = excinfo.value
    while exc is not None:
        message = re.sub(r"[\\/]+", "/", str(exc))
        if isinstance(exc, FileNotFoundError) and "data/processed" in message:
            return True
        if "No such file or directory" in message and "data/processed" in message:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if DATA_MARKER.exists() or not report.failed or call.excinfo is None:
        return
    if item.nodeid.replace("\\", "/") in INDIRECT or _missing_data_error(call.excinfo):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, REASON)


@pytest.fixture(autouse=True)
def no_local_model_keys(monkeypatch):
    """Offline tests never see this machine's model keys or key files."""
    from fieldshift.agent import providers
    monkeypatch.setattr(providers, "KEY_FILES", ())
    for name in providers.known_key_names() | {"FARMSIM_TOKEN"}:
        monkeypatch.delenv(name, raising=False)
