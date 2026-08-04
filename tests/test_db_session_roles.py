"""Role-based engine selection with single-DB fallback.

Serving reads (API, board) hit DATABASE_URL; pipeline/research hit
RESEARCH_DATABASE_URL, which falls back to DATABASE_URL when unset so a
single-DB dev/test setup keeps working unchanged.
"""
import src.db.session as sess

# _resolve_url reads os.environ live, so no module reload is needed — and
# reloading would re-run load_dotenv and repopulate a var the test just deleted.


def test_defaults_to_single_db_when_research_url_unset(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/serving")
    monkeypatch.delenv("RESEARCH_DATABASE_URL", raising=False)
    assert sess._resolve_url("research") == sess._resolve_url("serving")


def test_distinct_urls_when_both_set(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/serving")
    monkeypatch.setenv("RESEARCH_DATABASE_URL", "postgresql://u@h/research")
    assert sess._resolve_url("research").endswith("/research")
    assert sess._resolve_url("serving").endswith("/serving")


def test_serving_scope_is_session_scope_serving():
    assert callable(sess.serving_session_scope)
    assert callable(sess.session_scope)
