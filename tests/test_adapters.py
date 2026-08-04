"""PlatformAdapter protocol + ManualAdapter (draft-source abstraction)."""
from api.adapters.manual import ManualAdapter
from src.config import load_config


def test_manual_adapter_reads_history_as_draft_state():
    cfg = load_config()
    history = [[["pick", "P1", True]], [["skip", "_skip_2"]], [["pick", "P3", False]]]
    adapter = ManualAdapter(history, cfg)
    st = adapter.get_draft_state()
    assert st.picks == [("P1", True), (None, False), ("P3", False)]
    ls = adapter.get_league_settings()
    assert ls.teams == cfg.teams
    assert ls.rounds == cfg.roster.rounds
    assert ls.roster_slots == dict(cfg.roster.slots)


def test_manual_adapter_empty_history():
    cfg = load_config()
    adapter = ManualAdapter([], cfg)
    assert adapter.get_draft_state().picks == []
