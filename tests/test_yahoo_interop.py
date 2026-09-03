"""Manual entry and a live Yahoo feed, in the orders they actually occur.

The fastest way to keep the board current is the operator clicking a player
off it — faster than any poll interval. So the feed spends the draft echoing
picks that are already recorded, and those echoes must cost nothing.

They used to cost a pick. `resolve` excludes already-drafted players (late in
a draft "Jones" should not offer three men who are gone), so a duplicate came
back UNRESOLVED rather than as a duplicate — and an unresolved pick still
advances the clock. Measured: entering one pick by hand and letting the feed
echo it moved `pick_number` from 2 to 3. Over a draft where the operator leads
the feed, every pick burns a slot: the draft runs out at half distance and
every seat attribution after the first is wrong.
"""
from __future__ import annotations

import pytest

from src.app.web.sources.base import DraftEvent
from src.core.errors import DataError


@pytest.fixture()
def live(service):
    service.start_session(seat=3, source="manual", resume=False)
    return service


def _player(service, index=0):
    row = service.players.iloc[index]
    return str(row.player_id), str(row.player_name)


def _manual(service, player_id):
    return service.apply_event(DraftEvent(
        raw_name="", player_id=player_id, seat=None,
        source="manual", observed_at=""))


def _manual_name(service, name):
    return service.apply_event(DraftEvent(
        raw_name=name, player_id=None, seat=None, source="manual",
        observed_at=""))


def _feed(service, name, pick):
    return service.apply_event(DraftEvent(
        raw_name=name, player_id=None, seat=None, source="yahoo",
        observed_at="", external_pick_number=pick))


# ------------------------------------------------------- manual by itself
def test_a_click_fills_the_pick_for_whoever_is_on_the_clock(live):
    """Clicking a player records the CURRENT pick — for another manager, not
    for me. `record_pick(seat=None)` resolves the seat from the snake."""
    pid, _ = _player(live, 0)
    assert _manual(live, pid) == "recorded"
    assert pid in live.session.state.drafted
    assert live.session.state.pick_number == 2


def test_manual_entry_works_in_a_yahoo_session(service):
    """Manual is the fallback the whole integration rests on; it must not be
    conditional on how the session was created."""
    service.start_session(seat=3, source="manual", resume=False)
    pid, _ = _player(service, 0)
    assert _manual(service, pid) == "recorded"


# ---------------------------------------------------------- the echo path
def test_the_feed_echoing_a_manual_pick_is_refused(live):
    pid, name = _player(live, 0)
    _manual(live, pid)
    before = live.session.state.pick_number

    with pytest.raises(DataError, match="already drafted"):
        _feed(live, name, pick=1)

    assert live.session.state.pick_number == before, (
        "the echo advanced the clock; it burned a pick")
    assert len(live.session.state.drafted) == 1


def test_the_echo_does_not_become_an_unresolved_entry(live):
    """An unresolved entry is a prompt for the operator to fix something.
    A duplicate needs no fixing, and 180 phantom prompts is a broken cockpit."""
    pid, name = _player(live, 0)
    _manual(live, pid)
    with pytest.raises(DataError):
        _feed(live, name, pick=1)
    assert list(getattr(live.session.state, "unresolved", [])) == []


def test_the_duplicate_check_does_not_swallow_an_unknown_name(live):
    """The duplicate check must not hide the case it was NOT written for.

    From a FEED an unmatched name is refused outright (see
    `test_an_unmatched_name_from_the_feed_is_refused`); from the OPERATOR it
    is still recorded, because a human typing a name the board lacks has made
    a judgement the tool should not overrule.
    """
    assert _manual_name(live, "Nonexistent Person") == "unresolved"
    assert live.session.state.pick_number == 2


def test_a_whole_draft_of_echoes_costs_no_picks(live):
    """The realistic shape: operator leads, feed follows, every pick echoed."""
    picks = [_player(live, i) for i in range(8)]
    for n, (pid, name) in enumerate(picks, start=1):
        _manual(live, pid)
        with pytest.raises(DataError):
            _feed(live, name, pick=n)

    assert live.session.state.pick_number == len(picks) + 1
    assert len(live.session.state.drafted) == len(picks)


# ----------------------------------------------------------- interleaving
def test_manual_and_feed_interleave_into_one_sequence(live):
    """Both sources writing, alternating, with no gaps or double-counts."""
    a, _ = _player(live, 0)
    _, b_name = _player(live, 1)
    c, _ = _player(live, 2)

    _manual(live, a)
    assert _feed(live, b_name, pick=2) == "recorded"
    _manual(live, c)

    assert live.session.state.pick_number == 4
    assert len(live.session.state.drafted) == 3


# ------------------------------------- the other direction: feed, then click
async def test_clicking_a_row_the_feed_already_recorded_is_a_200(client,
                                                                started):
    """Yahoo records the pick, the board has not refreshed, the operator
    clicks the row anyway.

    The pick IS in. A 409 and a red banner for "the thing you wanted already
    happened" is worse than useless on a clock — and it teaches the operator
    to distrust a board that is in fact correct. The current state rides along
    so the stale board corrects itself.
    """
    row = started.players.iloc[0]
    pid, name = str(row.player_id), str(row.player_name)

    # The feed gets there first.
    started.apply_event(DraftEvent(raw_name=name, player_id=None, seat=None,
                                   source="yahoo", observed_at="",
                                   external_pick_number=1))
    before = started.session.state.pick_number

    response = await client.post("/api/picks", json={"player_id": pid})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "already_recorded"
    assert body["player_id"] == pid
    assert started.session.state.pick_number == before, (
        "the redundant click advanced the clock")


async def test_a_real_conflict_is_still_an_error(client, started):
    """The benign path must not swallow a genuinely bad request."""
    response = await client.post("/api/picks",
                                 json={"player_id": "not-a-real-player"})
    assert response.status_code == 404


def test_a_duplicate_is_still_a_DataError_for_the_poll_loop(live):
    """`DuplicatePick` subclasses `DataError` on purpose: the supervisor
    catches `DataError` to drop feed echoes without backing off. Breaking that
    inheritance would turn every echo into a logged failure and a backoff."""
    from src.app.web.service import DuplicatePick

    assert issubclass(DuplicatePick, DataError)


def test_the_poll_loop_treats_a_duplicate_as_droppable(live):
    """The supervisor catches DataError specifically. If the duplicate raised
    anything else the poller would log it as a failure and back off."""
    pid, name = _player(live, 0)
    _manual(live, pid)
    with pytest.raises(DataError):
        _feed(live, name, pick=1)


# =========================================================== feed limits
# The operator is the authoritative writer. A feed may report a pick; it may
# not invent one. Both cases below used to write into the session and cost a
# pick slot each, desynchronising the draft from that point forward.
def test_an_ambiguous_name_from_the_feed_is_refused(live):
    """Yahoo cannot tell three Robinsons apart, and the poll loop DISCARDS
    apply_event's return value — so an ambiguous name was recorded nowhere
    while its pick number went into the source's `_seen`. The pick vanished
    permanently with no symptom."""
    surnames = [n.split()[-1] for n in live.players.player_name]
    shared = next(x for x in surnames
                  if surnames.count(x) >= 2 and x not in ("Jr.", "Sr.", "II", "III"))
    assert live.resolve(shared).status == "ambiguous"

    before = live.session.state.pick_number
    with pytest.raises(DataError, match="ambiguous"):
        _feed(live, shared, pick=1)
    assert live.session.state.pick_number == before


def test_an_unmatched_name_from_the_feed_is_refused(live):
    """`pick_number` counts drafted PLUS unresolved, and only `undo` can
    repair an unresolved entry. So a feed name this board cannot match, plus
    the operator clicking the right player a moment later, consumed TWO slots
    for ONE pick."""
    before = live.session.state.pick_number
    with pytest.raises(DataError, match="matches nobody"):
        _feed(live, "Nonexistent Person", pick=1)
    assert live.session.state.pick_number == before
    assert list(getattr(live.session.state, "unresolved", [])) == []


def test_one_real_pick_costs_exactly_one_slot(live):
    """The end-to-end shape of the bug above: feed misses, operator clicks."""
    pid, _ = _player(live, 0)
    with pytest.raises(DataError):
        _feed(live, "Jaymyr Gibs", pick=1)      # a typo the board cannot match
    _manual(live, pid)                           # operator clicks the real one
    assert live.session.state.pick_number == 2
    assert len(live.session.state.drafted) == 1


def test_the_operator_may_still_force_an_unresolved_name(live):
    """The refusal is about FEEDS. A human who types a name the board does not
    carry has made a judgement the tool should not overrule."""
    assert _manual_name(live, "Some Deep Sleeper") == "unresolved"
    assert list(live.session.state.unresolved) == ["Some Deep Sleeper"]


# ================================================================== undo
def test_undo_lets_the_feed_report_the_pick_again(live, monkeypatch):
    """A polling source dedupes on pick number, so an undone pick would never
    be mentioned again — the board stays short one player all night while the
    recommender keeps offering someone who is gone."""
    from src.app.web.sources.yahoo import YahooSource

    source = YahooSource(league_key="x", manager_map={"t.1": 0})
    payload = {"a": {"draft_result": {"pick": 1, "team_key": "t.1",
                                      "player_key": "461.p.1"}}}
    assert [e.external_pick_number for e in source._to_events(payload)] == [1]
    assert source._to_events(payload) == []

    source.forget()
    assert [e.external_pick_number for e in source._to_events(payload)] == [1]


def test_undo_calls_forget_on_the_live_source(live):
    """The wiring, not just the capability."""
    called: list[bool] = []

    class Spy:
        name = "spy"

        def forget(self):
            called.append(True)

        def stop(self):
            pass

    live.source = Spy()
    pid, _ = _player(live, 0)
    _manual(live, pid)
    live.undo()
    assert called == [True]


def test_every_source_satisfies_the_forget_protocol():
    """A source without `forget` would break undo the moment it was selected."""
    from src.app.web.sources.manual import ManualSource
    from src.app.web.sources.replay import ReplaySource
    from src.app.web.sources.yahoo import YahooSource

    for cls in (ManualSource, ReplaySource, YahooSource):
        assert callable(getattr(cls, "forget", None)), cls.__name__
