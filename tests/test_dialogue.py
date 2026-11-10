"""Checks for the scripted visitor dialogue in agent.py: what gets asked, who answers, and that
answers only claim what a villager can actually perceive."""

from __future__ import annotations

import pytest
from sandbox.env import make_env
from sandbox.village import props

import agent as villager

NPCS = [f"player_{index}" for index in range(1, 11)]
QUESTIONS = (
    "Hello. How is your day going?",
    "Why does everyone go indoors before dark?",
    "Is the lantern lit?",
    "Are the lanterns lit?",
    "Is your stall open?",
    "Where is the inn?",
    "Where's the gardener?",
    "What are you doing?",
    "What do you sell?",
    "Do you like cheese?",
)


@pytest.mark.parametrize(
    ("text", "topic"),
    [
        ("Hello. How is your day going?", "how_are_you"),
        ("Why does everyone go indoors before dark?", "night_lore"),
        ("Is the lantern lit?", "prop_state"),
        ("Are the lanterns lit?", "prop_state"),
        ("Is the stall open?", "prop_state"),
        ("Where is the inn?", "where_place"),
        ("Where do you live?", "where_place"),
        ("Have you seen the blacksmith?", "where_person"),
        ("Where's the lamplighter?", "where_person"),
        ("Why is the bell ringing?", "bell"),
        ("What do you sell?", "trade_blacksmith"),
        ("What are you doing?", "doing_now"),
        ("What do you do?", "job"),
        ("What time is it?", "time"),
        ("Thank you. I am glad to be here.", "thanks"),
        ("What does the elixir do?", "elixir"),
        ("Does the potion ward off evil spirits?", "elixir"),
        ("Who keeps watch at night?", "guardian"),
        ("Is it safe at night?", "guardian"),
        ("Is the sleepyhead secretly a warrior?", "warrior"),
        ("Why does the sleepyhead sleep all day?", "trade_sleeper"),
    ],
)
def test_questions_route_to_their_topic(text: str, topic: str) -> None:
    assert villager.match_topics(text)[0] == topic


def test_unrelated_chatter_matches_no_topic() -> None:
    assert villager.match_topics("Do you like cheese?") == []


@pytest.fixture(scope="module")
def first_tick():
    env = make_env({"seat_plan": "cast_10", "daynight": True})
    observations, _ = env.reset(seed=0)
    yield env, observations
    env.close()


def _ask(observations, player_id: str, text: str, to: str | None = "self") -> tuple[villager.Agent, list[str]]:
    npc = villager.Agent()
    npc.reset(0, observations[player_id])
    npc.act(_with_visitor(observations[player_id], waving=False))  # in earshot, as anyone talking is
    npc._greetings.outbox = []  # drop the on-sight hello, so only the answer is left to check
    recipient = player_id if to == "self" else to
    sent = npc.chat([{"from": "player_0", "to": recipient, "text": text, "tick": 1}])
    return npc, [message["text"] for message in sent if message["to"] in ("player_0", None)]


@pytest.mark.parametrize("player_id", NPCS)
def test_every_answer_fits_the_chat_cap_and_never_names_a_player_id(first_tick, player_id: str) -> None:
    _, observations = first_tick
    for question in QUESTIONS:
        _, replies = _ask(observations, player_id, question)
        assert len(replies) == 1, question
        assert len(replies[0]) <= villager.REPLY_CAP
        assert "player_" not in replies[0]


@pytest.mark.parametrize("player_id", [pid for pid in NPCS if pid != "player_10"])
def test_lantern_answers_only_claim_what_the_villager_can_see(first_tick, player_id: str) -> None:
    env, observations = first_tick
    npc, replies = _ask(observations, player_id, "Is the lantern lit?")
    lanterns = [prop for prop in props.all(observations[player_id]) if prop["type"] == "lantern"]
    meant = npc._dialogue._referent(lanterns, "Is the lantern lit?", npc._ctx)
    visible = props.state(observations[player_id], meant["id"])
    assert env.day.prop_states[meant["id"]] == "unlit"
    if visible is None:
        assert "can't see" in replies[0]
        assert "I can see it" not in replies[0]
    else:
        assert "is dark. I can see it from here." in replies[0]


def test_the_sleeper_mumbles_instead_of_answering(first_tick) -> None:
    _, observations = first_tick
    _, replies = _ask(observations, "player_10", "Is the lantern lit?")
    assert replies[0].startswith("Zzz")


def test_a_line_sent_to_someone_else_gets_no_answer(first_tick) -> None:
    _, observations = first_tick
    _, replies = _ask(observations, "player_3", "What do you sell?", to="player_1")
    assert replies == []


def test_a_broadcast_naming_a_role_is_answered_only_by_that_role(first_tick) -> None:
    _, observations = first_tick
    _, priest = _ask(observations, "player_3", "Blacksmith, what do you sell?", to=None)
    _, smith = _ask(observations, "player_1", "Blacksmith, what do you sell?", to=None)
    assert priest == []
    assert smith and "sell" in smith[0]


def test_role_questions_get_expert_answers_or_a_pointer_to_the_expert(first_tick) -> None:
    _, observations = first_tick
    _, smith = _ask(observations, "player_1", "What do you sell?")
    _, gardener = _ask(observations, "player_5", "What do you sell?")
    assert "tools" in smith[0]
    assert "blacksmith" in gardener[0]


def test_everyone_explains_the_night_with_the_elixir(first_tick) -> None:
    _, observations = first_tick
    for player_id in [pid for pid in NPCS if pid != "player_10"]:
        _, replies = _ask(observations, player_id, "Why does everyone go indoors before dark?")
        assert "elixir" in replies[0], player_id


def test_the_sleepyhead_keeps_his_warrior_past_a_secret(first_tick) -> None:
    _, observations = first_tick
    npc = villager.Agent()
    npc.reset(0, observations["player_10"])
    npc.act(_with_visitor(observations["player_10"], waving=False))
    npc._greetings.outbox = []
    npc._memory["mode"] = "wander"  # awake, as at night
    npc._memory["awake"] = True
    sent = npc.chat([{"from": "player_0", "to": "player_10", "text": "Are you a legendary warrior?", "tick": 1}])
    assert "Ha!" in sent[0]["text"]


def test_only_the_suspicious_wanderer_has_doubts_about_the_sleepyhead(first_tick) -> None:
    _, observations = first_tick
    _, suspicious = _ask(observations, villager.SUSPICIOUS_ID, "Is the sleepyhead a warrior?")
    _, trusting = _ask(observations, "player_9", "Is the sleepyhead a warrior?")
    assert "blacksmiths" in suspicious[0]  # his doubts point the visitor on to the blacksmiths
    assert "old stories" in trusting[0]


# --- Wave-to-talk sessions -------------------------------------------------------------------

import copy
import math


def _with_visitor(observation, gap: float = 2.0, waving: bool = True, facing_off: float = 0.0):
    """Return a copy of ``observation`` with the visitor standing ``gap`` metres in front of us,
    facing us (give or take ``facing_off`` degrees), waving or not."""

    seen = copy.deepcopy(dict(observation))
    me = seen["self"]
    heading = math.radians(float(me["heading"]))
    x = float(me["position"]["x"]) + gap * math.cos(heading)
    y = float(me["position"]["y"]) + gap * math.sin(heading)
    back = (math.degrees(heading) + 180.0 + facing_off) % 360.0
    visitor = {
        "id": "player_0",
        "position": {"x": x, "y": y},
        "heading": back,
        "moved": 0.0,
        "expression": {"type": "wave" if waving else "none", "target": "none"},
    }
    seen["seen"] = [*[p for p in seen["seen"] if p["id"] != "player_0"], visitor]
    seen["nearby"] = [*[p for p in seen["nearby"] if p["id"] != "player_0"], {"id": "player_0", "position": {"x": x, "y": y}}]
    return seen


def _session_npc(observations, player_id: str) -> villager.Agent:
    npc = villager.Agent()
    npc.reset(0, observations[player_id])
    return npc


def _lines(npc: villager.Agent, inbox=()) -> list[str]:
    return [m["text"] for m in npc.chat(list(inbox)) if m["to"] in ("player_0", None)]


def _drain(npc: villager.Agent, observation, ticks: int = 3) -> list[str]:
    """Let a few ticks pass (the visitor standing still, not waving), collecting what's said."""

    said = []
    for _ in range(ticks):
        npc.act(_with_visitor(observation, waving=False))
        said += _lines(npc)
    return said


def _pick(npc: villager.Agent, observation, number: str) -> list[str]:
    npc.act(_with_visitor(observation, waving=False))
    said = _lines(npc, [{"from": "player_0", "to": npc._player_id, "text": number, "tick": 1}])
    return said + _drain(npc, observation)


@pytest.mark.parametrize("player_id", [pid for pid in NPCS if pid != "player_10"])
def test_a_wave_opens_a_conversation_with_numbered_options(first_tick, player_id: str) -> None:
    _, observations = first_tick
    npc = _session_npc(observations, player_id)
    npc.act(_with_visitor(observations[player_id]))
    said = _lines(npc) + _drain(npc, observations[player_id])
    menu = said[-1]
    assert npc._dialogue.session is not None
    assert "1) “" in menu and menu.endswith("\n5) “End conversation”")
    assert villager.MENU_GAP in menu  # the options sit a blank line below what was said
    assert len(said) == 1  # the greeting and its options arrive together, in one message
    assert all(len(line) <= villager.REPLY_CAP for line in said)


def test_an_answered_option_is_dropped_until_a_new_conversation(first_tick) -> None:
    _, observations = first_tick
    obs = observations["player_1"]
    npc = _session_npc(observations, "player_1")
    npc.act(_with_visitor(obs))
    _lines(npc), _drain(npc, obs)

    answer = _pick(npc, obs, "1")
    assert "tools" in answer[0]
    assert "1) " not in " ".join(answer) and "2) " in " ".join(answer)

    again = _pick(npc, obs, "1")
    assert "already talked about that" in again[0]

    bye = _pick(npc, obs, "5")
    assert bye[0] == villager.BYE_LINES["blacksmith"]
    assert npc._dialogue.session is None

    npc.act(_with_visitor(obs))  # a fresh wave: a new session with every option back
    fresh = " ".join(_lines(npc) + _drain(npc, obs))
    assert "1) “What do you sell?”" in fresh


def test_when_every_option_is_used_only_the_way_out_is_left(first_tick) -> None:
    _, observations = first_tick
    obs = observations["player_3"]
    npc = _session_npc(observations, "player_3")
    npc.act(_with_visitor(obs))
    _lines(npc), _drain(npc, obs)
    for number in "123":
        _pick(npc, obs, number)
    last = " ".join(_pick(npc, obs, "4"))
    assert "That's all I've got for now!\n\n5) “End conversation”" in last


def test_waving_again_mid_conversation_keeps_the_used_options_used(first_tick) -> None:
    _, observations = first_tick
    obs = observations["player_5"]
    npc = _session_npc(observations, "player_5")
    npc.act(_with_visitor(obs))
    _lines(npc), _drain(npc, obs)
    _pick(npc, obs, "2")
    npc.act(_with_visitor(obs))  # wave again
    said = " ".join(_lines(npc) + _drain(npc, obs))
    assert said.startswith("Still here!")
    assert "2) " not in said and "1) " in said


def test_a_number_with_no_conversation_asks_for_a_wave(first_tick) -> None:
    _, observations = first_tick
    _, replies = _ask(observations, "player_1", "2")
    assert replies == ["Wave at me if you'd like a chat!"]


def test_a_wave_facing_away_does_not_open_a_conversation(first_tick) -> None:
    _, observations = first_tick
    npc = _session_npc(observations, "player_1")
    npc.act(_with_visitor(observations["player_1"], facing_off=150.0))
    assert npc._dialogue.session is None


def test_the_conversation_ends_once_the_visitor_walks_off(first_tick) -> None:
    _, observations = first_tick
    obs = observations["player_1"]
    npc = _session_npc(observations, "player_1")
    npc.act(_with_visitor(obs))
    away = copy.deepcopy(dict(obs))
    away["seen"] = [p for p in away["seen"] if p["id"] != "player_0"]
    away["nearby"] = [p for p in away["nearby"] if p["id"] != "player_0"]
    for tick in range(2, 2 + villager.SESSION_AWAY_TICKS + 2):
        away["tick"] = tick
        npc.act(away)
    assert npc._dialogue.session is None


def test_the_sleepyhead_asleep_mumbles_instead_of_opening_a_conversation(first_tick) -> None:
    _, observations = first_tick
    npc = _session_npc(observations, "player_10")
    npc.act(_with_visitor(observations["player_10"]))
    said = _lines(npc)
    assert npc._dialogue.session is None
    assert said and said[0].startswith("Zzz")


def test_only_the_villager_the_visitor_faces_takes_up_the_wave(first_tick) -> None:
    _, observations = first_tick
    obs = _with_visitor(observations["player_1"])
    visitor = next(p for p in obs["seen"] if p["id"] == "player_0")
    vx, vy = visitor["position"]["x"], visitor["position"]["y"]
    # Another villager stands 40 degrees round from us, and the visitor turns to face them instead.
    aside = math.radians(float(visitor["heading"]) + 40.0)
    other = {"x": vx + 2.0 * math.cos(aside), "y": vy + 2.0 * math.sin(aside)}
    visitor["heading"] = math.degrees(aside) % 360.0
    obs["seen"].append({"id": "player_2", "position": other, "heading": 0.0, "moved": 0.0,
                        "expression": {"type": "none", "target": "none"}})
    npc = _session_npc(observations, "player_1")
    npc.act(obs)
    assert npc._dialogue.session is None
    assert npc._dialogue._wave_target(npc._ctx, visitor) == "player_2"


def test_villagers_talk_to_each_other_only_directly(first_tick) -> None:
    _, observations = first_tick
    npc = _session_npc(observations, "player_1")
    npc.act(observations["player_1"])
    npc._greetings.say("player_9", "Hello, wanderer!")  # far off, out of earshot
    sent = npc.chat([])
    assert all(message["to"] is not None for message in sent)
    assert not any(message["text"] == "Hello, wanderer!" for message in sent)


def test_a_seated_worker_nods_back_briefly_without_leaving_the_bench_for_good(first_tick) -> None:
    _, observations = first_tick
    npc = _session_npc(observations, "player_1")
    npc.act(observations["player_1"])
    npc._memory["on_break"] = True
    npc._tick = 1
    npc._reply_on_break("player_8")
    sit = villager.action.stand(0.0, "use")
    gestures = [npc._tree._seated_gesture(npc._context(dict(observations["player_1"], tick=t)), sit)["action"] for t in (2, 3, 4)]
    assert gestures[:2] == [villager.EMOTE_IDS["nod"]] * 2 and gestures[2] == 1


def test_a_villager_stays_in_a_conversation_until_the_visitor_ends_it(first_tick) -> None:
    _, observations = first_tick
    obs = observations["player_1"]  # a blacksmith, pottering about home at dawn
    npc = _session_npc(observations, "player_1")
    npc.act(_with_visitor(obs))
    _lines(npc)
    for tick in range(2, 302):  # five minutes of silence: well past any old timeout
        order = npc.act(_with_visitor(dict(obs, tick=tick), waving=False))
        _lines(npc)
        assert order["speed"] == 0.0, tick
    assert npc._dialogue.session is not None
    npc.act(_with_visitor(dict(obs, tick=302), waving=False))
    _lines(npc, [{"from": "player_0", "to": "player_1", "text": "5", "tick": 302}])
    moving = [npc.act(_with_visitor(dict(obs, tick=tick), waving=False))["speed"] for tick in range(303, 343)]
    assert npc._dialogue.session is None
    assert any(speed > 0 for speed in moving)  # free to go about their day again


# --- The sleepyhead's story ------------------------------------------------------------------


def _option(npc: villager.Agent, label: str) -> str:
    return next(str(number) for number, text, _ in npc._dialogue.session.options if text == label)


def _talk_to(observations, player_id: str, awake: bool = False) -> tuple[villager.Agent, dict]:
    obs = observations[player_id]
    npc = _session_npc(observations, player_id)
    if awake:  # the sleepyhead on his night watch
        npc._memory.update(mode="wander", awake=True)
    npc.act(_with_visitor(obs))
    _lines(npc), _drain(npc, obs)
    return npc, obs


def _say(npc: villager.Agent, obs, text: str) -> list[str]:
    npc.act(_with_visitor(obs, waving=False))
    if npc._dialogue.voice == "sleeper":
        npc._memory.update(mode="wander", awake=True)
    said = _lines(npc, [{"from": "player_0", "to": npc._player_id, "text": text, "tick": 1}])
    return said + _drain(npc, obs)


def test_the_clue_trail_leads_from_priest_to_wanderer_to_blacksmith(first_tick) -> None:
    _, observations = first_tick
    priest, obs = _talk_to(observations, villager.STORY_PRIEST_ID)
    assert "wanderers" in _pick(priest, obs, _option(priest, "Any news of the sleepyhead?"))[0]
    wanderer, obs = _talk_to(observations, villager.SUSPICIOUS_ID)
    assert "blacksmiths" in _pick(wanderer, obs, _option(wanderer, "What do you make of the sleepyhead?"))[0]
    other, obs = _talk_to(observations, "player_2")
    assert "fellow blacksmith" in _pick(other, obs, _option(other, "Know the sleepyhead?"))[0]
    smith, obs = _talk_to(observations, villager.STORY_BLACKSMITH_ID)
    assert "sword" in _pick(smith, obs, _option(smith, "Know the sleepyhead?"))[0]


def test_the_other_priest_has_no_story_option(first_tick) -> None:
    _, observations = first_tick
    priest, _ = _talk_to(observations, "player_4")
    assert all(topic != "clue_priest" for _, _, topic in priest._dialogue.session.options)


def test_the_sleepyhead_only_owns_up_when_confronted_with_the_sword(first_tick) -> None:
    _, observations = first_tick
    sleeper, obs = _talk_to(observations, "player_10", awake=True)
    labels = [label for _, label, _ in sleeper._dialogue.session.options]
    assert villager.PAST_LABEL not in labels

    asked = _say(sleeper, obs, _option(sleeper, "I hear rumours about you."))
    assert asked[0] == villager.RUMOURS_ASK
    brushed = _say(sleeper, obs, "The wanderer says you walk like a soldier.")
    assert "bad knee" in brushed[0] and villager.PAST_LABEL not in " ".join(brushed)

    _say(sleeper, obs, _option(sleeper, "I hear rumours about you."))  # asking again is allowed
    owned_up = " ".join(_say(sleeper, obs, "The blacksmith told me about the sword."))
    assert "found me out" in owned_up
    assert f"5) “{villager.PAST_LABEL}”" in owned_up and "6) “End conversation”" in owned_up

    story = " ".join(_say(sleeper, obs, "5"))
    assert "Night Blade" in story

    _say(sleeper, obs, "6")  # end, then a fresh conversation: he still remembers
    sleeper.act(_with_visitor(obs))
    sleeper._memory.update(mode="wander", awake=True)
    again = " ".join(_lines(sleeper) + _drain(sleeper, obs))
    assert f"3) “{villager.PAST_LABEL}”" in again and "rumours" not in again
