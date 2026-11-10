"""Days at Three Branches villagers built from small, reusable behaviour blocks.

Every villager has one role. A role is a tree of behaviour blocks: each block's ``tick`` returns
the next order while it is running, or ``None`` once it has finished so its parent can move on.
The platform runs a separate ``Agent`` instance for every villager, so each instance builds only
its own tree in ``reset``.
"""

from __future__ import annotations

import heapq
import math
import random
import re
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from sandbox.observation_types import ThreeBranchesAction, ThreeBranchesObservation
from sandbox.village import action, day, geometry, layout, me, people, props

# --- Roles -------------------------------------------------------------------------------------

BLACKSMITH_IDS = ("player_1", "player_2")
PRIEST_IDS = ("player_3", "player_4")
GARDENER_IDS = ("player_5", "player_6")
WANDERER_IDS = ("player_7", "player_8", "player_9")
LAMPLIGHTER_ID = WANDERER_IDS[0]  # the wanderer who lights the village lanterns each evening
SUSPICIOUS_ID = WANDERER_IDS[1]  # the wanderer who suspects the sleepyhead is more than he seems
SLEEPER_IDS = ("player_10",)
VISITOR_ID = "player_0"
WORKER_IDS = (*BLACKSMITH_IDS, *PRIEST_IDS, *GARDENER_IDS)

ROLES = {
    **{player_id: "blacksmith" for player_id in BLACKSMITH_IDS},
    **{player_id: "priest" for player_id in PRIEST_IDS},
    **{player_id: "gardener" for player_id in GARDENER_IDS},
    **{player_id: "wanderer" for player_id in WANDERER_IDS},
    **{player_id: "sleeper" for player_id in SLEEPER_IDS},
}

# --- Tunables (ticks unless noted) -------------------------------------------------------------

STALL_WORK_TICKS = 60
REPAIR_BENCH_WORK_TICKS = 80
PUMP_TICKS = 12
SHRINE_TICKS = 120
PLOT_WORK_TICKS = 70
GREET_TICKS = 8
OBSERVE_TICKS = 60
OBSERVE_DISTANCE = 2.5  # metres a wanderer keeps from the worker it watches
CHAT_TICKS = 40
CHAT_DISTANCE = 1.8  # metres between two chatting wanderers
CHAT_SEE_RANGE = 6.0  # metres: a wanderer seen this close starts a chat
CHAT_HEAR_RANGE = 3.0  # metres: a wanderer heard this close starts a chat
CHAT_COOLDOWN_TICKS = 250
CHAT_EMOTES = ("wave", "laugh", "nod", "point", "shrug", "laugh", "nod")
LOST_TICKS = 20  # how long a wanderer looks for a vanished worker before moving on
STUCK_TICKS = 6  # ticks without movement before a walker re-plans around people
NUDGE_TICKS = 10  # ticks spent shuffling closer when a prop is not yet usable
CLEARANCE_CAP = 3  # cells: open ground beyond this far from obstacles all counts the same
WALL_PENALTY = (0.8, 0.25, 0.0)  # extra route cost for a cell 0, 1, or 2+ cells from an obstacle
LINE_SAMPLE_STEP = 0.25  # metres between the points checked along a straight-line shortcut
LOOKAHEAD_CELLS = 8  # how far along its route a walker looks for a straight-line shortcut
AVOID_DISTANCE = 1.6  # metres: someone this close and ahead makes a walker veer aside
AVOID_ANGLE = 35.0  # degrees either side of the walking direction that count as "ahead"
AVOID_TURN = 40.0  # degrees a walker veers to pass someone
AVOID_COMMIT_TICKS = 4  # keep veering the same way this long, so a walker doesn't wobble
STUCK_PROGRESS = 0.2  # metres: moving less than this in a tick counts as stuck
REACH_MARGIN = 0.35  # stay this far inside use reach when choosing a standing cell
BREAK_TICKS = 100  # how long a worker sits on a bench after each full work cycle
BREAK_BENCH_TRIES = 3  # benches a worker tries before resting on their feet instead
BREAK_CHAT_TICKS = 60  # how long a wanderer chats with a worker on break
BREAK_CHAT_LINE_TICKS = 15  # ticks between a wanderer's lines of small talk
BREAK_CHAT_RANGE = 10.0  # metres: a wanderer notices a worker on break this close
BREAK_CHAT_COOLDOWN_TICKS = 250
BREAK_CHAT_EMOTES = ("nod", "laugh", "point", "shrug")
SEATED_REPLY_EMOTES = ("nod", "laugh", "nod", "shrug")  # a worker on a bench answering small talk
# Ticks a seated worker emotes per reply. Emoting lets go of the bench, so this stays under the
# three missed ticks after which UseProp gives up its seat.
SEATED_GESTURE_TICKS = 2
BREAK_OFF_TICKS = 3  # a worker seen off the bench this long has ended their break
SMALL_TALK = (
    "How's the work going today?",
    "The village feels busy this morning.",
    "Have you heard the bell lately?",
    "Well, I'll let you rest. Enjoy your break!",
)
BREAK_REPLIES = {
    "blacksmith": (
        "Ah, the muscles need a rest.",
        "It's tough work at the forge.",
        "Busy, busy. The stalls need tending all day.",
        "Thanks, friend. Back to the forge soon.",
    ),
    "priest": (
        "Ah, it's good to sit down for a bit.",
        "I feel a divine presence today. A blessing indeed.",
        "Every day is a blessing, busy or not.",
        "Thank you. May your day continue to be well.",
    ),
    "gardener": (
        "Ah, just need a little rest.",
        "Those darn weeds never stop growing!",
        "Lucky us, the soil's good this year.",
        "Thanks! The gardens won't tend themselves.",
    ),
}
HELLO_COOLDOWN_TICKS = 200  # after greeting someone, ignore their hellos this long (stops echo loops)
HELLO_WAVE_TICKS = 6  # how long a villager waves back
HELLO_WAVE_RANGE = 8.0  # metres: a wave from further away is not meant for us
HELLO_WAVE_ANGLE = 60.0  # degrees: a waver must be facing within this of us
VISITOR_SPOT_RANGE = 8.0  # metres: a visitor seen this close gets a hello without greeting first
VISITOR_SPOT_COOLDOWN_TICKS = 600  # after greeting the visitor, wait this long before greeting on sight again
HELLO_WORDS = re.compile(r"\b(hello|hi|hey|hiya|howdy|greetings|good (morning|afternoon|evening|day))\b", re.I)
PLAYER_ID_PATTERN = re.compile(r"\bplayer_\d+\b")
# What villagers call each other in chat: by role, never by player id.
ROLE_TITLES = {
    "blacksmith": "blacksmith",
    "priest": "priest",
    "gardener": "gardener",
    "wanderer": "wanderer",
    "sleeper": "sleepyhead",
}
LAMPLIGHTER_TITLE = "lamplighter"
VISITOR_TITLE = "traveler"
TITLE_PATTERN = re.compile(
    r"\b(" + "|".join((*ROLE_TITLES.values(), LAMPLIGHTER_TITLE, VISITOR_TITLE)) + r")\b", re.I
)
HELLO_REPLIES = {
    "blacksmith": "Hello, {name}! Can't stop, it's hammer time!",
    "priest": "Hello, {name}! Blessings on this fine day.",
    "gardener": "Hello, {name}! It's gonna be a good harvest today, I can feel it!",
    "wanderer": "Hello, {name}! Lovely day for a stroll.",
    "sleeper": "Zzz... mm, hello {name}... zzz.",
    "sleeper_awake": "Hello, {name}! I'm finally up and about.",
    "off_duty": "Hello, {name}! No work right now, just taking it easy.",
    "lamplighter": "Hello, {name}! Can't stop, these lanterns won't light themselves.",
}
WAKE_TICKS = 10  # how long a startled sleeper jumps before setting off
WAKE_LINE = "Wha-?! Who's there? Alright, alright, I'm up!"
NIGHT_WAKE_LINE = "Mm? Night already? Time for my watch."
HOME_WANDER_RADIUS = 5  # cells around the doorway a villager potters about in at dawn
HOME_LINGER_TICKS = (20, 40)  # how long a villager lingers at each spot around home
HOME_EMOTES = ("none", "none", "sweep", "nod")
STROLL_PROPS = ("bench", "pump", "shrine", "stall")  # public spots an evening stroll visits
STROLL_LINGER_TICKS = (30, 60)
STROLL_EMOTES = ("none", "none", "nod", "shrug")
STROLL_CHAT_GAP_TICKS = 150  # a stroller chats at most this often, so a crowd can't hold them forever
EMOTE_TICKS = 8  # how long a lingering villager holds its emote before standing easy
BED_SPACING = 2  # cells kept between housemates' beds
LIGHT_TICKS = 6  # most ticks the lamplighter holds a dark lantern waiting for it to catch
LAMPLIGHTER_LINE = "Evening's drawing in. Time to light the lanterns!"


def title(player_id: str) -> str:
    """What villagers call someone in chat: their role, or "traveler" for the visitor."""

    if people.is_visitor(player_id):
        return VISITOR_TITLE
    if player_id == LAMPLIGHTER_ID:
        return LAMPLIGHTER_TITLE
    return ROLE_TITLES.get(ROLES.get(player_id, ""), "friend")

# Prop footprints in cells, (width, height) before an east/west facing swaps them.
PROP_FOOTPRINTS = {"bench": (2, 1), "stall": (2, 2), "repair_bench": (2, 1), "shrine": (3, 3), "plot": (4, 2), "pump": (1, 1)}
CIRCLE_PROPS = {"pump", "lantern"}

Cell = tuple[int, int]


# --- Navigation --------------------------------------------------------------------------------

_NAV_CACHE: dict[tuple, Navigator] = {}

CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _village_key(observation: ThreeBranchesObservation) -> tuple:
    """Identify a village layout so instances in one process can share its navigator."""

    spawn = layout.spawn(observation)
    placements = tuple(
        (prop["id"], int(prop["cell"]["x"]), int(prop["cell"]["y"])) for prop in props.all(observation)
    )
    return float(spawn["x"]), float(spawn["y"]), placements


class Navigator:
    """The village as a weighted walking graph, built once per day.

    Moves go in eight directions; a diagonal is allowed only when both L-shaped detours around it
    are legal, so it never clips a corner. Each move costs its length, scaled by how slow the
    ground is (so roads and paths are preferred over fields and reeds), plus a penalty for cells
    hugging a wall, prop, or tree. Routes therefore keep a little room from obstacles."""

    def __init__(self, observation: ThreeBranchesObservation):
        size = layout.frame(observation)
        self.cell_size = float(size["cell_size"])
        cells_x, cells_y = int(size["cells_x"]), int(size["cells_y"])
        walkable = {
            (x, y) for x in range(cells_x) for y in range(cells_y) if layout.walkable(observation, {"x": x, "y": y})
        }
        steps: dict[Cell, set[Cell]] = {}
        for x, y in walkable:
            here = {"x": x, "y": y}
            steps[(x, y)] = {
                (x + dx, y + dy)
                for dx, dy in CARDINALS
                if (x + dx, y + dy) in walkable and layout.can_step(observation, here, {"x": x + dx, "y": y + dy})
            }

        # Clearance: how many cells of open ground surround a cell before the nearest obstacle.
        self.clearance: dict[Cell, int] = {}
        frontier = deque()
        for cell in walkable:
            x, y = cell
            if any((x + dx, y + dy) not in walkable for dx, dy in (*CARDINALS, *DIAGONALS)):
                self.clearance[cell] = 0
                frontier.append(cell)
        while frontier:
            x, y = frontier.popleft()
            depth = self.clearance[(x, y)] + 1
            if depth > CLEARANCE_CAP:
                continue
            for dx, dy in (*CARDINALS, *DIAGONALS):
                neighbour = (x + dx, y + dy)
                if neighbour in walkable and neighbour not in self.clearance:
                    self.clearance[neighbour] = depth
                    frontier.append(neighbour)

        slowness = {}
        for cell in walkable:
            ground = layout.ground_at(observation, {"x": cell[0], "y": cell[1]})
            speed = layout.SPEED_LIMITS.get(ground or "", 1.0) or 1.0
            slowness[cell] = 1.0 / speed

        self.edges: dict[Cell, tuple[tuple[Cell, float], ...]] = {}
        for cell, cardinal in steps.items():
            x, y = cell
            moves = [(neighbour, 1.0) for neighbour in cardinal]
            for dx, dy in DIAGONALS:
                across, side_x, side_y = (x + dx, y + dy), (x + dx, y), (x, y + dy)
                if (
                    side_x in cardinal
                    and side_y in cardinal
                    and across in steps.get(side_x, ())
                    and across in steps.get(side_y, ())
                ):
                    moves.append((across, 1.41421356))
            self.edges[cell] = tuple(
                (
                    neighbour,
                    length * (slowness[cell] + slowness[neighbour]) / 2
                    + WALL_PENALTY[min(self.clearance.get(neighbour, CLEARANCE_CAP), len(WALL_PENALTY) - 1)],
                )
                for neighbour, length in moves
            )

    @classmethod
    def for_village(cls, observation: ThreeBranchesObservation) -> Navigator:
        key = _village_key(observation)
        navigator = _NAV_CACHE.get(key)
        if navigator is None:
            navigator = _NAV_CACHE[key] = cls(observation)
        return navigator

    def __contains__(self, cell: Cell) -> bool:
        return cell in self.edges

    def route(self, start: Cell, goals: frozenset[Cell], blocked: frozenset[Cell] = frozenset()) -> list[Cell] | None:
        """A* search for the cheapest route from ``start`` to any goal cell."""

        if start in goals:
            return [start]
        low_x = min(x for x, _ in goals)
        high_x = max(x for x, _ in goals)
        low_y = min(y for _, y in goals)
        high_y = max(y for _, y in goals)

        def estimate(cell: Cell) -> float:
            # Octile distance to the goals' bounding box: never more than the true cost.
            dx = max(low_x - cell[0], 0, cell[0] - high_x)
            dy = max(low_y - cell[1], 0, cell[1] - high_y)
            return max(dx, dy) + 0.41421356 * min(dx, dy)

        cost: dict[Cell, float] = {start: 0.0}
        previous: dict[Cell, Cell | None] = {start: None}
        queue = [(estimate(start), 0, start)]
        order = 0
        while queue:
            _, _, current = heapq.heappop(queue)
            if current in goals:
                route = []
                step: Cell | None = current
                while step is not None:
                    route.append(step)
                    step = previous[step]
                route.reverse()
                return route
            base = cost[current]
            for neighbour, weight in self.edges.get(current, ()):
                if neighbour in blocked and neighbour not in goals:
                    continue
                total = base + weight
                if total < cost.get(neighbour, float("inf")):
                    cost[neighbour] = total
                    previous[neighbour] = current
                    order += 1
                    heapq.heappush(queue, (total + estimate(neighbour), order, neighbour))
        return None

    def line_clear(self, start, end) -> bool:
        """Whether a villager can walk straight from ``start`` to ``end``: every point on the way
        lies in open ground with room to spare (cells next to an obstacle only count at the two
        ends, where the route itself already goes)."""

        sx, sy = float(start["x"]), float(start["y"])
        ex, ey = float(end["x"]), float(end["y"])
        length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
        samples = max(1, int(length / LINE_SAMPLE_STEP))
        size = self.cell_size
        first = (int(sx // size), int(sy // size))
        last = (int(ex // size), int(ey // size))
        for index in range(samples + 1):
            fraction = index / samples
            cell = (int((sx + (ex - sx) * fraction) // size), int((sy + (ey - sy) * fraction) // size))
            if cell not in self.edges:
                return False
            if cell not in (first, last) and self.clearance.get(cell, 0) < 1:
                return False
        return True

    def centre(self, cell: Cell) -> dict[str, float]:
        return {"x": (cell[0] + 0.5) * self.cell_size, "y": (cell[1] + 0.5) * self.cell_size}


def _prop_shape(observation: ThreeBranchesObservation, prop) -> tuple[str, float, float, float, float]:
    """Return a prop's collision shape as ``(kind, x, y, width, height)`` in metres."""

    cell_size = float(layout.frame(observation)["cell_size"])
    width, height = PROP_FOOTPRINTS.get(str(prop["type"]), (1, 1))
    if prop.get("facing") in {"east", "west"}:
        width, height = height, width
    x, y = int(prop["cell"]["x"]) * cell_size, int(prop["cell"]["y"]) * cell_size
    width, height = width * cell_size, height * cell_size
    if prop["type"] in CIRCLE_PROPS:
        diameter = min(width, height)
        return "circle", x + (width - diameter) / 2, y + (height - diameter) / 2, diameter, diameter
    return "box", x, y, width, height


def _prop_center(observation: ThreeBranchesObservation, prop) -> dict[str, float]:
    _, x, y, width, height = _prop_shape(observation, prop)
    return {"x": x + width / 2, "y": y + height / 2}


def _nearest_on_prop(observation: ThreeBranchesObservation, prop, point) -> tuple[dict[str, float], float]:
    """Return the closest point of a prop's shape to ``point`` and the distance to it."""

    kind, x, y, width, height = _prop_shape(observation, prop)
    px, py = float(point["x"]), float(point["y"])
    if kind == "circle":
        centre = {"x": x + width / 2, "y": y + height / 2}
        gap = geometry.distance(point, centre)
        radius = width / 2
        if gap == 0:
            return {"x": centre["x"] + radius, "y": centre["y"]}, 0.0
        scale = radius / gap
        nearest = {"x": centre["x"] + (px - centre["x"]) * scale, "y": centre["y"] + (py - centre["y"]) * scale}
        return nearest, max(0.0, gap - radius)
    nearest = {"x": min(max(px, x), x + width), "y": min(max(py, y), y + height)}
    return nearest, geometry.distance(point, nearest)


def _approach_cells(observation: ThreeBranchesObservation, graph, prop) -> frozenset[Cell]:
    """Return walkable cells whose centre is comfortably within use reach of a prop, with no wall
    in between, so a villager standing there can use it."""

    _, x, y, width, height = _prop_shape(observation, prop)
    reach = geometry.PROP_REACH
    low_x, low_y = int(x - reach) - 1, int(y - reach) - 1
    high_x, high_y = int(x + width + reach) + 1, int(y + height + reach) + 1
    for limit in (reach - REACH_MARGIN, reach):
        found = set()
        for cx in range(low_x, high_x + 1):
            for cy in range(low_y, high_y + 1):
                if (cx, cy) not in graph:
                    continue
                centre = layout.cell_center(observation, {"x": cx, "y": cy})
                nearest, gap = _nearest_on_prop(observation, prop, centre)
                if gap > limit:
                    continue
                # Test sight to a point just outside the prop edge, which is never inside a wall.
                if gap > 0:
                    pull = min(0.1, gap) / gap
                    nearest = {
                        "x": nearest["x"] + (centre["x"] - nearest["x"]) * pull,
                        "y": nearest["y"] + (centre["y"] - nearest["y"]) * pull,
                    }
                if layout.line_of_sight(observation, centre, nearest):
                    found.add((cx, cy))
        if found:
            return frozenset(found)
    return frozenset()


def _home_cells(observation: ThreeBranchesObservation, graph, building_id: str) -> frozenset[Cell]:
    """Return the walkable floor cells inside a building, skipping its wall ring."""

    building = layout.building(observation, building_id)
    if building is None:
        return frozenset()
    width, height = layout.BUILDING_SIZES[str(building["type"])]
    x0, y0 = int(building["cell"]["x"]), int(building["cell"]["y"])
    return frozenset(
        (x, y)
        for x in range(x0 + 1, x0 + width - 1)
        for y in range(y0 + 1, y0 + height - 1)
        if (x, y) in graph
    )


def _doorway_cell(observation: ThreeBranchesObservation, building_id: str) -> Cell | None:
    door = layout.doorway(observation, building_id)
    found = None if door is None else layout.cell_at(observation, door)
    return None if found is None else (found["x"], found["y"])


def _bed_cell(observation: ThreeBranchesObservation, graph, home_id: str, player_id: str) -> frozenset[Cell]:
    """Return the cell a villager sleeps on: as far from the doorway as the home allows, with
    housemates kept apart. Every instance computes the same beds, so they never pick the same one."""

    floor = _home_cells(observation, graph, home_id)
    door = _doorway_cell(observation, home_id)
    if not floor or door is None:
        return floor
    ordered = sorted(floor, key=lambda cell: (-math.dist(cell, door), cell))
    housemates = [str(entry["id"]) for entry in people.roster(observation) if entry["home"] == home_id]
    slot = housemates.index(player_id) if player_id in housemates else 0
    beds: list[Cell] = []
    for cell in ordered:
        if all(max(abs(cell[0] - x), abs(cell[1] - y)) >= BED_SPACING for x, y in beds):
            beds.append(cell)
            if len(beds) > slot:
                break
    return frozenset({beds[min(slot, len(beds) - 1)]})


def _home_area(observation: ThreeBranchesObservation, graph, home_id: str) -> frozenset[Cell]:
    """Return the cells in and just outside a home, where a villager potters about at dawn."""

    floor = _home_cells(observation, graph, home_id)
    door = _doorway_cell(observation, home_id)
    return floor if door is None else floor | _cells_near(graph, door, HOME_WANDER_RADIUS)


def _cells_near(graph, cell: Cell, radius: int) -> frozenset[Cell]:
    """Return walkable cells within a square ``radius`` of ``cell``, excluding the cell itself."""

    x0, y0 = cell
    return frozenset(
        (x, y)
        for x in range(x0 - radius, x0 + radius + 1)
        for y in range(y0 - radius, y0 + radius + 1)
        if (x, y) in graph and (x, y) != cell
    )


# --- Shared, deterministic work assignments ----------------------------------------------------


def _props_of(observation: ThreeBranchesObservation, prop_type: str) -> list[str]:
    return sorted(str(prop["id"]) for prop in props.all(observation) if prop["type"] == prop_type)


def _share(items: list[str], member_ids: tuple[str, ...], player_id: str) -> list[str]:
    """Split ``items`` round-robin between role members so each one owns its own share."""

    if not items or player_id not in member_ids:
        return []
    return items[member_ids.index(player_id) :: len(member_ids)]


def work_sites(observation: ThreeBranchesObservation, player_id: str) -> list[str]:
    """Return the ordered prop ids a worker visits. Every instance computes the same answer, which
    is how wanderers know where to look for a worker they cannot yet see."""

    role = ROLES.get(player_id)
    if role == "blacksmith":
        own = _share(_props_of(observation, "stall"), BLACKSMITH_IDS, player_id)
        repair = _props_of(observation, "repair_bench")[:1]
        # Opposite offsets keep the two blacksmiths from reaching the shared repair bench together.
        return repair + own if BLACKSMITH_IDS.index(player_id) == 0 else own + repair
    if role == "priest":
        pumps = _props_of(observation, "pump")[:1]
        shrines = _props_of(observation, "shrine")
        if not shrines:
            return pumps
        own = shrines[PRIEST_IDS.index(player_id) % len(shrines)]
        return [*pumps, own]
    if role == "gardener":
        return _share(_props_of(observation, "plot"), GARDENER_IDS, player_id)
    return []


# --- Greetings ---------------------------------------------------------------------------------


class Greetings:
    """One villager's hellos: who greeted us and still needs an answer, who we greeted recently,
    and the chat messages waiting to go out on our next ``chat`` turn."""

    def __init__(self, me_id: str):
        self.me_id = me_id
        self.pending: list[str] = []
        self.last_greeted: dict[str, int] = {}
        self.outbox: list[dict] = []
        self.hearers: frozenset[str] = frozenset()

    def heard_hello(self, from_id: str, tick: int) -> None:
        """Note a hello from someone, unless we greeted each other moments ago."""

        if from_id == self.me_id or from_id in self.pending:
            return
        if tick - self.last_greeted.get(from_id, -HELLO_COOLDOWN_TICKS) < HELLO_COOLDOWN_TICKS:
            return
        self.pending.append(from_id)

    def say_hello(self, to_id: str, text: str, tick: int) -> None:
        """Queue a chat hello, routed as ``say`` routes it."""

        self.last_greeted[to_id] = tick
        self.say(to_id, text)

    def say(self, to_id: str, text: str) -> None:
        """Queue a chat line to one villager or the visitor, always as a direct message.

        Nothing is broadcast, so villagers' small talk stays out of the visitor's chat log; the
        visitor sees them emote at each other instead. Someone out of earshot couldn't hear the
        line anyway, so it is dropped."""

        if to_id not in self.hearers:
            return
        recipient = to_id
        self.outbox = [message for message in self.outbox if message["to"] != recipient]
        self.outbox.append({"to": recipient, "text": text})

    def is_hello_for_me(self, message: dict) -> bool:
        """Whether a chat message greets this villager: direct to us, or a broadcast that names
        nobody in particular, or names us by id or by our role."""

        text = str(message.get("text", ""))
        if not HELLO_WORDS.search(text):
            return False
        if message.get("to") == self.me_id:
            return True
        if message.get("to") is not None:
            return False
        ids = PLAYER_ID_PATTERN.findall(text)
        titles = {found.lower() for found in TITLE_PATTERN.findall(text)}
        if not ids and not titles:
            return True
        return self.me_id in ids or title(self.me_id) in titles


def spot_waves(ctx: Context) -> None:
    """Treat a nearby villager who waves while facing us as saying hello."""

    for person in people.seen(ctx.obs):
        if person["expression"]["type"] != "wave" or str(person["id"]) == VISITOR_ID:
            continue  # the visitor's wave opens a conversation instead: see ``Dialogue.watch``
        gap = geometry.distance(ctx.here, person["position"])
        if gap > HELLO_WAVE_RANGE:
            continue
        toward_me = geometry.heading_to(person["position"], ctx.here)
        off = abs((float(person["heading"]) - toward_me + 180.0) % 360.0 - 180.0)
        if off <= HELLO_WAVE_ANGLE:
            ctx.greetings.heard_hello(str(person["id"]), ctx.tick)


def spot_visitor(ctx: Context) -> None:
    """Greet the visitor on catching sight of them close by, unless we greeted them lately."""

    visitor = next((person for person in people.seen(ctx.obs) if people.is_visitor(str(person["id"]))), None)
    if visitor is None or geometry.distance(ctx.here, visitor["position"]) > VISITOR_SPOT_RANGE:
        return
    if VISITOR_ID not in ctx.greetings.hearers:
        return  # wait until they're in earshot, so the hello goes straight to them
    since = ctx.tick - ctx.greetings.last_greeted.get(VISITOR_ID, -VISITOR_SPOT_COOLDOWN_TICKS)
    if since >= VISITOR_SPOT_COOLDOWN_TICKS:
        ctx.greetings.heard_hello(VISITOR_ID, ctx.tick)


# --- Behaviour blocks --------------------------------------------------------------------------


@dataclass
class Context:
    """Everything a block needs for one tick."""

    obs: ThreeBranchesObservation
    me_id: str
    here: dict[str, float]
    cell: Cell | None
    tick: int
    nav: Navigator
    approach: dict[str, frozenset[Cell]]
    greetings: Greetings
    memory: dict = field(default_factory=dict)

    @property
    def graph(self) -> Navigator:
        """The walkable cells, for membership tests such as ``cell in ctx.graph``."""

        return self.nav

    def stand(self, expression: str = "none") -> ThreeBranchesAction:
        return action.stand(me.heading(self.obs), expression)

    def face(self, point, expression: str = "none") -> ThreeBranchesAction:
        return action.stand(geometry.heading_to(self.here, point), expression)

    def approach_cells(self, prop_id: str) -> frozenset[Cell]:
        cells = self.approach.get(prop_id)
        if cells is None:
            prop = props.find(self.obs, prop_id)
            cells = frozenset() if prop is None else _approach_cells(self.obs, self.graph, prop)
            self.approach[prop_id] = cells
        return cells

    def people_cells(self) -> frozenset[Cell]:
        cells = set()
        for person in (*people.seen(self.obs), *people.nearby(self.obs)):
            if str(person["id"]) == self.me_id:
                continue
            found = layout.cell_at(self.obs, person["position"])
            if found is not None:
                cells.add((found["x"], found["y"]))
        return frozenset(cells)


class Block:
    """A behaviour block. ``tick`` returns an order while running and ``None`` once finished."""

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        raise NotImplementedError


class GoTo(Block):
    """Walk a planned route to any cell from ``goals_fn``. Each tick it heads for the furthest
    point along the route it can reach in a clear straight line, so it walks smooth lines and
    diagonals instead of cell-by-cell zig-zags, and veers around people just ahead. Re-plans when
    knocked off the route, and around people when it stops making progress."""

    def __init__(self, goals_fn: Callable[[Context], Iterable[Cell]]):
        self.goals_fn = goals_fn
        self.goals: frozenset[Cell] | None = None
        self.route: list[Cell] | None = None
        self.progress = 0  # index of the furthest route cell reached so far
        self.veer = 0.0  # degrees we are currently veering around someone
        self.veer_left = 0
        self.stuck = 0
        self.failed = False

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.goals is None:
            self.goals = frozenset(self.goals_fn(ctx))
        if not self.goals or ctx.cell is None:
            self.failed = True
            return None
        if ctx.cell in self.goals:
            return None

        self.stuck = self.stuck + 1 if self.route is not None and me.moved(ctx.obs) < STUCK_PROGRESS else 0
        if self.route is None or not self._track(ctx.cell) or self.stuck >= STUCK_TICKS:
            blocked = ctx.people_cells() if self.stuck >= STUCK_TICKS else frozenset()
            route = ctx.nav.route(ctx.cell, self.goals, blocked)
            if route is None and blocked:
                route = ctx.nav.route(ctx.cell, self.goals)
            if route is None:
                self.failed = True
                return None
            self.route, self.progress, self.stuck = route, 0, 0

        heading = geometry.heading_to(ctx.here, self._aim(ctx))
        if self.veer_left > 0 and _open_ahead(ctx, heading + self.veer):
            self.veer_left -= 1
        else:
            self.veer = _avoid_people(ctx, heading)
            self.veer_left = AVOID_COMMIT_TICKS if self.veer else 0
        return action.walk(geometry.wrap(heading + self.veer), 1.0, "none")

    def _track(self, cell: Cell) -> bool:
        """Advance progress to where we are on the route. Shortcuts may leave us beside the route
        rather than on it; only straying further than one cell away counts as being off route."""

        window = self.route[self.progress : self.progress + LOOKAHEAD_CELLS + 2]
        for offset in range(len(window) - 1, -1, -1):
            if window[offset] == cell:
                self.progress += offset
                return True
        return any(max(abs(cell[0] - x), abs(cell[1] - y)) <= 1 for x, y in window[:4])

    def _aim(self, ctx: Context) -> dict[str, float]:
        last = len(self.route) - 1
        for index in range(min(self.progress + LOOKAHEAD_CELLS, last), self.progress, -1):
            point = ctx.nav.centre(self.route[index])
            if ctx.nav.line_clear(ctx.here, point):
                return point
        return ctx.nav.centre(self.route[min(self.progress + 1, last)])


def _open_ahead(ctx: Context, heading: float) -> bool:
    """Whether the next stride in this direction is open ground."""

    ahead = {
        "x": ctx.here["x"] + 0.8 * math.cos(math.radians(heading)),
        "y": ctx.here["y"] + 0.8 * math.sin(math.radians(heading)),
    }
    return ctx.nav.line_clear(ctx.here, ahead)


def _avoid_people(ctx: Context, heading: float) -> float:
    """Return how many degrees to veer to pass someone just ahead (0 when the way is clear or no
    side step is open ground)."""

    for person in people.seen(ctx.obs):
        if str(person["id"]) == ctx.me_id:
            continue
        gap = geometry.distance(ctx.here, person["position"])
        if gap > AVOID_DISTANCE:
            continue
        bearing = geometry.heading_to(ctx.here, person["position"])
        off = (bearing - heading + 180.0) % 360.0 - 180.0
        if abs(off) > AVOID_ANGLE:
            continue
        # Someone is ahead: try veering away from their side first, then the other way.
        for turn in ((-AVOID_TURN, AVOID_TURN) if off >= 0 else (AVOID_TURN, -AVOID_TURN)):
            if _open_ahead(ctx, heading + turn):
                return turn
        break
    return 0.0


class UseProp(Block):
    """Face a prop and hold ``use`` on it for ``ticks``. Shuffles closer first if the engine would
    not yet pick this prop, and gives up if it never becomes usable. ``blocked`` is set when it
    gives up because somebody else is holding the prop.

    ``face_fn`` may return a point to face instead of the prop: turning does not let go of it."""

    def __init__(
        self,
        prop_id: str,
        ticks: int,
        skip_if_busy: bool = False,
        face_fn: Callable[[Context], dict[str, float] | None] | None = None,
    ):
        self.prop_id = prop_id
        self.left = ticks
        self.skip_if_busy = skip_if_busy
        self.face_fn = face_fn
        self.started = False
        self.blocked = False
        self.misses = 0
        self.nudges = 0

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        prop = props.find(ctx.obs, self.prop_id)
        if prop is None or self.left <= 0:
            return None
        centre = _prop_center(ctx.obs, prop)
        if not self.started and self.skip_if_busy and props.state(ctx.obs, self.prop_id) == "busy":
            return None

        usable = props.usable(ctx.obs)
        if usable is None or usable["id"] != self.prop_id:
            self.nudges += 1
            if self.nudges > NUDGE_TICKS:
                return None
            nearest, _ = _nearest_on_prop(ctx.obs, prop, ctx.here)
            return action.walk(geometry.heading_to(ctx.here, nearest), 0.3, "none")

        if self.started:
            # Our expression shows last tick's result: someone else holding the prop leaves it "none".
            holding = me.expression(ctx.obs)
            self.misses = 0 if holding["type"] == "use" and holding["target"] == self.prop_id else self.misses + 1
            if self.misses >= 3:
                self.blocked = True
                return None
        self.started = True
        self.left -= 1
        look_at = self.face_fn(ctx) if self.face_fn is not None else None
        return ctx.face(look_at or centre, "use")


class Pause(Block):
    """Stand still with an expression for a number of ticks."""

    def __init__(self, ticks: int, expression: str = "none"):
        self.left = ticks
        self.expression = expression

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.left <= 0:
            return None
        self.left -= 1
        return ctx.stand(self.expression)


class Sequence(Block):
    """Run child blocks one after another."""

    def __init__(self, *children: Block):
        self.children = list(children)

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        while self.children:
            order = self.children[0].tick(ctx)
            if order is not None:
                return order
            self.children.pop(0)
        return None


class Repeat(Block):
    """Rebuild a child from ``factory`` each time it finishes, forever."""

    def __init__(self, factory: Callable[[], Block]):
        self.factory = factory
        self.child = factory()

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        # A few rebuilds per tick lets a child that finishes instantly hand over without idling,
        # while a routine with nothing to do cannot spin forever.
        for _ in range(4):
            order = self.child.tick(ctx)
            if order is not None:
                return order
            self.child = self.factory()
        return ctx.stand()


class Interrupt(Block):
    """Run ``routine`` until ``trigger`` offers a higher-priority block, run that block to
    completion, then resume the routine where it left off."""

    def __init__(self, trigger: Callable[[Context], Block | None], routine: Block):
        self.trigger = trigger
        self.routine = routine
        self.active: Block | None = None

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.active is None:
            self.active = self.trigger(ctx)
        if self.active is not None:
            order = self.active.tick(ctx)
            if order is not None:
                return order
            self.active = None
        return self.routine.tick(ctx)


Activity = tuple[str, str, Callable[[Context], Block]]


class ByPhase(Block):
    """Run the activity ``schedule`` names for the current phase of the day.

    ``schedule`` maps a phase to ``(activity, mode, factory)``, with ``"day"`` as the fallback. A
    fresh block is built only when the activity changes, so one spanning several phases carries on
    uninterrupted. ``mode`` tells other blocks what the villager is up to: ``"work"``, ``"wander"``,
    or ``"sleep"``; a block may change it later, as a sleeper does on waking."""

    def __init__(self, schedule: dict[str, Activity]):
        self.schedule = schedule
        self.activity: str | None = None
        self.child: Block | None = None

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        activity, mode, factory = self.schedule.get(day.phase(ctx.obs)) or self.schedule["day"]
        if activity != self.activity:
            self.activity, self.child = activity, factory(ctx)
            ctx.memory["mode"] = mode
            ctx.memory["activity"] = activity  # what the villager says they're up to when asked
            # A break cut short by the hour must not leave the worker looking like they're on one.
            for key in ("on_break", "break_talker", "break_lines_said"):
                ctx.memory.pop(key, None)
        return self.child.tick(ctx)


class ReplyToHellos(Block):
    """Wrap a role's routine so the villager answers every hello in chat and waves back.

    How much the reply interrupts the routine follows the villager's current mode: wandering turns
    to face the greeter and waves; working keeps at it and only waves while walking (a wave mid-use
    would drop the prop); sleeping mumbles a reply without waking up.
    """

    def __init__(self, routine: Block, role: str):
        self.routine = routine
        self.role = role
        self.wave_left = 0
        self.wave_at: str | None = None

    def _manner(self, ctx: Context) -> tuple[str, str]:
        """Return ``(mode, reply)`` for answering a hello right now."""

        default = {"wanderer": "wander", "sleeper": "sleep"}.get(self.role, "work")
        mode = ctx.memory.get("mode", default)
        if mode == "sleep":
            dozing = me.expression(ctx.obs)["type"] == "sleep"
            if dozing or (self.role == "sleeper" and not ctx.memory.get("awake")):
                return mode, HELLO_REPLIES["sleeper"]
            mode = "wander"  # still on the way to bed
        if self.role == "sleeper":
            return mode, HELLO_REPLIES["sleeper_awake"]
        if mode == "wander" and self.role != "wanderer":
            return mode, HELLO_REPLIES["off_duty"]
        if mode == "work" and self.role == "wanderer":
            return mode, HELLO_REPLIES["lamplighter"]
        return mode, HELLO_REPLIES.get(self.role, HELLO_REPLIES["wanderer"])

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        return self._notice_visitor(ctx, self._seated_gesture(ctx, self._tick(ctx)))

    def _seated_gesture(self, ctx: Context, order: ThreeBranchesAction | None) -> ThreeBranchesAction | None:
        """A worker on a bench nods or laughs back at small talk for a tick or two. Only on a
        bench: leaving a stall even for a tick would close it."""

        seated = ctx.memory.get("seated_gesture")
        if order is None or order["action"] != 1 or not ctx.memory.get("on_break"):
            return order
        if seated is None or seated[1] < ctx.tick:
            return order
        return {**order, "action": EMOTE_IDS[seated[0]]}

    def _notice_visitor(self, ctx: Context, order: ThreeBranchesAction | None) -> ThreeBranchesAction | None:
        """Look at the visitor while they're within earshot, so we see them wave. Standing orders
        just turn: using a prop depends on where we stand, not which way we face. And when the
        visitor walks right up to us, we stop for a moment to look, even mid-walk."""

        if order is None or ctx.memory.get("mode") == "sleep":
            return order
        visitor = next((person for person in people.nearby(ctx.obs) if str(person["id"]) == VISITOR_ID), None)
        gap = geometry.distance(ctx.here, visitor["position"]) if visitor is not None else math.inf
        if gap <= NOTICE_RANGE:
            if ctx.tick > ctx.memory.get("notice_until", -1) + NOTICE_REARM_TICKS:
                ctx.memory["notice_armed"] = True  # still close a while later: look again
            if ctx.memory.get("notice_armed", True):
                ctx.memory["notice_armed"] = False
                ctx.memory["notice_until"] = ctx.tick + NOTICE_TICKS
        elif gap > NOTICE_RANGE + 1.0:
            ctx.memory["notice_armed"] = True  # they've stepped away, so their next approach counts again
        if visitor is None:
            return order
        noticing = ctx.memory.get("notice_until", -1) >= ctx.tick
        toward = geometry.wrap(geometry.heading_to(ctx.here, visitor["position"]))
        if order["action"] not in (0, 1, EMOTE_IDS["wave"]):
            # Mid-chat with another villager: just a glance when the visitor walks up, so their
            # wave is seen, then back to facing whoever we're talking to.
            return {**order, "heading": toward, "speed": 0.0} if noticing else order
        if float(order["speed"]) > 0:
            if order["action"] == 1 or not noticing:
                return order
            return {**order, "heading": toward, "speed": 0.0}
        return {**order, "heading": toward}

    def _stay_for_visitor(self, ctx: Context, gesture: str, in_session: bool) -> ThreeBranchesAction | None:
        """Stay put, facing the visitor, while we're talking with them. In an open conversation
        every villager stays, whatever they were doing, until the visitor ends it: a worker at a
        prop keeps working there (``_notice_visitor`` turns them to face the visitor), but doesn't
        set off for the next job. A wanderer answering a free-typed question stays a short while."""

        if me.expression(ctx.obs)["type"] == "use":
            order = self.routine.tick(ctx)
            if order is not None and order["action"] == 1:
                return order
        visitor = people.find(ctx.obs, VISITOR_ID)
        if visitor is None:
            # Out of sight a moment: wait where we are for them to come back.
            return ctx.stand() if in_session else None
        return ctx.face(visitor["position"], gesture)

    def _tick(self, ctx: Context) -> ThreeBranchesAction | None:
        spot_waves(ctx)
        mode, reply = self._manner(ctx)
        if mode != "sleep":
            spot_visitor(ctx)
        greetings = ctx.greetings
        while greetings.pending:
            greeter = greetings.pending.pop(0)
            text = reply.format(name=title(greeter))
            if greeter == VISITOR_ID and mode != "sleep":
                text += WAVE_HINT  # tell the visitor how to start a conversation
            greetings.say_hello(greeter, text, ctx.tick)
            self.wave_left, self.wave_at = HELLO_WAVE_TICKS, greeter

        if mode == "sleep":
            return self.routine.tick(ctx)
        gesture = _gesture(ctx)
        in_session = ctx.memory.get("in_session", False)
        if in_session or (self.wave_left <= 0 and mode == "wander" and ctx.memory.get("talking_until", -1) >= ctx.tick):
            expression = gesture
            if self.wave_left > 0:
                # Someone else said hello: wave back from where we stand, without leaving the visitor.
                self.wave_left -= 1
                expression = "wave"
            held = self._stay_for_visitor(ctx, expression, in_session)
            if held is not None:
                return held
        if self.wave_left <= 0:
            order = self.routine.tick(ctx)
            if gesture == "none" or order is None or order["action"] == 1:
                return order
            return action.walk(order["heading"], order["speed"], gesture)
        self.wave_left -= 1

        if mode == "wander":
            greeter = people.find(ctx.obs, self.wave_at)
            if greeter is not None:
                return ctx.face(greeter["position"], "wave")

        order = self.routine.tick(ctx)
        if order is None or order["action"] == 1:  # 1 is "use": keep working, the chat reply is enough
            return order
        return action.walk(order["heading"], order["speed"], "wave")


# --- Role-specific blocks ----------------------------------------------------------------------


def work_at(prop_id: str, ticks: int, skip_if_busy: bool = False) -> Block:
    """Walk to a prop, then use it for a while."""

    return Sequence(GoTo(lambda ctx: ctx.approach_cells(prop_id)), UseProp(prop_id, ticks, skip_if_busy))


def rotation(steps: list[Callable[[], Block]]) -> Block:
    """Loop through a list of step factories in order, forever."""

    if not steps:
        return Pause(1)
    cursor = {"next": 0}

    def next_step() -> Block:
        step = steps[cursor["next"] % len(steps)]
        cursor["next"] += 1
        return step()

    return Repeat(next_step)


class TakeBreak(Block):
    """After a full work cycle: find a free bench, sit there for a while, then hand back to work.
    A bench seen to be occupied on the way, or held by someone on arrival, is swapped for the next
    nearest. If no bench works out, the worker rests on their feet instead."""

    def __init__(self):
        self.tried: set[str] = set()
        self.bench: str | None = None
        self.walk: GoTo | None = None
        self.sit: UseProp | None = None
        self.rest: Pause | None = None

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        order = self._run(ctx)
        ctx.memory["on_break"] = order is not None
        if order is None:
            ctx.memory.pop("break_talker", None)
            ctx.memory.pop("break_lines_said", None)
        return order

    def _run(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.rest is not None:
            return self.rest.tick(ctx)
        while True:
            if self.bench is None:
                self.bench = self._pick_bench(ctx)
                if self.bench is None:
                    self.rest = Pause(BREAK_TICKS // 2)
                    return self.rest.tick(ctx)
                self.tried.add(self.bench)
                self.walk = GoTo(lambda c, bench=self.bench: c.approach_cells(bench))
                self.sit = UseProp(self.bench, BREAK_TICKS, face_fn=_face_talker)

            if self.walk is not None:
                if props.state(ctx.obs, self.bench) == "occupied":
                    self.bench = None
                    continue
                order = self.walk.tick(ctx)
                if order is not None:
                    return order
                if self.walk.failed:
                    self.bench = None
                    continue
                self.walk = None

            order = self.sit.tick(ctx)
            if order is not None:
                return order
            if self.sit.started and not self.sit.blocked:
                return None  # sat out the whole break
            self.bench = None

    def _pick_bench(self, ctx: Context) -> str | None:
        if len(self.tried) >= BREAK_BENCH_TRIES:
            return None
        benches = [
            prop
            for prop in props.all(ctx.obs)
            if prop["type"] == "bench" and prop["id"] not in self.tried and props.state(ctx.obs, prop["id"]) != "occupied"
        ]
        if not benches:
            return None
        nearest = min(benches, key=lambda prop: geometry.distance(ctx.here, _prop_center(ctx.obs, prop)))
        return str(nearest["id"])


def _face_talker(ctx: Context) -> dict[str, float] | None:
    """While seated on a break, turn toward the wanderer who is chatting with us."""

    talker = ctx.memory.get("break_talker")
    person = people.find(ctx.obs, talker) if talker else None
    return None if person is None else person["position"]


def on_break(observation: ThreeBranchesObservation, person) -> bool:
    """Whether a seen worker is on break: sitting on a bench, which nobody uses for work."""

    expression = person.get("expression")
    if not expression or expression["type"] != "use":
        return False
    prop = props.find(observation, expression["target"])
    return prop is not None and prop["type"] == "bench"


class Follow:
    """Close in on a person who may be moving: walk straight at them when the way is clear and
    short, otherwise follow a route to the cells around them, refreshed as they move."""

    def __init__(self):
        self.walker: GoTo | None = None
        self.age = 0

    def step(self, ctx: Context, target) -> ThreeBranchesAction:
        gap = geometry.distance(ctx.here, target)
        if gap < 6.0 and ctx.nav.line_clear(ctx.here, target):
            return action.walk(geometry.heading_to(ctx.here, target), 1.0, "none")
        target_cell = layout.cell_at(ctx.obs, target)
        if target_cell is None:
            return ctx.stand()
        self.age += 1
        if self.walker is None or self.age > 15:
            cell = (target_cell["x"], target_cell["y"])
            self.walker = GoTo(lambda c, cell=cell: _cells_near(c.graph, cell, 2))
            self.age = 0
        order = self.walker.tick(ctx)
        return order if order is not None else ctx.face(target)

    def resume(self, ctx: Context) -> ThreeBranchesAction:
        """Keep walking toward where the person was last seen while they are out of sight."""

        order = self.walker.tick(ctx) if self.walker is not None else None
        return order if order is not None else ctx.stand()


class VisitWorker(Block):
    """Find a worker at one of their work sites, wave hello, then stand and watch them work. A
    worker found on a break gets a chat instead."""

    def __init__(self, worker_id: str, sites: list[str]):
        self.worker_id = worker_id
        self.sites = sites[:2]  # look in at most two places before moving on
        self.phase = "search"
        self.search: GoTo | None = None
        self.follow = Follow()
        self.chat: BreakChat | None = None
        self.lost = 0
        self.left = 0

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.chat is not None:
            return self.chat.tick(ctx)

        worker = people.find(ctx.obs, self.worker_id)
        if self.phase == "search":
            if worker is not None:
                self.phase = "approach"
            else:
                return self._search(ctx)

        if worker is None:
            self.lost += 1
            if self.lost > LOST_TICKS:
                return None
            return self.follow.resume(ctx) if self.phase == "approach" else ctx.stand()
        self.lost = 0

        if on_break(ctx.obs, worker) and _break_chat_allowed(ctx, self.worker_id):
            self.chat = BreakChat(self.worker_id)
            return self.chat.tick(ctx)

        gap = geometry.distance(ctx.here, worker["position"])
        if self.phase == "approach":
            if gap <= OBSERVE_DISTANCE:
                self.phase, self.left = "greet", GREET_TICKS
                ctx.greetings.say_hello(self.worker_id, f"Hello, {title(self.worker_id)}! Mind if I watch?", ctx.tick)
            else:
                return self.follow.step(ctx, worker["position"])
        if self.phase == "greet":
            if self.left > 0:
                self.left -= 1
                return ctx.face(worker["position"], "wave")
            self.phase, self.left = "observe", OBSERVE_TICKS
        if self.left <= 0:
            return None
        self.left -= 1
        if gap > OBSERVE_DISTANCE * 2:
            # The worker moved on to their next job: tag along and keep watching.
            return self.follow.step(ctx, worker["position"])
        nod = self.left % 25 == 0
        return ctx.face(worker["position"], "nod" if nod else "none")

    def _search(self, ctx: Context) -> ThreeBranchesAction | None:
        while self.sites:
            if self.search is None:
                site = self.sites[0]
                self.search = GoTo(lambda c, site=site: c.approach_cells(site))
            order = self.search.tick(ctx)
            if order is not None:
                return order
            self.sites.pop(0)
            self.search = None
        return None


class Chat(Block):
    """Walk up to another wanderer and trade emotes with them for a while."""

    def __init__(self, partner_id: str):
        self.partner_id = partner_id
        self.left = CHAT_TICKS
        self.approach = 20
        self.lost = 0
        self.greeted = False

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if not self.greeted:
            self.greeted = True
            fellow = "fellow " if title(self.partner_id) == title(ctx.me_id) else ""
            greeting = f"Hello, {fellow}{title(self.partner_id)}! Fancy meeting you here."
            ctx.greetings.say_hello(self.partner_id, greeting, ctx.tick)
        partner = people.find(ctx.obs, self.partner_id)
        if partner is None:
            self.lost += 1
            return None if self.lost > 5 else ctx.stand()
        self.lost = 0
        gap = geometry.distance(ctx.here, partner["position"])
        if gap > CHAT_DISTANCE and self.approach > 0:
            self.approach -= 1
            if ctx.nav.line_clear(ctx.here, partner["position"]):
                return action.walk(geometry.heading_to(ctx.here, partner["position"]), 0.8, "wave")
            return ctx.face(partner["position"], "wave")
        if gap > CHAT_SEE_RANGE or self.left <= 0:
            return None
        self.left -= 1
        emote = CHAT_EMOTES[(CHAT_TICKS - self.left) // 6 % len(CHAT_EMOTES)]
        return ctx.face(partner["position"], emote)


class BreakChat(Block):
    """Walk over to a worker sitting out their break and make small talk until it ends. The worker
    stays seated, so they answer in chat while the wanderer does the emoting."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.follow = Follow()
        self.phase = "approach"
        self.left = BREAK_CHAT_TICKS
        self.line = 0
        self.lost = 0
        self.off_bench = 0

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        worker = people.find(ctx.obs, self.worker_id)
        if worker is None:
            self.lost += 1
            return None if self.lost > 5 else self.follow.resume(ctx)
        self.lost = 0
        # A seen record shows whether they are still sitting; a heard one only shows where they are.
        # A tick or two off the bench is them nodding back at us, not the end of their break.
        if "expression" in worker and not on_break(ctx.obs, worker):
            self.off_bench += 1
            if self.off_bench >= BREAK_OFF_TICKS:
                return None  # the break is over
        else:
            self.off_bench = 0

        gap = geometry.distance(ctx.here, worker["position"])
        if self.phase == "approach":
            if gap > CHAT_DISTANCE + 0.4:
                return self.follow.step(ctx, worker["position"])
            self.phase = "talk"
            ctx.greetings.say_hello(self.worker_id, f"Hello, {title(self.worker_id)}! Taking a break?", ctx.tick)
            ctx.memory.setdefault("break_chats", {})[self.worker_id] = ctx.tick + BREAK_CHAT_COOLDOWN_TICKS
            return ctx.face(worker["position"], "wave")

        if self.left <= 0:
            return None
        self.left -= 1
        talked = BREAK_CHAT_TICKS - self.left
        if talked % BREAK_CHAT_LINE_TICKS == 0 and self.line < len(SMALL_TALK):
            ctx.greetings.say(self.worker_id, SMALL_TALK[self.line])
            self.line += 1
        emote = BREAK_CHAT_EMOTES[talked // 8 % len(BREAK_CHAT_EMOTES)]
        return ctx.face(worker["position"], emote)


class LightLanterns(Block):
    """Make a round of every lantern in the village, lighting each one still dark, nearest first.

    A lantern toggles whenever someone new takes hold of it, so grabbing a lit one would put it
    out. The lamplighter crosses off any lantern it sees already lit, and holds a dark one only
    until it shows lit. Finishes once every lantern is lit or has been tried."""

    def __init__(self):
        self.left: list[str] | None = None
        self.target: str | None = None
        self.walk: GoTo | None = None
        self.light: UseProp | None = None

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.left is None:
            self.left = [str(prop["id"]) for prop in props.all(ctx.obs) if prop["type"] == "lantern"]
            if VISITOR_ID in ctx.greetings.hearers:  # said aloud for a visitor nearby, not broadcast
                ctx.greetings.say(VISITOR_ID, LAMPLIGHTER_LINE)
        if self.light is not None and self.light.started and props.state(ctx.obs, self.target) == "lit":
            # Remembered so the lamplighter can later say which lanterns they lit, and when.
            ctx.memory.setdefault("lit_by_me", {}).setdefault(self.target, (ctx.tick, day.phase(ctx.obs)))
        self.left = [lantern for lantern in self.left if props.state(ctx.obs, lantern) != "lit"]

        while self.left:
            if self.target not in self.left:
                self.target = min(
                    self.left, key=lambda lantern: geometry.distance(ctx.here, _prop_center(ctx.obs, props.find(ctx.obs, lantern)))
                )
                self.walk = GoTo(lambda c, lantern=self.target: c.approach_cells(lantern))
                self.light = UseProp(self.target, LIGHT_TICKS)
            if self.walk is not None:
                order = self.walk.tick(ctx)
                if order is not None:
                    return order
                if self.walk.failed:
                    self.left.remove(self.target)
                    continue
                self.walk = None
            order = self.light.tick(ctx)
            if order is not None:
                return order
            self.left.remove(self.target)  # it never caught (or someone held it): move on
        return None


def lamplighter_evening(observation: ThreeBranchesObservation, player_id: str, rng: random.Random) -> Block:
    """Light the lanterns, then stroll for what is left of the evening, or go to bed if night has
    fallen meanwhile."""

    home = me.home(observation)
    after = ByPhase({
        "day": ("stroll", "wander", lambda ctx: evening_stroll(observation, rng)),
        "night": ("sleep", "sleep", lambda ctx: sleep_at_home(player_id, home)),
    })
    return Sequence(LightLanterns(), after)


def _break_chat_allowed(ctx: Context, worker_id: str) -> bool:
    return ctx.memory.setdefault("break_chats", {}).get(worker_id, 0) <= ctx.tick


def is_wandering(observation: ThreeBranchesObservation, person) -> bool:
    """Whether someone is out wandering: a wanderer before nightfall, a worker off duty in the
    evening, or a sleeper who has been woken up. A sleeper only ever sleeps indoors, so one seen
    out of doors must be awake."""

    role = ROLES.get(str(person["id"]))
    phase = day.phase(observation)
    if str(person["id"]) == LAMPLIGHTER_ID and phase == "evening":
        return False  # busy with the lanterns
    if role == "wanderer":
        return phase != "night"
    if role in ("blacksmith", "priest", "gardener"):
        return phase == "evening"
    return role == "sleeper" and layout.building_at(observation, person["position"]) is None


def wanderer_social_trigger(ctx: Context) -> Block | None:
    """Interrupt a wanderer's rounds to chat with a nearby wanderer, or with a worker on break."""

    cooldowns: dict[str, int] = ctx.memory.setdefault("chat_cooldowns", {})
    candidates = [
        (person, CHAT_SEE_RANGE) for person in people.seen(ctx.obs)
    ] + [(person, CHAT_HEAR_RANGE) for person in people.nearby(ctx.obs)]
    for person, reach in candidates:
        other = str(person["id"])
        if other == ctx.me_id or not is_wandering(ctx.obs, person):
            continue
        if cooldowns.get(other, 0) > ctx.tick:
            continue
        if geometry.distance(ctx.here, person["position"]) <= reach:
            cooldowns[other] = ctx.tick + CHAT_COOLDOWN_TICKS
            return Chat(other)

    for person in people.seen(ctx.obs):
        worker = str(person["id"])
        if ROLES.get(worker) not in ("blacksmith", "priest", "gardener") or not on_break(ctx.obs, person):
            continue
        if not _break_chat_allowed(ctx, worker):
            continue
        if geometry.distance(ctx.here, person["position"]) <= BREAK_CHAT_RANGE:
            ctx.memory["break_chats"][worker] = ctx.tick + BREAK_CHAT_COOLDOWN_TICKS
            return BreakChat(worker)
    return None


# --- Role trees --------------------------------------------------------------------------------


def build_role(observation: ThreeBranchesObservation, player_id: str, rng: random.Random) -> Block:
    """Return this villager's whole tree: their role's day, wrapped so they answer hellos."""

    role = ROLES.get(player_id, "wanderer")
    return ReplyToHellos(build_routine(observation, player_id, role, rng), role)


def build_routine(observation: ThreeBranchesObservation, player_id: str, role: str, rng: random.Random) -> Block:
    """Return the role's schedule through the day. Everyone but the sleeper potters about home at
    dawn and goes to bed at night; a season without day and night runs the ``"day"`` entry."""

    home = me.home(observation)
    rest: Activity = ("sleep", "sleep", lambda ctx: sleep_at_home(player_id, home))
    potter: Activity = ("home", "wander", lambda ctx: wander_home(observation, home, rng))

    if role in ("blacksmith", "priest", "gardener"):
        work: Activity = ("work", "work", lambda ctx: work_routine(observation, player_id, role))
        stroll: Activity = ("stroll", "wander", lambda ctx: evening_stroll(observation, rng))
        return ByPhase({"dawn": potter, "day": work, "morning": work, "midday": work, "evening": stroll, "night": rest})

    if role == "sleeper":

        def nightfall(ctx: Context) -> Block:
            # Woken earlier, they turn in with everyone else; slept all day, they get up now.
            if ctx.memory.get("awake"):
                return sleep_at_home(player_id, home)
            return NightOwl(wandering(observation, player_id, rng))

        dozing: Activity = ("day", "sleep", lambda ctx: Sleeper(observation, player_id, rng))
        night: Activity = ("night", "sleep", nightfall)
        return ByPhase({"dawn": dozing, "day": dozing, "morning": dozing, "midday": dozing, "evening": dozing, "night": night})

    out: Activity = ("out", "wander", lambda ctx: wandering(observation, player_id, rng))
    evening = out
    night = rest
    if player_id == LAMPLIGHTER_ID:
        # One activity across evening and night, so a round still under way at nightfall finishes.
        evening = night = ("lamps", "work", lambda ctx: lamplighter_evening(observation, player_id, rng))
    return ByPhase({"dawn": potter, "day": out, "morning": out, "midday": out, "evening": evening, "night": night})


def work_routine(observation: ThreeBranchesObservation, player_id: str, role: str) -> Block:
    """Loop through the worker's sites, with a break on a bench after each full lap."""

    sites = work_sites(observation, player_id)
    if role in ("blacksmith", "priest", "gardener"):
        repair = set(_props_of(observation, "repair_bench"))
        ticks = {
            "repair_bench": REPAIR_BENCH_WORK_TICKS,
            "stall": STALL_WORK_TICKS,
            "pump": PUMP_TICKS,
            "shrine": SHRINE_TICKS,
            "plot": PLOT_WORK_TICKS,
        }
        steps = [
            lambda site=site: work_at(
                site, ticks.get(str(props.find(observation, site)["type"]), STALL_WORK_TICKS), skip_if_busy=site in repair
            )
            for site in sites
        ]
        # One full lap of the work sites, then a break on a bench, then back to work.
        return rotation(steps + [TakeBreak] if steps else [])
    return rotation([])


SLEEP_ACTION = action.stand(0.0, "sleep")["action"]


def startled_by_visitor(ctx: Context) -> dict[str, float] | None:
    """Return where the visitor is if we can see them doing the startle emote close by."""

    for person in people.seen(ctx.obs):
        if not people.is_visitor(str(person["id"])) or person["expression"]["type"] != "startle":
            continue
        if geometry.distance(ctx.here, person["position"]) <= geometry.HEARING_RANGE:
            return person["position"]
    return None


class Sleeper(Block):
    """Sleep at home all day, turning over to face a visitor heard nearby. If the visitor startles
    them, they jump awake and spend the rest of the day wandering like a wanderer."""

    def __init__(self, observation: ThreeBranchesObservation, player_id: str, rng: random.Random):
        self.asleep = sleep_at_home(player_id, me.home(observation))
        self.wake_into = lambda: wandering(observation, player_id, rng)
        self.awake: Block | None = None
        self.jolt = 0
        self.visitor_at: dict[str, float] | None = None

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.awake is None:
            self.visitor_at = startled_by_visitor(ctx)
            if self.visitor_at is not None:
                ctx.memory["awake"] = True
                ctx.memory["mode"] = "wander"
                self.awake, self.jolt = self.wake_into(), WAKE_TICKS
                ctx.greetings.say(VISITOR_ID, WAKE_LINE)

        if self.awake is None:
            order = self.asleep.tick(ctx)
            visitor = people.find(ctx.obs, VISITOR_ID)
            if order is not None and order["action"] == SLEEP_ACTION and visitor is not None:
                # Stir and roll over toward the noise, so a startle right here is seen.
                return ctx.face(visitor["position"], "sleep")
            return order

        if self.jolt > 0:
            self.jolt -= 1
            return ctx.face(self.visitor_at, "startle")
        return self.awake.tick(ctx)


class NightOwl(Block):
    """A sleeper who slept the whole day through wakes at nightfall, shakes off the sleep, and
    heads out to wander."""

    def __init__(self, wake_into: Block):
        self.wake_into = wake_into
        self.stretch = WAKE_TICKS
        self.woke = False

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if not self.woke:
            self.woke = True
            ctx.memory["awake"] = True
            ctx.memory["mode"] = "wander"
            if VISITOR_ID in ctx.greetings.hearers:  # said aloud for a visitor nearby, not broadcast
                ctx.greetings.say(VISITOR_ID, NIGHT_WAKE_LINE)
        if self.stretch > 0:
            self.stretch -= 1
            return ctx.stand("shake_head")
        return self.wake_into.tick(ctx)


def sleep_at_home(player_id: str, home: str) -> Block:
    """Walk to our own bed at home, then sleep there."""

    def bed(ctx: Context) -> Iterable[Cell]:
        return _bed_cell(ctx.obs, ctx.graph, home, player_id) if home != "none" else ()

    return Sequence(GoTo(bed), Repeat(lambda: Pause(100, "sleep")))


def wander_home(observation: ThreeBranchesObservation, home: str, rng: random.Random) -> Block:
    """Potter about in and just outside home: walk to a spot, linger there a while, repeat."""

    area = sorted(_home_area(observation, Navigator.for_village(observation), home)) if home != "none" else []

    def next_spot() -> Block:
        if not area:
            return Pause(30)
        spot = rng.choice(area)
        return Sequence(
            GoTo(lambda ctx, spot=spot: (spot,)),
            Pause(EMOTE_TICKS, rng.choice(HOME_EMOTES)),
            Pause(rng.randint(*HOME_LINGER_TICKS)),
        )

    return Repeat(next_spot)


def evening_stroll(observation: ThreeBranchesObservation, rng: random.Random) -> Block:
    """Stroll between the village's public spots and linger at each, stopping to chat with anyone
    else out wandering."""

    spots = [str(prop["id"]) for prop in props.all(observation) if prop["type"] in STROLL_PROPS]

    def next_spot() -> Block:
        if not spots:
            return Pause(30)
        spot = rng.choice(spots)
        return Sequence(
            GoTo(lambda ctx, spot=spot: ctx.approach_cells(spot)),
            Pause(EMOTE_TICKS, rng.choice(STROLL_EMOTES)),
            Pause(rng.randint(*STROLL_LINGER_TICKS)),
        )

    def chat_now_and_then(ctx: Context) -> Block | None:
        if ctx.memory.get("next_stroll_chat", 0) > ctx.tick:
            return None
        chat = wanderer_social_trigger(ctx)
        if chat is not None:
            ctx.memory["next_stroll_chat"] = ctx.tick + STROLL_CHAT_GAP_TICKS
        return chat

    return Interrupt(chat_now_and_then, Repeat(next_spot))


def wandering(observation: ThreeBranchesObservation, player_id: str, rng: random.Random) -> Block:
    """Wander like a wanderer: rounds of the workers while they work, a stroll once they stop."""

    rounds: Activity = ("rounds", "wander", lambda ctx: wanderer_routine(observation, player_id))
    stroll: Activity = ("stroll", "wander", lambda ctx: evening_stroll(observation, rng))
    return ByPhase({"dawn": stroll, "day": rounds, "morning": rounds, "midday": rounds, "evening": stroll, "night": stroll})


def wanderer_routine(observation: ThreeBranchesObservation, player_id: str) -> Block:
    """Visit each worker in turn, with chats with other wanderers and with workers on a break
    taking priority."""

    roster_ids = {str(entry["id"]) for entry in people.roster(observation)}
    workers = [worker for worker in WORKER_IDS if worker in roster_ids]
    # Each wanderer starts its rounds with a different worker; a woken sleeper takes its own slot.
    offset = WANDERER_IDS.index(player_id) * 2 if player_id in WANDERER_IDS else 1
    workers = workers[offset:] + workers[:offset]
    cursor = {"next": 0, "site": {}}

    def next_visit() -> Block:
        if not workers:
            return Pause(30, "none")
        worker = workers[cursor["next"] % len(workers)]
        cursor["next"] += 1
        worker_sites = work_sites(observation, worker)
        if not worker_sites:
            return Pause(1)
        start = cursor["site"].get(worker, 0)
        cursor["site"][worker] = start + 1
        ordered = worker_sites[start % len(worker_sites) :] + worker_sites[: start % len(worker_sites)]
        return VisitWorker(worker, ordered)

    return Interrupt(wanderer_social_trigger, Repeat(next_visit))


# --- Conversation ------------------------------------------------------------------------------
#
# The visitor can ask any villager questions in chat. Each villager answers in their own voice,
# from what they have actually perceived today (``Knowledge``) plus what anyone living here would
# know: the village layout, who does which job, and the daily routine. Anything a villager cannot
# know is said plainly, or offered as a hedged guess from how the village works.

REPLY_CAP = 200  # the platform drops chat messages longer than this
FRESH_TICKS = 60  # a sighting this recent counts as "a moment ago"
CONVERSE_HOLD_TICKS = 40  # a wandering villager stays put this long after the visitor last spoke
GESTURE_TICKS = 6  # how long a villager nods or waves after answering
SAME_DISTANCE = 0.05  # metres: two villagers this close to equally near count as tied

COMPASS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")

PROP_NOUNS = {
    "lantern": "lantern",
    "stall": "stall",
    "bench": "bench",
    "shrine": "shrine",
    "board": "notice board",
    "plot": "vegetable patch",
    "hearth": "hearth in the inn",
    "repair_bench": "repair bench in the shed",
    "pump": "pump",
    "bell": "bell",
}
PROP_PLURALS = {
    "lantern": "lanterns",
    "stall": "stalls",
    "bench": "benches",
    "shrine": "shrines",
    "plot": "vegetable patches",
}
ACTIVE_STATES = {
    "lantern": "lit",
    "hearth": "lit",
    "stall": "open",
    "bench": "occupied",
    "shrine": "tended",
    "plot": "tended",
    "repair_bench": "busy",
    "pump": "flowing",
    "bell": "ringing",
}
STATE_WORDS = {"unlit": "dark", "occupied": "taken", "empty": "free", "busy": "in use", "flowing": "running", "silent": "quiet"}
STATE_OVERRIDES = {("pump", "idle"): "still", ("repair_bench", "idle"): "free"}
NOT_LANDMARKS = {"lantern", "bench", "hearth", "repair_bench"}  # too many, or tucked indoors
PHASE_WHEN = {
    "dawn": "at dawn",
    "morning": "this morning",
    "midday": "around midday",
    "evening": "this evening",
    "night": "earlier tonight",
    "day": "earlier today",
}
ORDINALS = ("first", "second", "third", "fourth", "fifth")
SIDE_NAMED = ("shrine", "plot")  # named by which side of the village they're on


def _words(pattern: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + pattern + r")\b", re.I)


# Things a question can be about. The earliest mention in the sentence wins.
PROP_WORDS = (
    ("repair_bench", _words(r"repair ?bench(?:es)?|work ?bench(?:es)?")),
    ("board", _words(r"notice ?boards?|boards?")),
    ("lantern", _words(r"lanterns?|lamps?|torch(?:es)?|street ?lights?|lights?")),
    ("stall", _words(r"stalls?|market|shops?")),
    ("shrine", _words(r"shrines?")),
    ("plot", _words(r"plots?|vegetable patch(?:es)?|patch(?:es)?|gardens?")),
    ("pump", _words(r"pump|well")),
    ("bench", _words(r"bench(?:es)?|seats?")),
    ("hearth", _words(r"hearth|fireplace")),
    ("bell", _words(r"bell")),
)
PLACE_WORDS = (
    ("inn", _words(r"inn|tavern|pub")),
    ("shed", _words(r"shed|workshop")),
    ("home", _words(r"home|house|live")),
)
PERSON_WORDS = (
    ("lamplighter", _words(r"lamplighters?")),
    ("sleeper", _words(r"sleepyhead|sleeper")),
    ("blacksmith", _words(r"blacksmiths?|smiths?")),
    ("priest", _words(r"priests?")),
    ("gardener", _words(r"gardeners?")),
    ("wanderer", _words(r"wanderers?")),
)
WHERE = _words(r"where|which way|how do i get|how to get|directions?|find|look(?:ing)? for")
SEEN = _words(r"seen|spotted")
STATE_ASK = _words(
    r"lit|unlit|dark|burning|on|off|out|open|closed|shut|tended|untended|overgrown|weeded|wild|busy|free|"
    r"taken|occupied|empty|in use|flowing|running|ringing|silent|quiet|state|status|condition|how(?:'s| is| are)"
)
BELL_ASK = _words(r"ring(?:ing|s)?|rang|why|what|hear")
PLURAL = _words(r"lanterns|lamps|torches|lights|stalls|shrines|plots|patches|gardens|benches|all|every|any|they|them")
YOUR = _words(r"your|yours")
TOPIC_PATTERNS = {
    "help": _words(r"help|what can i ask|what should i ask|anything to (?:ask|talk)|what can we talk"),
    "warrior": _words(
        r"warriors?|fighters?|legend(?:s|ary)?|swords?|hero(?:es)?|battles?|secrets?|mysterious|suspicious|"
        r"fight(?:ing)?|knights?|soldiers?"
    ),
    "guardian": _words(
        r"guard(?:s|ian|ing)?|keeps? watch|keeping watch|night ?watch|on watch|watchman|lookout|"
        r"protect(?:s|ing|ion)?|safe at night|is it safe|who watches"
    ),
    "elixir": _words(
        r"elixirs?|potions?|brew(?:ed|s)?|drink|drank|magic|(?:good )?luck(?:y)?|evil spirits?|spirits?|"
        r"ward(?:s|ing)? off|side effects?|sleepy|drowsy|tired"
    ),
    "night_lore": _words(
        r"indoors|inside|go(?:es|ing)? (?:home|in|inside|to bed)|head(?:s|ing)? (?:home|in)|before dark|after dark|"
        r"at night|nightfall|night-?time|dusk|curfew|the dark|gets dark"
    ),
    "village": _words(
        r"this (?:place|village|town)|three branches|the village|your village|about (?:the|this|your) "
        r"(?:village|town|place)|live here|lived here|history|who lives"
    ),
    "trade_blacksmith": _words(
        r"forge|forging|smithing|anvil|hammer(?:ing)?|metal|iron|steel|sell|selling|wares|buy|for sale|repairs?|"
        r"mend(?:ing)?|fix(?:ing)?|tools?|nails|horseshoes?|what do you make"
    ),
    "trade_priest": _words(
        r"pray(?:er|ers|ing)?|bless(?:ing|ings|ed)?|gods?|faith|offerings?|holy|worship|"
        r"what(?:'s| is) the shrine|shrines? for|holy water"
    ),
    "trade_gardener": _words(r"grow(?:ing)?|crops?|vegetables|harvest|seeds?|soil|weeds?|weeding|plants?|planting|gardening"),
    "trade_lamplighter": _words(r"lamplighting|who lights|light(?:ing)? the (?:lanterns|lamps)|lantern round"),
    "trade_sleeper": _words(r"sleep(?:ing|s)?|asleep|nap(?:ping)?|snor\w*|awake|wake up|night owl|nocturnal"),
    "trade_wanderer": _words(r"where are you (?:going|headed|off to)|where you off to|stroll(?:ing)?|wander(?:ing)?"),
    "doing_now": _words(r"what are you (?:doing|up to|working on)|what(?:'re| are) you up to|what you doing|whatcha|are you busy"),
    "how_are_you": _words(
        r"how (?:are|r) (?:you|ya)|how(?:'s| is| has) (?:your|the) day|how(?:'s| is) it going|"
        r"how(?:'s| is) (?:life|work|things|everything)|how do you do|how have you been|you (?:ok|okay|alright|well)"
    ),
    "job": _words(
        r"what do you do|what(?:'s| is) your (?:job|trade|work|role|name)|who are you|your name|"
        r"what are you(?! doing| up)|do you work|occupation"
    ),
    "time": _words(r"what time|time is it|what hour|time of day|how late|is it (?:late|early|morning|evening|night|day)"),
    "thanks": _words(r"thanks?|thank you|cheers|appreciate it|glad to be here"),
    "bye": _words(r"bye|goodbye|good ?night|farewell|see you|see ya|take care|so long"),
}
TRADE_ROLES = {
    "trade_blacksmith": "blacksmith",
    "trade_priest": "priest",
    "trade_gardener": "gardener",
    "trade_lamplighter": "lamplighter",
    "trade_sleeper": "sleeper",
    "trade_wanderer": "wanderer",
}
# A name at the start ("Blacksmith, ...", "Hey priest!") or end (", gardener?") addresses someone.
ADDRESS_START = re.compile(
    r"^\W*(?:(?:hello|hi|hey|hiya|oi|excuse me|good (?:morning|afternoon|evening|day))\W+)?(?:the |mr\.? |mrs\.? |miss )?("
    + "|".join((*ROLE_TITLES.values(), LAMPLIGHTER_TITLE))
    + r")\b",
    re.I,
)
ADDRESS_END = re.compile(r",\s*(?:the )?(" + "|".join((*ROLE_TITLES.values(), LAMPLIGHTER_TITLE)) + r")\W*$", re.I)


def match_topics(text: str) -> list[str]:
    """Return up to two topics a visitor's line asks about, in the order they come up."""

    found: dict[str, int] = {}
    spans: list[tuple[int, int]] = []
    prop = _earliest(PROP_WORDS, text)
    place = _earliest(PLACE_WORDS + PROP_WORDS, text)
    person = _earliest(PERSON_WORDS, text)
    where = WHERE.search(text)
    if where and place:
        found["where_place"] = where.start()
        spans.append(place[1].span())
    elif (where or SEEN.search(text)) and person:
        found["where_person"] = (where or SEEN.search(text)).start()
        spans.append(person[1].span())
    elif prop:
        if prop[0] == "bell" and BELL_ASK.search(text):
            found["bell"] = prop[1].start()
            spans.append(prop[1].span())
        elif STATE_ASK.search(text):
            found["prop_state"] = prop[1].start()
            spans.append(prop[1].span())
    if person:
        spans.append(person[1].span())  # "the lamplighter" names a person, not a lamplighting question
    for topic, pattern in TOPIC_PATTERNS.items():
        hit = pattern.search(text)
        if hit and topic not in found and not any(hit.start() < end and start < hit.end() for start, end in spans):
            found[topic] = hit.start()
            spans.append(hit.span())  # "safe at night" is about the guard, not also about night
    if "guardian" in found or "elixir" in found:
        found.pop("night_lore", None)  # "who keeps watch at night?" already gets the lore it needs
    return [topic for topic, _ in sorted(found.items(), key=lambda item: item[1])][:2]


def _earliest(table, text: str):
    """Return ``(key, match)`` for the entry mentioned first in ``text``, or ``None``."""

    best = None
    for key, pattern in table:
        hit = pattern.search(text)
        if hit and (best is None or hit.start() < best[1].start()):
            best = (key, hit)
    return best


def compass(heading: float) -> str:
    return COMPASS[int(((heading + 22.5) % 360.0) // 45.0)]


def state_word(prop_type: str, state: str) -> str:
    return STATE_OVERRIDES.get((prop_type, state)) or STATE_WORDS.get(state, state)


def _how_many(count: int, total: int) -> str:
    if count == 0:
        return "none"
    if count == total and total > 1:
        return "all"
    return str(count)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _fit(text: str, limit: int = REPLY_CAP) -> str:
    """Squash spaces (keeping line breaks) and trim to the chat cap, at a sentence end where possible."""

    text = "\n".join(" ".join(line.split()) for line in text.strip().split("\n"))
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind("\n"))
    return cut[: end + 1].rstrip() if end > 0 else cut[: limit - 3].rstrip() + "..."


def _gesture(ctx: Context) -> str:
    """The nod or wave a villager is giving after their last chat reply, or ``"none"``."""

    gesture = ctx.memory.get("gesture")
    return gesture[0] if gesture is not None and gesture[1] >= ctx.tick else "none"


def voice_of(player_id: str) -> str:
    """Whose voice a villager speaks in: their role, with the lamplighter set apart."""

    return "lamplighter" if player_id == LAMPLIGHTER_ID else ROLES.get(player_id, "wanderer")


# --- What each voice says. Plain tables: rewrite any line freely. ------------------------------
# Lines in a tuple are used in turn, so asking twice gets a different answer.

JOB_LINES = {
    "blacksmith": (
        "I'm one of the village blacksmiths. I keep the market stalls and mend tools at the repair bench in the shed.",
        "Folk just call me the blacksmith. Stalls by day, repairs in between.",
    ),
    "priest": (
        "I'm one of the priests. We brewed the elixir the whole village drank, and I tend my shrine.",
        "Folk just call me the priest. The shrines, and the elixir, are in our care.",
    ),
    "gardener": (
        "I'm a gardener. I keep the vegetable patches weeded and watered.",
        "Folk just call me the gardener. Muddy hands, every day.",
    ),
    "wanderer": (
        "I'm a wanderer! I walk the village and keep folk company while they work.",
        "Folk just call me the wanderer. Somebody has to see how everyone's doing!",
    ),
    "lamplighter": (
        "I wander by day, and at dusk I'm the lamplighter. I light the road so everyone's home before the elixir takes hold.",
        "Folk call me the lamplighter. Come evening, you'll see why.",
    ),
    "sleeper": (
        "Folk call me the sleepyhead. I sleep all day and keep watch over the village at night.",
        "The sleepyhead, at your service. Usually from my bed. Just a simple, sleepy villager!",
    ),
}
HOW_WORK = {
    "blacksmith": ("Busy, busy! The stalls don't keep themselves.", "Hot, honest work. Can't complain."),
    "priest": ("Peaceful, thank you. Tending the shrine settles the mind.", "Blessed, as every day is."),
    "gardener": ("Good! The weeds are losing today.", "Muddy hands, happy heart!"),
    "wanderer": ("Lovely! I've been doing my rounds, seeing how everyone's getting on.", "Can't complain. Plenty to see in Three Branches."),
    "lamplighter": ("Lovely, thanks! Just enjoying the day before my lantern round.", "Good! Saving my legs for the lanterns tonight."),
    "sleeper": ("Wide awake, for once! Someone shook me out of bed.", "Up in daylight? Strange feeling. The sun's awfully bright."),
}
HOW_ON_WATCH = (
    "Wide awake, as ever after dark. Someone has to keep watch!",
    "Fresh as a daisy. The night air suits me.",
)
HOW_BREAK = "Better now I'm off my feet for a bit."
HOW_OFF = "Work's done, so I'm enjoying the evening air before the elixir makes me drowsy."
HOW_STROLL = "Lovely, thanks! Just out for a stroll."
HOW_HOME = "Early yet. Just pottering about home before the day starts."
HOW_TIRED = "Sleepy! The elixir's kicking in. Ready for my bed."
DOING_USE = {
    "stall": "Tending my stall.",
    "repair_bench": "Mending tools at the repair bench.",
    "pump": "Drawing water at the pump.",
    "shrine": "Tending the shrine, so its blessing holds.",
    "plot": "Weeding this patch.",
    "bench": "Sitting a spell on this bench.",
    "lantern": "Lighting this lantern!",
}
DOING_ACTIVITY = {
    "work": "On my way to my next job.",
    "rounds": "Doing my rounds, visiting folk while they work.",
    "stroll": "Just taking a stroll.",
    "home": "Pottering about at home.",
    "sleep": "Heading home to bed before the elixir gets me.",
}
DOING_ON_WATCH = "Keeping watch over the village while everyone sleeps."
TRADE_LINES = {
    "blacksmith": (
        "We sell tools, nails and horseshoes at the market stalls. Anything broken comes to the repair bench in the shed.",
        "I open up my stalls in the market and mend tools in the shed. Need anything fixed?",
    ),
    "priest": (
        "A tended shrine keeps the village blessed, but the blessing fades unless someone tends it again before long.",
        "First I draw water at the pump, then I tend my shrine. Simple, steady work.",
    ),
    "gardener": (
        "Beans, turnips, a few cabbages. Leave a patch alone too long and it goes wild with weeds.",
        "Weeding, watering, more weeding. The patches never stay tidy for long!",
    ),
    "wanderer": (
        "Where am I off to? Wherever someone's working. I like to watch and have a chat.",
        "I wander, that's the whole job! Past the stalls, the shrines, the patches, and round again.",
    ),
    "lamplighter": (
        "At dusk I walk the road and light every lantern, nearest first.",
        "A lantern only takes a touch to light, but there are a good many of them.",
    ),
    "sleeper": (
        "Sleep all day? Blame the elixir. I, er, botched mine a little. Now I just can't seem to sleep at night.",
        "By day I sleep. By night I keep watch over the village. Somebody has to!",
    ),
}
TRADE_DEFER = {
    "blacksmith": "That's blacksmith business. You'll find them at the market stalls or the shed.",
    "priest": "Best ask a priest. They're usually at the pump or a shrine.",
    "gardener": "A gardener would know. Try the vegetable patches.",
    "wanderer": "Ask a wanderer, if you can catch one!",
    "lamplighter": "That'd be the lamplighter. They light the road lanterns every dusk.",
    "sleeper": "That's the sleepyhead. He botched his elixir, so he sleeps all day and keeps watch at night.",
}
# The village lore: the priests brewed an elixir that the whole village drank. It brings good luck
# and wards off evil spirits, but makes everyone very sleepy at night, so everyone heads home at
# dark. The sleepyhead botched his elixir and became nocturnal, so he guards the village at night.
# Nobody knows he is a legendary warrior, though one wanderer (SUSPICIOUS_ID) has their doubts.
NIGHT_LORE = {
    "blacksmith": (
        "It's the priests' elixir. Keeps our luck up and the evil spirits off, but after dark I can't keep my eyes open.",
    ),
    "priest": (
        "We priests brewed an elixir for the village. It brings good luck and wards off evil spirits, but by night it makes everyone very sleepy.",
        "A small price for the elixir's protection. Sleepy nights, safe village.",
    ),
    "gardener": (
        "The elixir! We all drank it. Lucky harvests, no evil spirits, but by nightfall it knocks us out like a sack of turnips!",
    ),
    "wanderer": (
        "We all drank the priests' elixir. Good luck, no evil spirits, and very sleepy nights. So it's home before dark!",
    ),
    "lamplighter": (
        "The priests' elixir makes us all drowsy at night. That's why I light the lanterns at dusk, to see everyone home before it takes hold.",
    ),
    "sleeper": (
        "Everyone else drank their elixir properly. I... didn't. So while they sleep, I keep watch.",
    ),
}
ELIXIR_LINES = {
    "blacksmith": (
        "The priests' elixir? Good luck and no evil spirits. I'll drink to that! Mind you, I'm out like a snuffed forge come nightfall.",
    ),
    "priest": (
        "We priests brewed it and the whole village drank it. It brings good luck and wards off evil spirits. Its one side effect: very sleepy nights.",
        "The recipe stays with us priests. Well... the sleepyhead's batch went wrong, but that was his own doing.",
    ),
    "gardener": (
        "Ever since we drank the priests' elixir, the harvests have been lucky and no spirit's come near. Sleepy nights are a small price!",
    ),
    "wanderer": (
        "The whole village drank the priests' elixir. Lucky days, no evil spirits, and very sleepy nights.",
    ),
    "lamplighter": (
        "The priests' elixir keeps the evil spirits away, but it makes us all drowsy after dark. That's why I light the lanterns early.",
    ),
    "sleeper": (
        "Ah, the elixir... I may have spilled some of mine. And stirred in the wrong thing. Now I'm wide awake all night instead.",
    ),
}
GUARDIAN_LINES = {
    "sleeper": (
        "At night I keep watch while everyone sleeps. I'm awake anyway, thanks to my elixir!",
        "Nothing gets past me at night. Not that anything ever tries. Funny, that.",
    ),
    "priest": (
        "The elixir wards off the evil spirits, and the sleepyhead keeps watch at night in case anything slips by.",
    ),
    "default": (
        "The sleepyhead keeps watch at night. His elixir went wrong, so he's awake while the rest of us sleep.",
    ),
}
WARRIOR_LINES = {
    "sleeper": (
        "Me? A warrior? Ha! I can barely stay awake past sunrise.",
        "Legendary? The only legend about me is how long I can sleep. Why... why do you ask?",
    ),
    "blacksmith": ("A warrior? In Three Branches? The closest thing we've got is my hammer.",),
    "default": ("A legendary warrior? Here? Only in the old stories, traveler.",),
}
# The one wanderer who has doubts about the sleepyhead. Everyone else suspects nothing.
SUSPICIOUS_LINES = {
    "warrior": (
        "Between you and me? He walks his night rounds like a soldier, and I've seen steel glint at his belt. Ask the blacksmiths where that came from.",
        "Don't tell him I said so, but he carries a blade at night. A blacksmith would know more.",
    ),
    "guardian": (
        "The sleepyhead keeps watch at night. Strange how no spirit ever gets past him. Almost like he's done this before...",
    ),
    "elixir": (
        "Lucky, isn't it, that the one batch that went wrong was his? Leaves him awake to guard us. Bit too lucky, if you ask me.",
    ),
    "trade_sleeper": (
        "The sleepyhead says he botched his elixir. Clumsy, he says. Yet I've never once seen him trip...",
    ),
}
# LORE (placeholder): what the bell means to the village.
BELL_LORE = "It calls the village together."
TIME_LINES = {
    "dawn": "It's dawn. The sun's barely up.",
    "morning": "Morning. Work's in full swing.",
    "midday": "Around midday.",
    "evening": "Evening already. The lanterns get lit about now, before the elixir makes us drowsy.",
    "night": "Night-time. The elixir has most folk fast asleep.",
    "day": "Daytime, and a fine one.",
}
THANKS_LINES = {
    "blacksmith": "Any time, traveler.",
    "priest": "Bless you, traveler.",
    "gardener": "My pleasure!",
    "wanderer": "Glad to have you here!",
    "lamplighter": "You're welcome. Mind the lanterns after dark!",
    "sleeper": "Mm, no trouble at all.",
}
BYE_LINES = {
    "blacksmith": "Safe travels! Come back if anything needs mending.",
    "priest": "Go well, and may the shrines watch over you.",
    "gardener": "Bye now! Mind the weeds.",
    "wanderer": "See you around, I'm sure!",
    "lamplighter": "Farewell! The road's well lit at night.",
    "sleeper": "Bye... time for a nap.",
}
SUGGESTIONS = {
    "blacksmith": ("what I sell", "whether my stall's open"),
    "priest": ("about the elixir", "what the shrine is for"),
    "gardener": ("what I'm growing", "whether a patch needs weeding"),
    "wanderer": ("where someone is", "about the elixir"),
    "lamplighter": ("whether a lantern is lit", "why we all head home at dark"),
    "sleeper": ("why I sleep all day", "who keeps watch at night"),
}
ASIDES = {"blacksmith": "Aye.", "priest": "", "gardener": "Ooh!", "wanderer": "Let me think.", "lamplighter": "", "sleeper": "Mm."}
HELP_LINE = "Ask me what I do, where something is, whether a lantern's lit, about the elixir, or why we all head home at dark."

# --- Wave-to-talk sessions. The visitor waves at a villager, who greets them and offers numbered
# options; the visitor replies with a number. Each option is answered once per session, and the
# last option always ends the conversation. ------------------------------------------------------

SESSION_AWAY_TICKS = 40  # a villager waits this long (10 s) for a visitor who walked off mid-conversation
QUEUE_LIMIT = 6  # lines waiting to go out to the visitor, one per tick
END_LABEL = "End conversation"
MENU_GAP = "\n\n"  # the blank line between what a villager says and the options under it
OPTION_NUMBER = re.compile(r"^\W*(?:option\s*)?#?(\d{1,2})\W*$", re.I)
WAVE_HINT = " Wave if you'd like a chat!"
NOTICE_RANGE = 3.5  # metres: a visitor walking up this close makes a villager stop and look...
NOTICE_TICKS = 16  # ...for this long (4 s), so the villager sees them wave
NOTICE_REARM_TICKS = 40  # a visitor who stays close gets another look after this long
EMOTE_IDS = {name: index + 2 for index, name in enumerate(action.EMOTES)}
NIGHT_WATCH_BYE = "Off you go, then. Get indoors. I'll keep watch."
# --- The sleepyhead's secret: a trail of clues for the visitor to follow. -----------------------
# A priest has overheard the suspicious wanderer; the wanderer has seen a blade and points to the
# blacksmiths; one blacksmith made him a sword. Villagers can't share memory, so the sleepyhead
# can't know what the visitor has heard: the visitor has to confront him with it. Mentioning the
# sword (or the blacksmith) unlocks his story. He remembers that for the rest of the day.
STORY_PRIEST_ID = PRIEST_IDS[0]
STORY_BLACKSMITH_ID = BLACKSMITH_IDS[0]
PRIEST_CLUE = (
    "The sleepyhead? Funny you ask. I overheard one of the wanderers whispering about him. "
    "The nosy one, always watching folk. Find them, they'll tell you more."
)
BLACKSMITH_CLUE = (
    "Aye, I gave him a sword for his night watch. Thought he'd cut his own foot off! "
    "But he holds it like he's done it a hundred times. Odd, for a sleepyhead."
)
BLACKSMITH_DEFER = "Not well. But my fellow blacksmith made him something for his night watch. Ask them about it."
RUMOURS_ASK = "R-rumours? About me? What... what have you heard?"
EVIDENCE = _words(r"swords?|blades?|blacksmiths?|smiths?")
DENIALS = (
    (_words(r"walks?|walking|soldiers?|wanderers?|steel|belt"), "My walk? I've a bad knee, that's all. Nothing to it."),
    (_words(r"priests?"), "The priests gossip more than they pray. Pay them no mind."),
)
DENIAL_DEFAULT = "Pfft. Idle gossip. I'm just a sleepy old villager."
CONFESSION = "The sword... the blacksmith told you, didn't they? Blast. Fine, you've found me out. Keep your voice down."
PAST_LABEL = "Tell me about your past."
WARRIOR_PAST = (
    "Long ago I was a knight of the old guard. They called me the Night Blade. I fought for years, until I'd had my fill of battle.",
    "I came here for a quiet life. Botching the elixir was a real accident! But an old soldier makes a fine night guard. Not a word to anyone, mind.",
)
# (label the visitor picks, topic it's answered as). The label is also the question that is answered.
DIALOGUE_OPTIONS = {
    "blacksmith": (
        ("What do you sell?", "trade_blacksmith"),
        ("Is your stall open?", "prop_state"),
        ("Why does everyone go in at dark?", "night_lore"),
        ("Know the sleepyhead?", "clue_blacksmith"),
    ),
    "priest_story": (
        ("What does the elixir do?", "elixir"),
        ("Is the shrine tended?", "prop_state"),
        ("Who keeps watch at night?", "guardian"),
        ("Any news of the sleepyhead?", "clue_priest"),
    ),
    "priest": (
        ("What does the elixir do?", "elixir"),
        ("Is the shrine tended?", "prop_state"),
        ("Who keeps watch at night?", "guardian"),
        ("What are you up to?", "doing_now"),
    ),
    "gardener": (
        ("What are you growing?", "trade_gardener"),
        ("Is your patch tended?", "prop_state"),
        ("Why does everyone go in at dark?", "night_lore"),
        ("Seen the blacksmith?", "where_person"),
    ),
    "wanderer": (
        ("Is the lantern lit?", "prop_state"),
        ("Where's the inn?", "where_place"),
        ("Who keeps watch at night?", "guardian"),
        ("What is this place?", "village"),
    ),
    "suspicious": (
        ("Is the lantern lit?", "prop_state"),
        ("What do you make of the sleepyhead?", "warrior"),
        ("Who keeps watch at night?", "guardian"),
        ("What is this place?", "village"),
    ),
    "lamplighter": (
        ("Are the lanterns lit?", "prop_state"),
        ("Why light them at dusk?", "night_lore"),
        ("What do you do by day?", "job"),
        ("Who keeps watch at night?", "guardian"),
    ),
    "sleeper": (
        ("Why do you sleep all day?", "trade_sleeper"),
        ("Who keeps watch at night?", "guardian"),
        ("I hear rumours about you.", "rumours"),
        ("What's the elixir?", "elixir"),
    ),
    "sleeper_unlocked": (  # once the visitor has confronted him with the sword
        ("Why do you sleep all day?", "trade_sleeper"),
        ("Who keeps watch at night?", "guardian"),
        (PAST_LABEL, "warrior_past"),
        ("What's the elixir?", "elixir"),
    ),
}
# How each villager opens a conversation, by what they're doing: see ``Dialogue._situation``.
OPENINGS = {
    "blacksmith": {
        "work": "Hello, traveler! Busy at the stalls, but I've got a minute.",
        "off": "Evening, traveler! Work's done, so I've got time to talk.",
    },
    "priest": {
        "work": "Blessings, traveler. What brings you to me?",
        "off": "Good evening, traveler. Walk a while with me?",
    },
    "gardener": {
        "work": "Oh, hello! Mind the mud. What can I do for you?",
        "off": "Evening! My hands are clean for once. What's on your mind?",
    },
    "wanderer": {
        "work": "Oh, a new face! Fancy a chat, traveler?",
        "stroll": "Hello again, traveler! Out for a stroll too?",
    },
    "lamplighter": {
        "lamps": "Evening! Walk with me, I've lanterns to light.",
        "work": "Hello, traveler! Come evening, I'm the lamplighter.",
        "stroll": "Hello, traveler! The road's lit, so I've got time to chat.",
    },
    "sleeper": {
        "night": "Halt! Who goes... oh, a traveler. Out late, aren't you?",
        "work": "Ugh, daylight. You woke me, traveler. What is it?",
    },
}
SHARED_OPENINGS = {
    "break": "Taking a breather, but sit, sit! What's on your mind?",
    "home": "Morning, traveler! An early start for you too?",
    "tired": "Mm, you caught me heading to bed. Quickly, then!",
}


@dataclass
class Session:
    """One conversation with the visitor: the options on offer and which have been answered."""

    options: list[tuple[int, str, str]]  # (number, label, topic)
    end: int  # the number that ends the conversation
    last_heard: int  # tick the visitor last spoke (or waved) to us
    last_near: int  # tick the visitor was last within earshot
    used: set[int] = field(default_factory=set)
    awaiting_rumour: bool = False  # the sleepyhead has asked what the visitor heard about him


class Knowledge:
    """What one villager has perceived today: each prop's last seen state, and where each person
    was last seen or heard. Answers come from here, never from anything the villager can't know."""

    def __init__(self):
        self.props: dict[str, tuple[str, int, str]] = {}
        self.people: dict[str, tuple[dict[str, float], int, str]] = {}

    def update(self, observation: ThreeBranchesObservation, tick: int) -> None:
        phase = day.phase(observation)
        for record in props.seen(observation):
            self.props[str(record["prop"])] = (str(record["state"]), tick, phase)
        for person in (*people.seen(observation), *people.nearby(observation)):
            where = {"x": float(person["position"]["x"]), "y": float(person["position"]["y"])}
            self.people[str(person["id"])] = (where, tick, phase)


class Dialogue:
    """One villager's side of conversations with the visitor."""

    def __init__(self, observation: ThreeBranchesObservation, player_id: str):
        self.me_id = player_id
        self.voice = voice_of(player_id)
        self.suspicious = player_id == SUSPICIOUS_ID
        self.knowledge = Knowledge()
        self.asked: dict[str, int] = {}
        size = layout.frame(observation)
        cell = float(size["cell_size"])
        self.centre = {"x": int(size["cells_x"]) * cell / 2, "y": int(size["cells_y"]) * cell / 2}
        self.counts: dict[str, int] = {}
        for prop in props.all(observation):
            self.counts[str(prop["type"])] = self.counts.get(str(prop["type"]), 0) + 1
        self.landmarks = self._landmarks(observation)
        self.names = self._names(observation)
        self.session: Session | None = None
        self.story_unlocked = False  # the sleepyhead has owned up to his past, for the rest of the day
        self.queue: deque[str] = deque()  # lines for the visitor, sent one per tick
        self.visitor_waving = False  # whether the visitor was waving last tick, to catch each new wave

    # --- Sessions --------------------------------------------------------------------------

    def watch(self, ctx: Context) -> None:
        """Each tick: open a session when the visitor waves at us, and close one that has run its
        course (the visitor left, went quiet, or waved at someone else, or we fell asleep)."""

        visitor = next((person for person in people.seen(ctx.obs) if str(person["id"]) == VISITOR_ID), None)
        waving = visitor is not None and visitor["expression"]["type"] == "wave"
        new_wave = waving and not self.visitor_waving
        self.visitor_waving = waving

        session = self.session
        if session is not None:
            # We stay put for as long as the conversation is open (see ReplyToHellos), so only the
            # visitor can end it: by picking End, waving at someone else, or walking off for good.
            if any(str(person["id"]) == VISITOR_ID for person in people.nearby(ctx.obs)):
                session.last_near = ctx.tick
            if self.asleep(ctx):
                self.say(self._mumble([], ""))
                self.end()
            elif ctx.tick - session.last_near > SESSION_AWAY_TICKS:
                self.end()
                self.queue.clear()  # they've gone: anything unsaid would be stale when they're back
            else:
                ctx.memory["talking_until"] = ctx.tick + CONVERSE_HOLD_TICKS

        if new_wave:
            target = self._wave_target(ctx, visitor)
            if target == self.me_id:
                self.open(ctx)
            elif target is not None and self.session is not None:
                self.end()  # they've turned to talk to someone else
        ctx.memory["in_session"] = self.session is not None  # tells ReplyToHellos to stay put

    def _wave_target(self, ctx: Context, visitor) -> str | None:
        """Which villager a visitor's wave is meant for: the one within earshot they face most
        directly. Every villager works this out from what it can perceive."""

        where = visitor["position"]
        heading = float(visitor["heading"])

        def off(point) -> float:
            return abs((heading - geometry.heading_to(where, point) + 180.0) % 360.0 - 180.0)

        candidates = []
        if geometry.distance(ctx.here, where) <= geometry.HEARING_RANGE and off(ctx.here) <= HELLO_WAVE_ANGLE:
            candidates.append((off(ctx.here), self.me_id))
        for person in (*people.seen(ctx.obs), *people.nearby(ctx.obs)):
            other = str(person["id"])
            if other in (self.me_id, VISITOR_ID) or not people.is_npc(other):
                continue
            point = person["position"]
            if geometry.distance(point, where) <= geometry.HEARING_RANGE and off(point) <= HELLO_WAVE_ANGLE:
                candidates.append((off(point), other))
        return min(candidates)[1] if candidates else None

    def open(self, ctx: Context) -> None:
        """Greet the visitor and offer this villager's options. A villager already in a session
        just offers what's left; one who is asleep only mumbles."""

        if self.asleep(ctx):
            self.say(self._mumble([], ""))
            return
        ctx.memory["gesture"] = ("wave", ctx.tick + GESTURE_TICKS)
        ctx.memory["talking_until"] = ctx.tick + CONVERSE_HOLD_TICKS
        if self.session is not None:
            self.session.last_heard = ctx.tick
            self.offer("Still here!")
            return
        options = DIALOGUE_OPTIONS[self._options_key()]
        self.session = Session(
            options=[(number, label, topic) for number, (label, topic) in enumerate(options, start=1)],
            end=len(options) + 1,
            last_heard=ctx.tick,
            last_near=ctx.tick,
        )
        ctx.greetings.last_greeted[VISITOR_ID] = ctx.tick  # no separate hello on top of the opening
        self.offer(self._opening(ctx))

    def _options_key(self) -> str:
        """Which set of options this villager offers: their role's, or a story variant."""

        if self.me_id == STORY_PRIEST_ID:
            return "priest_story"
        if self.suspicious:
            return "suspicious"
        if self.voice == "sleeper" and self.story_unlocked:
            return "sleeper_unlocked"
        return self.voice

    def _hear_rumour(self, text: str, ctx: Context) -> bool:
        """The sleepyhead, confronted: the sword gives him away; anything else he brushes off.
        Returns whether this line was about the rumours at all."""

        session = self.session
        if self.voice != "sleeper" or session is None or self.story_unlocked:
            return False
        if EVIDENCE.search(text):
            self._confess(ctx)
            return True
        if not session.awaiting_rumour:
            return False
        session.awaiting_rumour = False
        denial = next((line for pattern, line in DENIALS if pattern.search(text)), DENIAL_DEFAULT)
        ctx.memory["gesture"] = ("shake_head", ctx.tick + GESTURE_TICKS)
        self.offer(denial)
        return True

    def _confess(self, ctx: Context) -> None:
        """Found out: mark the rumours answered and add his story, renumbering only End."""

        session = self.session
        self.story_unlocked = True
        session.awaiting_rumour = False
        session.used.update(number for number, _, topic in session.options if topic == "rumours")
        session.options.append((session.end, PAST_LABEL, "warrior_past"))
        session.end += 1
        ctx.memory["gesture"] = ("shrug", ctx.tick + GESTURE_TICKS)
        self.offer(CONFESSION)

    def end(self, ctx: Context | None = None) -> None:
        """Close the conversation. When the visitor ends it (``ctx`` given), we're free to go at once."""

        self.session = None
        if ctx is not None:
            ctx.memory["talking_until"] = -1
            ctx.memory["in_session"] = False

    def _choose(self, number: int, ctx: Context) -> None:
        """Answer the option the visitor picked, then offer what's left."""

        session = self.session
        session.last_heard = ctx.tick
        if self.asleep(ctx):
            self.say(self._mumble([], ""))
            self.end()
            return
        if number == session.end:
            on_watch = self.voice == "sleeper" and day.phase(ctx.obs) == "night"
            self.say(NIGHT_WATCH_BYE if on_watch else BYE_LINES[self.voice])
            ctx.memory["gesture"] = ("wave", ctx.tick + GESTURE_TICKS)
            self.end(ctx)
            return
        chosen = next((option for option in session.options if option[0] == number), None)
        if chosen is None:
            self.offer("Pick one of these:")
            return
        if number in session.used:
            self.offer("We've already talked about that.")
            return
        _, label, topic = chosen
        session.awaiting_rumour = False
        if topic == "rumours":
            # Not used up: he waits, nervously, to hear what the visitor knows. Ask again any time.
            session.awaiting_rumour = True
            ctx.memory["gesture"] = ("startle", ctx.tick + GESTURE_TICKS)
            self.say(RUMOURS_ASK)
            return
        session.used.add(number)
        ctx.memory["gesture"] = ("nod", ctx.tick + GESTURE_TICKS)
        answer = self._answer(topic, label, ctx)
        if self._all_used():
            self.offer(answer, "That's all I've got for now!")
        else:
            self.offer(answer, "Anything else?")

    def _all_used(self) -> bool:
        return all(number in self.session.used for number, _, _ in self.session.options)

    def _menu(self) -> str:
        """The options still on offer, one per line, each in quotes: 1) “What do you sell?”"""

        session = self.session
        offered = [f"{number}) “{label}”" for number, label, _ in session.options if number not in session.used]
        return "\n".join((*offered, f"{session.end}) “{END_LABEL}”"))

    def offer(self, comment: str, lead: str = "") -> None:
        """Say ``comment`` (and ``lead``, such as "Anything else?"), with the options still on offer
        a blank line below. When it all won't fit in one message, the options follow in a second
        one, under ``lead`` if there is one."""

        menu = self._menu()
        whole = f"{' '.join(part for part in (comment, lead) if part)}{MENU_GAP}{menu}"
        if len(whole) <= REPLY_CAP:
            self.say(whole)
        else:
            self.say(comment)
            self.say(f"{lead}{MENU_GAP}{menu}" if lead else menu)

    def _opening(self, ctx: Context) -> str:
        if self.voice == "sleeper":
            return OPENINGS["sleeper"]["night" if day.phase(ctx.obs) == "night" else "work"]
        situation = self._situation(ctx)
        lines = OPENINGS[self.voice]
        return lines.get(situation) or SHARED_OPENINGS.get(situation) or lines["work"]

    def say(self, text: str) -> None:
        """Queue a line for the visitor. ``Agent.chat`` sends one per tick, so none overwrite."""

        if len(self.queue) < QUEUE_LIMIT:
            self.queue.append(_fit(text))

    def next_line(self) -> str | None:
        return self.queue.popleft() if self.queue else None

    # --- Who answers -----------------------------------------------------------------------

    def respond(self, message: dict, ctx: Context) -> bool:
        """Handle one visitor line: an option number, or a free-typed question. Returns ``False``
        to leave a plain broadcast hello to the usual hello handling."""

        text = str(message.get("text", ""))
        to = message.get("to")
        choice = OPTION_NUMBER.match(text)
        if choice:
            if self.session is not None and to in (None, self.me_id):
                self._choose(int(choice.group(1)), ctx)
            elif to == self.me_id:
                self.say("Wave at me if you'd like a chat!")
            return True
        topics = match_topics(text)
        hello = bool(HELLO_WORDS.search(text))
        if not self.addressed(message, ctx):
            return bool(topics)
        ctx.memory["talking_until"] = ctx.tick + CONVERSE_HOLD_TICKS
        if self.session is not None:
            self.session.last_heard = ctx.tick
        if not self.asleep(ctx) and self._hear_rumour(text, ctx):
            return True
        if not topics and hello:
            if to == self.me_id and self.session is None and not self.asleep(ctx):
                self.open(ctx)  # a hello straight to us works as well as a wave
                return True
            return False
        if self.asleep(ctx):
            self.say(self._mumble(topics, text))
            return True
        reply = " ".join(self._answer(topic, text, ctx) for topic in topics) if topics else self._fallback()
        since = ctx.tick - ctx.greetings.last_greeted.get(VISITOR_ID, -HELLO_COOLDOWN_TICKS)
        if hello and since >= HELLO_COOLDOWN_TICKS:
            ctx.greetings.last_greeted[VISITOR_ID] = ctx.tick
            reply = f"Hello, {VISITOR_TITLE}! {reply}"
            ctx.memory["gesture"] = ("wave", ctx.tick + GESTURE_TICKS)
        else:
            ctx.memory["gesture"] = ("nod", ctx.tick + GESTURE_TICKS)
        self.say(reply)
        if "bye" in topics:
            self.end(ctx)
        return True

    def addressed(self, message: dict, ctx: Context) -> bool:
        """Whether a visitor line is for us: sent to us, or a broadcast that calls us by role, or
        that comes while we're in a session with them, or an unnamed broadcast when no villager we
        can perceive is nearer the visitor than we are."""

        to = message.get("to")
        if to is not None:
            return to == self.me_id
        text = str(message.get("text", ""))
        named = ADDRESS_START.search(text) or ADDRESS_END.search(text)
        if named:
            return named.group(1).lower() == title(self.me_id)
        if self.session is not None:
            return True
        visitor = people.find(ctx.obs, VISITOR_ID)
        if visitor is None:
            return True
        mine = geometry.distance(ctx.here, visitor["position"])
        for person in (*people.seen(ctx.obs), *people.nearby(ctx.obs)):
            other = str(person["id"])
            if other in (self.me_id, VISITOR_ID) or not people.is_npc(other):
                continue
            theirs = geometry.distance(person["position"], visitor["position"])
            if theirs < mine - SAME_DISTANCE or (abs(theirs - mine) <= SAME_DISTANCE and other < self.me_id):
                return False
        return True

    def asleep(self, ctx: Context) -> bool:
        # Before the routine's first tick sets a mode, assume the role's usual one (as _manner does).
        default = {"wanderer": "wander", "lamplighter": "wander", "sleeper": "sleep"}.get(self.voice, "work")
        if ctx.memory.get("mode", default) != "sleep":
            return False
        if me.expression(ctx.obs)["type"] == "sleep":
            return True
        return self.voice == "sleeper" and not ctx.memory.get("awake")

    # --- Answers ---------------------------------------------------------------------------

    def _answer(self, topic: str, text: str, ctx: Context) -> str:
        if self.suspicious and topic in SUSPICIOUS_LINES:
            return self._pick("suspicious_" + topic, SUSPICIOUS_LINES[topic])
        if topic in TRADE_ROLES:
            return self._trade(TRADE_ROLES[topic], ctx)
        return getattr(self, "_a_" + topic)(text, ctx)

    def _pick(self, key: str, lines: tuple[str, ...]) -> str:
        count = self.asked.get(key, 0)
        self.asked[key] = count + 1
        return lines[count % len(lines)]

    def _aside(self, line: str) -> str:
        aside = ASIDES.get(self.voice, "")
        return f"{aside} {line}" if aside else line

    def _a_help(self, text: str, ctx: Context) -> str:
        return HELP_LINE

    def _a_thanks(self, text: str, ctx: Context) -> str:
        return THANKS_LINES[self.voice]

    def _a_bye(self, text: str, ctx: Context) -> str:
        return BYE_LINES[self.voice]

    def _a_job(self, text: str, ctx: Context) -> str:
        return self._pick("job", JOB_LINES[self.voice])

    def _a_night_lore(self, text: str, ctx: Context) -> str:
        return self._pick("night_lore", NIGHT_LORE[self.voice])

    def _a_elixir(self, text: str, ctx: Context) -> str:
        return self._pick("elixir", ELIXIR_LINES[self.voice])

    def _a_guardian(self, text: str, ctx: Context) -> str:
        return self._pick("guardian", GUARDIAN_LINES.get(self.voice, GUARDIAN_LINES["default"]))

    def _a_clue_priest(self, text: str, ctx: Context) -> str:
        return PRIEST_CLUE

    def _a_clue_blacksmith(self, text: str, ctx: Context) -> str:
        return BLACKSMITH_CLUE if self.me_id == STORY_BLACKSMITH_ID else BLACKSMITH_DEFER

    def _a_warrior_past(self, text: str, ctx: Context) -> str:
        """His story, in two parts: the first said now, the second under the options that follow."""

        self.say(WARRIOR_PAST[0])
        return WARRIOR_PAST[1]

    def _a_warrior(self, text: str, ctx: Context) -> str:
        if self.voice == "sleeper" and self.story_unlocked:
            return WARRIOR_PAST[1]  # no point denying it now
        return self._pick("warrior", WARRIOR_LINES.get(self.voice, WARRIOR_LINES["default"]))

    def _a_time(self, text: str, ctx: Context) -> str:
        return TIME_LINES.get(day.phase(ctx.obs), TIME_LINES["day"])

    def _a_village(self, text: str, ctx: Context) -> str:
        homes = sum(1 for building in layout.buildings(ctx.obs) if building["type"] == "home")
        shrines, patches = self.counts.get("shrine", 0), self.counts.get("plot", 0)
        facts = (
            f"This is Three Branches: {homes} houses, an inn, a shed, a market, {shrines} shrines and "
            f"{patches} vegetable patches."
        )
        lore = "We're small, but lucky. The priests' elixir keeps the evil spirits away, and the sleepyhead watches over us at night."
        return self._pick("village", (facts, lore))

    def _a_how_are_you(self, text: str, ctx: Context) -> str:
        if self.voice == "sleeper" and day.phase(ctx.obs) == "night":
            return self._pick("how_on_watch", HOW_ON_WATCH)
        situation = self._situation(ctx)
        if situation == "break":
            return HOW_BREAK
        if situation == "lamps":
            left = len(self._dark_lanterns(ctx))
            return f"Busy! {left} lanterns still to light." if left else "Content. Every lantern's lit."
        if situation == "home":
            return HOW_HOME
        if situation == "tired":
            return HOW_TIRED
        if situation == "off":
            return HOW_OFF
        if situation == "stroll":
            return HOW_STROLL
        return self._pick("how", HOW_WORK[self.voice])

    def _a_doing_now(self, text: str, ctx: Context) -> str:
        holding = me.expression(ctx.obs)
        if holding["type"] == "use":
            prop = props.find(ctx.obs, holding["target"])
            if prop is not None and str(prop["type"]) in DOING_USE:
                return DOING_USE[str(prop["type"])]
        if self.voice == "sleeper" and day.phase(ctx.obs) == "night":
            return DOING_ON_WATCH
        situation = self._situation(ctx)
        if situation == "break":
            return "Taking a breather. I'm off to find a bench."
        if situation == "lamps":
            left = len(self._dark_lanterns(ctx))
            return f"Lighting the lanterns. {left} to go!" if left else "All the lanterns are lit, so I'm done for tonight."
        activity = ctx.memory.get("activity", "")
        return DOING_ACTIVITY.get(activity, "Not much. Just enjoying the day.")

    def _situation(self, ctx: Context) -> str:
        """Sum up what we're doing as one of: break, lamps, home, tired, off, or work."""

        if ctx.memory.get("on_break"):
            return "break"
        activity = ctx.memory.get("activity", "")
        if activity == "lamps":
            return "lamps"
        if activity == "home":
            return "home"
        if activity == "sleep" or (day.phase(ctx.obs) == "night" and self.voice != "sleeper"):
            return "tired"
        if activity == "stroll":
            return "off" if self.voice in ("blacksmith", "priest", "gardener") else "stroll"
        return "work"

    def _trade(self, role: str, ctx: Context) -> str:
        own = role == self.voice or (role == "wanderer" and self.voice == "lamplighter")
        if not own:
            return TRADE_DEFER[role]
        line = self._pick("trade_" + role, TRADE_LINES[self.voice if role != "wanderer" else "wanderer"])
        if role == "lamplighter":
            lit = len(ctx.memory.get("lit_by_me", {}))
            if lit:
                line += f" I've lit {lit} tonight."
            elif day.phase(ctx.obs) not in ("evening", "night"):
                line += " Come evening, you'll see me at it."
        return line

    def _a_bell(self, text: str, ctx: Context) -> str:
        if day.bell_ringing(ctx.obs):
            return f"Hear that? Someone's ringing the bell! {BELL_LORE}"
        bell = next((prop for prop in props.all(ctx.obs) if prop["type"] == "bell"), None)
        if bell is None:
            return "We've no bell in this village."
        where = self.directions(ctx, _prop_center(ctx.obs, bell))
        return f"It's quiet just now. The bell is {where}. {BELL_LORE}"

    def _a_prop_state(self, text: str, ctx: Context) -> str:
        found = _earliest(PROP_WORDS, text)
        prop_type = found[0]
        if prop_type == "bell":
            return self._a_bell(text, ctx)
        if prop_type == "board":
            return "It's just the notice board. Nothing to light or open, but have a read!"
        of_type = [prop for prop in props.all(ctx.obs) if prop["type"] == prop_type]
        if not of_type:
            return f"We've no {PROP_PLURALS.get(prop_type, PROP_NOUNS[prop_type] + 's')} in this village."
        if len(of_type) > 1 and PLURAL.search(text) and not YOUR.search(text):
            return self._summary(prop_type, of_type, ctx)
        report = self._report(self._referent(of_type, text, ctx), ctx)
        own = set(work_sites(ctx.obs, self.me_id))
        if YOUR.search(text) and not any(prop["id"] in own for prop in of_type):
            return f"I don't keep a {PROP_NOUNS[prop_type]} myself. {report}"
        return self._aside(report)

    def _referent(self, of_type: list, text: str, ctx: Context):
        """Pick which prop the visitor means: one of ours if they say "your", otherwise the one
        nearest to them (or to us, if we can't tell where they are)."""

        if YOUR.search(text):
            own = set(work_sites(ctx.obs, self.me_id))
            ours = [prop for prop in of_type if prop["id"] in own]
            of_type = ours or of_type
        visitor = people.find(ctx.obs, VISITOR_ID)
        anchor = visitor["position"] if visitor is not None else ctx.here
        return min(of_type, key=lambda prop: geometry.distance(anchor, _prop_center(ctx.obs, prop)))

    def _report(self, prop, ctx: Context) -> str:
        """Say what we know about one prop's state, and how we know it."""

        prop_id, prop_type = str(prop["id"]), str(prop["type"])
        name = self.describe(ctx.obs, prop)
        now = props.state(ctx.obs, prop_id)
        lit_by_me = ctx.memory.get("lit_by_me", {})
        if now is not None:
            line = f"{_cap(name)} is {state_word(prop_type, now)}. I can see it from here."
            if now == "lit" and prop_id in lit_by_me:
                line += " Lit it myself!"
            elif prop_type == "lantern" and now != "lit" and ctx.memory.get("activity") == "lamps":
                line += " I'll get to it soon."
            return line
        if prop_id in lit_by_me:
            tick, phase = lit_by_me[prop_id]
            return f"I lit {name} myself {self.when(ctx, tick, phase)}, so it should still be burning."
        seen = self.knowledge.props.get(prop_id)
        if seen is not None:
            state, tick, phase = seen
            line = f"{_cap(name)} was {state_word(prop_type, state)} when I saw it {self.when(ctx, tick, phase)}."
            hedge = self._hedge(prop_type, ctx) if phase != day.phase(ctx.obs) else ""
            return f"{line} Can't see it from here, mind. {hedge}".strip()
        return f"I can't see {name} from here, and I haven't passed it today. {self._hedge(prop_type, ctx)}".strip()

    def _summary(self, prop_type: str, of_type: list, ctx: Context) -> str:
        """Sum up what we know about every prop of one kind."""

        active, plural, total = ACTIVE_STATES[prop_type], PROP_PLURALS[prop_type], len(of_type)
        word = state_word(prop_type, active)
        visible = [props.state(ctx.obs, str(prop["id"])) for prop in of_type]
        visible = [state for state in visible if state is not None]
        known = [self.knowledge.props[str(prop["id"])] for prop in of_type if str(prop["id"]) in self.knowledge.props]
        parts = []
        if visible:
            showing = sum(state == active for state in visible)
            count = _how_many(showing, len(visible))
            if len(visible) == 1:
                parts.append(f"I can see one from here, and it's {state_word(prop_type, visible[0])}.")
            else:
                parts.append(f"I can see {len(visible)} from here, and {count} of them {'is' if count == '1' else 'are'} {word}.")
        lit = len(ctx.memory.get("lit_by_me", {})) if prop_type == "lantern" else 0
        since = "today"
        if prop_type == "lantern" and day.phase(ctx.obs) in ("evening", "night"):
            # Lanterns seen in daylight say nothing about now, so count only those seen since dusk.
            known = [entry for entry in known if entry[2] in ("evening", "night")]
            since = "since dusk"
        if lit:
            parts.append(f"I've lit all {total} myself tonight." if lit == total else f"I've lit {lit} of the {total} myself tonight.")
        elif len(known) > len(visible):
            showing = sum(state == active for state, _, _ in known)
            count = _how_many(showing, len(known))
            verb = "was" if count == "1" else "were"
            parts.append(f"Of the {len(known)} I've seen {since}, {count} {verb} {word} last I looked.")
        if not parts:
            parts.append(f"I haven't seen any of the {plural} {since}, so I couldn't say.")
        if len(known) < total and not lit:
            parts.append(self._hedge(prop_type, ctx, plural=True))
        return " ".join(part for part in parts if part)

    def _hedge(self, prop_type: str, ctx: Context, plural: bool = False) -> str:
        """A guess from how the village works, for when we can't see for ourselves."""

        phase = day.phase(ctx.obs)
        it = "they're" if plural else "it's"
        if prop_type == "lantern":
            if phase in ("evening", "night"):
                if self.voice == "lamplighter":
                    return "I'll get to it on my round." if ctx.memory.get("activity") == "lamps" else ""
                return f"The lamplighter goes round at dusk, so {it} likely lit by now."
            if phase == "day":
                return ""
            return "They stay dark until the lamplighter's round at dusk."
        return {
            "stall": "The blacksmiths open and close them as they work.",
            "shrine": "The priests tend them through the day, but a blessing fades if it's left too long.",
            "plot": "The gardeners weed them through the day. Left alone, they go wild.",
            "pump": "It only runs while someone's working it." if not plural else "",
            "bench": "Folk come and go from the benches all day.",
            "repair_bench": "The blacksmiths take turns at it.",
        }.get(prop_type, "")

    def _dark_lanterns(self, ctx: Context) -> list[str]:
        lit_by_me = ctx.memory.get("lit_by_me", {})
        return [
            str(prop["id"])
            for prop in props.all(ctx.obs)
            if prop["type"] == "lantern"
            and prop["id"] not in lit_by_me
            and self.knowledge.props.get(str(prop["id"]), ("unlit",))[0] != "lit"
        ]

    def _a_where_place(self, text: str, ctx: Context) -> str:
        kind, _ = _earliest(PLACE_WORDS + PROP_WORDS, text)
        if kind == "home":
            home = me.home(ctx.obs)
            point = layout.doorway(ctx.obs, home) or layout.building_center(ctx.obs, home)
            if point is None:
                return "I don't have a home here."
            return f"My house is {self.directions(ctx, point)}."
        if kind in ("inn", "shed"):
            building = next((b for b in layout.buildings(ctx.obs) if b["type"] == kind), None)
            if building is None:
                return f"We've no {kind} in this village."
            point = layout.doorway(ctx.obs, str(building["id"])) or layout.building_center(ctx.obs, str(building["id"]))
            return f"The {kind} is {self.directions(ctx, point)}."
        of_type = [prop for prop in props.all(ctx.obs) if prop["type"] == kind]
        if not of_type:
            return f"We've no {PROP_NOUNS[kind]} in this village."
        nearest = min(of_type, key=lambda prop: geometry.distance(ctx.here, _prop_center(ctx.obs, prop)))
        name = "the market" if kind == "stall" else self.describe(ctx.obs, nearest)
        lead = "The nearest one, " if len(of_type) > 1 and kind != "stall" else ""
        where = self.directions(ctx, _prop_center(ctx.obs, nearest))
        return f"{lead}{name}, is {where}." if lead else f"{_cap(name)} is {where}."

    def _a_where_person(self, text: str, ctx: Context) -> str:
        role, _ = _earliest(PERSON_WORDS, text)
        if role == "lamplighter":
            ids = [LAMPLIGHTER_ID]
        elif role == "wanderer":
            ids = [pid for pid in WANDERER_IDS if pid != LAMPLIGHTER_ID]
        else:
            ids = [pid for pid, held in ROLES.items() if held == role]
        roster = {str(entry["id"]) for entry in people.roster(ctx.obs)}
        others = [pid for pid in ids if pid in roster and pid != self.me_id]
        label = LAMPLIGHTER_TITLE if role == "lamplighter" else ROLE_TITLES[role]
        if not others:
            return "That's me!" if self.me_id in ids else f"There's no {label} in this village."
        if self.me_id in ids:
            label = f"other {label}"
        sightings = [(self.knowledge.people[pid], pid) for pid in others if pid in self.knowledge.people]
        if sightings:
            (where, tick, phase), _ = max(sightings, key=lambda item: item[0][1])
            if tick == ctx.tick:
                gap = geometry.distance(ctx.here, where)
                heading = compass(geometry.heading_to(ctx.here, where))
                span = "a metre" if gap < 1.5 else f"{round(gap)} metres"
                who = f"The {label}'s" if self.me_id in ids else "There's one"
                return self._aside(f"{who} right over there, about {span} {heading} of me.")
            return self._aside(f"I saw the {label} {self.near(where)} {self.when(ctx, tick, phase)}.")
        return f"I haven't seen the {label} today. {self._guess(role, ctx)}"

    def _guess(self, role: str, ctx: Context) -> str:
        """Where someone usually is at this hour, from the village routine."""

        phase = day.phase(ctx.obs)
        if role == "sleeper":
            if phase == "night":
                return "On watch, now it's night. He guards the village while the rest of us sleep."
            return "Asleep at home, I'd think. He sleeps the day away."
        if phase == "night":
            return "Asleep at home by now, I'd think."
        if phase == "dawn":
            return "Pottering about at home, most likely."
        if role == "lamplighter" and phase == "evening":
            return "Out lighting the lanterns along the road, I'd think."
        if role in ("wanderer", "lamplighter"):
            return "Could be anywhere. Wanderers wander! Usually wherever someone's working."
        if phase == "evening":
            return "Work's done for the day, so they'll be out strolling somewhere."
        return {
            "blacksmith": "At this hour they're usually at the market stalls or the repair bench in the shed.",
            "priest": "Usually at the pump or tending one of the shrines this time of day.",
            "gardener": "Out at the vegetable patches, I'd guess.",
        }[role]

    def _fallback(self) -> str:
        first, second = SUGGESTIONS[self.voice]
        return self._pick("fallback", (
            f"Hmm, not sure I follow. Try asking me {first}, or {second}.",
            f"Sorry, you've lost me. You could ask {first}?",
        ))

    def _mumble(self, topics: list[str], text: str) -> str:
        if self.voice != "sleeper":
            return "Mm... the elixir... so sleepy... ask me at dawn... zzz."
        topic = topics[0] if topics else ""
        if topic == "warrior":
            return "Zzz... mm... my sword... the old battle... zzz... wha? Nothing. I said nothing. Zzz."
        found = _earliest(PROP_WORDS, text)
        word = {
            "prop_state": PROP_PLURALS.get(found[0], PROP_NOUNS[found[0]]) if found else "",
            "where_place": "over there",
            "where_person": "they went that way",
            "night_lore": "night",
            "how_are_you": "fine",
        }.get(topic, "")
        return f"Zzz... {word}... mm, five more minutes... zzz." if word else "Zzz... mm, five more minutes... zzz."

    # --- Naming places ---------------------------------------------------------------------

    def _landmarks(self, observation: ThreeBranchesObservation) -> list[tuple[str, str, dict[str, float]]]:
        """Return ``(kind, name, position, prop_id)`` for the places villagers give directions by."""

        marks = []
        home = me.home(observation)
        roster = people.roster(observation)
        for building in layout.buildings(observation):
            building_id, kind = str(building["id"]), str(building["type"])
            centre = layout.building_center(observation, building_id)
            if centre is None:
                continue
            if kind != "home":
                name = f"the {kind}"
            elif building_id == home:
                name = "my house"
            else:
                residents = sorted({title(str(entry["id"])) for entry in roster if entry["home"] == building_id})
                name = f"the {' and '.join(residents)}'s house" if residents else "one of the houses"
            marks.append(("building", name, centre, ""))
        stalls = []
        for prop in props.all(observation):
            kind = str(prop["type"])
            if kind == "stall":
                stalls.append(_prop_center(observation, prop))
            elif kind not in NOT_LANDMARKS:
                marks.append((kind, self._own_name(observation, prop), _prop_center(observation, prop), str(prop["id"])))
        if stalls:  # the stalls stand together, so the market is one landmark at their middle
            middle = {"x": sum(s["x"] for s in stalls) / len(stalls), "y": sum(s["y"] for s in stalls) / len(stalls)}
            marks.append(("stall", "the market", middle, ""))
        return marks

    def _names(self, observation: ThreeBranchesObservation) -> dict[str, str]:
        """Name every prop so the visitor can tell which one we mean. Shrines and vegetable
        patches are named first, since they double as landmarks for naming everything else."""

        names: dict[str, str] = {}
        groups: dict[str, list] = {}
        for prop in props.all(observation):
            kind = str(prop["type"])
            if self.counts.get(kind, 0) <= 1:
                names[str(prop["id"])] = f"the {PROP_NOUNS[kind]}"
            else:
                groups.setdefault(kind, []).append(prop)
        for kind in sorted(groups, key=lambda kind: kind not in SIDE_NAMED):
            names.update(self._name_group(observation, kind, groups[kind]))
            if kind in SIDE_NAMED:
                # Only a shrine or patch with a plain name ("the east shrine") makes a landmark, so
                # nothing else gets a tangled name like "the lantern by the shrine by the bell".
                self.landmarks = [mark for mark in self.landmarks if mark[3] not in names or names[mark[3]] == mark[1]]
        return names

    def _name_group(self, observation: ThreeBranchesObservation, kind: str, group: list) -> dict[str, str]:
        """Give each prop of one kind its own name. A name two props would share is made more
        precise, and if they still clash, they are counted outward ("the nearer lantern ...")."""

        noun = PROP_NOUNS[kind]

        def centre(prop):
            return _prop_center(observation, prop)

        if kind in SIDE_NAMED:  # "the east shrine", else "the shrine by the inn"
            first = {str(prop["id"]): self._own_name(observation, prop) for prop in group}

            def sharper(prop):
                return f"the {noun} {self.near(centre(prop), exclude=kind, precise=True)}"
        else:  # "the lantern by the bell", else "the lantern just west of the bell"
            first = {str(prop["id"]): f"the {noun} {self.near(centre(prop), exclude=kind)}" for prop in group}

            def sharper(prop):
                return f"the {noun} {self.near(centre(prop), exclude=kind, precise=True)}"

        names = {}
        for prop in group:
            prop_id = str(prop["id"])
            clash = sum(name == first[prop_id] for name in first.values()) > 1
            names[prop_id] = sharper(prop) if clash else first[prop_id]
        shared: dict[str, list] = {}
        for prop in group:
            shared.setdefault(names[str(prop["id"])], []).append(prop)
        for name, members in shared.items():
            if len(members) == 1:
                continue
            mark = self._nearest_mark(centre(members[0]), exclude=kind)
            origin = mark[2] if mark is not None else self.centre
            members.sort(key=lambda prop: geometry.distance(origin, centre(prop)))
            order = ("nearer", "farther") if len(members) == 2 else ORDINALS
            for index, prop in enumerate(members):
                rank = order[index] if index < len(order) else f"number {index + 1}"
                names[str(prop["id"])] = name.replace("the ", f"the {rank} ", 1)
        return names

    def _nearest_mark(self, point, exclude: str = ""):
        marks = [mark for mark in self.landmarks if mark[0] != exclude]
        return min(marks, key=lambda mark: geometry.distance(mark[2], point)) if marks else None

    def _own_name(self, observation: ThreeBranchesObservation, prop) -> str:
        """Name a prop by itself: "the pump", or "the east shrine" when there are several."""

        kind = str(prop["type"])
        if self.counts.get(kind, 0) <= 1:
            return f"the {PROP_NOUNS[kind]}"
        side = compass(geometry.heading_to(self.centre, _prop_center(observation, prop)))
        return f"the {side} {PROP_NOUNS[kind]}"

    def describe(self, observation: ThreeBranchesObservation, prop) -> str:
        """Name a prop so the visitor can tell which one we mean."""

        return self.names.get(str(prop["id"])) or self._own_name(observation, prop)

    def near(self, point, exclude: str = "", precise: bool = False) -> str:
        """Place a point by its nearest landmark: "by the inn", "west of the pump". ``precise``
        always gives the compass direction from the landmark, to tell close neighbours apart."""

        mark = self._nearest_mark(point, exclude)
        if mark is None:
            return "out on the road"
        _, name, where, _ = mark
        gap = geometry.distance(where, point)
        side = compass(geometry.heading_to(where, point))
        if precise:
            return f"just {side} of {name}" if gap < 10 else f"{side} of {name}"
        if gap < 3:
            return f"right by {name}"
        if gap < 10:
            return f"by {name}"
        if gap < 20:
            return f"near {name}"
        return f"{side} of {name}"

    def directions(self, ctx: Context, point) -> str:
        gap = geometry.distance(ctx.here, point)
        if gap < 5:
            return "just here, a few steps away"
        metres = max(5, int(round(gap / 5.0)) * 5)
        return f"about {metres} metres {compass(geometry.heading_to(ctx.here, point))} of here"

    def when(self, ctx: Context, tick: int, phase: str) -> str:
        if ctx.tick - tick <= FRESH_TICKS:
            return "a moment ago"
        if phase == day.phase(ctx.obs):
            return "a little while ago"
        return PHASE_WHEN.get(phase, "earlier today")


# --- Agent -------------------------------------------------------------------------------------


class Agent:
    """One villager. Its role decides what it does through each phase of the day."""

    def reset(self, seed: int, observation: ThreeBranchesObservation) -> None:
        """Build the walk graph and this villager's behaviour tree before the day starts."""

        self._player_id = me.player_id(observation)
        self._nav = Navigator.for_village(observation)
        self._approach: dict[str, frozenset[Cell]] = {}
        self._memory: dict = {}
        self._greetings = Greetings(self._player_id)
        self._tick = 0
        self._tree = build_role(observation, self._player_id, me.rng(observation, seed))
        self._dialogue = Dialogue(observation, self._player_id)
        self._ctx: Context | None = None

    def _context(self, observation: ThreeBranchesObservation) -> Context:
        here = me.position(observation)
        found = layout.cell_at(observation, here)
        return Context(
            obs=observation,
            me_id=self._player_id,
            here=here,
            cell=None if found is None else (found["x"], found["y"]),
            tick=day.tick(observation),
            nav=self._nav,
            approach=self._approach,
            greetings=self._greetings,
            memory=self._memory,
        )

    def act(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Advance this villager's behaviour tree by one tick."""

        ctx = self._context(observation)
        self._tick = ctx.tick
        self._ctx = ctx  # chat runs right after act, and answers from what we perceive now
        self._dialogue.knowledge.update(observation, ctx.tick)
        self._dialogue.watch(ctx)  # a wave from the visitor opens a conversation
        # Everyone we can hear can hear us, so these are the villagers we can message directly.
        self._greetings.hearers = frozenset(str(person["id"]) for person in people.nearby(observation))
        order = self._tree.tick(ctx)
        return order if order is not None else ctx.stand()

    def chat(self, inbox: list[dict]) -> list[dict] | None:
        """Answer the visitor's option picks and questions, note any hellos for the next ``act`` to
        answer, chat back to wanderers who talk to us on a break, and send everything queued so far."""

        for message in inbox or ():
            if not isinstance(message, dict):
                continue
            sender = str(message.get("from"))
            if sender == VISITOR_ID and self._ctx is not None and self._dialogue.respond(message, self._ctx):
                continue
            if self._memory.get("on_break") and ROLES.get(sender) in ("wanderer", "sleeper"):
                self._memory["break_talker"] = sender
            if self._greetings.is_hello_for_me(message):
                self._greetings.heard_hello(sender, self._tick)
            elif self._memory.get("on_break") and ROLES.get(sender) in ("wanderer", "sleeper") and message.get("to") == self._player_id:
                self._reply_on_break(sender)
        # Held, not lost, while the visitor is a step out of earshot.
        line = self._dialogue.next_line() if VISITOR_ID in self._greetings.hearers else None
        if line is not None:
            # One message per recipient per tick, so longer exchanges go out over a few ticks.
            self._greetings.say(VISITOR_ID, line)
        outgoing, self._greetings.outbox = self._greetings.outbox, []
        return outgoing

    def _reply_on_break(self, wanderer_id: str) -> None:
        """Answer a wanderer's small talk with the next of this role's break lines."""

        lines = BREAK_REPLIES.get(ROLES.get(self._player_id, ""), BREAK_REPLIES["gardener"])
        said = self._memory.get("break_lines_said", 0)
        self._memory["break_lines_said"] = said + 1
        self._greetings.say(wanderer_id, lines[said % len(lines)])
        # The line itself is private, so show it: a quick nod or laugh from the bench.
        emote = SEATED_REPLY_EMOTES[said % len(SEATED_REPLY_EMOTES)]
        self._memory["seated_gesture"] = (emote, self._tick + SEATED_GESTURE_TICKS)
