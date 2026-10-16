"""Source-derived passage gates: coord triggers that push the player back.

A coord trigger whose script, on some branch, walks the player off the trigger
tile while leaving the trigger armed is a gate: the Saffron gate guards ("I'm
on guard duty. Gee, I'm thirsty, though!"), Pewter's gym guide, the Route 23
badge checks or Emerald's Route 111 sandstorm. Branches that disarm the trigger
(writing its own variable) or simply release the player are pass branches; the
conditions along them (flags, bag items, variables, party moves, defeated
trainers) are the requirements.

The analysis is a small symbolic execution of the pinned pret scripts: it
follows goto/call/fallthrough, the generic ``compare``/``goto_if`` encoding of
all conditional macros, ``switch``/``case`` and constant ``setvar`` values. It
never runs on the emulator and never writes RAM. Runtime evaluation reads the
same flags, variables and bag a player can observe.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Protocol, cast

from core.navigation.source_data import ROOT, MapKey, SOURCE_COMMITS, load_map_catalog, source_root

XY = tuple[int, int]
# An operand is ("const", value), ("var", name), ("result", source...) or
# ("unknown", text). A literal says the comparison result (0 <, 1 =, 2 >) of
# two operands, or a flag's value (0/1), lies in a set of accepted results.
Operand = tuple[Any, ...]
Literal_ = tuple[Any, ...]
Conjunction = tuple[Literal_, ...]

_JUMP_RESULTS = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2}),
                 3: frozenset({0, 1}), 4: frozenset({1, 2}), 5: frozenset({0, 2})}
_CONDITION_NAMES = {"lt": 0, "eq": 1, "gt": 2, "le": 3, "ge": 4, "ne": 5}
_PLAYER_IDS = frozenset({"OBJ_EVENT_ID_PLAYER", "LOCALID_PLAYER", "0xFF", "0xff", "255"})
_RESULT_COMMANDS = frozenset({"checkmoney", "checkcoins", "getpartysize", "checkplayergender", "random",
                              "checkitemspace", "checkpcitem", "checkitemtype", "checkdecor", "checkdecorspace",
                              "checkpokeblock", "checkspace", "getpricereduction", "checkfieldmove"})
_CHOICE_COMMANDS = frozenset({"yesnobox", "multichoice", "multichoicedefault", "multichoicegrid",
                              "dynmultichoice"})
_MAX_PATHS = 4096
_MAX_STEPS = 4000


@dataclass(frozen=True)
class ScriptGate:
    """One trigger tile that blocks while armed unless a pass branch applies."""
    map_name: str
    map_key: MapKey
    tile: XY
    elevation: int
    var: str
    var_value: int
    script: str
    pass_conditions: tuple[Conjunction, ...]
    pushback: XY
    requirement: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"map_name": self.map_name, "map_key": list(self.map_key), "tile": list(self.tile),
                "elevation": self.elevation, "var": self.var, "var_value": self.var_value,
                "script": self.script, "pass_conditions": [[_literal_to_json(item) for item in conjunction]
                                                           for conjunction in self.pass_conditions],
                "pushback": list(self.pushback), "requirement": self.requirement}

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> ScriptGate:
        conditions = tuple(tuple(_literal_from_json(item) for item in conjunction)
                           for conjunction in row["pass_conditions"])
        return cls(str(row["map_name"]), (int(row["map_key"][0]), int(row["map_key"][1])),
                   (int(row["tile"][0]), int(row["tile"][1])), int(row["elevation"]), str(row["var"]),
                   int(row["var_value"]), str(row["script"]), conditions,
                   (int(row["pushback"][0]), int(row["pushback"][1])), str(row.get("requirement", "")))


def _to_json(value: Any) -> Any:
    if isinstance(value, frozenset):
        return {"set": sorted(value)}
    if isinstance(value, tuple):
        return [_to_json(item) for item in value]
    return value


def _from_json(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"set"}:
        return frozenset(int(item) for item in value["set"])
    if isinstance(value, list):
        return tuple(_from_json(item) for item in value)
    return value


def _literal_to_json(literal: Literal_) -> Any:
    return _to_json(literal)


def _literal_from_json(raw: Any) -> Literal_:
    value = _from_json(raw)
    if not isinstance(value, tuple):
        raise ValueError("Invalid gate literal")
    return value


# ---------------------------------------------------------------------------
# Source parsing


@dataclass
class _Program:
    blocks: dict[str, list[tuple[str, tuple[str, ...]]]] = field(default_factory=dict)
    fallthrough: dict[str, str] = field(default_factory=dict)


def _split_arguments(text: str) -> tuple[str, ...]:
    """Macro arguments; GAS also accepts blanks between plain symbols
    (``call_if_not_defeated TRAINER_X Label`` in Rocket Hideout B1F)."""
    parts: list[str] = []
    for part in (piece.strip() for piece in text.split(",")):
        if not part:
            continue
        parts.extend(part.split() if re.fullmatch(r"[\w.]+(?:\s+[\w.]+)+", part) else [part])
    return tuple(parts)


def _parse_program(root: Path) -> _Program:
    program = _Program()
    files = sorted((root / "data" / "maps").rglob("*.inc")) + sorted((root / "data" / "scripts").rglob("*.inc"))
    event_scripts = root / "data" / "event_scripts.s"
    if event_scripts.is_file():
        files.append(event_scripts)
    for path in files:
        previous: str | None = None
        label: str | None = None
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = re.sub(r"/\*.*?\*/", "", raw.split("@")[0].split("//")[0]).strip()
            if not line:
                continue
            match = re.match(r"^(\w+)::?\s*(.*)$", line)
            if match and not line.startswith("."):
                label = match[1]
                program.blocks.setdefault(label, [])
                if previous is not None and previous not in program.fallthrough:
                    program.fallthrough[previous] = label
                previous = label
                line = match[2].strip()
                if not line:
                    continue
            if label is None:
                continue
            parts = line.split(None, 1)
            command = parts[0]
            arguments = _split_arguments(parts[1]) if len(parts) > 1 else ()
            program.blocks[label].append((command, arguments))
    return program


_DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _movement_offset(program: _Program, label: str) -> XY:
    """Net player displacement of a movement script; in-place steps are zero."""
    dx = dy = 0
    seen: set[str] = set()
    current: str | None = label
    while current is not None and current not in seen:
        seen.add(current)
        for command, _ in program.blocks.get(current, []):
            name = command.lower()
            if name == "step_end":
                return dx, dy
            match = re.search(r"_(up|down|left|right)$", name)
            if match is None or "in_place" in name or name.startswith(("face", "lock", "unlock", "emote", "set_")):
                continue
            head = name.split("_", 1)[0]
            if head not in ("walk", "slide", "player", "jump", "ride", "run", "acro", "fly"):
                continue
            distance = 2 if name.startswith("jump_2_") else 1
            step = _DIRECTIONS[match[1]]
            dx, dy = dx + step[0] * distance, dy + step[1] * distance
        current = program.fallthrough.get(current)
    return dx, dy


# ---------------------------------------------------------------------------
# Symbolic execution


@dataclass(frozen=True)
class _PathOutcome:
    conditions: frozenset[Literal_]
    displacement: XY
    disarmed: bool
    warped: bool
    written_flags: frozenset[str] = frozenset()
    written_vars: tuple[tuple[str, int], ...] = ()
    tiles: tuple[tuple[int, int, int], ...] = ()


@dataclass
class _State:
    label: str
    index: int
    stack: tuple[tuple[str, int], ...]
    conditions: frozenset[Literal_]
    values: dict[str, Operand]
    flags: dict[str, bool]
    compare: tuple[Any, ...] | None
    displacement: XY
    disarmed: bool
    steps: int
    written: frozenset[str] = frozenset()
    tiles: tuple[tuple[int, int, int], ...] = ()

    def fork(self) -> _State:
        return _State(self.label, self.index, self.stack, self.conditions, dict(self.values), dict(self.flags),
                      self.compare, self.displacement, self.disarmed, self.steps, self.written, self.tiles)


class _Analyzer:
    def __init__(self, program: _Program, constants: Mapping[str, int]) -> None:
        self.program = program
        self.constants = constants

    def _number(self, text: str) -> int | None:
        token = text.strip()
        if token in ("TRUE", "YES"):
            return 1
        if token in ("FALSE", "NO"):
            return 0
        try:
            return int(token, 0)
        except ValueError:
            pass
        value = self.constants.get(token)
        return int(value) if value is not None else None

    def _is_var(self, text: str) -> bool:
        return text.startswith("VAR_")

    def _operand(self, state: _State, text: str) -> Operand:
        if self._is_var(text):
            known = state.values.get(text)
            if known is not None:
                return known
            return ("var", text)
        number = self._number(text)
        if number is not None:
            if 0x4000 <= number < 0x4100 or 0x8000 <= number <= 0x8016:
                names = [name for name, value in self.constants.items() if value == number and name.startswith("VAR_")]
                if names:
                    return self._operand(state, sorted(names)[0])
            return ("const", number)
        return ("unknown", text)

    @staticmethod
    def _compare_known(left: Operand, right: Operand) -> int | None:
        if left[0] == "const" and right[0] == "const":
            a, b = int(left[1]), int(right[1])
            return 0 if a < b else (1 if a == b else 2)
        return None

    def _branch(self, state: _State, condition: int, destination: str, call: bool,
                frontier: list[_State]) -> None:
        """Fork on the comparison register; record literals for unknown operands."""
        accepted = _JUMP_RESULTS.get(condition)
        register = state.compare
        if accepted is None or register is None:
            # An unknown condition can go either way.
            taken = state.fork()
            self._jump(taken, destination, call)
            frontier.append(taken)
            return
        if register[0] == "flag":
            flag = str(register[1])
            known = state.flags.get(flag)
            if known is not None:
                if int(known) in accepted:
                    self._jump(state, destination, call)
                return
            if accepted >= {0, 1}:
                self._jump(state, destination, call)
                return
            if not accepted & {0, 1}:
                return
            jump_value = 1 if 1 in accepted else 0
            taken = state.fork()
            taken.conditions = taken.conditions | {("flag", flag, frozenset({jump_value}))}
            taken.flags[flag] = bool(jump_value)
            self._jump(taken, destination, call)
            frontier.append(taken)
            state.conditions = state.conditions | {("flag", flag, frozenset({1 - jump_value}))}
            state.flags[flag] = not bool(jump_value)
            return
        left, right = cast(Operand, register[1]), cast(Operand, register[2])
        known_result = self._compare_known(left, right)
        if known_result is not None:
            if known_result in accepted:
                self._jump(state, destination, call)
            return
        complement = frozenset({0, 1, 2}) - accepted
        taken = state.fork()
        taken.conditions = taken.conditions | {("cmp", left, right, accepted)}
        self._jump(taken, destination, call)
        frontier.append(taken)
        state.conditions = state.conditions | {("cmp", left, right, complement)}

    def _jump(self, state: _State, destination: str, call: bool) -> None:
        if call:
            state.stack = state.stack + ((state.label, state.index),)
        state.label, state.index = destination, 0

    def explore(self, entry: str, trigger_var: str, trigger_value: int) -> list[_PathOutcome]:
        outcomes: list[_PathOutcome] = []
        frontier = [_State(entry, 0, (), frozenset(), {}, {}, None, (0, 0), False, 0)]
        paths = 0
        while frontier and paths < _MAX_PATHS:
            state = frontier.pop()
            paths += 1
            outcome = self._run(state, trigger_var, trigger_value, frontier)
            if outcome is not None:
                outcomes.append(outcome)
        return outcomes

    def _set_value(self, state: _State, name: str, value: Operand, trigger_var: str, trigger_value: int) -> None:
        state.values[name] = value
        if name == trigger_var and not (value[0] == "const" and int(value[1]) == trigger_value):
            state.disarmed = True

    def _run(self, state: _State, trigger_var: str, trigger_value: int,
             frontier: list[_State]) -> _PathOutcome | None:
        while state.steps < _MAX_STEPS:
            state.steps += 1
            block = self.program.blocks.get(state.label)
            if block is None:
                return None  # unresolvable symbol: no claim about this branch
            if state.index >= len(block):
                following = self.program.fallthrough.get(state.label)
                if following is None:
                    return self._finish(state, warped=False)
                state.label, state.index = following, 0
                continue
            command, arguments = block[state.index]
            state.index += 1
            name = command.lower()
            if name in ("end", "releaseall_end"):
                return self._finish(state, warped=False)
            if name == "return":
                if not state.stack:
                    return self._finish(state, warped=False)
                state.stack, (state.label, state.index) = state.stack[:-1], state.stack[-1]
                continue
            if name in ("goto", "goto_std") and arguments:
                state.label, state.index = arguments[0], 0
                continue
            if name == "call" and arguments:
                self._jump(state, arguments[0], True)
                continue
            if name.startswith("warp"):
                return self._finish(state, warped=True)
            if name in ("goto_if_set", "goto_if_unset", "call_if_set", "call_if_unset",
                        "vgoto_if_set", "vgoto_if_unset") and len(arguments) >= 2:
                state.compare = ("flag", arguments[0])
                self._branch(state, 1 if name.endswith("_set") else 0, arguments[1], name.startswith("call"), frontier)
                continue
            if name in ("goto_if_defeated", "goto_if_not_defeated", "call_if_defeated",
                        "call_if_not_defeated") and len(arguments) >= 2:
                state.compare = ("flag", "TRAINER:" + arguments[0])
                self._branch(state, 0 if "not_defeated" in name else 1, arguments[1], name.startswith("call"), frontier)
                continue
            match = re.fullmatch(r"v?(goto|call)_if_(lt|eq|gt|le|ge|ne)", name)
            if match is not None and arguments:
                condition = _CONDITION_NAMES[match[2]]
                if len(arguments) >= 3:
                    state.compare = ("cmp", self._operand(state, arguments[0]), self._operand(state, arguments[1]))
                    destination = arguments[2]
                else:
                    destination = arguments[-1]
                self._branch(state, condition, destination, match[1] == "call", frontier)
                continue
            if name in ("goto_if", "call_if") and len(arguments) >= 2:
                raw_condition = self._number(arguments[0])
                self._branch(state, -1 if raw_condition is None else raw_condition, arguments[1],
                             name == "call_if", frontier)
                continue
            if name == "goto_if_questlog":
                continue  # quest-log playback never runs during play
            if name in ("compare", "compare_var_to_value", "compare_var_to_var") and len(arguments) >= 2:
                state.compare = ("cmp", self._operand(state, arguments[0]), self._operand(state, arguments[1]))
                continue
            if name == "checkflag" and arguments:
                state.compare = ("flag", arguments[0])
                continue
            if name == "checktrainerflag" and arguments:
                state.compare = ("flag", "TRAINER:" + arguments[0])
                continue
            if name == "switch" and arguments:
                self._set_value(state, "VAR_0x8000", self._operand(state, arguments[0]), trigger_var, trigger_value)
                continue
            if name == "case" and len(arguments) >= 2:
                state.compare = ("cmp", self._operand(state, "VAR_0x8000"), self._operand(state, arguments[0]))
                self._branch(state, 1, arguments[1], False, frontier)
                continue
            if name == "setflag" and arguments:
                state.flags[arguments[0]] = True
                state.written = state.written | {arguments[0]}
                continue
            if name == "clearflag" and arguments:
                state.flags[arguments[0]] = False
                continue
            if name == "setvar" and len(arguments) >= 2:
                number = self._number(arguments[1])
                value: Operand = ("const", number) if number is not None else ("unknown", arguments[1])
                self._set_value(state, arguments[0], value, trigger_var, trigger_value)
                continue
            if name == "copyvar" and len(arguments) >= 2:
                self._set_value(state, arguments[0], self._operand(state, arguments[1]), trigger_var, trigger_value)
                continue
            if name in ("addvar", "subvar") and len(arguments) >= 2:
                current = state.values.get(arguments[0])
                amount = self._number(arguments[1])
                if current is not None and current[0] == "const" and amount is not None:
                    updated = int(current[1]) + (amount if name == "addvar" else -amount)
                    self._set_value(state, arguments[0], ("const", updated), trigger_var, trigger_value)
                else:
                    self._set_value(state, arguments[0], ("unknown", name), trigger_var, trigger_value)
                continue
            if name == "checkitem" and arguments:
                quantity = self._number(arguments[1]) if len(arguments) > 1 else 1
                state.values["VAR_RESULT"] = ("result", "item", arguments[0], quantity or 1)
                continue
            if name == "checkpartymove" and arguments:
                state.values["VAR_RESULT"] = ("result", "partymove", arguments[0])
                continue
            if name == "specialvar" and len(arguments) >= 2:
                state.values[arguments[0]] = ("unknown", "special:" + arguments[1])
                continue
            if name == "special":
                state.values["VAR_RESULT"] = ("unknown", "special:" + (arguments[0] if arguments else ""))
                continue
            if name in _CHOICE_COMMANDS or (name == "msgbox" and len(arguments) > 1 and arguments[1] == "MSGBOX_YESNO"):
                state.values["VAR_RESULT"] = ("result", "choice", name)
                continue
            if name in _RESULT_COMMANDS:
                state.values["VAR_RESULT"] = ("unknown", name)
                continue
            if name == "setmetatile" and len(arguments) >= 4:
                x, y, impassable = (self._number(arguments[0]), self._number(arguments[1]),
                                    self._number(arguments[3]))
                if x is not None and y is not None and impassable is not None:
                    state.tiles = state.tiles + ((x, y, int(bool(impassable))),)
                continue
            if name in ("applymovement", "applywaitmovement") and len(arguments) >= 2:
                if arguments[0] in _PLAYER_IDS:
                    offset = _movement_offset(self.program, arguments[1])
                    state.displacement = (state.displacement[0] + offset[0], state.displacement[1] + offset[1])
                continue
        return None

    @staticmethod
    def _finish(state: _State, warped: bool) -> _PathOutcome:
        variables = tuple(sorted((name, int(value[1])) for name, value in state.values.items()
                                 if value[0] == "const" and name.startswith("VAR_") and not name.startswith("VAR_0x")))
        return _PathOutcome(state.conditions, state.displacement, state.disarmed, warped, state.written, variables,
                            state.tiles)


def _describe(conditions: Iterable[Conjunction]) -> str:
    parts: list[str] = []
    for conjunction in conditions:
        words: list[str] = []
        for literal in conjunction:
            if literal[0] == "flag":
                words.append(("" if 1 in literal[2] else "no ") + str(literal[1]))
            elif literal[0] == "cmp":
                left, right, accepted = literal[1], literal[2], literal[3]
                if left[0] == "result" and left[1] == "item":
                    has = (1 in accepted) == (right[0] == "const" and int(right[1]) == 1)
                    words.append(("" if has else "sin ") + str(left[2]))
                elif left[0] == "result" and left[1] == "partymove":
                    words.append("equipo con " + str(left[2]))
                elif left[0] == "var":
                    words.append(f"{left[1]} {sorted(accepted)} {right[1] if len(right) > 1 else ''}")
                else:
                    words.append("condición de script")
        parts.append(" y ".join(words) if words else "siempre")
    return " o ".join(parts)


def analyze_gates(game: str, root: Path | None = None) -> tuple[ScriptGate, ...]:
    """Every pushback trigger of a pinned decomp tree, in map/tile order."""
    root = root or source_root(game)
    catalog = load_map_catalog(game)
    program = _parse_program(root)
    constants = dict(catalog.constants)
    analyzer = _Analyzer(program, constants)
    gates: list[ScriptGate] = []
    for key in sorted(catalog.maps):
        raw = catalog.maps[key]
        for event in raw.get("coord_events", []) or []:
            if str(event.get("type", "trigger")) != "trigger":
                continue
            script = str(event.get("script", ""))
            if script in ("", "0", "0x0") or script not in program.blocks:
                continue
            var = str(event.get("var", "0"))
            value = analyzer._number(str(event.get("var_value", "0")))
            if value is None:
                continue
            trigger_var = var if var.startswith("VAR_") else ""
            outcomes = analyzer.explore(script, trigger_var, value)
            blocking = [outcome for outcome in outcomes
                        if not outcome.warped and not outcome.disarmed and outcome.displacement != (0, 0)]
            if not blocking:
                continue
            passing = sorted({tuple(sorted(outcome.conditions, key=repr)) for outcome in outcomes
                              if outcome not in blocking}, key=repr)
            pushback = blocking[0].displacement
            conditions: tuple[Conjunction, ...] = tuple(passing)
            gates.append(ScriptGate(str(raw["name"]), key, (int(event["x"]), int(event["y"])),
                                    int(event.get("elevation", 0) or 0), trigger_var, value, script,
                                    conditions, pushback, _describe(conditions)))
    return tuple(gates)


def _script_trainer(program: _Program, label: str) -> str | None:
    """The TRAINER_* constant of the first trainerbattle reachable from a script."""
    seen: set[str] = set()
    current: str | None = label
    steps = 0
    while current is not None and current not in seen and steps < 64:
        seen.add(current)
        for command, arguments in program.blocks.get(current, []):
            steps += 1
            name = command.lower()
            if name.startswith("trainerbattle") and arguments:
                trainer = arguments[1] if name == "trainerbattle" and len(arguments) > 1 else arguments[0]
                return trainer if trainer.startswith("TRAINER_") else None
            if name in ("goto", "call") and arguments:
                current = arguments[0]
                break
            if name in ("end", "return"):
                return None
        else:
            current = program.fallthrough.get(current)
    return None


def analyze_trainers(game: str, root: Path | None = None) -> dict[str, dict[str, str]]:
    """Map name -> object local id -> TRAINER_* constant, for sight and defeat flags."""
    root = root or source_root(game)
    catalog = load_map_catalog(game)
    program = _parse_program(root)
    result: dict[str, dict[str, str]] = {}
    for key in sorted(catalog.maps):
        raw = catalog.maps[key]
        for index, event in enumerate(raw.get("object_events", []) or []):
            if str(event.get("trainer_type", "TRAINER_TYPE_NONE")) in ("TRAINER_TYPE_NONE", "0"):
                continue
            trainer = _script_trainer(program, str(event.get("script", "")))
            if trainer is not None:
                result.setdefault(str(raw["name"]), {})[str(index + 1)] = trainer
    return result


def _reaches(program: _Program, label: str, command_name: str, limit: int = 64, argument: str = "") -> bool:
    """Whether any branch from ``label`` executes ``command_name`` (with ``argument``)."""
    pending, seen = [label], set()
    while pending and len(seen) < limit:
        current = pending.pop()
        if current in seen or current not in program.blocks:
            continue
        seen.add(current)
        ended = False
        for command, arguments in program.blocks[current]:
            name = command.lower()
            if name == command_name and (not argument or argument in arguments):
                return True
            if (name in ("goto", "call") or name.startswith(("goto_if", "call_if")) or name == "case") and arguments:
                pending.append(arguments[-1])
            if name in ("goto", "end", "return"):
                ended = True
                break
        if not ended and current in program.fallthrough:
            pending.append(program.fallthrough[current])
    return False


def _elevator_floors(program: _Program, script: str,
                     constants: Mapping[str, int]) -> tuple[list[dict[str, Any]], list[str]]:
    """Menu row -> dynamic warp destination of a floor-select script.

    Only rows of a ``multichoice`` menu count: the cases of the first
    ``switch VAR_RESULT`` after it are the rows the field controller can pick.
    Persistent flags tested with ``goto_if_unset`` towards a branch that never
    sets the warp (the Lift Key) are the panel's requirements.
    """
    floors: dict[int, dict[str, Any]] = {}
    required: list[str] = []
    pending: list[tuple[str, int | None, bool]] = [(script, None, False)]
    seen: set[tuple[str, int | None, bool]] = set()
    while pending and len(seen) < 256:
        label, choice, menu = pending.pop()
        if (label, choice, menu) in seen or label not in program.blocks:
            continue
        seen.add((label, choice, menu))
        ended = False
        for command, arguments in program.blocks[label]:
            name = command.lower()
            if name.startswith("multichoice"):
                menu = True
            elif name == "case" and len(arguments) >= 2:
                value = int(arguments[0]) if arguments[0].isdecimal() else constants.get(arguments[0])
                row = value if menu and choice is None and value is not None and value < 32 else choice
                if menu and choice is None and row is None:
                    continue  # B pressed / out of menu: no floor
                pending.append((arguments[1], row, menu))
            elif name == "setdynamicwarp" and len(arguments) >= 2 and choice is not None:
                # formatwarp: (map, warp) | (map, x, y) | (map, warp, x, y).
                try:
                    values = [int(argument, 0) for argument in arguments[1:4]]
                except ValueError:
                    continue
                row: dict[str, Any] = {"choice": choice, "map": arguments[0]}
                if len(values) == 1:
                    row["warp_id"] = values[0]
                else:
                    row["xy"] = values[-2:]
                floors.setdefault(choice, row)
            elif name.startswith(("goto_if", "call_if")) and arguments:
                flag = arguments[0]
                if (name == "goto_if_unset" and flag.startswith("FLAG_") and not flag.startswith("FLAG_TEMP") and
                        flag not in required and not _reaches(program, arguments[-1], "setdynamicwarp")):
                    required.append(flag)
                pending.append((arguments[-1], choice, menu))
            elif name in ("goto", "call") and arguments:
                pending.append((arguments[0], choice, menu))
            if name in ("goto", "end", "return"):
                ended = True
                break
        if not ended and label in program.fallthrough:
            pending.append((program.fallthrough[label], choice, menu))
    return [floors[row] for row in sorted(floors)], required


def analyze_elevators(game: str, root: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Map name -> floor-select panels of maps whose exits are dynamic warps."""
    root = root or source_root(game)
    catalog = load_map_catalog(game)
    program = _parse_program(root)
    result: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(catalog.maps):
        raw = catalog.maps[key]
        if not any(str(warp.get("dest_map")) == "MAP_DYNAMIC" for warp in raw.get("warp_events", []) or []):
            continue
        for target in catalog.targets(str(raw["name"])):
            if target["action"] != "interact":
                continue
            floors, required = _elevator_floors(program, str(target["script"]), catalog.constants)
            for floor in floors:
                destination = catalog.by_id.get(str(floor["map"]))
                warps = catalog.maps[destination]["warp_events"] if destination is not None else []
                if "xy" not in floor and 0 <= int(floor.get("warp_id", -1)) < len(warps):
                    warp = warps[int(floor.pop("warp_id"))]
                    floor["xy"] = [int(warp["x"]), int(warp["y"])]
            floors = [floor for floor in floors if "xy" in floor and str(floor["map"]) in catalog.by_id]
            if floors:
                result.setdefault(str(raw["name"]), []).append(
                    {"script": str(target["script"]), "xy": [int(target["x"]), int(target["y"])],
                     "npc_local_id": int(target["npc_local_id"]), "floors": floors, "required_flags": required})
    return result


def analyze_overlays(game: str, root: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Map name -> metatiles its on-load script places, per branch condition.

    Rocket Hideout B1F walls its elevator corridor while a grunt is undefeated:
    the source layout alone shows an open floor, and plans through it fail live.
    """
    root = root or source_root(game)
    catalog = load_map_catalog(game)
    program = _parse_program(root)
    analyzer = _Analyzer(program, dict(catalog.constants))
    result: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(catalog.maps):
        name = str(catalog.maps[key]["name"])
        for command, arguments in program.blocks.get(name + "_MapScripts", []):
            if command != "map_script" or len(arguments) < 2 or arguments[0] != "MAP_SCRIPT_ON_LOAD":
                continue
            rows: dict[tuple[Any, ...], dict[str, Any]] = {}
            for outcome in analyzer.explore(arguments[1], "", 0):
                if not outcome.tiles:
                    continue
                conditions = tuple(sorted(outcome.conditions, key=repr))
                final: dict[tuple[int, int], int] = {}
                for x, y, impassable in outcome.tiles:
                    final[(x, y)] = impassable
                rows[conditions] = {"conditions": [_literal_to_json(literal) for literal in conditions],
                                    "tiles": [[x, y, value] for (x, y), value in sorted(final.items())]}
            if rows:
                result[name] = [rows[conditions] for conditions in sorted(rows, key=repr)]
    return result


@lru_cache(maxsize=2)
def load_overlays(game: str) -> dict[str, tuple[tuple[Conjunction, tuple[tuple[int, int, int], ...]], ...]]:
    path = gate_resource_path(game)
    raw: Any = None
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8")).get("overlays")
    if not isinstance(raw, dict) and (source_root(game) / "data" / "maps").is_dir():
        raw = analyze_overlays(game)
    if not isinstance(raw, dict):
        return {}
    return {str(name): tuple((tuple(_literal_from_json(item) for item in row["conditions"]),
                              tuple((int(x), int(y), int(value)) for x, y, value in row["tiles"]))
                             for row in rows)
            for name, rows in raw.items()}


def map_overlay(game: str, map_name: str, world: GateWorld | None) -> tuple[tuple[int, int, int], ...]:
    """Collision overrides of the on-load script branch the world satisfies (if known)."""
    if world is None:
        return ()
    for conditions, tiles in load_overlays(game).get(map_name, ()):
        if all(_literal_value(world, literal) is True for literal in conditions):
            return tiles
    return ()


def analyze_floor_switches(game: str, root: Path | None = None) -> dict[str, list[list[int]]]:
    """Map name -> step triggers whose scripts redraw the room (Mauville's barriers)."""
    root = root or source_root(game)
    catalog = load_map_catalog(game)
    program = _parse_program(root)
    result: dict[str, list[list[int]]] = {}
    for key in sorted(catalog.maps):
        raw = catalog.maps[key]
        for event in raw.get("coord_events", []) or []:
            script = str(event.get("script", ""))
            if str(event.get("type", "trigger")) != "trigger" or script in ("", "0", "0x0"):
                continue
            if _reaches(program, script, "setmetatile") or _reaches(program, script, "special", argument="DrawWholeMapView"):
                result.setdefault(str(raw["name"]), []).append([int(event["x"]), int(event["y"])])
    return result


@lru_cache(maxsize=2)
def load_floor_switches(game: str) -> dict[str, list[list[int]]]:
    path = gate_resource_path(game)
    if path.is_file():
        switches = json.loads(path.read_text(encoding="utf-8")).get("floor_switches")
        if isinstance(switches, dict):
            return cast(dict[str, list[list[int]]], switches)
    if (source_root(game) / "data" / "maps").is_dir():
        return analyze_floor_switches(game)
    return {}


@lru_cache(maxsize=2)
def load_elevators(game: str) -> dict[str, list[dict[str, Any]]]:
    path = gate_resource_path(game)
    if path.is_file():
        elevators = json.loads(path.read_text(encoding="utf-8")).get("elevators")
        if isinstance(elevators, dict):
            return cast(dict[str, list[dict[str, Any]]], elevators)
    if (source_root(game) / "data" / "maps").is_dir():
        return analyze_elevators(game)
    return {}


@lru_cache(maxsize=2)
def _program(game: str) -> _Program | None:
    root = source_root(game)
    return _parse_program(root) if (root / "data" / "maps").is_dir() else None


def completion_requirements(game: str, script: str, flag: str = "", variable: str = "",
                            minimum: int = 1) -> list[Conjunction] | None:
    """Conditions on the branches of an event script that achieve an effect.

    ``None`` when the script cannot be analysed (no sources, or no branch sets
    the flag/variable here, e.g. it happens after a battle callback): the
    caller then has no claim about what the event wants.
    """
    program = _program(game)
    if program is None or script not in program.blocks:
        return None
    analyzer = _Analyzer(program, dict(load_map_catalog(game).constants))
    outcomes = analyzer.explore(script, "", 0)
    good = [outcome for outcome in outcomes
            if (flag and flag in outcome.written_flags) or
            (variable and dict(outcome.written_vars).get(variable, -1) >= minimum)]
    if not good:
        return None
    return sorted({tuple(sorted(outcome.conditions, key=repr)) for outcome in good}, key=repr)


def unmet_requirements(conjunctions: Iterable[Conjunction], world: GateWorld) -> list[tuple[str, str]]:
    """(kind, name) of flags to set or items to obtain, for the easiest branch.

    A branch already satisfied (or with only unobservable literals) means the
    event is not waiting for anything obtainable: the result is empty.
    """
    best: list[tuple[str, str]] | None = None
    for conjunction in conjunctions:
        needs: list[tuple[str, str]] = []
        feasible = True
        for literal in conjunction:
            value = _literal_value(world, literal)
            if value is not False:
                continue
            if literal[0] == "flag" and 1 in literal[2] and not str(literal[1]).startswith("TRAINER:"):
                needs.append(("flag", str(literal[1])))
            elif (literal[0] == "cmp" and literal[1][0] == "result" and literal[1][1] == "item" and
                  (1 in literal[3]) == (literal[2][0] == "const" and int(literal[2][1]) == 1)):
                needs.append(("item", str(literal[1][2])))
            else:
                feasible = False  # a condition no event of ours can satisfy directly
        if not feasible:
            continue
        if not needs:
            return []
        if best is None or len(needs) < len(best):
            best = needs
    return best or []


def gate_resource_path(game: str) -> Path:
    return ROOT / "data" / f"gates_{game}.json"


def export_gates(game: str, path: Path | None = None) -> Path:
    gates = analyze_gates(game)
    target = path or gate_resource_path(game)
    data = {"game": game, "source_commit": SOURCE_COMMITS[game],
            "description": "Coord triggers that push the player back while armed; pass conditions per branch. "
                           "Trainers: object local id -> TRAINER_* constant (defeat flag, sight cone). "
                           "Elevators: floor-select panels, menu row -> dynamic warp destination. "
                           "Floor switches: step triggers that redraw the room. "
                           "Overlays: collision placed by on-load scripts, per branch condition.",
            "gates": [gate.to_json() for gate in gates], "trainers": analyze_trainers(game),
            "elevators": analyze_elevators(game), "floor_switches": analyze_floor_switches(game),
            "overlays": analyze_overlays(game)}
    target.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


@lru_cache(maxsize=2)
def load_gates(game: str) -> tuple[ScriptGate, ...]:
    """Exported knowledge first; analyze the vendored sources when it is absent."""
    path = gate_resource_path(game)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("game") != game or data.get("source_commit") != SOURCE_COMMITS[game]:
            raise ValueError("Gate knowledge does not match the pinned game source")
        return tuple(ScriptGate.from_json(row) for row in data["gates"])
    if (source_root(game) / "data" / "maps").is_dir():
        return analyze_gates(game)
    return ()


@lru_cache(maxsize=2)
def load_trainers(game: str) -> dict[str, dict[str, str]]:
    path = gate_resource_path(game)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        trainers = data.get("trainers")
        if isinstance(trainers, dict):
            return {str(name): {str(local): str(value) for local, value in rows.items()}
                    for name, rows in trainers.items() if isinstance(rows, dict)}
    if (source_root(game) / "data" / "maps").is_dir():
        return analyze_trainers(game)
    return {}


@lru_cache(maxsize=2)
def gates_by_map(game: str) -> dict[MapKey, tuple[ScriptGate, ...]]:
    grouped: dict[MapKey, list[ScriptGate]] = {}
    for gate in load_gates(game):
        grouped.setdefault(gate.map_key, []).append(gate)
    return {key: tuple(values) for key, values in grouped.items()}


# ---------------------------------------------------------------------------
# Runtime evaluation


class GateWorld(Protocol):
    """Observable state a gate condition may depend on."""

    def flag(self, name: str) -> bool | None: ...

    def var(self, name: str) -> int | None: ...

    def item_count(self, name: str) -> int | None: ...

    def party_move_slot(self, name: str) -> int | None: ...


def _operand_value(world: GateWorld, operand: Operand) -> int | None:
    kind = operand[0]
    if kind == "const":
        return int(operand[1])
    if kind == "var":
        return world.var(str(operand[1]))
    if kind == "result":
        if operand[1] == "item":
            count = world.item_count(str(operand[2]))
            return None if count is None else int(count >= int(operand[3]))
        if operand[1] == "partymove":
            return world.party_move_slot(str(operand[2]))
    return None


def _literal_value(world: GateWorld, literal: Literal_) -> bool | None:
    if literal[0] == "flag":
        state = world.flag(str(literal[1]))
        return None if state is None else int(state) in literal[2]
    if literal[0] == "cmp":
        left, right, accepted = literal[1], literal[2], literal[3]
        if left[0] == "result" and left[1] == "choice" or right[0] == "result" and right[1] == "choice":
            return True  # the player answers the menu; the field controller chooses
        a, b = _operand_value(world, left), _operand_value(world, right)
        if a is None or b is None:
            return None
        return (0 if a < b else (1 if a == b else 2)) in accepted
    return None


def gate_state(gate: ScriptGate, world: GateWorld) -> bool | None:
    """True when closed, False when passable, None when an operand is unobservable."""
    if gate.var:
        current = world.var(gate.var)
        if current is not None and current != gate.var_value:
            return False  # the trigger is disarmed
    unknown = False
    for conjunction in gate.pass_conditions:
        values = [_literal_value(world, literal) for literal in conjunction]
        if all(value is True for value in values):
            return False
        if not any(value is False for value in values):
            unknown = True
    return None if unknown else True


class RamGateWorld:
    """Flags, variables, bag and party read once from RAM for one evaluation pass.

    Temporary flags and variables reset whenever a map is entered, so they keep
    their live values only for the map the player stands on.
    """

    def __init__(self, bridge: Any, game: str, current_map: bool) -> None:
        catalog = load_map_catalog(game)
        self.constants = catalog.constants
        self.current_map = current_map
        memory = bridge.memory_map
        base = bridge.get_save_block1_dynamic_ptr()
        self._flags = bytes(bridge.read_memory(base + memory.flags_offset, memory.vars_offset - memory.flags_offset))
        self._vars = bytes(bridge.read_memory(base + memory.vars_offset, 0x200))
        self._inventory: dict[int, int] | None = None
        self._party: list[dict[str, Any]] | None = None
        self._bridge = bridge
        self._temp_flag_end = int(self.constants.get("TEMP_FLAGS_END", 0x1F))
        self._trainer_start = int(self.constants.get("TRAINER_FLAGS_START", 0x500))

    def flag(self, name: str) -> bool | None:
        if name.startswith("TRAINER:"):
            trainer = self.constants.get(name.split(":", 1)[1])
            flag_id = None if trainer is None else self._trainer_start + int(trainer)
        else:
            flag_id = self.constants.get(name)
            if flag_id is None and name.startswith("0x"):
                flag_id = int(name, 16)
        if flag_id is None:
            return None
        if flag_id <= self._temp_flag_end and not self.current_map:
            return False
        index = int(flag_id)
        if not 0 <= index < len(self._flags) * 8:
            return None
        return bool(self._flags[index >> 3] & (1 << (index & 7)))

    def var(self, name: str) -> int | None:
        identifier = self.constants.get(name)
        if identifier is None:
            return None
        identifier = int(identifier)
        if not 0x4000 <= identifier < 0x4100:
            return None  # special variables live outside the save block
        if identifier < 0x4010 and not self.current_map:
            return 0  # VAR_TEMP_* restart at zero on entry
        offset = 2 * (identifier - 0x4000)
        return int.from_bytes(self._vars[offset:offset + 2], "little")

    def item_count(self, name: str) -> int | None:
        item = self.constants.get(name)
        if item is None:
            return None
        if self._inventory is None:
            try:
                self._inventory = {int(key): int(value) for key, value in self._bridge.get_inventory().items()}
            except (ValueError, RuntimeError):
                return None
        return int(self._inventory.get(int(item), 0))

    def party_move_slot(self, name: str) -> int | None:
        move = self.constants.get(name)
        if move is None:
            return None
        if self._party is None:
            try:
                self._party = list(self._bridge.get_decrypted_player_party())
            except (ValueError, RuntimeError):
                return None
        for index, mon in enumerate(self._party):
            if int(move) in [int(value) for value in mon.get("moves", [])]:
                return index
        return 6


def closed_gate_tiles(game: str, key: MapKey, world: GateWorld | None,
                      learned: Iterable[XY] = ()) -> frozenset[XY]:
    """Trigger tiles that would push the player back now.

    Unobservable conditions are optimistic (passable); runtime pushback
    learning closes them only after they actually block.
    """
    closed: set[XY] = set(learned)
    if world is None:
        return frozenset(closed)
    for gate in gates_by_map(game).get(key, ()):
        if gate_state(gate, world) is True:
            closed.add(gate.tile)
    return frozenset(closed)


def gate_signature(game: str, key: MapKey, world: GateWorld | None) -> tuple[XY, ...]:
    """Hashable summary of the closed gates of a map, for route caches."""
    return tuple(sorted(closed_gate_tiles(game, key, world)))


def gate_at(game: str, key: MapKey, tile: XY) -> ScriptGate | None:
    return next((gate for gate in gates_by_map(game).get(key, ()) if gate.tile == tile), None)


def unlocking_flags(gate: ScriptGate) -> tuple[str, ...]:
    """Flags whose setting satisfies some pass branch on its own (for planning)."""
    names: list[str] = []
    for conjunction in gate.pass_conditions:
        flags = [literal for literal in conjunction if literal[0] == "flag" and 1 in literal[2]]
        if len(flags) == 1 and len(conjunction) == 1:
            names.append(str(flags[0][1]))
    return tuple(dict.fromkeys(names))


def describe_gate(gate: ScriptGate) -> dict[str, Any]:
    return {"map": gate.map_name, "tile": list(gate.tile), "script": gate.script,
            "requirement": gate.requirement or "evento de historia"}


__all__ = ["ScriptGate", "RamGateWorld", "GateWorld", "analyze_gates", "analyze_trainers", "closed_gate_tiles",
           "describe_gate", "export_gates", "gate_at", "gate_signature", "gate_state", "gates_by_map", "load_gates",
           "load_trainers", "unlocking_flags"]


def _main(argv: Iterable[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Export source-derived pushback gates")
    parser.add_argument("--game", choices=("firered", "emerald", "both"), default="both")
    args = parser.parse_args(list(argv) if argv is not None else None)
    for game in ("firered", "emerald") if args.game == "both" else (args.game,):
        path = export_gates(game)
        count = len(json.loads(path.read_text(encoding="utf-8"))["gates"])
        print(f"{game}: {count} gates -> {path}")


if __name__ == "__main__":
    _main()
