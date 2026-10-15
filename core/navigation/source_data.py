"""Compile pinned pret map/event metadata into portable campaign resources.

The compiler reads official JSON, binary layouts and assembly event scripts. It
does not inject event flags into an emulator or infer progress from a target name.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMITS = {"firered": "037335f4c725d7c9aecdac87066f2002b4bd7e14",
                  "emerald": "731ad5bfd6e6f265508d0efcca0ba42f9dcf5881"}
MapKey = tuple[int, int]


def source_root(game: str) -> Path:
    if game not in SOURCE_COMMITS:
        raise ValueError("Unsupported generation-three game")
    return ROOT / "native" / "vendor" / ("pokefirered" if game == "firered" else "pokeemerald")


def _integer(expression: str, constants: dict[str, int]) -> int:
    def evaluate(node: ast.expr) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return constants[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.LShift):
                return left << right
            if isinstance(node.op, ast.BitOr):
                return left | right
        raise ValueError("Not an integer constant expression")
    return evaluate(ast.parse(expression, mode="eval").body)


def parse_constants(root: Path) -> dict[str, int]:
    """Resolve arithmetic defines to a fixed point; never execute source text."""
    pending: dict[str, str] = {}
    headers: list[str] = []
    for path in (root / "include" / "constants").glob("*.h"):
        header = path.read_text(encoding="utf-8")
        headers.append(header)
        for line in header.splitlines():
            match = re.match(r"\s*#define\s+(\w+)\s+(.+)", line)
            if match:
                pending[match[1]] = match[2].split("//")[0].split("/*")[0].strip()
    result: dict[str, int] = {}
    enums_read = False
    while pending:
        resolved: list[str] = []
        for name, expression in pending.items():
            try:
                result[name] = _integer(expression, result)
                resolved.append(name)
            except (KeyError, SyntaxError, ValueError, TypeError):
                continue
        if not resolved:
            if not enums_read:
                enums_read = True
                for header in headers:
                    header = re.sub(r"/\*.*?\*/|//[^\n]*", "", header, flags=re.DOTALL)
                    for body in re.findall(r"enum(?:\s+\w+)?\s*\{([^}]+)\}", header):
                        counter = 0
                        for entry in body.split(","):
                            match = re.fullmatch(r"\s*(\w+)\s*(?:=\s*(.+))?\s*", entry)
                            if not match:
                                continue
                            try:
                                if match[2]:
                                    counter = _integer(match[2].strip(), result)
                                result[match[1]], counter = counter, counter + 1
                            except (KeyError, SyntaxError, ValueError, TypeError):
                                continue
                continue
            break
        for name in resolved:
            del pending[name]
    # Emerald's behaviors are a simple implicit enum.
    text = (root / "include" / "constants" / "metatile_behaviors.h").read_text(encoding="utf-8")
    if "enum {" in text:
        count = 0
        for match in re.finditer(r"^\s*(MB_\w+)(?:\s*=\s*(\w+))?\s*,", text, re.MULTILINE):
            if match[2]:
                count = _integer(match[2], result)
            result[match[1]], count = count, count + 1
    return result


@dataclass(frozen=True)
class ScriptEffect:
    set_flags: frozenset[str]
    clear_flags: frozenset[str]
    set_vars: tuple[tuple[str, str], ...]
    items: frozenset[str]
    references: frozenset[str]


def parse_scripts(root: Path) -> dict[str, ScriptEffect]:
    blocks: dict[str, list[str]] = {}
    for base in (root / "data" / "maps", root / "data" / "scripts"):
        for path in base.rglob("*.inc"):
            label: str | None = None
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.split("@")[0].strip()
                match = re.match(r"^(\w+)::?", line)
                if match:
                    label = match[1]
                    blocks.setdefault(label, [])
                elif label:
                    blocks[label].append(line)
    result: dict[str, ScriptEffect] = {}
    for label, lines in blocks.items():
        text = "\n".join(lines)
        references = frozenset(token for token in re.findall(r"\b\w+\b", text)
                               if token in blocks and token != label)
        result[label] = ScriptEffect(
            frozenset(re.findall(r"\bsetflag\s+(FLAG_\w+)", text)),
            frozenset(re.findall(r"\bclearflag\s+(FLAG_\w+)", text)),
            tuple(re.findall(r"\bsetvar\s+(VAR_\w+)\s*,\s*(\w+)", text)),
            frozenset(re.findall(r"\b(?:giveitem|giveitem_msg|finditem|givemeitem)\s+(?:\w+\s*,\s*)?(ITEM_\w+)", text)),
            references)
    return result


class SourceMapCatalog:
    """Complete map IDs, event targets and script-derived completion evidence."""

    def __init__(self, game: str, use_export: bool = True) -> None:
        self.game, self.root = game, source_root(game)
        self._grids: dict[str, str] = {}
        self._behaviors: dict[str, str] = {}
        self._effects_cache: dict[str, ScriptEffect] = {}
        self.layouts: dict[str, dict[str, Any]]
        self.maps: dict[MapKey, dict[str, Any]]
        self.by_name: dict[str, MapKey]
        self.by_id: dict[str, MapKey]
        self.land_encounters: dict[str, dict[str, Any]]
        exported = ROOT / "data" / f"maps_{game}.json"
        if use_export and exported.exists():
            resource = json.loads(exported.read_text(encoding="utf-8"))
            if resource["game"] != game or resource["source_commit"] != SOURCE_COMMITS[game]:
                raise ValueError("Campaign resource does not match pinned game source")
            self.constants = {str(k): int(v) for k, v in resource["constants"].items()}
            self.scripts = {str(label): ScriptEffect(frozenset(effect["set_flags"]), frozenset(effect["clear_flags"]),
                                                   tuple((str(k), str(v)) for k, v in effect["set_vars"]),
                                                   frozenset(effect["items"]), frozenset(effect["references"]))
                            for label, effect in resource["scripts"].items()}
            self.layouts = resource["layouts"]
            self.maps = {(int(raw["map_group"]), int(raw["map_num"])): raw for raw in resource["maps"]}
            self.by_name = {str(raw["name"]): key for key, raw in self.maps.items()}
            self.by_id = {str(raw["id"]): key for key, raw in self.maps.items()}
            self._grids, self._behaviors = resource["grids"], resource["behaviors"]
            self.land_encounters = resource["land_encounters"]
            return
        self.constants = parse_constants(self.root)
        self.scripts = parse_scripts(self.root)
        raw_groups = json.loads((self.root / "data" / "maps" / "map_groups.json").read_text(encoding="utf-8"))
        layouts_data = json.loads((self.root / "data" / "layouts" / "layouts.json").read_text(encoding="utf-8"))
        self.layouts = {str(entry["id"]): entry for entry in layouts_data["layouts"] if "id" in entry}
        self.maps = {}
        self.by_name = {}
        self.by_id = {}
        for group, group_name in enumerate(raw_groups["group_order"]):
            for number, name in enumerate(raw_groups[group_name]):
                raw: dict[str, Any] = json.loads((self.root / "data" / "maps" / name / "map.json").read_text(encoding="utf-8"))
                layout = self.layouts[str(raw["layout"])]
                raw.update(map_group=group, map_num=number, width=int(layout["width"]), height=int(layout["height"]))
                for events_key in ("warp_events", "object_events", "coord_events", "bg_events"):
                    raw.setdefault(events_key, [])
                self.maps[group, number], self.by_name[name], self.by_id[str(raw["id"])] = raw, (group, number), (group, number)
        self.land_encounters = {}
        encounters = json.loads((self.root / "src" / "data" / "wild_encounters.json").read_text(encoding="utf-8"))
        for encounter_group in encounters["wild_encounter_groups"]:
            if not encounter_group.get("for_maps"):
                continue
            weights = next(entry["encounter_rates"] for entry in encounter_group["fields"] if entry["type"] == "land_mons")
            for encounter in encounter_group["encounters"]:
                if "land_mons" not in encounter or (game == "firered" and "LeafGreen" in str(encounter["base_label"])):
                    continue
                land = dict(encounter["land_mons"])
                land["weights"] = weights
                self.land_encounters[str(encounter["map"])] = land

    def effects(self, script: str) -> ScriptEffect:
        cached = self._effects_cache.get(script)
        if cached is not None:
            return cached
        seen: set[str] = set()
        frontier = [script]
        flags: set[str] = set()
        clear: set[str] = set()
        variables: list[tuple[str, str]] = []
        items: set[str] = set()
        while frontier:
            label = frontier.pop()
            if label in seen or label not in self.scripts:
                continue
            seen.add(label)
            effect = self.scripts[label]
            flags.update(effect.set_flags)
            clear.update(effect.clear_flags)
            variables.extend(effect.set_vars)
            items.update(effect.items)
            frontier.extend(effect.references)
        result = ScriptEffect(frozenset(flags), frozenset(clear), tuple(variables), frozenset(items), frozenset(seen))
        self._effects_cache[script] = result
        return result

    def targets(self, map_name: str) -> list[dict[str, Any]]:
        raw = self.maps[self.by_name[map_name]]
        targets: list[dict[str, Any]] = []
        for key, action in (("object_events", "interact"), ("coord_events", "trigger"), ("bg_events", "interact")):
            for index, entry in enumerate(raw.get(key, [])):
                if str(entry.get("script", "0")) in ("0", "0x0"):
                    continue
                local_id = index + 1 if key == "object_events" else 0
                targets.append(dict(entry, action=action, npc_local_id=local_id, map_group=raw["map_group"],
                                    map_num=raw["map_num"], map_name=map_name))
        return targets

    def find_target(self, map_name: str, script_pattern: str | None = None,
                    effect_flag: str | None = None) -> dict[str, Any]:
        targets = self.targets(map_name)
        if script_pattern:
            targets = [target for target in targets if re.search(script_pattern, str(target["script"]), re.IGNORECASE)]
        if effect_flag:
            targets = [target for target in targets if effect_flag in self.effects(str(target["script"])).set_flags]
        if not targets:
            raise ValueError(f"No source-backed {map_name} event matching {script_pattern!r}/{effect_flag!r}")
        # Object interactions are preferred to incidental trigger duplicates.
        return sorted(targets, key=lambda target: (target["action"] != "interact", target["x"], target["y"]))[0]

    @lru_cache(maxsize=1024)
    def grid(self, map_key: MapKey) -> NDArray[np.uint16]:
        raw = self.maps[map_key]
        layout = self.layouts[str(raw["layout"])]
        image = (bytes.fromhex(self._grids[str(raw["name"])]) if self._grids else
                 (self.root / str(layout["blockdata_filepath"])).read_bytes())
        expected = int(raw["width"]) * int(raw["height"]) * 2
        if len(image) != expected:
            raise ValueError("Pinned source layout dimensions do not match binary")
        return np.frombuffer(image, dtype="<u2").reshape(int(raw["height"]), int(raw["width"])).copy()

    @lru_cache(maxsize=1024)
    def behavior_grid(self, map_key: MapKey) -> NDArray[np.uint16]:
        """Decode each tileset's attribute table; tile IDs are never behaviors."""
        raw = self.maps[map_key]
        if self._behaviors:
            image = bytes.fromhex(self._behaviors[str(raw["name"])])
            return np.frombuffer(image, dtype="<u2").reshape(int(raw["height"]), int(raw["width"])).copy()
        layout = self.layouts[str(raw["layout"])]
        headers = (self.root / "src" / "data" / "tilesets" / "headers.h").read_text(encoding="utf-8")
        metatiles = (self.root / "src" / "data" / "tilesets" / "metatiles.h").read_text(encoding="utf-8")
        tile_ids = self.grid(map_key) & 0x3FF
        result = np.zeros_like(tile_ids)
        cutoff = 640 if self.game == "firered" else 512
        for property_name, lower, upper in (("primary_tileset", 0, cutoff), ("secondary_tileset", cutoff, 1024)):
            name = str(layout[property_name])
            definition = re.search(r"\b" + re.escape(name) + r"\s*=\s*\{(.*?)\};", headers, re.DOTALL)
            if definition is None:
                raise ValueError(f"Missing source tileset {name}")
            attribute = re.search(r"\.metatileAttributes\s*=\s*(\w+)", definition[1])
            if attribute is None:
                raise ValueError("Tileset lacks an attribute table")
            path = re.search(r"\b" + re.escape(attribute[1]) + r'\[\]\s*=\s*INCBIN_U(?:16|32)\("([^"]+)"\)', metatiles)
            if path is None:
                raise ValueError("Attribute table source missing")
            data = np.frombuffer((self.root / path[1]).read_bytes(), dtype="<u4" if self.game == "firered" else "<u2")
            mask = (tile_ids >= lower) & (tile_ids < upper)
            indices = tile_ids[mask] - lower
            if indices.size and int(indices.max()) >= len(data):
                raise ValueError("Map references an absent tileset attribute")
            result[mask] = data[indices] & (0x1FF if self.game == "firered" else 0xFF)
        return result

    def grass_tiles(self, map_key: MapKey) -> list[tuple[int, int]]:
        behavior = self.behavior_grid(map_key)
        locations = np.argwhere((behavior == 2) | (behavior == 3))
        return [(int(x), int(y)) for y, x in locations]

    def export(self, path: Path) -> None:
        """Offline, reproducible portable resource with original source labels."""
        data = {"game": self.game, "source_commit": SOURCE_COMMITS[self.game],
                "constants": self.constants, "maps": list(self.maps.values()),
                "layouts": self.layouts,
                "land_encounters": self.land_encounters,
                "grids": {str(raw["name"]): self.grid(key).tobytes().hex() for key, raw in self.maps.items()},
                "behaviors": {str(raw["name"]): self.behavior_grid(key).tobytes().hex() for key, raw in self.maps.items()},
                "scripts": {label: {"set_flags": sorted(effect.set_flags), "clear_flags": sorted(effect.clear_flags),
                                    "set_vars": list(effect.set_vars), "items": sorted(effect.items),
                                    "references": sorted(effect.references)} for label, effect in self.scripts.items()}}
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


@lru_cache(maxsize=2)
def load_map_catalog(game: str) -> SourceMapCatalog:
    return SourceMapCatalog(game)

