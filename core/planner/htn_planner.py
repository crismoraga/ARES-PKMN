"""Readonly, source-backed FireRed and Emerald campaign task decomposition.

Targets are the official map/script events. Persistent flags, variables and bag
items certify tasks; merely reaching an NPC never counts as completing a boss.
Campaign planning does not write RAM. Synthetic callers may supply an explicit
small DAG without borrowing the real games' completion conditions.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from core.bridge.ffi_mgba import GBABridgeProtocol
from core.game_data import Game, RomCatalog
from core.navigation.source_data import ROOT, MapKey, SourceMapCatalog, load_map_catalog
from core.navigation.constrained_astar import GbaConstrainedAStar, Matrix
from core.navigation.puzzles import plan_sootopolis_ice_commands, plan_strength_commands
from core.navigation.warp_topology import HierarchicalWarpGraph


@dataclass(frozen=True)
class CompletionPredicate:
    kind: Literal["flag", "var_at_least", "item", "party_species", "map"]
    identifier: int
    value: int = 1

    def satisfied(self, bridge: GBABridgeProtocol, party: list[dict[str, Any]]) -> bool:
        if self.kind == "flag":
            return bridge.get_event_flag(self.identifier) == bool(self.value)
        if self.kind == "var_at_least":
            if not 0x4000 <= self.identifier < 0x4100:
                raise ValueError("Only persistent game variables are valid campaign evidence")
            address = bridge.get_save_block1_dynamic_ptr() + bridge.memory_map.vars_offset + 2 * (self.identifier - 0x4000)
            return bridge.read_u16(address) >= self.value
        if self.kind == "item":
            return bridge.get_inventory().get(self.identifier, 0) >= self.value
        if self.kind == "party_species":
            return any(int(mon["species"]) == self.identifier for mon in party)
        coords = bridge.get_player_coords_and_map()
        return (coords["map_group"], coords["map_num"]) == (self.identifier, self.value)


@dataclass(frozen=True)
class CampaignMilestone:
    # The first eight fields preserve the PDF's public construction API.
    milestone_id: str
    description: str
    completion_flag_id: int
    target_map_group: int
    target_map_num: int
    target_coords: tuple[int, int]
    required_avg_level: int
    required_types: list[int]
    dependencies: tuple[str, ...] = ()
    action: str = "interact"
    script: str = ""
    npc_local_id: int = 0
    completion_conditions: tuple[CompletionPredicate, ...] = ()
    required_items: tuple[int, ...] = ()
    required_moves: tuple[int, ...] = ()
    map_name: str = ""
    flag_name: str = ""
    puzzle: str = ""
    menu_choice: int = 0

    @property
    def map_key(self) -> MapKey:
        return self.target_map_group, self.target_map_num

    def complete(self, bridge: GBABridgeProtocol, party: list[dict[str, Any]]) -> bool:
        if self.completion_flag_id >= 0 and not bridge.get_event_flag(self.completion_flag_id):
            return False
        if self.completion_flag_id < 0 and not self.completion_conditions:
            raise ValueError(f"Task {self.milestone_id} has no completion evidence")
        return all(condition.satisfied(bridge, party) for condition in self.completion_conditions)


class _CampaignBuilder:
    def __init__(self, game: str) -> None:
        self.catalog = load_map_catalog(game)
        self.nodes: list[CampaignMilestone] = []

    def task(self, name: str, map_name: str, pattern: str, flag: str = "", level: int = 0,
             types: tuple[int, ...] = (), dependencies: tuple[str, ...] | None = None,
             variable: tuple[str, int] | None = None, item: str = "",
             moves: tuple[int, ...] = (), puzzle: str = "", action: str | None = None,
             choice: int = 0, trainer: str = "") -> None:
        catalog = self.catalog
        candidates = [target for target in catalog.targets(map_name)
                      if re.search(pattern, str(target["script"]), re.IGNORECASE)]
        if candidates:
            # Keep original ordering: the first alternate event is canonical.
            target = candidates[0]
        else:
            # Map-script scenes (Champion/HOF) trigger automatically on entering.
            matches = [label for label in catalog.scripts if label.startswith(map_name + "_") and
                       re.search(pattern, label, re.IGNORECASE)]
            if not matches:
                raise ValueError(f"Campaign target {name} absent in pinned source {map_name}/{pattern}")
            raw = catalog.maps[catalog.by_name[map_name]]
            warps = raw["warp_events"]
            if not warps:
                raise ValueError(f"Automatic scene {name} needs an entrance event")
            target = {"x": warps[0]["x"], "y": max(0, int(warps[0]["y"]) - 1),
                      "script": matches[0], "action": "trigger", "npc_local_id": 0}
        key = catalog.by_name[map_name]
        conditions: list[CompletionPredicate] = []
        if variable:
            conditions.append(CompletionPredicate("var_at_least", catalog.constants[variable[0]], variable[1]))
        if item:
            conditions.append(CompletionPredicate("item", catalog.constants[item]))
        flag_id = catalog.constants[flag] if flag else -1
        if trainer:
            flag_id = catalog.constants["TRAINER_FLAGS_START"] + catalog.constants[trainer]
            flag = "TRAINER_FLAGS_START + " + trainer
        dependencies = dependencies if dependencies is not None else ((self.nodes[-1].milestone_id,) if self.nodes else ())
        node = CampaignMilestone(name, name.replace("_", " "), flag_id, *key,
                                 (int(target["x"]), int(target["y"])), level, list(types), dependencies,
                                 action or str(target["action"]), str(target["script"]), int(target["npc_local_id"]),
                                 tuple(conditions), (), moves, map_name, flag, puzzle, choice)
        self.nodes.append(node)


def _firered_campaign() -> list[CampaignMilestone]:
    b = _CampaignBuilder("firered")
    t = b.task
    t("OAK_ROUTE1_TRIGGER", "PalletTown", "OakTriggerLeft", variable=("VAR_MAP_SCENE_PALLET_TOWN_OAK", 1))
    t("STARTER_BULBASAUR", "PalletTown_ProfessorOaksLab", "BulbasaurBall", "FLAG_SYS_POKEMON_GET")
    t("RIVAL_LAB", "PalletTown_ProfessorOaksLab", "RivalBattleTriggerMid", variable=("VAR_MAP_SCENE_PALLET_TOWN_PROFESSOR_OAKS_LAB", 4), level=5)
    t("OAK_PARCEL_PICKUP", "ViridianCity_Mart", "Clerk", variable=("VAR_MAP_SCENE_VIRIDIAN_CITY_MART", 1))
    t("OAK_PARCEL", "PalletTown_ProfessorOaksLab", "ProfOak", "FLAG_SYS_POKEDEX_GET")
    t("GYM_1_BROCK", "PewterCity_Gym", "Brock$", "FLAG_BADGE01_GET", 14, (11, 12))
    t("MT_MOON_FOSSIL_GUARD", "MtMoon_B2F", "Miguel$", level=18, trainer="TRAINER_SUPER_NERD_MIGUEL")
    t("MOUNT_MOON", "MtMoon_B2F", "HelixFossil$", "FLAG_GOT_FOSSIL_FROM_MT_MOON", 18)
    t("CERULEAN_RIVAL", "CeruleanCity", "RivalTriggerMid", variable=("VAR_MAP_SCENE_CERULEAN_CITY_RIVAL", 1), level=20)
    t("BILL_RESCUE", "Route25_SeaCottage", "Computer$", "FLAG_HELPED_BILL_IN_SEA_COTTAGE", puzzle="bill_teleporter")
    t("SS_TICKET", "Route25_SeaCottage", "Bill$", "FLAG_GOT_SS_TICKET")
    t("GYM_2_MISTY", "CeruleanCity_Gym", "Misty$", "FLAG_BADGE02_GET", 23, (12, 13))
    t("SS_ANNE_CUT", "SSAnne_CaptainsOffice", "Captain$", "FLAG_GOT_HM01", 24)
    t("SURGE_SWITCHES", "VermilionCity_Gym", "TrashCan", "FLAG_FOUND_BOTH_VERMILION_GYM_SWITCHES", moves=(15,), puzzle="vermilion_switches")
    t("GYM_3_SURGE", "VermilionCity_Gym", "Surge$", "FLAG_BADGE03_GET", 28, (4,), moves=(15,))
    t("CELADON_TEA", "CeladonCity_Condominiums_1F", "TeaWoman$", "FLAG_GOT_TEA", 29)
    t("ROCKET_POSTER_GUARD", "CeladonCity_GameCorner", "RocketGrunt$", level=30, trainer="TRAINER_TEAM_ROCKET_GRUNT_7")
    t("ROCKET_POSTER", "CeladonCity_GameCorner", "Poster", "FLAG_OPENED_ROCKET_HIDEOUT", 30)
    t("ROCKET_KEY_GUARD", "RocketHideout_B4F", "Grunt1$", level=32, trainer="TRAINER_TEAM_ROCKET_GRUNT_18", puzzle="rocket_spinners")
    t("ROCKET_LIFT_KEY", "RocketHideout_B4F", "LiftKey$", "FLAG_CAN_USE_ROCKET_HIDEOUT_LIFT", 32, puzzle="rocket_spinners")
    t("ROCKET_BOSS_GUARD_LEFT", "RocketHideout_B4F", "Grunt2$", level=33, trainer="TRAINER_TEAM_ROCKET_GRUNT_16")
    t("ROCKET_BOSS_GUARD_RIGHT", "RocketHideout_B4F", "Grunt3$", level=33, trainer="TRAINER_TEAM_ROCKET_GRUNT_17")
    t("ROCKET_GIOVANNI", "RocketHideout_B4F", "Giovanni$", level=34, trainer="TRAINER_BOSS_GIOVANNI")
    t("SILPH_SCOPE", "RocketHideout_B4F", "SilphScope$", item="ITEM_SILPH_SCOPE")
    t("GYM_4_ERIKA", "CeladonCity_Gym", "Erika$", "FLAG_BADGE04_GET", 34, (10, 2), moves=(15,))
    t("FUJI_RESCUE", "PokemonTower_7F", "Fuji$", "FLAG_RESCUED_MR_FUJI", 36)
    t("POKE_FLUTE", "LavenderTown_VolunteerPokemonHouse", "Fuji$", "FLAG_GOT_POKE_FLUTE")
    t("SAFARI_GOLD_TEETH", "SafariZone_West", "GoldTeeth", "FLAG_HIDE_SAFARI_ZONE_WEST_GOLD_TEETH", puzzle="safari_steps")
    t("HM_SURF", "SafariZone_SecretHouse", "Attendant$", "FLAG_GOT_HM03")
    t("HM_STRENGTH", "FuchsiaCity_WardensHouse", "Warden$", "FLAG_GOT_HM04")
    t("GYM_5_KOGA", "FuchsiaCity_Gym", "Koga$", "FLAG_BADGE05_GET", 43, (14, 4), puzzle="koga_invisible_walls")
    t("SILPH_CARD_KEY", "SilphCo_5F", "CardKey", "FLAG_HIDE_SILPH_CO_5F_CARD_KEY", 43)
    t("SILPH_GIOVANNI", "SilphCo_11F", "GiovanniTriggerLeft", "FLAG_HIDE_SAFFRON_ROCKETS", 45, puzzle="silph_teleporters")
    t("SILPH_CO", "SilphCo_11F", "President$", "FLAG_GOT_MASTER_BALL_FROM_SILPH")
    t("GYM_6_SABRINA", "SaffronCity_Gym", "Sabrina$", "FLAG_BADGE06_GET", 46, (6, 7, 17), puzzle="saffron_teleporters")
    t("MANSION_SECRET_KEY", "PokemonMansion_B1F", "SecretKey", "FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY", 47, moves=(57,), puzzle="mansion_statues")
    t("GYM_7_BLAINE", "CinnabarIsland_Gym", "Blaine$", "FLAG_BADGE07_GET", 49, (11, 4, 5), puzzle="cinnabar_quizzes")
    t("GYM_8_GIOVANNI", "ViridianCity_Gym", "Giovanni$", "FLAG_BADGE08_GET", 54, (11, 12, 15), puzzle="viridian_spinners")
    t("ELITE_LORELEI", "PokemonLeague_LoreleisRoom", "Lorelei$", "FLAG_DEFEATED_LORELEI", 62, (13, 6), moves=(57, 70), puzzle="victory_road_boulders")
    t("ELITE_BRUNO", "PokemonLeague_BrunosRoom", "Bruno$", "FLAG_DEFEATED_BRUNO", 62, (11, 14))
    t("ELITE_AGATHA", "PokemonLeague_AgathasRoom", "Agatha$", "FLAG_DEFEATED_AGATHA", 62, (14, 4))
    t("ELITE_LANCE", "PokemonLeague_LancesRoom", "Lance$", "FLAG_DEFEATED_LANCE", 64, (15, 13))
    t("CHAMPION", "PokemonLeague_ChampionsRoom", "EventScript_EnterRoom$", "FLAG_DEFEATED_CHAMP", 65, (11, 15, 13, 14))
    t("ELITE_FOUR", "PokemonLeague_HallOfFame", "EventScript_EnterRoom$", "FLAG_SYS_GAME_CLEAR", action="trigger")
    return b.nodes


def _emerald_campaign() -> list[CampaignMilestone]:
    b = _CampaignBuilder("emerald")
    t = b.task
    t("SET_CLOCK", "LittlerootTown_BrendansHouse_2F", "WallClock$", "FLAG_SET_WALL_CLOCK", puzzle="wall_clock")
    t("WATCH_TV_REPORT", "LittlerootTown_BrendansHouse_1F", "PlayersHouse_1F_EventScript_Mom$", variable=("VAR_LITTLEROOT_INTRO_STATE", 7))
    t("MEET_RIVAL", "LittlerootTown_MaysHouse_2F", "RivalsPokeBall$", variable=("VAR_LITTLEROOT_RIVAL_STATE", 3))
    t("BIRCH_RESCUE_TRIGGER", "Route101", "StartBirchRescue$", variable=("VAR_ROUTE101_STATE", 2))
    t("STARTER_MUDKIP", "Route101", "BirchsBag$", "FLAG_RESCUED_BIRCH", choice=2)
    t("BIRCH_LAB_STARTER", "LittlerootTown_ProfessorBirchsLab", "EventScript_Birch$", "FLAG_SYS_POKEMON_GET")
    t("RIVAL_ROUTE103", "Route103", "Rival", "FLAG_DEFEATED_RIVAL_ROUTE103", 7)
    t("POKEDEX", "LittlerootTown_ProfessorBirchsLab", "EventScript_Birch$", "FLAG_SYS_POKEDEX_GET")
    t("WALLY_TUTORIAL", "PetalburgCity_Gym", "Norman$", variable=("VAR_PETALBURG_GYM_STATE", 2))
    t("PETALBURG_WOODS_AQUA", "PetalburgWoods", "DevonResearcherLeft", variable=("VAR_PETALBURG_WOODS_STATE", 1), level=9)
    t("GYM_1_ROXANNE", "RustboroCity_Gym", "Roxanne$", "FLAG_BADGE01_GET", 16, (11, 12))
    t("DEVON_GOODS", "RusturfTunnel", "EventScript_Grunt$", "FLAG_RECOVERED_DEVON_GOODS", 17)
    t("DEVON_RETURN", "RustboroCity", "DevonEmployee1", "FLAG_RETURNED_DEVON_GOODS")
    t("DEVON_POKENAV", "RustboroCity_DevonCorp_3F", "MrStone$", "FLAG_RECEIVED_POKENAV")
    t("HM_CUT", "RustboroCity_CuttersHouse", "Cutter", "FLAG_RECEIVED_HM_CUT")
    t("STEVEN_LETTER", "GraniteCave_StevensRoom", "Steven$", "FLAG_DELIVERED_STEVEN_LETTER")
    t("GYM_2_BRAWLY", "DewfordTown_Gym", "Brawly$", "FLAG_BADGE02_GET", 20, (2, 14))
    t("STERN_GOODS", "SlateportCity_OceanicMuseum_2F", "Stern", "FLAG_DELIVERED_DEVON_GOODS", 21)
    t("RIVAL_ROUTE110", "Route110", "RivalTrigger", "FLAG_HIDE_ROUTE_110_RIVAL", 22)
    t("WALLY_MAUVILLE", "MauvilleCity", "Wally", "FLAG_DEFEATED_WALLY_MAUVILLE", 23)
    t("GYM_3_WATTSON", "MauvilleCity_Gym", "Wattson$", "FLAG_BADGE03_GET", 25, (4,), puzzle="mauville_switches")
    t("HM_ROCK_SMASH", "MauvilleCity_House1", "RockSmashDude$", "FLAG_RECEIVED_HM_ROCK_SMASH")
    t("METEOR_FALLS_TEAMS", "MeteorFalls_1F_1R", "MagmaStealsMeteoriteScene$", variable=("VAR_METEOR_FALLS_STATE", 1), moves=(249,))
    t("MT_CHIMNEY", "MtChimney", "Maxie$", "FLAG_DEFEATED_EVIL_TEAM_MT_CHIMNEY", 28)
    t("GYM_4_FLANNERY", "LavaridgeTown_Gym_1F", "Flannery$", "FLAG_BADGE04_GET", 31, (11, 4), puzzle="lavaridge_holes")
    t("HM_STRENGTH", "RusturfTunnel", "^EventScript_RockSmash$", "FLAG_RECEIVED_HM_STRENGTH", moves=(249,))
    t("PETALBURG_RANDALL", "PetalburgCity_Gym", "Randall$", level=32, trainer="TRAINER_RANDALL", puzzle="petalburg_rooms")
    t("PETALBURG_PARKER", "PetalburgCity_Gym", "Parker$", level=32, trainer="TRAINER_PARKER", puzzle="petalburg_rooms")
    t("PETALBURG_JODY", "PetalburgCity_Gym", "Jody$", level=33, trainer="TRAINER_JODY", puzzle="petalburg_rooms")
    t("GYM_5_NORMAN", "PetalburgCity_Gym", "Norman$", "FLAG_BADGE05_GET", 33, (1,), puzzle="petalburg_rooms")
    t("HM_SURF", "PetalburgCity_WallysHouse", "Dad|Father", "FLAG_RECEIVED_HM_SURF")
    t("WEATHER_INSTITUTE", "Route119_WeatherInstitute_2F", "Aqua|Shelly", "FLAG_HIDE_ROUTE_119_TEAM_AQUA", 35, moves=(57,))
    t("HM_FLY", "Route119", "RivalTrigger", "FLAG_RECEIVED_HM_FLY", 36)
    t("DEVON_SCOPE", "Route120", "Steven$", "FLAG_RECEIVED_DEVON_SCOPE", 36)
    t("GYM_6_WINONA", "FortreeCity_Gym", "Winona$", "FLAG_BADGE06_GET", 38, (13, 15), puzzle="fortree_turnstiles")
    t("MT_PYRE_EMBLEM", "MtPyre_Summit", "OldWoman|OldLady|ArchieMaxieTrigger0", item="ITEM_MAGMA_EMBLEM", puzzle="mt_pyre_team_scene")
    t("MAGMA_HIDEOUT", "MagmaHideout_4F", "Maxie$", "FLAG_GROUDON_AWAKENED_MAGMA_HIDEOUT", 40, moves=(70, 249))
    t("SLATEPORT_SUBMARINE", "SlateportCity_Harbor", "Stern|Captain|Archie", "FLAG_TEAM_AQUA_ESCAPED_IN_SUBMARINE")
    t("AQUA_HIDEOUT", "AquaHideout_B2F", "Grunt|Matt", "FLAG_HIDE_AQUA_HIDEOUT_GRUNTS", 42, puzzle="aqua_teleporters")
    t("GYM_7_TATE_LIZA", "MossdeepCity_Gym", "TateAndLiza$", "FLAG_BADGE07_GET", 43, (11, 7, 17), puzzle="mossdeep_arrows")
    t("SPACE_CENTER", "MossdeepCity_SpaceCenter_2F", "Steven$", "FLAG_DEFEATED_MAGMA_SPACE_CENTER", 44)
    t("HM_DIVE", "MossdeepCity_StevensHouse", "Steven$", "FLAG_RECEIVED_HM_DIVE")
    t("SEAFLOOR_CAVERN", "SeafloorCavern_Room9", "Archie|Kyogre|Trigger", "FLAG_HIDE_SEAFLOOR_CAVERN_AQUA_GRUNTS", 45, moves=(57, 70, 249, 291), puzzle="seafloor_boulders")
    t("WALLACE_ORIGIN", "CaveOfOrigin_B1F", "Wallace$", "FLAG_WALLACE_GOES_TO_SKY_PILLAR", choice=2)
    t("RAYQUAZA_AWAKEN", "SkyPillar_Top", "AwakenRayquaza$", variable=("VAR_SKY_PILLAR_STATE", 1), puzzle="sky_pillar_cracked_floor")
    t("SOOTOPOLIS_MAXIE", "SootopolisCity", "Maxie$", "FLAG_MET_MAXIE_SOOTOPOLIS")
    t("SOOTOPOLIS_TRUCE", "SootopolisCity", "Archie$", "FLAG_SOOTOPOLIS_ARCHIE_MAXIE_LEAVE")
    t("HM_WATERFALL", "SootopolisCity", "Wallace$", "FLAG_RECEIVED_HM_WATERFALL")
    t("GYM_8_JUAN", "SootopolisCity_Gym_1F", "Juan$", "FLAG_BADGE08_GET", 47, (12, 13), puzzle="sootopolis_ice")
    t("VICTORY_ROAD_WALLY", "VictoryRoad_1F", "Wally", "FLAG_DEFEATED_WALLY_VICTORY_ROAD", 52, moves=(57, 70, 249, 127), puzzle="victory_road_boulders")
    t("ELITE_SIDNEY", "EverGrandeCity_SidneysRoom", "Sidney$", "FLAG_DEFEATED_ELITE_4_SIDNEY", 57, (1, 6))
    t("ELITE_PHOEBE", "EverGrandeCity_PhoebesRoom", "Phoebe$", "FLAG_DEFEATED_ELITE_4_PHOEBE", 57, (7, 17))
    t("ELITE_GLACIA", "EverGrandeCity_GlaciasRoom", "Glacia$", "FLAG_DEFEATED_ELITE_4_GLACIA", 58, (1, 13))
    t("ELITE_DRAKE", "EverGrandeCity_DrakesRoom", "Drake$", "FLAG_DEFEATED_ELITE_4_DRAKE", 60, (15, 16))
    t("CHAMPION", "EverGrandeCity_ChampionsRoom", "EventScript_EnterRoom$", level=62, types=(12, 13), trainer="TRAINER_WALLACE")
    # Game clear is set by the Hall of Fame routine, not merely a victory screen.
    t("ELITE_FOUR", "EverGrandeCity_HallOfFame", "EventScript_EnterHallOfFame$", "FLAG_SYS_GAME_CLEAR", action="trigger")
    return b.nodes


class HTNCampaignPlanner:
    FIRERED_CAMPAIGN_DAG: list[CampaignMilestone] = []
    EMERALD_CAMPAIGN_DAG: list[CampaignMilestone] = []

    def __init__(self, is_emerald: bool = False, milestones: Sequence[CampaignMilestone] | None = None,
                 graph: HierarchicalWarpGraph | None = None) -> None:
        self.game = "emerald" if is_emerald else "firered"
        self.custom_campaign = milestones is not None
        self.catalog: SourceMapCatalog | None = None if self.custom_campaign else load_map_catalog(self.game)
        self.milestones = list(milestones) if milestones is not None else (_emerald_campaign() if is_emerald else _firered_campaign())
        if not self.custom_campaign:
            if is_emerald:
                self.EMERALD_CAMPAIGN_DAG = self.milestones
            else:
                self.FIRERED_CAMPAIGN_DAG = self.milestones
        self.graph = graph or (HierarchicalWarpGraph() if self.custom_campaign else HierarchicalWarpGraph.from_source(self.game))
        self.maps = self.catalog.maps if self.catalog else {}
        self._rom_catalog: RomCatalog | None = None
        self._validate_dag()

    @property
    def campaign_dag(self) -> tuple[CampaignMilestone, ...]:
        return tuple(self.milestones)

    def _validate_dag(self) -> None:
        seen: set[str] = set()
        for task in self.milestones:
            if task.milestone_id in seen or any(parent not in seen for parent in task.dependencies):
                raise ValueError("Campaign is not a unique topologically ordered DAG")
            if task.completion_flag_id < 0 and not task.completion_conditions:
                raise ValueError("Campaign requires observable completion evidence")
            seen.add(task.milestone_id)

    def completed_milestones(self, bridge: GBABridgeProtocol, party: list[dict[str, Any]]) -> frozenset[str]:
        completed: set[str] = set()
        for task in self.milestones:
            if all(parent in completed for parent in task.dependencies) and task.complete(bridge, party):
                completed.add(task.milestone_id)
        return frozenset(completed)

    def _healing_directive(self, bridge: GBABridgeProtocol, hp: int, max_hp: int,
                           completed: frozenset[str]) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": "HEAL_POKECENTER", "reason": f"Party health {hp}/{max_hp}", "action": "interact"}
        if self.catalog is None:
            return result
        coords = bridge.get_player_coords_and_map()
        current = (coords["map_group"], coords["map_num"])
        candidates: list[tuple[float, dict[str, Any]]] = []
        for key, raw in self.catalog.maps.items():
            if "PokemonCenter_1F" not in str(raw["name"]):
                continue
            nurse = [target for target in self.catalog.targets(str(raw["name"])) if "Nurse" in str(target["script"])]
            if not nurse:
                continue
            route = self.graph.find_route(current, key, completed, (coords["x"], coords["y"]))
            if route is not None:
                candidates.append((sum(edge.cost for edge in route), nurse[0]))
        if candidates:
            target = min(candidates, key=lambda value: value[0])[1]
            result.update(dest=(target["map_group"], target["map_num"], (target["x"], target["y"])),
                          script=target["script"], npc_local_id=target["npc_local_id"])
        return result

    def _training_directive(self, bridge: GBABridgeProtocol, task: CampaignMilestone,
                             completed: frozenset[str], party: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": "OPTIMAL_TRAINING_CURRICULUM", "target_level": task.required_avg_level,
                                  "milestone": task.milestone_id, "required_types": task.required_types,
                                  "dest": (*task.map_key, task.target_coords)}
        if self.catalog is None:
            return result
        coords = bridge.get_player_coords_and_map()
        current = (coords["map_group"], coords["map_num"])
        candidates: list[tuple[float, MapKey, tuple[int, int]]] = []
        for key, raw in self.catalog.maps.items():
            if raw["map_type"] not in ("MAP_TYPE_ROUTE", "MAP_TYPE_TOWN", "MAP_TYPE_CITY"):
                continue
            if key != current and any(name not in completed for name in self.training_map_prerequisites(str(raw["name"]))):
                continue
            grass = self.catalog.grass_tiles(key)
            packed = self.catalog.grid(key)
            grass = [point for point in grass if int(packed[point[1], point[0]]) >> 10 & 3 == 0]
            if not grass:
                continue
            if key == current:
                # Prefer another grass square, so grinding actually generates steps.
                grass = [point for point in grass if point != (coords["x"], coords["y"])]
                if not grass:
                    continue
                _, live_collision, live_elevation = bridge.extract_live_map_matrices()
                local = GbaConstrainedAStar(live_collision, live_elevation, bridge.extract_behavior_map(), game=self.game)
                local.register_object_events(bridge.get_object_events(), bridge.read_u8(bridge.memory_map.avatar + 5))
                grass.sort(key=lambda xy: abs(xy[0] - coords["x"]) + abs(xy[1] - coords["y"]))
                point = next((xy for xy in grass if local.find_optimal_path((coords["x"], coords["y"], coords["z"]), xy) is not None), None)
                if point is None:
                    continue
                candidates.append((0, key, point))
                continue
            route = self.graph.find_route(current, key, completed, (coords["x"], coords["y"]))
            if route is not None:
                entry = route[-1].destination_xy if route else (coords["x"], coords["y"])
                source_planner = GbaConstrainedAStar(cast(Matrix, (packed >> 10) & 3), cast(Matrix, packed >> 12),
                                                    self.catalog.behavior_grid(key), game=self.game)
                grass.sort(key=lambda xy: abs(xy[0] - entry[0]) + abs(xy[1] - entry[1]))
                if not 0 <= entry[1] < packed.shape[0] or not 0 <= entry[0] < packed.shape[1]:
                    continue
                origin = entry[0], entry[1], int(packed[entry[1], entry[0]]) >> 12
                point = next((xy for xy in grass if source_planner.find_optimal_path(origin, xy) is not None), None)
                if point is None:
                    continue
                cost = sum(edge.cost for edge in route) + abs(point[0] - entry[0]) + abs(point[1] - entry[1])
                candidates.append((cost + len(route), key, point))
        if candidates:
            _, key, point = min(candidates, key=lambda entry: entry[0])
            result.update(dest=(*key, point), action="train", training_route=self.catalog.maps[key]["name"])
            if self._rom_catalog is None:
                self._rom_catalog = RomCatalog(cast(Game, self.game))
            choices: list[dict[str, Any]] = []
            hp_ratio = sum(int(mon["hp"]) for mon in party) / max(1, sum(int(mon["maxHP"]) for mon in party))
            player_level = max((int(mon["level"]) for mon in party), default=5)
            for travel, key, point in sorted(candidates, key=lambda entry: entry[0])[:12]:
                raw = self.catalog.maps[key]
                land = self.catalog.land_encounters.get(str(raw["id"]))
                if land is None:
                    continue
                expected_xp = expected_level = 0.0
                total_weight = max(1, sum(int(weight) for weight in land["weights"]))
                for mon, weight in zip(land["mons"], land["weights"], strict=True):
                    level = (int(mon["min_level"]) + int(mon["max_level"])) / 2
                    species = self._rom_catalog.species(self.catalog.constants[str(mon["species"])])
                    expected_xp += species.experience_yield * level / 7 * int(weight) / total_weight
                    expected_level += level * int(weight) / total_weight
                # This is an explicit conservative prior, not a safety proof.
                # Observed route outcomes update the curriculum's Beta posterior.
                risk = min(1.0, max(0.005, 1 / (1 + math.exp(min(30.0, 4 * (player_level - expected_level) / max(1, expected_level))))
                                     * (1.5 - min(1.0, hp_ratio))))
                duration = max(3.0, 60.0 / max(1, int(land["encounter_rate"])) + 2 * expected_level / max(1, player_level))
                choices.append({"route": str(raw["name"]), "map_group": key[0], "map_num": key[1], "coords": point,
                                "expected_experience": expected_xp, "duration_seconds": duration,
                                "faint_probability": risk, "travel_seconds": travel * 16 / 60,
                                "risk_model": "level_and_hp_prior; update with observed outcomes"})
            result["training_candidates"] = choices
        return result

    def training_map_prerequisites(self, map_name: str) -> tuple[str, ...]:
        """Story gates for source training routes, separate from geometric reach.

        The current map is additionally checked against the live RAM grid. A
        static topology route alone cannot prove the player has passed a blocking
        guide, Snorlax, Cut tree, Rock Smash boulder or Surf crossing.
        """
        match = re.match(r"Route(\d+)(?:_|$)", map_name)
        if match is None:
            return ()
        route = int(match[1])
        if self.game == "firered":
            if route in (1, 2, 22):
                return ()
            if route in (3, 4):
                return ("GYM_1_BROCK",)
            if route in (24, 25):
                return ("CERULEAN_RIVAL",)
            if route in (5, 6, 11):
                return ("SS_TICKET",)
            if route in (7, 8, 9, 10, 16):
                return ("GYM_2_MISTY", "SS_ANNE_CUT")
            if route in (12, 13, 14, 15, 17, 18):
                return ("POKE_FLUTE",)
            if route in (19, 20, 21):
                return ("GYM_5_KOGA", "HM_SURF")
            if route == 23:
                return ("GYM_8_GIOVANNI",)
            return ("ELITE_FOUR",)
        if route in (101, 102, 103):
            return ()
        if route in (104, 116):
            return ("WALLY_TUTORIAL",)
        if route in (110, 117):
            return ("STERN_GOODS",)
        if route in (111, 112, 113, 114, 115):
            return ("GYM_3_WATTSON", "HM_ROCK_SMASH")
        if 118 <= route <= 123:
            return ("GYM_5_NORMAN", "HM_SURF")
        if 124 <= route <= 134:
            return ("AQUA_HIDEOUT", "HM_SURF")
        return ("ELITE_FOUR",)

    def puzzle_actions(self, bridge: GBABridgeProtocol) -> list[dict[str, Any]]:
        """Source-defined local switches, doors and blockers for replanning.

        Runtime callers execute candidate actions and observe RAM/map changes.
        This never labels a puzzle solved just because a switch was approached.
        """
        if self.catalog is None:
            return []
        coords = bridge.get_player_coords_and_map()
        key = coords["map_group"], coords["map_num"]
        raw = self.catalog.maps.get(key)
        if raw is None:
            return []
        targets = self.catalog.targets(str(raw["name"]))
        patterns = r"(?:TrashCan|Statue$|Switch\d|Computer$|Door(?:\d|$)|Strength|RockSmash|CutTree|Quiz|Puzzle)"
        return [{"mode": "EXECUTE_PUZZLE", "dest": (*key, (int(target["x"]), int(target["y"]))),
                 "action": target["action"], "script": target["script"], "npc_local_id": target["npc_local_id"]}
                for target in targets if re.search(patterns, str(target["script"]))]

    def plan_puzzle_actions(self, bridge: GBABridgeProtocol, goal: tuple[int, int] | None = None,
                             max_states: int = 25_000) -> list[dict[str, Any]] | None:
        """Physical command plan: pressure plates, boulder access, ice or portals.

        Each command specifies its expected origin/target or an interactable NPC.
        Execute until a script/map transition and replan from fresh RAM. None
        means no legal plan was found; an empty list means no commands are needed.
        Native completion still requires the campaign's flags/vars, independently.
        """
        if self.catalog is None:
            return None
        coords = bridge.get_player_coords_and_map()
        key = coords["map_group"], coords["map_num"]
        if self.catalog.maps.get(key, {}).get("name") == "SootopolisCity_Gym_1F":
            return plan_sootopolis_ice_commands(bridge, self.catalog, max_states=max_states)
        _, collision, elevation = bridge.extract_live_map_matrices()
        behavior = bridge.extract_behavior_map()
        walk_goal = goal if goal is not None and 0 <= goal[1] < collision.shape[0] and 0 <= goal[0] < collision.shape[1] and int(collision[goal[1], goal[0]]) == 0 else None
        strength = plan_strength_commands(bridge, self.catalog, exit_tile=walk_goal, max_states=max_states)
        if strength is not None:
            return strength
        if goal is None:
            return None
        route = self.graph.find_intra_map_route(key, (coords["x"], coords["y"], coords["z"]), goal,
                                                collision, elevation, behavior,
                                                self.completed_milestones(bridge, bridge.get_decrypted_player_party()), bridge)
        if route is None:
            return None
        return [{"mode": "EXECUTE_PUZZLE", "dest": (*key, edge.source_xy), "action": edge.activation,
                 "direction": edge.direction, "script": edge.script, "npc_local_id": edge.npc_local_id,
                 "menu_choice": edge.menu_choice, "menu_default": edge.menu_default,
                 "puzzle": "same_map_portal", "expected_destination": edge.destination_xy} for edge in route]

    def _task_target(self, bridge: GBABridgeProtocol, task: CampaignMilestone) -> dict[str, Any] | None:
        if self.catalog is None:
            return None
        if task.milestone_id == "BILL_RESCUE" and not bridge.get_event_flag(2):
            targets = [target for target in self.catalog.targets("Route25_SeaCottage") if str(target["script"]).endswith("_Bill")]
            for target in targets:
                flag = self.catalog.constants.get(str(target.get("flag", "0")))
                if flag is None or not bridge.get_event_flag(flag):
                    return target
        if task.milestone_id == "SURGE_SWITCHES":
            coords = bridge.get_player_coords_and_map()
            if (coords["map_group"], coords["map_num"]) != task.map_key:
                return None
            var_offset = 2 if bridge.get_event_flag(1) else 0
            switch = bridge.read_u16(bridge.get_save_block1_dynamic_ptr() + bridge.memory_map.vars_offset + var_offset)
            if 1 <= switch <= 15:
                return self.catalog.find_target("VermilionCity_Gym", rf"TrashCan{switch}$")
        return None

    def get_active_directive(self, bridge: GBABridgeProtocol, party_summary: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.custom_campaign and bridge.memory_map.game != self.game:
            raise ValueError("Planner campaign and bridge ROM differ")
        completed = self.completed_milestones(bridge, party_summary)
        hp = sum(int(mon["hp"]) for mon in party_summary)
        max_hp = max(1, sum(int(mon["maxHP"]) for mon in party_summary))
        if party_summary and hp / max_hp < 0.4 and not bridge.is_in_battle()[0]:
            return self._healing_directive(bridge, hp, max_hp, completed)
        for task in self.milestones:
            if task.milestone_id in completed or any(parent not in completed for parent in task.dependencies):
                continue
            level = max((int(mon["level"]) for mon in party_summary), default=0)
            if party_summary and level < task.required_avg_level:
                return self._training_directive(bridge, task, completed, party_summary)
            missing_moves = [move for move in task.required_moves
                             if not any(move in mon["moves"] for mon in party_summary)]
            if missing_moves:
                return {"mode": "TEACH_FIELD_MOVE", "milestone": task.milestone_id, "moves": missing_moves,
                        "action": "teach", "dest": (*task.map_key, task.target_coords)}
            result: dict[str, Any] = {"mode": "EXECUTE_MILESTONE", "milestone": task.milestone_id,
                    "dest": (*task.map_key, task.target_coords), "action": task.action, "script": task.script,
                    "npc_local_id": task.npc_local_id, "map_name": task.map_name, "puzzle": task.puzzle,
                    "menu_choice": task.menu_choice, "required_types": task.required_types,
                    "completion_flag_id": task.completion_flag_id}
            target = self._task_target(bridge, task)
            if target is not None:
                result.update(dest=(target["map_group"], target["map_num"], (target["x"], target["y"])),
                              action=target["action"], script=target["script"], npc_local_id=target["npc_local_id"])
            return result
        if len(completed) == len(self.milestones):
            return {"mode": "POSTGAME_COMPLETE", "reason": "All campaign conditions observed", "completed": sorted(completed)}
        raise RuntimeError("Campaign has unresolved dependencies; progress observations are inconsistent")

    def export_campaign(self, path: str | Path) -> None:
        data = {"game": self.game, "milestones": [asdict(task) for task in self.milestones],
                "certification": "source-backed DAG; native execution evidence required separately"}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

