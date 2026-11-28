"""Simulated live runs for developing and testing the live dashboard.

Everything this script publishes is FAKE and is marked as such (``"demo": true``, run ids ``demo-*``, the
agent string says "simulado"); the dashboard shows a SIMULADO label on these cards. Frames are the real
GAL screenshots (``artifacts/ares_u/e3/shot_<game>.png``) scrolled to look alive, or a synthetic battle
screen; numbers come from a seeded random process, not from any agent.

    python -m scripts.live_demo_publisher                      # 7 runs, forever, 1 publish/s
    python -m scripts.live_demo_publisher --duration 300 --clean
    python -m scripts.live_demo_publisher --runs rojo,league --interval 0.5
    python -m scripts.live_demo_publisher --with-stale         # plus one run that stops (stale card)

``--clean`` removes the ``demo-*`` directories this script created when it exits.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import random
import shutil
import time
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from core.live_status import LIVE_ROOT, publish
from scripts.live_dashboard import GAMES, real_multiplier

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "artifacts" / "ares_u" / "e3"
PREFIX = "demo-"
SIZES = {"gb": (160, 144), "gbc": (160, 144), "gba": (240, 160), "nds": (256, 192), "sim": (240, 160)}
EMU_FPS = {"gb": 3600.0, "gbc": 3200.0, "gba": 240.0, "nds": 120.0, "sim": 900.0}
TYPE_RGB = {"normal": (168, 167, 122), "fire": (238, 129, 48), "water": (99, 144, 240), "electric": (247, 208, 44),
            "grass": (122, 199, 76), "ice": (150, 217, 214), "fighting": (194, 46, 40), "poison": (163, 62, 161),
            "ground": (226, 191, 101), "flying": (169, 143, 243), "psychic": (249, 85, 135), "bug": (166, 185, 26),
            "rock": (182, 161, 54), "ghost": (115, 87, 151), "dragon": (111, 53, 252), "dark": (112, 87, 70),
            "steel": (183, 183, 206)}

# (name, type, power) - Spanish move names; power 0 = status move.
MOVES: dict[str, tuple[str, int]] = {
    "Placaje": ("normal", 40), "Arañazo": ("normal", 40), "Ataque Rápido": ("normal", 40), "Gruñido": ("normal", 0),
    "Ascuas": ("fire", 40), "Lanzallamas": ("fire", 90), "Rueda Fuego": ("fire", 60), "Pistola Agua": ("water", 40),
    "Burbuja": ("water", 40), "Hidrobomba": ("water", 110), "Látigo Cepa": ("grass", 45), "Hoja Afilada": ("grass", 55),
    "Impactrueno": ("electric", 40), "Chispa": ("electric", 65), "Tornado": ("flying", 40), "Ataque Ala": ("flying", 60),
    "Picotazo Ven": ("poison", 15), "Doble Patada": ("fighting", 30), "Golpe Karate": ("fighting", 50),
    "Confusión": ("psychic", 50), "Lanzarrocas": ("rock", 50), "Disparo Lodo": ("ground", 55), "Mordisco": ("dark", 60),
    "Bofetón Lodo": ("ground", 20), "Picadura": ("bug", 60), "Lengüetazo": ("ghost", 30), "Paralizador": ("grass", 0),
    "Onda Trueno": ("electric", 0), "Tóxico": ("poison", 0),
}
STATUS_EFFECT = {"Tóxico": "tox", "Onda Trueno": "par", "Paralizador": "par"}


def multiplier(move_type: str, defenders: tuple[str, ...]) -> float:
    value = real_multiplier(move_type, list(defenders))
    return 1.0 if value is None else value


@dataclass(frozen=True)
class Script:
    region: str
    party: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]   # species, types, moves
    wild: tuple[tuple[str, tuple[str, ...]], ...]
    milestones: tuple[tuple[str, bool], ...]                         # (description, is_badge)


KANTO = Script(
    "Kanto",
    (("Charmeleon", ("fire",), ("Arañazo", "Ascuas", "Gruñido", "Rueda Fuego")),
     ("Pidgeotto", ("normal", "flying"), ("Placaje", "Tornado", "Ataque Rápido", "Ataque Ala")),
     ("Nidorino", ("poison",), ("Picotazo Ven", "Doble Patada", "Placaje", "Tóxico")),
     ("Pikachu", ("electric",), ("Impactrueno", "Ataque Rápido", "Onda Trueno", "Chispa"))),
    (("Rattata", ("normal",)), ("Pidgey", ("normal", "flying")), ("Caterpie", ("bug",)), ("Geodude", ("rock", "ground")),
     ("Oddish", ("grass", "poison")), ("Zubat", ("poison", "flying")), ("Staryu", ("water",)), ("Sandshrew", ("ground",)),
     ("Ekans", ("poison",)), ("Voltorb", ("electric",)), ("Bellsprout", ("grass", "poison")), ("Mankey", ("fighting",))),
    (("Elegir inicial en el laboratorio de Oak", False), ("Entregar el paquete de Oak", False),
     ("Cruzar el Bosque Verde", False), ("Medalla Roca (Brock, Ciudad Plateada)", True),
     ("Atravesar el Monte Moon", False), ("Medalla Cascada (Misty, Ciudad Celeste)", True),
     ("Visitar a Bill en la Ruta 25", False), ("S.S. Anne: obtener Corte", False),
     ("Medalla Trueno (Lt. Surge, Ciudad Carmín)", True), ("Cruzar el Túnel Roca", False),
     ("Medalla Arcoíris (Erika, Ciudad Azulona)", True), ("Guarida Rocket: Scope Silph", False),
     ("Torre Pokémon (Pueblo Lavanda)", False), ("Medalla Alma (Koga, Ciudad Fucsia)", True)))
JOHTO = Script(
    "Johto",
    (("Quilava", ("fire",), ("Placaje", "Ascuas", "Ataque Rápido", "Rueda Fuego")),
     ("Hoothoot", ("normal", "flying"), ("Placaje", "Confusión", "Tornado", "Gruñido")),
     ("Geodude", ("rock", "ground"), ("Placaje", "Lanzarrocas", "Bofetón Lodo", "Gruñido")),
     ("Flaaffy", ("electric",), ("Placaje", "Impactrueno", "Onda Trueno", "Chispa"))),
    (("Sentret", ("normal",)), ("Hoothoot", ("normal", "flying")), ("Spinarak", ("bug", "poison")),
     ("Bellsprout", ("grass", "poison")), ("Zubat", ("poison", "flying")), ("Wooper", ("water", "ground")),
     ("Gastly", ("ghost", "poison")), ("Ledyba", ("bug", "flying")), ("Miltank", ("normal",)), ("Onix", ("rock", "ground"))),
    (("Recibir el inicial del Prof. Elm", False), ("Huevo misterioso del Sr. Pokémon", False),
     ("Torre Bellsprout", False), ("Medalla Céfiro (Pegaso, Ciudad Malva)", True), ("Cueva Unión", False),
     ("Pozo Slowpoke", False), ("Medalla Colmena (Antón, Pueblo Azalea)", True), ("Encinar: obtener Corte", False),
     ("Medalla Planicie (Blanca, Ciudad Trigal)", True), ("Torre Quemada", False),
     ("Medalla Niebla (Morti, Ciudad Iris)", True)))
HOENN = Script(
    "Hoenn",
    (("Marshtomp", ("water", "ground"), ("Placaje", "Pistola Agua", "Bofetón Lodo", "Disparo Lodo")),
     ("Taillow", ("normal", "flying"), ("Ataque Rápido", "Ataque Ala", "Gruñido", "Placaje")),
     ("Kirlia", ("psychic",), ("Confusión", "Gruñido", "Placaje", "Onda Trueno")),
     ("Shroomish", ("grass",), ("Placaje", "Látigo Cepa", "Paralizador", "Hoja Afilada"))),
    (("Zigzagoon", ("normal",)), ("Poochyena", ("dark",)), ("Wurmple", ("bug",)), ("Lotad", ("water", "grass")),
     ("Seedot", ("grass",)), ("Wingull", ("water", "flying")), ("Makuhita", ("fighting",)), ("Nosepass", ("rock",)),
     ("Magnemite", ("electric", "steel")), ("Electrike", ("electric",)), ("Numel", ("fire", "ground"))),
    (("Elegir inicial y rescatar al Prof. Abedul", False), ("Combate con Aura en la Ruta 103", False),
     ("Bosque Petalia: Equipo Aqua", False), ("Medalla Piedra (Petra, Ciudad Férrica)", True),
     ("Cueva Granito: carta para Máximo", False), ("Medalla Puño (Marcial, Pueblo Azuliza)", True),
     ("Ciudad Portual: Astillero", False), ("Medalla Dinamo (Erico, Ciudad Malvalona)", True),
     ("Monte Cenizo", False), ("Medalla Calor (Candela, Pueblo Lavacalda)", True),
     ("Medalla Equilibrio (Norman, Ciudad Petalia)", True)))
SINNOH = Script(
    "Sinnoh",
    (("Monferno", ("fire", "fighting"), ("Arañazo", "Ascuas", "Golpe Karate", "Rueda Fuego")),
     ("Staravia", ("normal", "flying"), ("Placaje", "Ataque Rápido", "Ataque Ala", "Gruñido")),
     ("Luxio", ("electric",), ("Placaje", "Chispa", "Mordisco", "Onda Trueno")),
     ("Budew", ("grass", "poison"), ("Látigo Cepa", "Paralizador", "Tóxico", "Hoja Afilada"))),
    (("Bidoof", ("normal",)), ("Starly", ("normal", "flying")), ("Kricketot", ("bug",)), ("Shinx", ("electric",)),
     ("Geodude", ("rock", "ground")), ("Cranidos", ("rock",)), ("Machop", ("fighting",)), ("Psyduck", ("water",)),
     ("Budew", ("grass", "poison")), ("Zubat", ("poison", "flying")), ("Buizel", ("water",))),
    (("Elegir inicial en el Lago Veraz", False), ("Recibir la Pokédex del Prof. Serbal", False),
     ("Ciudad Jubileo: Pokéreloj", False), ("Mina Pirita", False), ("Medalla Lignito (Roco, Ciudad Pirita)", True),
     ("Valle Eólico: Equipo Galaxia", False), ("Bosque Vetusto", False),
     ("Medalla Bosque (Gardenia, Ciudad Vetusta)", True), ("Ruta 206: bicicleta", False),
     ("Medalla Reliquia (Fantina, Ciudad Corazón)", True)))
UNOVA = Script(
    "Teselia",
    (("Pignite", ("fire", "fighting"), ("Placaje", "Ascuas", "Golpe Karate", "Rueda Fuego")),
     ("Tranquill", ("normal", "flying"), ("Ataque Rápido", "Ataque Ala", "Gruñido", "Tornado")),
     ("Blitzle", ("electric",), ("Chispa", "Ataque Rápido", "Onda Trueno", "Impactrueno")),
     ("Panpour", ("water",), ("Arañazo", "Pistola Agua", "Mordisco", "Gruñido"))),
    (("Patrat", ("normal",)), ("Lillipup", ("normal",)), ("Purrloin", ("dark",)), ("Pidove", ("normal", "flying")),
     ("Sewaddle", ("bug", "grass")), ("Venipede", ("bug", "poison")), ("Roggenrola", ("rock",)),
     ("Woobat", ("psychic", "flying")), ("Drilbur", ("ground",)), ("Pansage", ("grass",)), ("Pansear", ("fire",))),
    (("Elegir inicial (Prof. Encina)", False), ("Ruta 1: combate con Cheren y Bel", False),
     ("Pueblo Terracota: discurso del Equipo Plasma", False), ("Primera medalla (Ciudad Gres)", True),
     ("Solar de los Sueños", False), ("Segunda medalla (Aloe, Ciudad Esmalte)", True), ("Bosque Azulejo", False),
     ("Tercera medalla (Camus, Ciudad Porcelana)", True), ("Cuarta medalla (Camila, Ciudad Mayólica)", True)))
SCRIPTS = {"rojo": KANTO, "firered": KANTO, "cristal": JOHTO, "heartgold": JOHTO, "emerald": HOENN,
           "platino": SINNOH, "negro": UNOVA, "negro2": UNOVA}
DEFAULT_RUNS = ("rojo", "cristal", "firered", "emerald", "platino", "negro", "league")
RUN_KIND = {"firered": "campaign", "emerald": "campaign"}


def _font(size: int) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):  # Pillow without FreeType
        return ImageFont.load_default()


def base_frame(game: str, platform: str) -> NDArray[np.uint8]:
    """The game's real GAL screenshot when available, otherwise a gradient of the right size."""
    path = SHOTS / f"shot_{game}.png"
    if path.is_file():
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    width, height = SIZES.get(platform, (240, 160))
    ys, xs = np.mgrid[0:height, 0:width]
    gradient = np.stack([(xs * 255 // width), (ys * 255 // height), np.full_like(xs, 140)], axis=-1)
    return cast(NDArray[np.uint8], gradient.astype(np.uint8))


def hp_color(fraction: float) -> tuple[int, int, int]:
    return (72, 200, 88) if fraction > 0.5 else ((232, 184, 40) if fraction > 0.2 else (224, 64, 56))


@dataclass
class Mon:
    species: str
    types: tuple[str, ...]
    moves: tuple[str, ...]
    level: int
    hp: int = 0
    max_hp: int = 0
    status: str = ""

    def __post_init__(self) -> None:
        self.max_hp = self.max_hp or 12 + self.level * 3
        self.hp = self.hp or self.max_hp

    def view(self) -> dict[str, Any]:
        return {"species": self.species, "level": self.level, "hp": self.hp, "max_hp": self.max_hp,
                "types": list(self.types), "status": self.status,
                "moves": [{"name": name, "type": MOVES[name][0], "power": MOVES[name][1]} for name in self.moves]}


@dataclass
class Battle:
    enemy: Mon
    wild: bool
    turn: int = 0
    choice: str = ""
    reason: str = ""
    effectiveness: float = 1.0
    scores: dict[str, float] = field(default_factory=dict)
    log: str = ""


class DemoRun:
    """One simulated playthrough (seeded random process; nothing here comes from an agent)."""

    def __init__(self, game: str, rng: random.Random, sequence: int, inherited: dict[str, dict[str, float]]) -> None:
        meta = GAMES[game]
        self.game, self.rng, self.sequence = game, rng, sequence
        self.platform = str(meta["platform"])
        self.kind = RUN_KIND.get(game, "universal")
        self.run_id = f"{PREFIX}{self.kind}-{game}"
        self.script = SCRIPTS[game]
        self.base = base_frame(game, self.platform)
        self.frames = rng.randrange(0, 4000)
        self.fps = EMU_FPS[self.platform] * rng.uniform(0.8, 1.2)
        self.milestones: list[str] = []
        self.badges = 0
        self.maps = 1
        self.party = [Mon(species, types, moves, level=5 + rng.randrange(0, 3) - index)
                      for index, (species, types, moves) in enumerate(self.script.party[:2])]
        self.battle: Battle | None = None
        self.stats = {"battles": 0, "wins": 0, "losses": 0, "stalls": 0, "recoveries": 0, "heals": 0}
        self.events: list[str] = []
        self.chart: dict[str, dict[str, float]] = {atk: dict(row) for atk, row in inherited.items()}
        self.inherited = sum(len(row) for row in self.chart.values())
        self.skills = ["caminar", "hablar", "menús"]
        self.stuck = 0
        self.progress = 0.0
        self.pace = rng.uniform(0.035, 0.07)  # milestone progress per simulated second
        self.started_at = time.time()
        self.pos = (0, 0)
        self.heading = (1, 0)
        self.event("Inicio de partida" + (f" con {self.inherited} celdas de la tabla de tipos heredadas"
                                         if self.inherited else ""))

    # ---------------------------------------------------------------- simulation
    def event(self, text: str) -> None:
        self.events.append(f"f{self.frames:,} · {text}".replace(",", "."))
        self.events = self.events[-20:]

    @property
    def next_milestone(self) -> tuple[str, bool] | None:
        index = len(self.milestones)
        return self.script.milestones[index] if index < len(self.script.milestones) else None

    def goal(self) -> str:
        if self.battle is not None:
            return f"Vencer a {self.battle.enemy.species} (Nv {self.battle.enemy.level})"
        if self.stuck:
            return "Salir del atasco: probar ruta alternativa"
        lead = self.party[0]
        if lead.hp < lead.max_hp * 0.3:
            return "Curar al equipo en el Centro Pokémon"
        upcoming = self.next_milestone
        return upcoming[0] if upcoming else "Explorar y entrenar al equipo"

    def step(self, seconds: float) -> None:
        self.frames += int(self.fps * seconds)
        if self.battle is not None:
            self._battle_turn()
            return
        if self.stuck:
            self.stuck -= 1
            if self.stuck == 0:
                self.stats["recoveries"] += 1
                self.event("Recuperado del atasco: ruta alternativa encontrada")
            return
        roll = self.rng.random()
        if roll < 0.16:
            self._start_battle()
        elif roll < 0.185:
            self.stuck = self.rng.randrange(2, 5)
            self.stats["stalls"] += 1
            self.event("Atasco detectado: misma casilla durante 600 fotogramas")
        else:
            self._walk(seconds)

    def _walk(self, seconds: float) -> None:
        lead = self.party[0]
        if lead.hp < lead.max_hp * 0.3:
            for mon in self.party:
                mon.hp, mon.status = mon.max_hp, ""
            self.stats["heals"] += 1
            self.event("Equipo curado en el Centro Pokémon")
            return
        if self.rng.random() < 0.1:
            self.heading = self.rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        self.pos = (self.pos[0] + self.heading[0] * 4, self.pos[1] + self.heading[1] * 4)
        if self.rng.random() < 0.08:
            self.maps += 1
            self.event(f"Mapa nuevo descubierto (#{self.maps})")
        self.progress += self.pace * seconds * self.rng.uniform(0.5, 1.5)
        upcoming = self.next_milestone
        if upcoming is not None and self.progress >= 1.0:
            self.progress = 0.0
            self.milestones.append(upcoming[0])
            self.badges += int(upcoming[1])
            self.event(("¡MEDALLA! " if upcoming[1] else "Hito completado: ") + upcoming[0])
            if len(self.party) < len(self.script.party) and self.rng.random() < 0.6:
                species, types, moves = self.script.party[len(self.party)]
                self.party.append(Mon(species, types, moves, level=max(3, self.party[0].level - 4)))
                self.event(f"Captura: {species} se une al equipo")
            if self.rng.random() < 0.5:
                self.skills.append(self.rng.choice(("Corte", "Surf", "Fuerza", "Destello", "bicicleta", "tienda",
                                                    "PC de Bill", "MT")))

    def _start_battle(self) -> None:
        species, types = self.rng.choice(self.script.wild)
        level = max(2, self.party[0].level + self.rng.randrange(-3, 2))
        wild = self.rng.random() < 0.75
        self.battle = Battle(Mon(species, types, (), level), wild)
        self.stats["battles"] += 1
        self.event(("Combate salvaje: " if wild else "Combate contra entrenador: ") + f"{species} Nv {level}")

    def _choose(self, battle: Battle, me: Mon) -> tuple[str, float]:
        scores: dict[str, float] = {}
        best, best_mult = me.moves[0], 1.0
        for name in me.moves:
            move_type, power = MOVES[name]
            mult = multiplier(move_type, battle.enemy.types) if power else 1.0
            stab = 1.5 if move_type in me.types else 1.0
            if power:
                value = power * mult * stab / 150.0
            else:
                value = 0.25 if name in STATUS_EFFECT and not battle.enemy.status else 0.03
            scores[name] = round(min(1.0, value), 3)
            if scores[name] >= scores.get(best, -1.0):
                best, best_mult = name, mult
        if len(self.party) > 1:
            other = self.party[1]
            mult = max(multiplier(MOVES[n][0], battle.enemy.types) for n in other.moves if MOVES[n][1])
            scores[f"Cambiar a {other.species}"] = round(min(1.0, 0.3 * mult * (1.4 if me.hp < me.max_hp * 0.3
                                                                                 else 0.6)), 3)
        battle.scores = dict(sorted(scores.items(), key=lambda item: -item[1]))
        return best, best_mult

    def _battle_turn(self) -> None:
        battle = self.battle
        assert battle is not None
        me = self.party[0]
        battle.turn += 1
        move, mult = self._choose(battle, me)
        move_type, power = MOVES[move]
        battle.choice, battle.effectiveness = move, mult
        effect = {0.0: "no le afecta", 0.25: "muy poco eficaz", 0.5: "poco eficaz", 1.0: "neutro",
                  2.0: "súper eficaz", 4.0: "súper eficaz x4"}.get(mult, f"x{mult:g}")
        enemy_types = "/".join(battle.enemy.types)
        if power:
            battle.reason = (f"{move} ({move_type}, potencia {power}) contra {battle.enemy.species} ({enemy_types}): "
                             f"{effect}. Mejor daño esperado entre {len(me.moves)} movimientos"
                             + ("; mismo tipo que el usuario (STAB x1.5)." if move_type in me.types else "."))
        else:
            battle.reason = (f"{move}: {battle.enemy.species} aún no tiene estado alterado; "
                             "desgastarlo compensa más que un ataque débil.")
            battle.enemy.status = STATUS_EFFECT.get(move, battle.enemy.status)
        if battle.enemy.status == "tox":
            battle.enemy.hp = max(0, battle.enemy.hp - max(1, battle.enemy.max_hp // 8))
        if power:
            damage = int(max(1, battle.enemy.max_hp * power * mult / 160 * self.rng.uniform(0.7, 1.15)))
            battle.enemy.hp = max(0, battle.enemy.hp - damage)
            if len(battle.enemy.types) == 1:
                self.chart.setdefault(move_type, {})[battle.enemy.types[0]] = mult
        battle.log = f"¡{me.species} usó {move}!"
        if battle.enemy.hp == 0:
            self.stats["wins"] += 1
            self.event(f"Victoria contra {battle.enemy.species} con {move}")
            if self.rng.random() < 0.35:
                me.level += 1
                me.max_hp += 3
                me.hp = min(me.max_hp, me.hp + 3)
                self.event(f"{me.species} sube al nivel {me.level}")
            self.battle = None
            return
        me.hp = max(0, me.hp - int(me.max_hp * self.rng.uniform(0.03, 0.22)))
        if me.hp == 0:
            self.stats["losses"] += 1
            self.event(f"Derrota contra {battle.enemy.species}: volver al Centro Pokémon")
            for mon in self.party:
                mon.hp = mon.max_hp
            self.battle = None

    # -------------------------------------------------------------------- views
    def status(self) -> dict[str, Any]:
        battle = None
        if self.battle is not None:
            battle = {"me": self.party[0].view(), "enemy": self.battle.enemy.view(), "choice": self.battle.choice,
                      "reason": self.battle.reason, "effectiveness": self.battle.effectiveness,
                      "scores": self.battle.scores, "turn": self.battle.turn, "is_wild": self.battle.wild}
        cells = sum(len(row) for row in self.chart.values())
        return {"kind": self.kind, "game": self.game, "generation": GAMES[self.game]["generation"],
                "platform": self.platform, "demo": True, "sequence": self.sequence, "started_at": self.started_at,
                "agent": "DEMO · datos simulados (no es el agente real)",
                "frames": self.frames, "fps": round(self.fps * self.rng.uniform(0.93, 1.07), 1),
                "badges": self.badges, "milestones": list(self.milestones), "maps_visited": self.maps,
                "party": [mon.view() for mon in self.party], "goal": self.goal(), "battle": battle,
                "stats": dict(self.stats),
                "knowledge": {"items": cells + len(self.skills), "inherited": self.inherited,
                              "type_chart": self.chart, "skills": list(self.skills)},
                "events": list(self.events)}

    def frame(self) -> NDArray[np.uint8]:
        if self.battle is not None:
            return battle_frame(self.base.shape[1], self.base.shape[0], self.party[0], self.battle.enemy,
                                self.battle.log or f"¿Qué hará {self.party[0].species}?")
        shifted = np.roll(self.base, shift=(self.pos[1] % self.base.shape[0], self.pos[0] % self.base.shape[1]),
                          axis=(0, 1)).copy()
        if self.stuck:
            shifted[..., 0] = np.minimum(255, shifted[..., 0].astype(np.int32) + 40).astype(np.uint8)
        height, width = shifted.shape[:2]
        cy, cx = height // 2, width // 2
        shifted[cy - 5:cy + 5, cx - 4:cx + 4] = (232, 64, 48)
        shifted[cy - 5:cy - 2, cx - 4:cx + 4] = (40, 40, 40)
        return shifted


def battle_frame(width: int, height: int, me: Mon, enemy: Mon, text: str) -> NDArray[np.uint8]:
    """Synthetic battle screen (two platforms, HP boxes, text box)."""
    image = Image.new("RGB", (width, height), (232, 240, 216))
    draw = ImageDraw.Draw(image)
    font = _font(max(8, height // 16))
    box_h = height // 4
    field_h = height - box_h
    draw.ellipse((width * 0.52, field_h * 0.30, width * 0.95, field_h * 0.48), fill=(200, 216, 168))
    draw.ellipse((width * 0.02, field_h * 0.80, width * 0.50, field_h * 0.98), fill=(200, 216, 168))
    enemy_rgb = TYPE_RGB.get(enemy.types[0], (160, 160, 160))
    me_rgb = TYPE_RGB.get(me.types[0], (160, 160, 160))
    draw.ellipse((width * 0.64, field_h * 0.08, width * 0.84, field_h * 0.40), fill=enemy_rgb, outline=(40, 40, 40))
    draw.rounded_rectangle((width * 0.12, field_h * 0.52, width * 0.38, field_h * 0.92), radius=6, fill=me_rgb,
                           outline=(40, 40, 40))

    def hp_box(x: float, y: float, mon: Mon) -> None:
        w = width * 0.44
        draw.rounded_rectangle((x, y, x + w, y + height * 0.15), radius=4, fill=(248, 248, 240), outline=(60, 60, 60))
        draw.text((x + 4, y + 1), f"{mon.species.upper()}  Nv{mon.level}", fill=(30, 30, 30), font=font)
        fraction = mon.hp / max(1, mon.max_hp)
        bar_y = y + height * 0.095
        draw.rectangle((x + 4, bar_y, x + w - 6, bar_y + 4), fill=(80, 80, 80))
        draw.rectangle((x + 5, bar_y + 1, x + 5 + (w - 12) * fraction, bar_y + 3), fill=hp_color(fraction))

    hp_box(width * 0.03, field_h * 0.05, enemy)
    hp_box(width * 0.53, field_h * 0.62, me)
    draw.rectangle((0, field_h, width - 1, height - 1), fill=(40, 48, 72), outline=(200, 168, 72), width=2)
    draw.text((8, field_h + 6), text, fill=(248, 248, 248), font=font)
    return np.asarray(image, dtype=np.uint8)


RIVALS = (
    ("Kamikaze", "Ataque total: siempre el golpe de más daño, nunca cambia",
     {"agresividad": 0.95, "cambios": 0.02, "estados": 0.03, "mejoras": 0.0, "riesgo": 0.9}, 1480),
    ("Envenenador", "Reparte estados (tóxico, parálisis, sueño) y desgasta",
     {"agresividad": 0.35, "cambios": 0.15, "estados": 0.85, "mejoras": 0.1, "riesgo": 0.3}, 1520),
    ("Muro", "Defensivo: resiste por tipo, se cura y alarga el combate",
     {"agresividad": 0.2, "cambios": 0.45, "estados": 0.3, "mejoras": 0.2, "riesgo": 0.1}, 1450),
    ("Pivote", "Cambia constantemente buscando ventaja de tipos",
     {"agresividad": 0.5, "cambios": 0.8, "estados": 0.15, "mejoras": 0.05, "riesgo": 0.35}, 1550),
    ("Barredor", "Se potencia (Danza Espada, Maquinación) y luego barre",
     {"agresividad": 0.7, "cambios": 0.1, "estados": 0.05, "mejoras": 0.9, "riesgo": 0.75}, 1530),
    ("Caótico", "Política de alta entropía: imprevisible a propósito",
     {"agresividad": 0.5, "cambios": 0.5, "estados": 0.5, "mejoras": 0.5, "riesgo": 0.5}, 1350),
    ("Greedy", "Heurística de daño inmediato (línea base del proyecto)",
     {"agresividad": 0.9, "cambios": 0.05, "estados": 0.05, "mejoras": 0.0, "riesgo": 0.6}, 1500),
    ("E1-PPO", "Cerebro E1 (Transformer por entidades, PPO self-play)",
     {"agresividad": 0.6, "cambios": 0.3, "estados": 0.2, "mejoras": 0.2, "riesgo": 0.4}, 1600),
)


class DemoLeague:
    """Simulated league: Elo random walk toward each rival's hidden strength; cross-play from the Elo gap."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.run_id = f"{PREFIX}league"
        self.generation = 0
        self.matches = 0
        self.elo = {name: 1500.0 for name, _, _, _ in RIVALS}
        self.history: list[dict[str, Any]] = [{"generation": 0, "elo": dict(self.elo)}]
        self.events: list[str] = []
        self.started_at = time.time()
        self.pair = (RIVALS[0][0], RIVALS[1][0])
        self.base = base_frame("emerald", "gba")

    def step(self, seconds: float) -> None:
        for _ in range(int(40 * seconds)):
            (a, _, _, skill_a), (b, _, _, skill_b) = self.rng.sample(RIVALS, 2)
            expected = 1 / (1 + 10 ** ((self.elo[b] - self.elo[a]) / 400))
            true_p = 1 / (1 + 10 ** ((skill_b - skill_a) / 400))
            score = 1.0 if self.rng.random() < true_p else 0.0
            self.elo[a] += 8 * (score - expected)
            self.elo[b] -= 8 * (score - expected)
            self.matches += 1
            self.pair = (a, b)
        if self.rng.random() < 0.12:
            self.generation += 1
            self.history.append({"generation": self.generation, "elo": {k: round(v, 1) for k, v in self.elo.items()}})
            leader = max(self.elo, key=lambda name: self.elo[name])
            self.events.append(f"Generación {self.generation}: líder {leader} ({self.elo[leader]:.0f} Elo)")
            self.events = self.events[-20:]

    def status(self) -> dict[str, Any]:
        names = [name for name, _, _, _ in RIVALS]
        matrix = [[None if a == b else round(1 / (1 + 10 ** ((self.elo[b] - self.elo[a]) / 400)), 3) for b in names]
                  for a in names]
        champion = max(self.elo, key=lambda name: self.elo[name])
        league = {"generation": self.generation, "matches": self.matches,
                  "rivals": [{"name": name, "description": text, "profile": profile, "elo": round(self.elo[name], 1)}
                             for name, text, profile, _ in RIVALS],
                  "crossplay": {"names": names, "values": matrix, "metric": "win_rate"},
                  "champion": {"name": champion, "elo": round(self.elo[champion], 1)},
                  "champion_vs_e1": {"win_rate": round(matrix[names.index(champion)][names.index("E1-PPO")] or 0.5, 3)
                                     if champion != "E1-PPO" else 0.5,
                                     "battles": self.matches // 8, "note": "simulado"},
                  "elo_history": self.history[-40:]}
        return {"kind": "league", "game": "stadium", "generation": 3, "platform": "sim", "demo": True,
                "agent": "DEMO · liga simulada (no es la liga real)", "frames": self.matches * 30,
                "fps": 1200.0, "started_at": self.started_at,
                "goal": f"Generación {self.generation}: {self.pair[0]} contra {self.pair[1]}",
                "stats": {"battles": self.matches, "generation": self.generation},
                "league": league, "events": list(self.events)}

    def frame(self) -> NDArray[np.uint8]:
        a, b = self.pair
        me = Mon(a, ("fire",), ("Lanzallamas",), 50, hp=int(self.rng.uniform(20, 160)), max_hp=160)
        enemy = Mon(b, ("water",), ("Surf",), 50, hp=int(self.rng.uniform(20, 160)), max_hp=160)
        return battle_frame(240, 160, me, enemy, f"Liga · generación {self.generation}")


class DemoWorld:
    """All simulated runs; ``step`` advances and publishes every run once."""

    def __init__(self, root: Path = LIVE_ROOT, games: tuple[str, ...] = DEFAULT_RUNS, seed: int = 0,
                 speed: float = 1.0) -> None:
        self.root, self.speed = root, speed
        rng = random.Random(seed)
        self.runs: list[DemoRun] = []
        self.league: DemoLeague | None = None
        learned: dict[str, dict[str, float]] = {}
        chart = real_rows()
        sequence = 0
        for game in games:
            if game == "league":
                self.league = DemoLeague(random.Random(rng.random()))
                continue
            sequence += 1
            self.runs.append(DemoRun(game, random.Random(rng.random()), sequence, learned))
            # The next simulated game "inherits" a few type-chart cells (illustrative only).
            for attacker in rng.sample(sorted(chart), 3):
                defenders = rng.sample(sorted(chart[attacker]), 3)
                learned.setdefault(attacker, {}).update({d: chart[attacker][d] for d in defenders})
        self.created: set[str] = set()

    def ids(self) -> list[str]:
        return [run.run_id for run in self.runs] + ([self.league.run_id] if self.league else [])

    def step(self, seconds: float = 1.0) -> None:
        simulated = seconds * self.speed
        for run in self.runs:
            run.step(simulated)
            publish(run.run_id, run.status(), frame=run.frame(), root=self.root)
            self.created.add(run.run_id)
        if self.league is not None:
            self.league.step(simulated)
            publish(self.league.run_id, self.league.status(), frame=self.league.frame(), root=self.root)
            self.created.add(self.league.run_id)

    def publish_stale(self) -> str:
        """One extra run published once and never again: the dashboard marks it stale after 30 s."""
        run = DemoRun("heartgold", random.Random(7), 99, {})
        for _ in range(30):
            run.step(5.0)
        run_id = f"{PREFIX}universal-heartgold"
        publish(run_id, {**run.status(), "goal": "(ejecución detenida: ejemplo de tarjeta inactiva)"},
                frame=run.frame(), root=self.root)
        self.created.add(run_id)
        return run_id

    def clean(self) -> None:
        for run_id in sorted(self.created):
            if run_id.startswith(PREFIX):
                shutil.rmtree(self.root / run_id, ignore_errors=True)


def real_rows() -> dict[str, dict[str, float]]:
    """Real single-type chart rows (attacker -> defender -> multiplier) for generation 3."""
    types = ("normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying",
             "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel")
    return {a: {d: cast_float(real_multiplier(a, [d])) for d in types} for a in types}


def cast_float(value: float | None) -> float:
    return 1.0 if value is None else value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publica ejecuciones SIMULADAS para probar el panel en vivo")
    parser.add_argument("--root", type=Path, default=LIVE_ROOT)
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS),
                        help="juegos separados por comas (rojo,cristal,firered,emerald,platino,heartgold,negro,"
                             "negro2) y/o 'league'")
    parser.add_argument("--interval", type=float, default=1.0, help="segundos reales entre publicaciones")
    parser.add_argument("--speed", type=float, default=1.0, help="multiplicador del tiempo simulado")
    parser.add_argument("--duration", type=float, default=0.0, help="segundos (0 = sin límite)")
    parser.add_argument("--ticks", type=int, default=0, help="número de publicaciones (0 = sin límite)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-stale", action="store_true", help="añadir una ejecución detenida (tarjeta inactiva)")
    parser.add_argument("--clean", action="store_true", help="borrar los directorios demo-* creados al salir")
    args = parser.parse_args(argv)
    games = tuple(name.strip() for name in str(args.runs).split(",") if name.strip())
    unknown = [name for name in games if name != "league" and name not in SCRIPTS]
    if unknown:
        parser.error(f"juegos desconocidos: {unknown}")
    world = DemoWorld(Path(args.root), games, seed=int(args.seed), speed=float(args.speed))
    if args.with_stale:
        world.publish_stale()
    print(f"[demo] publicando {', '.join(world.ids())} en {args.root} cada {args.interval}s "
          f"(DATOS SIMULADOS; Ctrl+C para parar)", flush=True)
    started, ticks = time.monotonic(), 0
    try:
        while True:
            tick_start = time.monotonic()
            world.step(float(args.interval))
            ticks += 1
            if args.ticks and ticks >= args.ticks:
                break
            if args.duration and time.monotonic() - started >= args.duration:
                break
            time.sleep(max(0.0, float(args.interval) - (time.monotonic() - tick_start)))
    except KeyboardInterrupt:
        pass
    finally:
        if args.clean:
            world.clean()
            print("[demo] directorios demo-* eliminados", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
