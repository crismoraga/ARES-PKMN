"""Spanish play-by-play of stadium events, for the live dashboard."""
from __future__ import annotations

from collections.abc import Sequence

from .dex import TYPE_NAMES, Dex
from .engine import STAT_NAMES, Event

_STAT_ES = {"atk": "Ataque", "def": "Defensa", "spe": "Velocidad", "spa": "At. Esp.", "spd": "Def. Esp.",
            "acc": "Precisión", "eva": "Evasión", "hp": "PS"}
_STATUS_ES = {"psn": "fue envenenado", "tox": "fue gravemente envenenado", "brn": "se quemó", "par": "quedó paralizado",
              "frz": "se congeló", "slp": "se durmió"}
_RESIDUAL_ES = {"poison": "el veneno", "toxic": "el veneno grave", "burn": "la quemadura", "sandstorm": "la tormenta de arena",
                "hail": "el granizo", "leech_seed": "las drenadoras", "recoil": "el retroceso", "spikes": "las púas",
                "confusion": "la confusión", "curse": "la maldición", "trap": "el atrapamiento", "crash": "el golpe fallido",
                "rough_skin": "Piel Tosca", "liquid_ooze": "Viscosecreción"}
_WEATHER_ES = {"rain": "Empezó a llover", "sun": "El sol pega fuerte", "sandstorm": "Se desató una tormenta de arena",
               "hail": "Empezó a granizar"}


def _name(dex: Dex, species: int) -> str:
    return dex.species[species].name.title() if 0 <= species < len(dex.species) else f"#{species}"


def narrate(events: Sequence[Event], dex: Dex, labels: tuple[str, str]) -> list[str]:
    """One Spanish line per relevant event; ``labels`` are the two sides' names."""
    active: dict[int, int] = {}
    lines: list[str] = []
    for event in events:
        kind = event[0]
        if kind == "switch":
            active[int(event[1])] = int(event[3])
            lines.append(f"{labels[event[1]]} saca a {_name(dex, int(event[3]))}.")
            continue
        side = int(event[1]) if len(event) > 1 and isinstance(event[1], int) else 0
        who = f"{_name(dex, active.get(side, 0))} ({labels[side]})" if side in active else labels[side]
        if kind == "move":
            lines.append(f"{who} usó {dex.moves[int(event[3])].name.title()}.")
        elif kind == "hit":
            _, target, damage, label, critical, move_type, _, species, substitute = event
            parts = []
            if critical:
                parts.append("¡Golpe crítico!")
            if label == "super":
                parts.append("¡Es súper eficaz!")
            elif label == "resisted":
                parts.append("No es muy eficaz...")
            tail = " al sustituto" if substitute else ""
            parts.append(f"{_name(dex, int(species))} pierde {damage} PS{tail} ({TYPE_NAMES[int(move_type)]}).")
            lines.append(" ".join(parts))
        elif kind == "noeffect":
            lines.append(f"No afecta a {_name(dex, int(event[4]))} ({TYPE_NAMES[int(event[2])]}).")
        elif kind == "miss":
            lines.append(f"¡El ataque de {who} falló!")
        elif kind == "fail":
            lines.append(f"Pero falló ({dex.moves[int(event[2])].name.title()}).")
        elif kind == "faint":
            lines.append(f"¡{who} se debilitó!")
        elif kind == "status":
            lines.append(f"{who} {_STATUS_ES.get(str(event[2]), str(event[2]))}.")
        elif kind == "boost":
            stat = _STAT_ES.get(STAT_NAMES[int(event[2])], STAT_NAMES[int(event[2])])
            delta = int(event[3])
            verb = "subió" if delta > 0 else "bajó"
            lines.append(f"{stat} de {who} {verb}{' mucho' if abs(delta) >= 2 else ''}.")
        elif kind == "residual":
            lines.append(f"{who} sufre {event[2]} PS por {_RESIDUAL_ES.get(str(event[3]), str(event[3]))}.")
        elif kind == "heal":
            lines.append(f"{who} recupera {event[2]} PS.")
        elif kind == "ability":
            lines.append(f"[Habilidad] {who} revela su habilidad (#{event[2]}).")
        elif kind == "weather":
            lines.append(_WEATHER_ES.get(str(event[1]), str(event[1])) + ".")
        elif kind in ("paralyzed", "asleep", "frozen", "flinched", "confusion_hit", "protected", "recharge", "charging"):
            text = {"paralyzed": "está paralizado y no se mueve", "asleep": "está dormido", "frozen": "está congelado",
                    "flinched": "retrocedió", "confusion_hit": "se hirió en su confusión", "protected": "se protegió",
                    "recharge": "necesita recargar", "charging": "está cargando energía"}[kind]
            lines.append(f"{who} {text}.")
        elif kind == "unsupported":
            lines.append(f"(efecto de {dex.moves[int(event[2])].name.title()} no modelado en el Stadium)")
        elif kind == "end":
            winner = int(event[1])
            lines.append("Empate." if winner == 2 else f"¡Gana {labels[winner]}!")
    return lines
