"""Spanish play-by-play of stadium events, and the frame timeline the live arena animates.

``narrate`` gives one line per relevant event. ``timeline`` pairs every line
with the state the audience should see at that moment (who is on the field,
every team member's HP and status), reconstructed from the events and
re-synchronised with the engine's own snapshot after each turn step, so the
arena can never drift from the real battle.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .dex import TYPE_NAMES, Dex
from .engine import BURN, FREEZE, PARALYSIS, POISON, SLEEP_MASK, STAT_NAMES, TOXIC, Event, MonSpec

Snapshot = tuple[int, tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]]  # event count, per-side state
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
_ABILITY_ES = {2: "Llovizna", 3: "Impulso", 5: "Robustez", 9: "Elec. Estática", 10: "Absorbe Elec", 11: "Absorbe Agua",
               16: "Cambio Color", 18: "Absorbe Fuego", 22: "Intimidación", 24: "Piel Tosca", 25: "Superguarda",
               26: "Levitación", 27: "Efecto Espora", 28: "Sincronía", 29: "Cuerpo Puro", 36: "Rastro",
               38: "Punto Tóxico", 43: "Insonorizar", 45: "Chorro Arena", 49: "Cuerpo Llama", 51: "Vista Lince",
               52: "Corte Fuerte", 54: "Ausente", 61: "Mudar", 64: "Viscosecreción", 70: "Sequía", 73: "Humo Blanco",
               15: "Insomnio", 72: "Espíritu Vital", 46: "Presión", 47: "Sebo", 65: "Espesura", 66: "Mar Llamas",
               67: "Torrente", 68: "Enjambre"}
_SIMPLE_ES = {"paralyzed": "está paralizado y no se mueve", "asleep": "está dormido", "frozen": "está congelado",
              "flinched": "retrocedió", "confusion_hit": "se hirió en su confusión", "protected": "se protegió",
              "recharge": "necesita recargar", "charging": "está cargando energía", "woke": "se despertó",
              "thawed": "se descongeló", "confused": "está confuso", "unconfused": "ya no está confuso",
              "seeded": "fue infectado con drenadoras", "trapped": "quedó atrapado", "substitute": "creó un sustituto",
              "substitute_broke": "perdió su sustituto", "protect": "se protege", "endure": "aguanta el golpe",
              "taunted": "cayó en la mofa", "encored": "sufre los efectos de otra vez", "yawn": "empieza a tener sueño",
              "cured": "se curó", "knocked_off": "perdió su objeto", "cursed": "fue maldecido"}


def _name(dex: Dex, species: int) -> str:
    return dex.species[species].name.title() if 0 <= species < len(dex.species) else f"#{species}"


def describe(event: Event, active: dict[int, int], dex: Dex, labels: tuple[str, str], short: bool = False) -> str | None:
    """One Spanish line for an event (None when it is not worth showing). Updates ``active`` on switches."""
    kind = event[0]
    if kind == "switch":
        active[int(event[1])] = int(event[3])
        return f"{labels[event[1]]} saca a {_name(dex, int(event[3]))}."
    side = int(event[1]) if len(event) > 1 and isinstance(event[1], int) and event[1] in (0, 1) else 0
    name = _name(dex, active[side]) if side in active else labels[side]
    who = name if short or side not in active else f"{name} ({labels[side]})"
    if kind == "move":
        return f"{who} usó {dex.moves[int(event[3])].name.title()}."
    if kind == "hit":
        _, _, damage, label, critical, move_type, _, species, substitute = event
        parts = []
        if critical:
            parts.append("¡Golpe crítico!")
        if label == "super":
            parts.append("¡Es súper eficaz!")
        elif label == "resisted":
            parts.append("No es muy eficaz...")
        tail = " al sustituto" if substitute else ""
        parts.append(f"{_name(dex, int(species))} pierde {damage} PS{tail} ({TYPE_NAMES[int(move_type)]}).")
        return " ".join(parts)
    if kind == "noeffect":
        return f"No afecta a {_name(dex, int(event[4]))} ({TYPE_NAMES[int(event[2])]})."
    if kind == "miss":
        return f"¡El ataque de {who} falló!"
    if kind == "fail":
        return "Pero falló."
    if kind == "faint":
        return f"¡{who} se debilitó!"
    if kind == "status":
        return f"{who} {_STATUS_ES.get(str(event[2]), str(event[2]))}."
    if kind == "boost":
        stat = _STAT_ES.get(STAT_NAMES[int(event[2])], STAT_NAMES[int(event[2])])
        delta = int(event[3])
        return f"{stat} de {who} {'subió' if delta > 0 else 'bajó'}{' mucho' if abs(delta) >= 2 else ''}."
    if kind == "residual":
        return f"{who} sufre {event[2]} PS por {_RESIDUAL_ES.get(str(event[3]), str(event[3]))}."
    if kind == "heal":
        return f"{who} recupera {event[2]} PS."
    if kind == "ability":
        return f"[{_ABILITY_ES.get(int(event[2]), f'Habilidad #{event[2]}')}] de {who}."
    if kind == "weather":
        return _WEATHER_ES.get(str(event[1]), str(event[1])) + "."
    if kind == "weather_end":
        return "El clima volvió a la normalidad."
    if kind == "multihit":
        return f"¡{event[2]} golpes!"
    if kind in _SIMPLE_ES:
        return f"{who} {_SIMPLE_ES[str(kind)]}."
    if kind == "unsupported":
        return f"(efecto de {dex.moves[int(event[2])].name.title()} no modelado en el Stadium)"
    if kind == "end":
        winner = int(event[1])
        return "Empate." if winner == 2 else f"¡Gana {labels[winner]}!"
    return None


def narrate(events: Sequence[Event], dex: Dex, labels: tuple[str, str]) -> list[str]:
    """One Spanish line per relevant event; ``labels`` are the two sides' names."""
    active: dict[int, int] = {}
    lines = []
    for event in events:
        line = describe(event, active, dex, labels)
        if line is not None:
            lines.append(line)
    return lines


def status_name(status: int) -> str:
    if status & SLEEP_MASK:
        return "slp"
    for bit, name in ((TOXIC, "tox"), (POISON, "psn"), (BURN, "brn"), (FREEZE, "frz"), (PARALYSIS, "par")):
        if status & bit:
            return name
    return ""


def timeline(events: Sequence[Event], steps: Sequence[Snapshot], teams: tuple[Sequence[MonSpec], Sequence[MonSpec]],
             dex: Dex, labels: tuple[str, str]) -> list[dict[str, Any]]:
    """Frames for the arena: text + who is active + HP and status of every team member."""
    max_hp = [[spec.stats[0] for spec in team] for team in teams]
    hp = [list(row) for row in max_hp]
    status = [[""] * len(team) for team in teams]
    on_field = [0, 0]
    species: dict[int, int] = {}
    frames: list[dict[str, Any]] = []
    step_at = {count: state for count, state in steps}
    for index, event in enumerate(events):
        kind = str(event[0])
        side = int(event[1]) if len(event) > 1 and isinstance(event[1], int) and event[1] in (0, 1) else -1
        effect: dict[str, Any] = {}
        if kind == "switch":
            on_field[side] = int(event[2])
        elif kind == "hit":
            if not event[8]:
                hp[side][on_field[side]] = max(0, hp[side][on_field[side]] - int(event[2]))
            effect = {"eff": str(event[3]), "crit": bool(event[4]), "type": int(event[5])}
        elif kind == "noeffect":
            effect = {"eff": "no_effect", "type": int(event[2])}
        elif kind == "residual":
            hp[side][on_field[side]] = max(0, hp[side][on_field[side]] - int(event[2]))
        elif kind == "heal":
            member = on_field[side]
            hp[side][member] = min(max_hp[side][member], hp[side][member] + int(event[2]))
        elif kind == "faint":
            hp[side][int(event[2])] = 0
            status[side][int(event[2])] = ""
        elif kind == "status":
            status[side][on_field[side]] = str(event[2])
        elif kind == "move":
            move = dex.moves[int(event[3])]
            effect = {"move": move.name.title(), "type": move.move_type, "power": move.power}
        synced = step_at.get(index + 1)
        if synced is not None:  # the engine's own state after this turn step wins over the reconstruction
            for own in (0, 1):
                on_field[own] = synced[own][0]
                hp[own] = list(synced[own][1])
                status[own] = [status_name(value) for value in synced[own][2]]
        text = describe(event, species, dex, labels, short=True)
        if text is None:
            continue
        frame: dict[str, Any] = {"t": text, "k": kind, "s": side, "a": list(on_field),
                                 "hp": [list(hp[0]), list(hp[1])], "st": [list(status[0]), list(status[1])]}
        if effect:
            frame["fx"] = effect
        frames.append(frame)
    return frames
