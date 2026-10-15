"""Show both real emulators and their campaign status in a read-only window."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from collections.abc import Mapping
from typing import cast

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from core.game_data import ROOT
from core.runtime_view import LiveCampaignReader, LiveCampaignSnapshot, event_caption, nested_mapping, status_number

BACKGROUND = (15, 20, 31)
PANEL = (23, 32, 46)
TEXT = (233, 239, 247)
MUTED = (159, 179, 197)
GREEN = (104, 218, 159)
GOLD = (242, 203, 112)
RED = (241, 134, 134)


def _fit_text(font: pygame.font.Font, value: str, width: int) -> str:
    if font.size(value)[0] <= width:
        return value
    trimmed = value
    while trimmed and font.size(trimmed + "…")[0] > width:
        trimmed = trimmed[:-1]
    return trimmed + "…"


def _text(surface: pygame.Surface, font: pygame.font.Font, value: str,
          x: int, y: int, width: int, color: tuple[int, int, int] = TEXT) -> None:
    surface.blit(font.render(_fit_text(font, value, width), True, color), (x, y))


def _party_caption(status: Mapping[str, object]) -> str:
    value = status.get("party")
    if not isinstance(value, list):
        return "Equipo: esperando la lectura de la partida"
    captions: list[str] = []
    for mon in value[:6]:
        if isinstance(mon, Mapping):
            values = cast(Mapping[str, object], mon)
            captions.append(f"#{status_number(values, 'species')} N{status_number(values, 'level')} "
                            f"{status_number(values, 'hp')}/{status_number(values, 'maxHP')} PS")
    return "Equipo: " + "  ·  ".join(captions)


def _draw_panel(surface: pygame.Surface, rect: pygame.Rect, snapshot: LiveCampaignSnapshot,
                reader: LiveCampaignReader, title_font: pygame.font.Font,
                body_font: pygame.font.Font, small_font: pygame.font.Font) -> None:
    pygame.draw.rect(surface, PANEL, rect, border_radius=14)
    x, y, width = rect.x + 14, rect.y + 12, rect.width - 28
    title = "Pokémon FireRed" if snapshot.game == "firered" else "Pokémon Emerald"
    _text(surface, title_font, title, x, y, width)
    backend = str(snapshot.status.get("backend", reader.backend))
    native = backend == "native"
    indicator = "ROM real · mGBA" if native else "Laboratorio sintético"
    _text(surface, small_font, indicator, x, y + 34, width, GREEN if native else GOLD)
    viewport_y = y + 63
    viewport_width = width
    viewport_height = viewport_width * 160 // 240
    viewport = pygame.Rect(x, viewport_y, viewport_width, viewport_height)
    pygame.draw.rect(surface, (4, 7, 12), viewport)
    if snapshot.rgb is not None:
        image = pygame.image.frombuffer(snapshot.rgb, (240, 160), "RGB")
        surface.blit(pygame.transform.scale(image, viewport.size), viewport.topleft)
    else:
        _text(surface, body_font, "Esperando el emulador…", x + 20, viewport.centery - 12, width - 40, MUTED)
    y = viewport.bottom + 13
    status, metrics = snapshot.status, nested_mapping(snapshot.status, "metrics")
    running = bool(status.get("running", False))
    error = str(status.get("error", ""))
    state = "En ejecución" if running else ("Detenido" if status else "Esperando datos")
    if status.get("complete") or status.get("native_campaign_certified"):
        state = "Campaña completada y verificada"
    if error:
        state = "Error: " + error
    if snapshot.rgb is not None and snapshot.age_seconds > 3 and running:
        state = f"Última imagen hace {snapshot.age_seconds:.1f} s"
    _text(surface, body_font, f"{state} · {status_number(status, 'frames'):,} fotogramas", x, y, width,
          RED if error else (GREEN if running else GOLD))
    _text(surface, body_font, f"Victorias {status_number(metrics, 'wins')}  Derrotas {status_number(metrics, 'losses')}  "
          f"Debilitados {status_number(metrics, 'faints')}  Capturas {status_number(metrics, 'captures')}",
          x, y + 27, width)
    directive, position = nested_mapping(status, "directive"), nested_mapping(status, "position")
    _text(surface, small_font, f"Objetivo: {directive.get('milestone', directive.get('mode', 'esperando'))}", x, y + 56, width, GOLD)
    _text(surface, small_font, f"Mapa {position.get('map_group', '?')}:{position.get('map_num', '?')}  "
          f"Posición ({position.get('x', '?')}, {position.get('y', '?')}, {position.get('z', '?')})  "
          f"Menú: {status.get('menu', '—')}", x, y + 80, width, MUTED)
    _text(surface, small_font, _party_caption(status), x, y + 104, width)
    _text(surface, small_font, f"Decisiones con prueba {status_number(metrics, 'proved_decisions')}  "
          f"Sin prueba {status_number(metrics, 'unproven_decisions')}  Guardados {status_number(metrics, 'checkpoints')}",
          x, y + 128, width, MUTED)
    _text(surface, small_font, "Actividad reciente", x, y + 159, width, MUTED)
    for index, event in enumerate(reader.events[-5:]):
        line_y = y + 184 + index * 22
        if line_y + 20 < rect.bottom - 10:
            _text(surface, small_font, event_caption(event), x, line_y, width, MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "artifacts/campaigns")
    parser.add_argument("--backend", choices=("native", "synthetic"), default="native")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--snapshot", type=Path,
                        help="Guarda cada2seg una imagen PNG de la ventana mostrada")
    args = parser.parse_args()
    if not 1 <= args.scale <= 6:
        parser.error("--scale debe estar entre1 y6")
    pygame.init()
    try:
        info = pygame.display.Info()
        desired_width = 2 * 240 * int(args.scale) + 110
        desired_height = 160 * int(args.scale) + 450
        width = min(desired_width, max(960, info.current_w - 60))
        height = min(desired_height, max(720, info.current_h - 80))
        surface = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        caption = ("ARES · Emuladores reales en vivo · FireRed + Emerald" if args.backend == "native"
                   else "ARES · Laboratorio sintético en vivo · FireRed + Emerald")
        pygame.display.set_caption(caption)
        title_font = pygame.font.SysFont("Segoe UI", 27, bold=True)
        body_font = pygame.font.SysFont("Segoe UI", 20)
        small_font = pygame.font.SysFont("Consolas", 16)
        readers = [LiveCampaignReader(args.checkpoint_root.resolve(), game, str(args.backend))
                   for game in ("firered", "emerald")]
        clock = pygame.time.Clock()
        snapshot_path = cast(Path | None, args.snapshot)
        if snapshot_path is not None:
            snapshot_path = snapshot_path.resolve()
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        last_snapshot = 0.0
        active = True
        while active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    active = False
            surface.fill(BACKGROUND)
            screen_width, screen_height = surface.get_size()
            _text(surface, small_font, "ARES · progreso en tiempo real. Cerrar esta ventana solo cierra el visor.",
                  18, 10, screen_width - 36, MUTED)
            panel_width = (screen_width - 54) // 2
            for index, reader in enumerate(readers):
                snapshot = reader.poll()
                reader.poll_events()
                rect = pygame.Rect(18 + index * (panel_width + 18), 40, panel_width, screen_height - 55)
                _draw_panel(surface, rect, snapshot, reader, title_font, body_font, small_font)
            pygame.display.flip()
            now = time.monotonic()
            if snapshot_path is not None and now - last_snapshot >= 2:
                # This is the displayed pygame surface, including both actual
                # worker framebuffers and HUD. It never synthesizes game pixels.
                temporary = snapshot_path.with_name(snapshot_path.stem + ".pending.png")
                try:
                    pygame.image.save(surface, str(temporary))
                    os.replace(temporary, snapshot_path)
                except OSError:
                    pass  # viewers may briefly hold the previous PNG on Windows
                last_snapshot = now
            clock.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
