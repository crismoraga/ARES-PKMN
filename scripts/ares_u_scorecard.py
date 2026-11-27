"""Tablero de veredictos: cada pregunta del proyecto con su métrica, su criterio de éxito y su resultado.

How ARES-U is measured, and how we know whether a result is good, in one place. Every row is computed
from the protocol outputs (nothing typed by hand) and states:

* the question, the metric (with its unit) and the baseline it is compared with;
* the success criterion, fixed before the result is looked at (E1: commit 53d61ec; E5: 392d951 and
  3e6e549; E3/E4: "the 95% interval excludes the reference value");
* the result with its 95% interval and the verdict: PASS (criterion met), FAIL (not met), PARTIAL (met in
  k of n games) or PENDING (runs not finished).

Intervals: battles use the paired battle-level bootstrap of ``ares_u.battle.paired``; exploration uses a
hierarchical bootstrap (resample seeds, then episodes within each seed; the same seeds for every arm, so
arms stay paired by seed). Exploration scores are normalised per game as (R - random) / (scratch - random):
0 = random play, 1 = the current practice (train that game from scratch with the same budget).

    python -m scripts.ares_u_scorecard
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "ares_u"
E3, E4 = ART / "e3", ART / "e4"
CHAIN = ("rojo", "cristal", "emerald", "platino", "negro")
HELD_OUT = ("heartgold", "negro2")
E3_GAMES = ("rojo", "cristal", "platino", "heartgold", "negro", "negro2")
TITLE = {"rojo": "Rojo", "cristal": "Cristal", "emerald": "Esmeralda", "platino": "Platino", "negro": "Negro",
         "heartgold": "HeartGold", "negro2": "Negro 2"}
DRAWS = 4000
PASS, FAIL, PARTIAL, PENDING = "PASS", "FAIL", "PARTIAL", "PENDING"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def episodes(path: Path, key: str = "screens") -> NDArray[np.float64] | None:
    data = load(path)
    return None if data is None or key not in data else np.asarray(data[key], dtype=np.float64)


def baseline_path(game: str, method: str, seed: int) -> Path:
    if game == "emerald":
        return E4 / "baselines" / "emerald" / f"seed{seed}" / f"eval_{method}.json"
    return E3 / game / f"seed{seed}" / f"eval_{method}.json"


def arms_for(paths: Mapping[str, Callable[..., Path]], key: str = "screens") -> dict[str, list[NDArray[np.float64]]] | None:
    """Per-arm list of per-seed episode arrays, over the seeds where *every* arm exists (paired seeds)."""
    collected: dict[str, list[NDArray[np.float64]]] = {arm: [] for arm in paths}
    for seed in range(10):
        values = {arm: episodes(fn(seed), key) for arm, fn in paths.items()}
        if all(v is not None and len(v) for v in values.values()):
            for arm, v in values.items():
                collected[arm].append(v)  # type: ignore[arg-type]
    return collected if collected[next(iter(paths))] else None


def hboot(stat: Callable[[dict[str, float]], float], arms: dict[str, list[NDArray[np.float64]]], seed: int = 0
          ) -> tuple[float, float, float, int]:
    """(point, low, high, seeds) of ``stat`` with a hierarchical bootstrap (seeds, then episodes)."""
    rng = np.random.default_rng(seed)
    count = len(next(iter(arms.values())))
    point = stat({arm: float(np.mean([v.mean() for v in values])) for arm, values in arms.items()})
    samples = []
    for _ in range(DRAWS):
        chosen = rng.integers(count, size=count)
        means = {arm: float(np.mean([rng.choice(values[i], len(values[i])).mean() for i in chosen]))
                 for arm, values in arms.items()}
        value = stat(means)
        if np.isfinite(value):
            samples.append(value)
    low, high = np.percentile(samples, [2.5, 97.5]) if samples else (np.nan, np.nan)
    return point, float(low), float(high), count


def normalised(agent: str) -> Callable[[dict[str, float]], float]:
    def stat(m: dict[str, float]) -> float:
        span = m["scratch"] - m["random"]
        return (m[agent] - m["random"]) / span if abs(span) > 1e-9 else float("nan")
    return stat


def ci_text(point: float, low: float, high: float, digits: int = 2) -> str:
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def verdict_count(passed: int, total: int) -> str:
    if total == 0:
        return PENDING
    return PASS if passed == total else FAIL if passed == 0 else PARTIAL


# ----------------------------------------------------------------------------- rows

def row(question: str, metric: str, criterion: str, result: str, verdict: str, source: str) -> dict[str, str]:
    return {"question": question, "metric": metric, "criterion": criterion, "result": result, "verdict": verdict,
            "source": source}


def metric_validity() -> dict[str, str]:
    summary = load(E3 / "summary.json")
    rho = summary["validation"]["spearman_rho"] if summary else None
    n = summary["validation"]["episodes"] if summary else 0
    return row("¿La métrica de exploración mide algo real?", "Spearman entre pantallas únicas y celdas RAM visitadas",
               "$\\rho \\geq 0.7$", f"$\\rho={rho:.2f}$ ($n={n}$)" if rho is not None else "--",
               PENDING if rho is None else PASS if rho >= 0.7 else FAIL, "E3")


def battle_rows() -> list[dict[str, str]]:
    paired = load(ART / "battle" / "final" / "paired.json")
    rows = []
    specs = (("H1", "¿El combate se transfiere a especies nunca vistas?", "puntaje PPO $-$ 0.5 contra Greedy, Gen2$^\\ast$/Gen3$^\\ast$"),
             ("H3", "¿La representación por atributos supera al DQN del curso?", "puntaje PPO $-$ DQN, Gen2$^\\ast$/Gen3$^\\ast$"),
             ("H2", "¿Quitar los IDs de especie ayuda a generalizar?", "puntaje PPO $-$ PPO+IDs, Gen2$^\\ast$/Gen3$^\\ast$"))
    for hid, question, metric in specs:
        tests = [t for t in (paired or {}).get("family", []) if t["id"] == hid]
        if not tests:
            rows.append(row(question, metric, "dif.\\ $>0$, $p_{\\mathrm{Holm}}<0.05$", "--", PENDING, "E1"))
            continue
        passed = sum(1 for t in tests if t["diff"] > 0 and t["p_holm"] < 0.05)
        result = "; ".join(f"{t['split'].replace('gen', 'G')}: {t['diff']:+.3f} ($p_H$={t['p_holm']:.2g})" for t in tests)
        rows.append(row(question, metric, "dif.\\ $>0$, $p_{\\mathrm{Holm}}<0.05$", result,
                        verdict_count(passed, len(tests)), "E1"))
    return rows


def cross_console() -> dict[str, str]:
    passed, total, parts = 0, 0, []
    for game in E3_GAMES:
        arms = arms_for({"zero_shot": lambda s, g=game: E3 / g / f"seed{s}" / "eval_zero_shot.json",
                         "random": lambda s, g=game: E3 / g / f"seed{s}" / "eval_random.json"})
        if arms is None:
            continue
        point, low, high, _ = hboot(lambda m: m["zero_shot"] - m["random"], arms)
        total += 1
        passed += int(low > 0)
        parts.append(f"{TITLE[game]} {point:+.0f}")
    return row("¿Lo aprendido en GBA sirve en otras consolas sin reentrenar?",
               "pantallas únicas: zero-shot $-$ azar, por juego nuevo", "IC95\\% $>0$",
               f"{passed}/{total}: " + ", ".join(parts) if total else "--", verdict_count(passed, total), "E3")


def curriculum_rows() -> list[dict[str, str]]:
    jump_pass = jump_total = gain_pass = gain_total = held_pass = held_total = 0
    jumps, gains, helds, forgets = [], [], [], []
    forget_values = []
    for stage, game in enumerate(CHAIN):
        base = {"random": lambda s, g=game: baseline_path(g, "random", s),
                "scratch": lambda s, g=game: baseline_path(g, "scratch", s)}
        if stage > 0:
            arms = arms_for({**base, "agent": lambda s, st=stage - 1, g=game:
                             E4 / "finetune" / f"seed{s}" / f"after{st}_on_{g}.json"})
            if arms is not None:
                point, low, high, _ = hboot(normalised("agent"), arms)
                jump_total += 1
                jump_pass += int(low > 0)
                jumps.append(f"{TITLE[game]} {point:.2f}")
            arms = arms_for({**base, "agent": lambda s, st=stage, g=game:
                             E4 / "finetune" / f"seed{s}" / f"after{st}_on_{g}.json"})
            if arms is not None:
                point, low, high, _ = hboot(normalised("agent"), arms)
                gain_total += 1
                gain_pass += int(low > 1)
                gains.append(f"{TITLE[game]} {ci_text(point, low, high)}")
        if stage < len(CHAIN) - 1:
            arms = arms_for({**base,
                             "own": lambda s, st=stage, g=game: E4 / "finetune" / f"seed{s}" / f"after{st}_on_{g}.json",
                             "last": lambda s, g=game: E4 / "finetune" / f"seed{s}" / f"after{len(CHAIN) - 1}_on_{g}.json"})
            if arms is not None:
                point, low, high, _ = hboot(lambda m: normalised("last")(m) - normalised("own")(m), arms)
                forget_values.append((point, low, high))
                forgets.append(f"{TITLE[game]} {point:+.2f}")
    for game in HELD_OUT:
        arms = arms_for({"random": lambda s, g=game: baseline_path(g, "random", s),
                         "scratch": lambda s, g=game: baseline_path(g, "scratch", s),
                         "agent": lambda s, g=game: E4 / "finetune" / f"seed{s}" / f"after{len(CHAIN) - 1}_on_{g}.json"})
        if arms is not None:
            point, low, high, _ = hboot(normalised("agent"), arms)
            held_total += 1
            held_pass += int(low > 0)
            helds.append(f"{TITLE[game]} {ci_text(point, low, high)}")
    forget_pass = sum(1 for _, low, _ in forget_values if low >= 0) + \
        sum(1 for _, low, high in forget_values if low < 0 <= high)
    return [
        row("¿Arranca mejor que el azar al llegar a cada generación? (transferencia hacia adelante)",
            "puntaje normalizado al llegar, sin entrenar en ese juego", "IC95\\% $>0$",
            f"{jump_pass}/{jump_total}: " + ", ".join(jumps) if jump_total else "--",
            verdict_count(jump_pass, jump_total), "E4"),
        row("¿Supera a la práctica actual (cada juego desde cero, mismo presupuesto)?",
            "puntaje normalizado tras entrenar ese juego", "IC95\\% $>1$",
            f"{gain_pass}/{gain_total}: " + "; ".join(gains) if gain_total else "--",
            verdict_count(gain_pass, gain_total), "E4"),
        row("¿Generaliza a juegos nunca entrenados?", "puntaje normalizado del agente final (HeartGold, Negro 2)",
            "IC95\\% $>0$", f"{held_pass}/{held_total}: " + "; ".join(helds) if held_total else "--",
            verdict_count(held_pass, held_total), "E4"),
        row("¿Recuerda los juegos anteriores? (transferencia hacia atrás)",
            "normalizado al final $-$ normalizado justo tras entrenarlo", "IC95\\% no bajo 0",
            f"{forget_pass}/{len(forget_values)}: " + ", ".join(forgets) if forget_values else "--",
            verdict_count(forget_pass, len(forget_values)), "E4"),
    ]


def places_row() -> dict[str, str]:
    """Distinct maps (RAM map id) per episode: closer to progress than screens (GB/GBC/GBA adapters)."""
    passed, total, parts = 0, 0, []
    for stage, game in enumerate(CHAIN[:3]):
        if stage == 0:
            continue
        arms = arms_for({"scratch": lambda s, g=game: baseline_path(g, "scratch", s),
                         "agent": lambda s, st=stage, g=game: E4 / "finetune" / f"seed{s}" / f"after{st}_on_{g}.json"},
                        key="maps")
        if arms is None:
            continue
        point, low, high, _ = hboot(lambda m: m["agent"] - m["scratch"], arms)
        total += 1
        passed += int(low > 0)
        parts.append(f"{TITLE[game]} {ci_text(point, low, high, 1)}")
    return row("¿Llega a más lugares, no solo a más pantallas?",
               "mapas distintos (id de mapa en RAM) por episodio: currículo $-$ desde cero", "IC95\\% $>0$",
               f"{passed}/{total}: " + "; ".join(parts) if total else "--", verdict_count(passed, total), "E4")


def evolution_rows() -> list[dict[str, str]]:
    rows = []
    v1 = load(ART / "e5_battle" / "summary.json")
    c6 = next((c for c in (v1 or {}).get("comparisons", []) if c["id"] == "C6"), None)
    rows.append(row("¿La evolución mejora generación tras generación sin olvidar la primera?",
                    "puntaje contra Greedy en Gen1: U$_2$ $-$ U$_0$", "dif.\\ $>0$, $p_{\\mathrm{Holm}}<0.05$",
                    f"{c6['diff']:+.3f} [{c6['ci'][0]:+.3f}, {c6['ci'][1]:+.3f}], $p_H$={c6['p_holm']:.2g}" if c6 else "--",
                    PENDING if c6 is None else PASS if c6["diff"] > 0 and c6["p_holm"] < 0.05 else FAIL, "E5-combate"))
    v2 = load(ART / "e5_battle" / "v2" / "summary.json")
    comps = [c for c in (v2 or {}).get("comparisons", []) if c["id"] in ("D1", "D2", "D3")]
    source, label = "E5-combate v2", "D1--D3"
    if not comps:
        comps = [c for c in (v1 or {}).get("comparisons", []) if c["id"] in ("C2", "C3", "C4")]
        source, label = "E5-combate v1", "C2--C4"
    passed = sum(1 for c in comps if c["diff"] > 0 and c["p_holm"] < 0.05)
    rows.append(row("¿La población que compite y se fusiona supera a un linaje único con el mismo cómputo?",
                    f"puntaje contra Greedy: unificado $-$ linaje único ({label})", "dif.\\ $>0$, $p_{\\mathrm{Holm}}<0.05$",
                    ", ".join(f"{c['split'].replace('gen', 'G')} {c['diff']:+.3f} ($p_H$={c['p_holm']:.2g})" for c in comps)
                    if comps else "--", verdict_count(passed, len(comps)), source))
    better = total = 0
    nav = load(ART / "e5" / "summary.json") or []
    for record in nav:
        total += 1
        better += int(record["test"]["unified"]["mean"] >= record["test"]["best_proposal"]["mean"])
    v2_tests = [load(p) for p in sorted((ART / "e5_battle" / "v2").glob("seed*/test.json"))]
    for test in v2_tests:
        for g, split in ((1, "gen2"), (2, "gen3")):
            children = test["children"].get(f"v2_gen{g}", {})
            if children:
                total += 1
                better += int(float(np.mean(test["vs_greedy"][f"U{g}v2"][split])) >= max(c["test"] for c in children.values()) - 1e-9)
    rows.append(row("¿Fusionar las propuestas no empeora a la mejor de ellas?",
                    "test: unificado $\\geq$ mejor propuesta individual, por generación", "en la mayoría de las generaciones",
                    f"{better}/{total} generaciones" if total else "--",
                    PENDING if total == 0 else PASS if better > total / 2 else FAIL, "E5"))
    rel = (v2 or {}).get("reliability", {})
    if rel.get("v2", {}).get("mean") is not None:
        r1, r2 = rel["v1"]["mean"], rel["v2"]["mean"]
        rows.append(row("¿La selección elige de verdad a los mejores? (medir la medición)",
                        "Spearman entre ranking de validación y puntaje de test de los hijos", "media $>0.5$",
                        f"v1 (torneo, 200/par): {r1:.2f}; v2 (panel, $10^4$ pareados): {r2:.2f}",
                        PASS if r2 > 0.5 else FAIL, "E5-combate v2"))
    else:
        rows.append(row("¿La selección elige de verdad a los mejores? (medir la medición)",
                        "Spearman entre ranking de validación y puntaje de test de los hijos", "media $>0.5$", "--",
                        PENDING, "E5-combate v2"))
    return rows


def build() -> list[dict[str, str]]:
    rows = [metric_validity(), *battle_rows(), cross_console(), *curriculum_rows(), places_row(), *evolution_rows()]
    rows.append(row("¿Termina el juego?", "medallas obtenidas", "$\\geq 1$ medalla", "0 (no se persigue la historia)",
                    FAIL, "--"))
    return rows


SYMBOL = {PASS: "\\textcolor{green!50!black}{$\\checkmark$}", FAIL: "\\textcolor{red!70!black}{$\\times$}",
          PARTIAL: "\\textcolor{orange!80!black}{parcial}", PENDING: "\\textcolor{gray}{pend.}"}
MD = {PASS: "✔", FAIL: "✘", PARTIAL: "◐", PENDING: "…"}


def main() -> None:
    rows = build()
    (ART / "scorecard.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Tablero de veredictos", "", "| | Pregunta | Métrica | Criterio | Resultado | Fuente |", "|---|---|---|---|---|---|"]
    md += [f"| {MD[r['verdict']]} | {r['question']} | {r['metric']} | {r['criterion']} | {r['result']} | {r['source']} |"
           for r in rows]
    (ART / "scorecard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex = ["\\begin{tabular}{@{}cL{0.25\\textwidth}L{0.25\\textwidth}L{0.11\\textwidth}L{0.29\\textwidth}l@{}}",
           "\\toprule", " & Pregunta & Métrica (vs.\\ línea base) & Criterio & Resultado (IC 95\\%) & Exp. \\\\", "\\midrule"]
    tex += [f"{SYMBOL[r['verdict']]} & {r['question']} & {r['metric']} & {r['criterion']} & {r['result']} & {r['source']} \\\\"
            for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    generated = ROOT / "informe" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "tab_score.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    counts = {key: sum(1 for r in rows if r["verdict"] == key) for key in (PASS, FAIL, PARTIAL, PENDING)}
    print(ART / "scorecard.md", counts)


if __name__ == "__main__":
    main()
