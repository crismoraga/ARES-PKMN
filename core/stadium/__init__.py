"""ARES-ADVERSARIAL: a Gen-3 battle stadium where adversarial agents coevolve.

The stadium is a research laboratory, never a campaign certificate: battles here
are simulated with the project's exact integer damage engine and ROM data, so
nothing in this package writes emulator RAM or touches campaign progress.

Layers (each importable on its own):

* ``dex``        - immutable species/move/type tables read once from the ROM.
* ``engine``     - deterministic singles battle engine (6v6, switches, status...).
* ``teams``      - teams harvested from real FireRed/Emerald campaign checkpoints.
* ``knowledge``  - what an agent has *observed* (learned type chart, revealed sets).
* ``features``   - option features and deep-policy observations (no hidden info).
* ``policies``   - evolvable option-scoring MLPs, baselines and the deep policy.
* ``match``      - duels with behaviour descriptors and decision traces.
* ``game_qd``    - GAME coevolution with Ranking/Pareto/Behavior/Random tasks.
* ``metrics``    - the six adversarial QD metrics of the paper plus Elo.
* ``learning``   - PPO, archive distillation and counterfactual lessons.
* ``league``     - the live two-sided ARES-ADVERSARIAL loop and its telemetry.
"""
