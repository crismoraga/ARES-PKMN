"""ARES-Battle (experiment E1): cross-generation transfer of a game-agnostic battle policy.

Modules:

* ``teams``    - random 3v3 level-50 teams per generation pool (Gen1 train / Gen1 held-out / Gen2 / Gen3).
* ``obs``      - ID-free entity/action tokenisation, the species-ID ablation and the flat DQN vector.
* ``env``      - a battle wrapper that tracks revealed information, rewards and anti-vice counters.
* ``agents``   - baselines (Random, Greedy, ARES prior expert) and neural agents.
* ``model``    - the entity Transformer actor-critic (hand-written pre-norm blocks) and the DQN MLP.
* ``rollout``  - multiprocessing CPU rollout/evaluation workers (forward passes only, never backward).
* ``ppo``      - PPO self-play with a league (annealed LR/entropy, target-KL, pipelined rollouts).
* ``bc``       - imitation warm-start: demonstrations of the ARES prior/Greedy on Gen1 train teams.
* ``dqn``      - the DQN baseline trained against the same opponent mix and step budget.
* ``evaluate`` - evaluation harness with Wilson intervals and anti-vice metrics.
* ``report``   - aggregation over seeds: markdown tables, statistics and figures.
"""
