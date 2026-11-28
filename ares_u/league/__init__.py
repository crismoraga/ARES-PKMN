"""ARES league: truly different rivals, measured diversity and a champion trained against all of them.

Modules:

* ``archetypes`` - hand-written battle styles (stall, setup sweeper, pivot, gambler, disruptor, game
  theorist, ...) as stadium policies built on public information only, plus the cheating lookahead.
* ``registry``   - rival specs (archetypes, E1 neural personalities, exploiters, snapshots) shipped to
  worker processes, and the controllers they become there.
* ``behaviour``  - per-side behaviour counters of a battle (switch / status / setup rates, damage per
  turn, type-effective choices) and the distances that prove rivals are not clones.
* ``ratings``    - Bradley-Terry ratings on the Elo scale, bootstrap intervals, non-transitive cycles.
* ``pfsp``       - prioritized fictitious self-play: payoff table and f(p) = (1 - p)^2 matchmaking.
* ``tasks``      - worker-side tasks (league rollouts, behaviour evaluations, decision probes).
* ``trainer``    - the league training loop: main agent + PFSP + periodically generated exploiters,
  with a live status published once per second (``core.live_status``).
"""
