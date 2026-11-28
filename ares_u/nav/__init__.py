"""ARES-Nav (experiment E2): pixel-based exploration with cross-game transfer FireRed -> Emerald.

Modules:
    env     -- torch-free emulator environment (universal 9-action space, pixel observations,
               RAM used only for metrics) and a multiprocessing vectorised wrapper.
    states  -- derivation and provenance of the fixed post-intro start savestates.
    models  -- conv encoder, conv autoencoder, PPO actor-critic and RND networks.
    train   -- frame collection, AE pretraining, PPO+RND training, evaluation and the protocol.
"""
