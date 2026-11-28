"""ARES-U universal player: one game-agnostic agent for any Pokemon game with a ``GameAdapter``.

The agent (``agent.UniversalAgent``) never contains game-specific logic: every RAM address, text
encoding and menu quirk lives in an adapter (``gen1.Gen1Adapter`` for Pokemon Rojo,
``gen2.Gen2Adapter`` for Pokemon Cristal). What the agent learns (type chart evidence, dialog
answers, menu label semantics, which kinds of exploration targets produce story progress, how
healing works) is stored in a persistent knowledge base (``knowledge.KnowledgeBase``,
``artifacts/universal/knowledge.json``) that is loaded when the agent starts a new game.
"""
