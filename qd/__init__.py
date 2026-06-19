"""QD-over-Skills: Quality-Diversity (MAP-Elites) search over SkillOpt skills.

Extends the frozen ``SkillOpt/`` fork (imported as ``skillopt``). See the
``QD-over-Skills/`` docs (BRIEF, SPEC, AGENTS) for the design and red lines.

Red lines (BRIEF §4): pure-API / no-GPU; K=1 must reduce exactly to SkillOpt
(regression test = this package's ``tests/test_k1_reduces_to_skillopt``);
per-cell strict ``>`` gate (ties reject); descriptor from trajectory τ only.
"""
