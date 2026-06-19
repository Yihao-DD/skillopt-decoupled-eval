"""Root pytest conftest — make ``import skillopt`` + ``import qd`` resolve
WITHOUT relying on a per-machine editable ``.pth``.

Imports must work on a CLEAN clone (the company side) and must survive the reorg
move ``SkillOpt/`` -> ``vendor/SkillOpt/``. We prepend to sys.path: (a) the repo
root (so ``import qd`` works regardless of pytest import mode) and (b) whichever
of ``vendor/SkillOpt`` (post-reorg) or ``SkillOpt`` (pre-reorg) actually contains
the ``skillopt`` package.  See REORG_PLAN.md (S10) / audit "K=1 path-fragility".
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# (a) repo root -> `import qd`
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# (b) the vendored SkillOpt fork (post-reorg) or legacy SkillOpt/ (pre-reorg) -> `import skillopt`
for _cand in ("vendor/SkillOpt", "SkillOpt"):
    _pkg_parent = _ROOT / _cand
    if (_pkg_parent / "skillopt" / "__init__.py").exists():
        if str(_pkg_parent) not in sys.path:
            sys.path.insert(0, str(_pkg_parent))
        break
