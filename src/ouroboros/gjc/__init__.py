"""GJC runtime artifact projection helpers."""

from ouroboros.gjc.artifacts import (
    GJC_SKILL_NAMESPACE,
    GjcSkillInstallResult,
    gjc_skills_root,
    has_managed_gjc_skill_projection,
    install_gjc_skills,
    remove_gjc_skills,
)

__all__ = [
    "GJC_SKILL_NAMESPACE",
    "GjcSkillInstallResult",
    "gjc_skills_root",
    "has_managed_gjc_skill_projection",
    "install_gjc_skills",
    "remove_gjc_skills",
]
