# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Package data resource access for SYMFLUENCE.

Handles loading base_settings and config_templates from package data
in both development (editable install) and production (site-packages) modes.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

# Python 3.9+ importlib.resources
if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    # Fallback for older Python versions (though we require 3.9+)
    from importlib_resources import files


def get_bundled_base_settings_dir(model_name: str) -> Path:
    """
    Get path to the *bundled* base settings directory for a model.

    This locates only this package's OWN data — the framework-owned
    ``symfluence/resources/base_settings/<model_name>`` directory that ships
    fixtures (``TEST``) and the legacy central layout for models whose package
    has not registered a ``R.base_settings`` anchor.

    It is deliberately NOT the general resolver: choosing between a model
    package's registered anchor and this bundled fallback is contract-tier
    policy and lives in :func:`symfluence.core.modeling.base_settings.
    get_base_settings_dir`. Keeping the policy out of here is what lets this
    package stay data-only and stop importing ``symfluence.core`` (the two
    packages used to import each other).

    Args:
        model_name: Directory name under ``resources/base_settings`` (used
            literally, no normalization).

    Returns:
        Path to the bundled base settings directory for the model

    Raises:
        FileNotFoundError: If the bundled directory doesn't exist
    """
    base_settings_root = files('symfluence.resources.base_settings')
    model_settings = base_settings_root / model_name

    # Convert Traversable to Path
    # In editable mode, this is already a Path
    # In installed mode, this is a Traversable that we convert
    if hasattr(model_settings, '__fspath__'):
        path = Path(model_settings)
    else:
        # For Traversable objects, convert to string then Path
        path = Path(str(model_settings))

    # Verify the directory exists
    if not path.exists():
        raise FileNotFoundError(
            f"Base settings directory for model '{model_name}' not found at: {path}"
        )

    return path


def get_config_template(template_name: str = 'config_template.yaml') -> Path:
    """
    Get path to a configuration template file.

    Args:
        template_name: Name of template file (default: 'config_template.yaml')
                      Available templates:
                      - config_template.yaml
                      - config_template_comprehensive.yaml
                      - config_template_comprehensive_nested.yaml
                      - config_quickstart_minimal_nested.yaml

    Returns:
        Path to the template file

    Raises:
        FileNotFoundError: If template doesn't exist

    Examples:
        >>> template = get_config_template()
        >>> comprehensive = get_config_template('config_template_comprehensive.yaml')
    """
    try:
        # Get the templates directory
        templates_root = files('symfluence.resources.config_templates')
        template_file = templates_root / template_name

        # Convert to Path
        if hasattr(template_file, '__fspath__'):
            path = Path(template_file)
        else:
            path = Path(str(template_file))

        # Verify file exists
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Template '{template_name}' not found at: {path}")

        return path

    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as e:
        # Provide helpful error message with available templates
        available = ['config_template.yaml', 'config_template_comprehensive.yaml',
                    'config_template_comprehensive_nested.yaml',
                    'config_quickstart_minimal_nested.yaml']
        raise FileNotFoundError(
            f"Config template '{template_name}' not found.\n"
            f"Available templates: {', '.join(available)}"
        ) from e


def list_config_templates() -> list[Path]:
    """
    List all available configuration templates.

    Returns:
        List of Paths to template files (sorted alphabetically)

    Examples:
        >>> templates = list_config_templates()
        >>> for t in templates:
        ...     print(t.name)
    """
    try:
        templates_root = files('symfluence.resources.config_templates')

        # Handle both installed and editable modes
        if hasattr(templates_root, '__fspath__'):
            # Editable mode - can use pathlib
            root_path = Path(templates_root)
            templates = [f for f in root_path.glob('*.yaml') if f.is_file()]
        else:
            # Installed mode - use Traversable API
            templates = []
            try:
                for item in templates_root.iterdir():
                    if item.name.endswith('.yaml') and not item.name.startswith('__'):
                        # Convert Traversable to Path
                        templates.append(Path(str(item)))
            except AttributeError:
                # Fallback: manually construct known templates
                known_templates = [
                    'config_template.yaml',
                    'config_template_comprehensive.yaml',
                    'config_template_comprehensive_nested.yaml',
                    'config_quickstart_minimal_nested.yaml'
                ]
                for name in known_templates:
                    try:
                        path = get_config_template(name)
                        templates.append(path)
                    except FileNotFoundError:
                        pass

        return sorted(templates, key=lambda p: p.name)

    except (FileNotFoundError, ModuleNotFoundError):
        return []


def get_system_deps_registry_path() -> Path:
    """
    Get path to the system dependencies YAML registry.

    Returns:
        Path to system_deps.yml

    Raises:
        FileNotFoundError: If registry file doesn't exist
    """
    try:
        registry_file = files('symfluence.resources') / 'system_deps.yml'

        if hasattr(registry_file, '__fspath__'):
            path = Path(registry_file)
        else:
            path = Path(str(registry_file))

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"System deps registry not found at: {path}")

        return path

    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as e:
        raise FileNotFoundError(
            "System dependency registry (system_deps.yml) not found in package resources."
        ) from e


def get_skills_dir() -> Path:
    """
    Get path to the packaged agent-skills directory.

    The skills are domain guides (one ``SKILL.md`` per skill) that the
    ``symfluence agent`` launcher exposes to an external coding-agent CLI.

    Returns:
        Path to the ``symfluence.resources.skills`` directory.

    Raises:
        FileNotFoundError: If the packaged skills directory is missing.
    """
    try:
        skills_root = files('symfluence.resources') / 'skills'

        if hasattr(skills_root, '__fspath__'):
            path = Path(skills_root)
        else:
            path = Path(str(skills_root))

        if not path.is_dir():
            raise FileNotFoundError(f"Packaged skills directory not found at: {path}")

        return path

    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as e:
        raise FileNotFoundError(
            "Packaged agent skills (symfluence.resources.skills) not found."
        ) from e


def get_agents_dir() -> Path:
    """
    Get path to the packaged subagent-definition directory.

    Each ``*.md`` file (YAML frontmatter + prompt body) describes one
    specialized subagent that ``symfluence agent launch`` exposes to host CLIs
    that support custom agents.

    Returns:
        Path to the ``symfluence.resources.agents`` directory.

    Raises:
        FileNotFoundError: If the packaged agents directory is missing.
    """
    try:
        agents_root = files('symfluence.resources') / 'agents'

        if hasattr(agents_root, '__fspath__'):
            path = Path(agents_root)
        else:
            path = Path(str(agents_root))

        if not path.is_dir():
            raise FileNotFoundError(f"Packaged agents directory not found at: {path}")

        return path

    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as e:
        raise FileNotFoundError(
            "Packaged subagent definitions (symfluence.resources.agents) not found."
        ) from e


def _user_cache_base() -> Path:
    """The current user's cache directory (no third-party dependency)."""
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA')
        return Path(base) if base else Path.home() / 'AppData' / 'Local'
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Caches'
    base = os.environ.get('XDG_CACHE_HOME')
    return Path(base) if base else Path.home() / '.cache'


def _ensure_private_dir(path: Path) -> Path:
    """Create ``path`` as a directory only its owner can use, or refuse it.

    The agent cache carries permission requests and their replies, job
    records and the skill files handed to the agent CLI. If another local
    user could write there they could approve a tool call on the user's
    behalf or swap a skill file, so the directory must belong to the current
    user and be closed to group and others (mode 0700). An existing directory
    that fails either test -- or is a symlink -- is refused, never "fixed":
    its contents may already have been tampered with.

    On Windows the per-user profile directory already carries a user-only
    ACL, and POSIX mode bits are not meaningful, so only creation is done.
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if sys.platform == 'win32':
        return path
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PermissionError(
            f"Refusing to use the agent cache at {path}: it is not a real directory."
        )
    if info.st_uid != os.getuid():
        raise PermissionError(
            f"Refusing to use the agent cache at {path}: it is owned by another user."
        )
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError(
            f"Refusing to use the agent cache at {path}: it is accessible to other "
            f"users (mode {stat.S_IMODE(info.st_mode):04o}). Remove it, or run "
            f"'chmod 700 {path}' after checking its contents."
        )
    return path


def agent_cache_root() -> Path:
    """The per-user directory where launch-time agent context is materialized.

    Lives under the user's cache directory (``SYMFLUENCE_AGENT_CACHE_DIR``
    overrides the location), is created with mode 0700, and is verified to be
    owned by the current user and closed to others on every call -- it used to
    be a fixed name in the shared temp directory. The path is deterministic
    because the TUI and the MCP server (a child of the agent CLI, not of the
    TUI) meet there.

    Raises:
        PermissionError: If the directory exists but is not private to the
            current user.
    """
    override = os.environ.get('SYMFLUENCE_AGENT_CACHE_DIR')
    path = Path(override) if override else _user_cache_base() / 'symfluence' / 'agent'
    return _ensure_private_dir(path)


def parse_frontmatter(md_file: Path) -> tuple[dict, str] | None:
    """Parse a ``---``-fenced YAML-frontmatter markdown file into (metadata, body).

    The single frontmatter parser for the packaged skills and subagent
    definitions — the CLI listing, the TUI panels, and launch-time priming all
    go through here so they can never disagree about the same file.

    Returns None when the file is unreadable, does not *start* with a
    frontmatter fence (a mid-document ``---`` ruler is not frontmatter), or the
    YAML block is invalid / not a mapping.
    """
    try:
        text = md_file.read_text(encoding='utf-8')
    except OSError:
        return None
    text = text.lstrip('\ufeff')  # tolerate a BOM
    if not text.startswith('---'):
        return None
    try:
        import yaml
        _, frontmatter, body = text.split('---', 2)
        meta = yaml.safe_load(frontmatter)
    except Exception:  # noqa: BLE001 — malformed frontmatter is "no frontmatter"
        return None
    if not isinstance(meta, dict):
        return None
    return meta, body.strip()


def _selected_skills(
    skills_dir: Path, skills: tuple[str, ...] | None,
) -> list[Path]:
    """The packaged skill directories to materialize (all when ``skills`` is None)."""
    return [
        skill for skill in sorted(skills_dir.iterdir())
        if (skill / 'SKILL.md').is_file()
        and (skills is None or skill.name in skills)
    ]


def _render_agents_md(
    skills_dir: Path,
    preamble: str | None = None,
    skills: tuple[str, ...] | None = None,
) -> str:
    """Render the packaged skills into a single neutral ``AGENTS.md`` document."""
    lines = []
    if preamble:
        lines.extend([preamble, "", "---", ""])
    lines += [
        "# SYMFLUENCE agent skills",
        "",
        "These are SYMFLUENCE domain guides. Read the relevant skill before acting "
        "on a SYMFLUENCE task (adding a data handler or model, debugging calibration, "
        "running the workflow).",
        "",
    ]
    for skill in _selected_skills(skills_dir, skills):
        lines.append(f"## {skill.name}")
        lines.append("")
        lines.append((skill / 'SKILL.md').read_text(encoding='utf-8').strip())
        lines.append("")
    return "\n".join(lines)


def prepare_agent_context(
    skills_mode: str,
    workdir: Path,
    preamble: str | None = None,
    skills: tuple[str, ...] | None = None,
    cache_scope: str | None = None,
) -> tuple[list[str], list[str], bool]:
    """
    Materialize the packaged skills for an external coding-agent CLI.

    Args:
        skills_mode: How the target CLI consumes skills.
            ``"claude_native"`` — lay the skills out as ``.claude/skills/`` in a
            cache directory and return ``--add-dir`` so Claude Code discovers them
            without touching the user's project.
            ``"agents_md"`` — write a neutral ``AGENTS.md`` into ``workdir`` (the
            convention honoured by Codex/Gemini and other tools). A SYMFLUENCE-
            generated ``AGENTS.md`` from an earlier launch is refreshed in place;
            a user-authored one is never touched (and ``delivered`` is False so
            callers can report the gap honestly).
        workdir: The directory the agent CLI is launched from.
        preamble: Optional block (agent identity / project context) prepended to
            the generated ``AGENTS.md``. Ignored in ``claude_native`` mode, where
            identity travels via the CLI's own system-prompt flag.
        skills: Packaged skill names to materialize; None means all of them.
        cache_scope: Optional subdirectory of the agent cache to materialize
            into (e.g. an agent-mode name), so differently-primed launches
            don't clobber each other's cache payloads.

    Returns:
        ``(extra_argv, messages, delivered)`` — extra arguments to pass to the
        CLI, human-readable lines for the caller to log, and whether the skills
        (and preamble, in ``agents_md`` mode) actually reached the CLI. Skill
        materialization is skipped entirely when ``SYMFLUENCE_NO_SKILLS`` is set.
    """
    if os.environ.get('SYMFLUENCE_NO_SKILLS'):
        return [], ["Skill materialization disabled via SYMFLUENCE_NO_SKILLS."], False

    skills_dir = get_skills_dir()

    if skills_mode == 'claude_native':
        cache_root = agent_cache_root()
        if cache_scope:
            cache_root = cache_root / cache_scope
        target = cache_root / '.claude' / 'skills'
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        count = 0
        for skill in _selected_skills(skills_dir, skills):
            # The whole skill directory travels, so reference/asset files
            # shipped alongside SKILL.md survive materialization.
            shutil.copytree(skill, target / skill.name)
            count += 1
        return (
            ['--add-dir', str(cache_root)],
            [f"Exposed {count} SYMFLUENCE skill(s) to the agent via {cache_root}."],
            True,
        )

    if skills_mode == 'agents_md':
        agents_md = workdir / 'AGENTS.md'
        if agents_md.exists():
            existing = agents_md.read_text(encoding='utf-8', errors='replace')
            if '# SYMFLUENCE agent skills' not in existing:
                # User-authored file: never touch it, and say plainly that the
                # SYMFLUENCE context did NOT reach the CLI.
                return [], [
                    f"AGENTS.md in {workdir} is not SYMFLUENCE-generated; left "
                    f"unchanged. SYMFLUENCE identity/skills were NOT injected — "
                    f"merge or remove it to let SYMFLUENCE regenerate."
                ], False
            # Ours from an earlier launch: refresh (preamble + skills may be stale).
            agents_md.write_text(
                _render_agents_md(skills_dir, preamble, skills), encoding='utf-8')
            return [], [f"Refreshed SYMFLUENCE context in {agents_md}."], True
        agents_md.write_text(
            _render_agents_md(skills_dir, preamble, skills), encoding='utf-8')
        return [], [f"Wrote SYMFLUENCE skills to {agents_md}."], True

    return [], [], False


def copy_config_template_to_project(destination: Path,
                                    template_name: str = 'config_template.yaml',
                                    output_name: str = None) -> Path:
    """
    Copy a config template from package data to a project directory.

    Args:
        destination: Destination directory path
        template_name: Name of template to copy (default: 'config_template.yaml')
        output_name: Output filename (default: same as template_name)

    Returns:
        Path to the copied config file

    Raises:
        FileNotFoundError: If template doesn't exist
        PermissionError: If destination is not writable

    Examples:
        >>> dest = Path('./my_project')
        >>> config_path = copy_config_template_to_project(dest, output_name='my_config.yaml')
    """
    template_path = get_config_template(template_name)

    # Create destination directory
    destination.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if output_name is None:
        output_name = template_name

    dest_file = destination / output_name

    # Copy template
    shutil.copy2(template_path, dest_file)

    return dest_file
