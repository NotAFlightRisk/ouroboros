#!/usr/bin/env python3
"""Keep evaluation/consensus schema, runtime wiring, and docs in agreement.

Every field in the user-facing ``EvaluationConfig`` and ``ConsensusConfig``
schema must have exactly one truthful disposition:

* production code reads it from the corresponding config section;
* ``docs/config-reference.md`` marks it as inert and names the effective
  control; or
* it is present in the explicit, justified schema-only allowlist below.

The scan is syntax-aware. It inspects Python attribute loads rather than source
text, and it reads structured Markdown table rows plus JSON contract markers.
That lets the same check catch both directions of drift: a new unwired field,
and a previously inert field that becomes wired while the reference still says
it does nothing.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ouroboros.config._model_defaults import (  # noqa: E402
    DEFAULT_CONSENSUS_OPUS_MODEL,
    DEFAULT_OPUS_MODEL,
)
from ouroboros.config.models import ConsensusConfig, EvaluationConfig  # noqa: E402


@dataclass(frozen=True, order=True)
class ConfigField:
    """One field in a named user-facing config section."""

    section: str
    name: str

    @property
    def dotted(self) -> str:
        return f"{self.section}.{self.name}"


@dataclass(frozen=True)
class ReferenceRow:
    """Relevant cells from one config-reference field row."""

    default: str
    description: str


@dataclass(frozen=True)
class InertMarker:
    """Machine-readable companion to one visible inert-field description."""

    field: ConfigField
    effective_control: str


@dataclass(frozen=True)
class ContractReport:
    """All violations found in one complete contract audit."""

    violations: tuple[str, ...]
    runtime_reads: frozenset[ConfigField]


TRACKED_SECTIONS = frozenset({"evaluation", "consensus"})
REFERENCE_PATH = Path("docs/config-reference.md")
_SECTION_HEADING = re.compile(r"^## `(?P<section>evaluation|consensus)`\s*$")
_FIELD_ROW = re.compile(r"^\|\s*`(?P<field>[a-z][a-z0-9_]*)`\s*\|")
_INERT_MARKER = re.compile(r"<!--\s*config-field-contract:\s*(?P<payload>\{[^\r\n]*\})\s*-->")

# Escape hatch for a deliberately schema-only field that should not be called
# inert in the public reference. Keep this small. Every entry must carry a
# concrete rationale; the audit rejects stale entries once production reads the
# field. The eight fields motivating #1998 are documented, so none belong here.
SCHEMA_ONLY_ALLOWLIST: Mapping[ConfigField, str] = {}

# Bounded guard for the two Opus identifiers that intentionally use different
# normalized forms. This catches a blanket replacement without turning the
# checker into a general-purpose documentation-value synchronizer.
DOCUMENTED_DEFAULTS: Mapping[ConfigField, str] = {
    ConfigField("evaluation", "semantic_model"): DEFAULT_OPUS_MODEL,
    ConfigField("consensus", "advocate_model"): DEFAULT_CONSENSUS_OPUS_MODEL,
}


def schema_fields() -> frozenset[ConfigField]:
    """Enumerate the authoritative Pydantic fields for the two sections."""

    return frozenset(
        ConfigField(section, name)
        for section, model in (
            ("evaluation", EvaluationConfig),
            ("consensus", ConsensusConfig),
        )
        for name in model.model_fields
    )


_CONFIG_ROOT = "<config-root>"
_CONFIG_NAME = re.compile(
    r"(?:^|_)(?:cfg|config|configs|configuration|settings)$",
    flags=re.IGNORECASE,
)
_CONFIG_FACTORY = re.compile(
    r"(?:^|_)(?:build|create|get|load|read|resolve)_(?:config|configuration|settings)$",
    flags=re.IGNORECASE,
)


def _looks_like_config_name(name: str) -> bool:
    return _CONFIG_NAME.search(name) is not None


def _callable_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda


def _is_generator_function(node: _FunctionNode) -> bool:
    if isinstance(node, ast.Lambda):
        return False

    class Finder(ast.NodeVisitor):
        found = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Yield(self, node: ast.Yield) -> None:
            self.found = True

        def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
            self.found = True

    finder = Finder()
    for statement in node.body:
        finder.visit(statement)
    return finder.found


@dataclass(frozen=True)
class _AbstractValue:
    """Possible config provenance plus bounded container/object shape."""

    origins: frozenset[str] = frozenset()
    items: tuple[_AbstractValue, ...] | None = None
    entries: tuple[tuple[str, _AbstractValue], ...] | None = None
    attributes: tuple[tuple[str, _AbstractValue], ...] | None = None
    identity: frozenset[int] = frozenset()
    literal: str | None = None
    string_value: str | None = None
    truth: bool | None = None
    classes: frozenset[ast.ClassDef] = frozenset()
    instance_classes: frozenset[ast.ClassDef] = frozenset()
    modules: frozenset[str] = frozenset()
    callables: tuple[_CallableTarget, ...] = ()
    serialized_sections: frozenset[str] = frozenset()
    accessed_attributes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _CallableTarget:
    """A tracked callable plus the receiver captured by attribute binding."""

    function: _FunctionNode
    receiver: _AbstractValue | None = None


@dataclass(frozen=True)
class _DeferredGenerator:
    """A generator body plus the bindings captured when it was created."""

    node: _FunctionNode | ast.GeneratorExp
    scoped: dict[str, _AbstractValue]


@dataclass(frozen=True)
class _DeferredCoroutine:
    """An async function body, deferred until its coroutine is awaited."""

    node: ast.AsyncFunctionDef
    scoped: dict[str, _AbstractValue]


_UNKNOWN_VALUE = _AbstractValue()
_STATIC_UNKNOWN = object()

_ANNOTATION_MODULE = "<annotation-config-module>"
_TYPE_CHECKING_FALSE = "<typing-type-checking-false>"
_TYPING_MODULE = "<typing-module>"
_OPERATOR_MODULE = "<operator-module>"
_ATTRGETTER_FACTORY = "<attrgetter-factory>"
_BUILTINS_MODULE = "<builtins-module>"
_GETATTR_BUILTIN = "<getattr-builtin>"
_EAGER_BUILTIN_CONSUMERS = frozenset(
    {
        "all",
        "any",
        "frozenset",
        "list",
        "max",
        "min",
        "next",
        "set",
        "sorted",
        "sum",
        "tuple",
    }
)
_LAZY_BUILTIN_CONSUMERS = frozenset({"enumerate", "filter", "iter", "map", "reversed", "zip"})
_ASYNC_BUILTIN_CONSUMERS = frozenset({"anext"})
_TRACKED_BUILTIN_CONSUMERS = (
    _EAGER_BUILTIN_CONSUMERS | _LAZY_BUILTIN_CONSUMERS | _ASYNC_BUILTIN_CONSUMERS
)
_EAGER_BUILTIN_CONSUMER_ORIGINS = frozenset(
    f"<builtin-consumer:{name}>" for name in _EAGER_BUILTIN_CONSUMERS
)
_EAGER_EXTERNAL_CONSUMER_ORIGINS = frozenset({"<external-consumer:collections.deque>"})
_EAGER_CONSUMER_ORIGINS = _EAGER_BUILTIN_CONSUMER_ORIGINS | _EAGER_EXTERNAL_CONSUMER_ORIGINS
_LAZY_BUILTIN_CONSUMER_ORIGINS = frozenset(
    f"<builtin-consumer:{name}>" for name in _LAZY_BUILTIN_CONSUMERS
)
_ASYNC_BUILTIN_CONSUMER_ORIGINS = frozenset(
    f"<builtin-consumer:{name}>" for name in _ASYNC_BUILTIN_CONSUMERS
)
_SECTION_ANNOTATIONS: Mapping[str, str] = {
    "EvaluationConfig": "evaluation",
    "ConsensusConfig": "consensus",
}
_CONFIG_ANNOTATION_MODULES = frozenset(
    {
        "ouroboros.config",
        "ouroboros.config.models",
    }
)


def _safe_constant_value(node: ast.AST) -> object:
    """Evaluate a deliberately small, side-effect-free constant expression."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_safe_constant_value(item) for item in node.elts]
        if any(value is _STATIC_UNKNOWN for value in values):
            return _STATIC_UNKNOWN
        if isinstance(node, ast.Tuple):
            return tuple(values)
        if isinstance(node, ast.List):
            return values
        try:
            return set(values)
        except TypeError:
            return _STATIC_UNKNOWN
    if isinstance(node, ast.UnaryOp):
        operand = _safe_constant_value(node.operand)
        if operand is _STATIC_UNKNOWN:
            return _STATIC_UNKNOWN
        try:
            if isinstance(node.op, ast.Not):
                return not operand
            if isinstance(node.op, ast.UAdd) and isinstance(operand, (int, float, complex)):
                return +operand
            if isinstance(node.op, ast.USub) and isinstance(operand, (int, float, complex)):
                return -operand
            if isinstance(node.op, ast.Invert) and isinstance(operand, int):
                return ~operand
        except (TypeError, ValueError, OverflowError):
            return _STATIC_UNKNOWN
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _safe_constant_value(node.left)
        right = _safe_constant_value(node.right)
        if left is _STATIC_UNKNOWN or right is _STATIC_UNKNOWN:
            return _STATIC_UNKNOWN
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        if isinstance(left, bytes) and isinstance(right, bytes):
            return left + right
        if isinstance(left, tuple) and isinstance(right, tuple):
            return left + right
        if (
            isinstance(left, (int, float, complex))
            and not isinstance(left, bool)
            and isinstance(right, (int, float, complex))
            and not isinstance(right, bool)
        ):
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
                continue
            if not isinstance(part, ast.FormattedValue):
                return _STATIC_UNKNOWN
            value = _safe_constant_value(part.value)
            if value is _STATIC_UNKNOWN:
                return _STATIC_UNKNOWN
            if part.conversion == 115:
                value = str(value)
            elif part.conversion == 114:
                value = repr(value)
            elif part.conversion == 97:
                value = ascii(value)
            elif part.conversion != -1:
                return _STATIC_UNKNOWN
            format_spec = ""
            if part.format_spec is not None:
                resolved_spec = _safe_constant_value(part.format_spec)
                if not isinstance(resolved_spec, str):
                    return _STATIC_UNKNOWN
                format_spec = resolved_spec
            try:
                parts.append(format(value, format_spec))
            except (TypeError, ValueError):
                return _STATIC_UNKNOWN
        return "".join(parts)
    if isinstance(node, ast.Compare):
        left = _safe_constant_value(node.left)
        if left is _STATIC_UNKNOWN:
            return _STATIC_UNKNOWN
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = _safe_constant_value(comparator)
            if right is _STATIC_UNKNOWN:
                return _STATIC_UNKNOWN
            try:
                if isinstance(operator, ast.Eq):
                    matched = left == right
                elif isinstance(operator, ast.NotEq):
                    matched = left != right
                elif isinstance(operator, ast.Lt):
                    matched = left < right
                elif isinstance(operator, ast.LtE):
                    matched = left <= right
                elif isinstance(operator, ast.Gt):
                    matched = left > right
                elif isinstance(operator, ast.GtE):
                    matched = left >= right
                elif isinstance(operator, ast.Is):
                    if not (
                        (left is None or type(left) is bool)
                        and (right is None or type(right) is bool)
                    ):
                        return _STATIC_UNKNOWN
                    matched = left is right
                elif isinstance(operator, ast.IsNot):
                    if not (
                        (left is None or type(left) is bool)
                        and (right is None or type(right) is bool)
                    ):
                        return _STATIC_UNKNOWN
                    matched = left is not right
                elif isinstance(operator, ast.In):
                    matched = left in right
                elif isinstance(operator, ast.NotIn):
                    matched = left not in right
                else:
                    return _STATIC_UNKNOWN
            except (TypeError, ValueError):
                return _STATIC_UNKNOWN
            if not matched:
                return False
            left = right
        return True
    return _STATIC_UNKNOWN


def _origin_value(*origins: str) -> _AbstractValue:
    return _AbstractValue(origins=frozenset(origins))


def _type_checking_value() -> _AbstractValue:
    return _AbstractValue(origins=frozenset({_TYPE_CHECKING_FALSE}), truth=False)


def _contained_origins(value: _AbstractValue) -> frozenset[str]:
    origins = set(value.origins)
    for item in value.items or ():
        origins.update(_contained_origins(item))
    for _, item in value.entries or ():
        origins.update(_contained_origins(item))
    for _, item in value.attributes or ():
        origins.update(_contained_origins(item))
    return frozenset(origins)


def _conservative_value(value: _AbstractValue) -> _AbstractValue:
    return _AbstractValue(
        origins=_contained_origins(value),
        classes=value.classes,
        instance_classes=value.instance_classes,
        modules=value.modules,
        callables=value.callables,
        serialized_sections=value.serialized_sections,
        accessed_attributes=value.accessed_attributes,
    )


def _join_values(*values: _AbstractValue) -> _AbstractValue:
    if not values:
        return _UNKNOWN_VALUE
    origins = frozenset().union(*(value.origins for value in values))

    known_items = [value.items for value in values if value.items is not None]
    items: tuple[_AbstractValue, ...] | None = None
    if known_items:
        lengths = {len(candidate) for candidate in known_items}
        if len(lengths) == 1:
            items = tuple(_join_values(*position) for position in zip(*known_items, strict=True))
        else:
            items = (_join_values(*(item for candidate in known_items for item in candidate)),)

    def join_named(
        groups: Iterable[tuple[tuple[str, _AbstractValue], ...] | None],
    ) -> tuple[tuple[str, _AbstractValue], ...] | None:
        known = [dict(group) for group in groups if group is not None]
        if not known:
            return None
        names = set().union(*(group.keys() for group in known))
        return tuple(
            (name, _join_values(*(group[name] for group in known if name in group)))
            for name in sorted(names)
        )

    def join_entries(
        groups: Iterable[tuple[tuple[str, _AbstractValue], ...] | None],
    ) -> tuple[tuple[str, _AbstractValue], ...] | None:
        raw = list(groups)
        if not any(group is not None for group in raw):
            return None
        mappings = [dict(group or ()) for group in raw]
        names = set().union(*(group.keys() for group in mappings))
        missing = _origin_value(_MAPPING_MISSING)
        joined: list[tuple[str, _AbstractValue]] = []
        for name in sorted(names):
            possibilities: list[_AbstractValue] = []
            for group in mappings:
                if name in group:
                    possibilities.append(group[name])
                elif name != _DYNAMIC_KEY and _DYNAMIC_KEY in group:
                    possibilities.append(_join_values(group[_DYNAMIC_KEY], missing))
                else:
                    possibilities.append(missing)
            joined.append((name, _join_values(*possibilities)))
        return tuple(joined)

    callables: list[_CallableTarget] = []
    for value in values:
        for target in value.callables:
            if target not in callables:
                callables.append(target)

    return _AbstractValue(
        origins=origins,
        items=items,
        entries=join_entries(value.entries for value in values),
        attributes=join_named(value.attributes for value in values),
        identity=frozenset().union(*(value.identity for value in values)),
        literal=(
            values[0].literal
            if all(value.literal == values[0].literal for value in values)
            else None
        ),
        string_value=(
            values[0].string_value
            if all(value.string_value == values[0].string_value for value in values)
            else None
        ),
        truth=(
            values[0].truth if all(value.truth == values[0].truth for value in values) else None
        ),
        classes=frozenset().union(*(value.classes for value in values)),
        instance_classes=frozenset().union(*(value.instance_classes for value in values)),
        modules=frozenset().union(*(value.modules for value in values)),
        callables=tuple(callables),
        serialized_sections=frozenset().union(*(value.serialized_sections for value in values)),
        accessed_attributes=frozenset().union(*(value.accessed_attributes for value in values)),
    )


def _key_token(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


_DYNAMIC_KEY = "**"
_MAPPING_MISSING = "<mapping-missing>"
_FunctionSet = frozenset[_FunctionNode]
_BindingSnapshot = tuple[
    dict[str, _AbstractValue],
    dict[str, _FunctionSet],
    dict[str, _AbstractValue],
]


@dataclass(frozen=True)
class _IndexedModule:
    """One parsed source module and its statically declared callables."""

    name: str
    package: str
    tree: ast.Module
    functions: dict[str, _FunctionSet]
    classes: dict[str, frozenset[ast.ClassDef]]


class _SourceIndex:
    """Resolve only explicit imports and class receivers inside the scanned tree."""

    def __init__(self, source_root: Path, trees: Mapping[Path, ast.Module]) -> None:
        self._root_name = source_root.name
        self._aliases: dict[str, _IndexedModule] = {}
        self._owners: dict[int, _IndexedModule] = {}
        self._paths: dict[Path, _IndexedModule] = {}
        self._reexports: dict[tuple[str, str], tuple[_IndexedModule, str]] = {}
        self._class_bases: dict[int, tuple[ast.ClassDef, ...]] = {}
        for path, tree in trees.items():
            relative = path.relative_to(source_root).with_suffix("")
            parts = list(relative.parts)
            is_package = bool(parts and parts[-1] == "__init__")
            if is_package:
                parts.pop()
            name = ".".join(parts) or self._root_name
            package = name if is_package else name.rpartition(".")[0]
            functions = {
                statement.name: frozenset({statement})
                for statement in tree.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            classes = {
                statement.name: frozenset({statement})
                for statement in tree.body
                if isinstance(statement, ast.ClassDef)
            }
            indexed = _IndexedModule(name, package, tree, functions, classes)
            self._paths[path] = indexed
            aliases = {name}
            if name != self._root_name and not name.startswith(f"{self._root_name}."):
                aliases.add(f"{self._root_name}.{name}")
            for alias in aliases:
                self._aliases[alias] = indexed
            for candidate in ast.walk(tree):
                if isinstance(
                    candidate,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
                ):
                    self._owners[id(candidate)] = indexed
        for indexed in self._paths.values():
            self._index_exports(indexed)
        for indexed in self._paths.values():
            self._index_class_bases(indexed)

    @staticmethod
    def _assigned_names(target: ast.AST) -> frozenset[str]:
        if isinstance(target, ast.Name):
            return frozenset({target.id})
        if isinstance(target, (ast.Tuple, ast.List)):
            return frozenset().union(
                *(_SourceIndex._assigned_names(element) for element in target.elts)
            )
        if isinstance(target, ast.Starred):
            return _SourceIndex._assigned_names(target.value)
        return frozenset()

    def _clear_export(self, module: _IndexedModule, name: str) -> None:
        module.functions.pop(name, None)
        module.classes.pop(name, None)
        self._reexports.pop((module.name, name), None)

    def _index_exports(self, module: _IndexedModule) -> None:
        """Index final explicit module bindings without suffix/name guessing."""

        module.functions.clear()
        module.classes.clear()
        for statement in module.tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._clear_export(module, statement.name)
                module.functions[statement.name] = frozenset({statement})
            elif isinstance(statement, ast.ClassDef):
                self._clear_export(module, statement.name)
                module.classes[statement.name] = frozenset({statement})
            elif isinstance(statement, ast.ImportFrom):
                target = self.resolve_module(statement.module, module, statement.level)
                for alias in statement.names:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    self._clear_export(module, bound)
                    if target is not None:
                        self._reexports[(module.name, bound)] = (target, alias.name)
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    self._clear_export(module, alias.asname or alias.name.partition(".")[0])
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
                )
                for target in targets:
                    for name in self._assigned_names(target):
                        self._clear_export(module, name)

    def resolve_functions(
        self,
        module: _IndexedModule,
        name: str,
        seen: frozenset[tuple[str, str]] = frozenset(),
    ) -> _FunctionSet:
        key = (module.name, name)
        if key in seen:
            return frozenset()
        direct = module.functions.get(name)
        if direct is not None:
            return direct
        edge = self._reexports.get(key)
        if edge is None:
            return frozenset()
        target, target_name = edge
        return self.resolve_functions(target, target_name, seen | {key})

    def resolve_classes(
        self,
        module: _IndexedModule,
        name: str,
        seen: frozenset[tuple[str, str]] = frozenset(),
    ) -> frozenset[ast.ClassDef]:
        key = (module.name, name)
        if key in seen:
            return frozenset()
        direct = module.classes.get(name)
        if direct is not None:
            return direct
        edge = self._reexports.get(key)
        if edge is None:
            return frozenset()
        target, target_name = edge
        return self.resolve_classes(target, target_name, seen | {key})

    def _index_class_bases(self, module: _IndexedModule) -> None:
        class_bindings: dict[str, frozenset[ast.ClassDef]] = {}
        module_bindings: dict[str, _IndexedModule] = {}
        for statement in module.tree.body:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    bound = alias.asname or alias.name.partition(".")[0]
                    imported_name = alias.name if alias.asname else alias.name.partition(".")[0]
                    imported = self.resolve_module(imported_name, module)
                    class_bindings.pop(bound, None)
                    if imported is not None:
                        module_bindings[bound] = imported
            elif isinstance(statement, ast.ImportFrom):
                imported = self.resolve_module(statement.module, module, statement.level)
                for alias in statement.names:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    module_bindings.pop(bound, None)
                    class_bindings[bound] = (
                        self.resolve_classes(imported, alias.name)
                        if imported is not None
                        else frozenset()
                    )
                    child = self.resolve_module(
                        f"{imported.name}.{alias.name}" if imported is not None else alias.name,
                        module,
                    )
                    if child is not None:
                        module_bindings[bound] = child
            elif isinstance(statement, ast.ClassDef):
                bases: list[ast.ClassDef] = []
                for base in statement.bases:
                    if isinstance(base, ast.Name):
                        for resolved in class_bindings.get(base.id, frozenset()):
                            if resolved not in bases:
                                bases.append(resolved)
                    elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                        imported = module_bindings.get(base.value.id)
                        if imported is not None:
                            for resolved in self.resolve_classes(imported, base.attr):
                                if resolved not in bases:
                                    bases.append(resolved)
                self._class_bases[id(statement)] = tuple(bases)
                class_bindings[statement.name] = frozenset({statement})
                module_bindings.pop(statement.name, None)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                class_bindings.pop(statement.name, None)
                module_bindings.pop(statement.name, None)
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
                )
                for target in targets:
                    for name in self._assigned_names(target):
                        class_bindings.pop(name, None)
                        module_bindings.pop(name, None)

    def module_for_path(self, path: Path) -> _IndexedModule:
        return self._paths[path]

    def owner(self, node: ast.AST) -> _IndexedModule | None:
        return self._owners.get(id(node))

    def resolve_module(
        self,
        module: str | None,
        current: _IndexedModule,
        level: int = 0,
    ) -> _IndexedModule | None:
        if level:
            package = current.package.split(".") if current.package else []
            keep = max(0, len(package) - level + 1)
            parts = [*package[:keep], *(module.split(".") if module else [])]
            name = ".".join(parts)
        else:
            name = module or ""
        direct = self._aliases.get(name)
        if direct is not None:
            return direct
        return None

    def methods(self, classes: Iterable[ast.ClassDef], name: str) -> _FunctionSet:
        """Resolve the first known implementation along each exact class lineage."""

        def inherited(class_node: ast.ClassDef, seen: frozenset[int]) -> _FunctionSet:
            if id(class_node) in seen:
                return frozenset()
            direct = frozenset(
                statement
                for statement in class_node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == name
            )
            if direct:
                return direct
            for base in self._class_bases.get(id(class_node), ()):
                resolved = inherited(base, seen | {id(class_node)})
                if resolved:
                    return resolved
            return frozenset()

        return frozenset().union(*(inherited(class_node, frozenset()) for class_node in classes))


@dataclass(frozen=True)
class _AbruptPath:
    """One non-fallthrough control path and its path-local payload."""

    kind: str
    bindings: _BindingSnapshot
    return_value: _AbstractValue | None = None
    exception_type: str | None = None
    exception_exclusions: frozenset[str] = frozenset()
    exception_upper_bound: str | None = None


@dataclass(frozen=True)
class _FlowResult:
    """Binding paths leaving one statement suite."""

    fallthrough: _BindingSnapshot | None
    abrupt: tuple[_AbruptPath, ...] = ()


class _RuntimeReadVisitor(ast.NodeVisitor):
    """Collect config reads with conservative flow- and binding-aware provenance."""

    def __init__(
        self,
        fields: frozenset[ConfigField],
        source_index: _SourceIndex,
        module: _IndexedModule,
    ) -> None:
        self._fields = fields
        self._source_index = source_index
        self._module = module
        # Explicit unknown values shadow name-based config inference.
        self._states: list[dict[str, _AbstractValue]] = [{}]
        self._annotations: list[dict[str, _AbstractValue]] = [{}]
        self._functions: list[dict[str, _FunctionSet]] = [{}]
        self._global_names: list[set[str]] = [set()]
        self._nonlocal_names: list[set[str]] = [set()]
        self._active_calls: set[int] = set()
        self._expression_cache: dict[int, _AbstractValue] = {}
        self._flow_abrupts: list[list[_AbruptPath]] = [[]]
        self._path_reachable = True
        self._exception_capture_depth = 0
        self._caught_exception_stack: list[tuple[_AbruptPath, ...]] = []
        self._function_body_depth = 0
        self._class_member_values: dict[int, tuple[_AbstractValue, ...]] = {}
        self._closure_bindings: dict[int, dict[str, _AbstractValue]] = {}
        self._deferred_generators: dict[int, _DeferredGenerator] = {}
        self._deferred_coroutines: dict[int, _DeferredCoroutine] = {}
        self._deferred_generator_positions: dict[int, int] = {}
        self._generator_consumer_modes: list[str] = []
        self._generator_skip_yields: list[int] = []
        self._generator_advanced_yields: list[int] = []
        self.reads: set[ConfigField] = set()

    def _name_value(self, name: str) -> _AbstractValue:
        for scope in reversed(self._states):
            if name in scope:
                return scope[name]
        if _looks_like_config_name(name):
            return _origin_value(_CONFIG_ROOT)
        return _UNKNOWN_VALUE

    def _name_is_bound(self, name: str) -> bool:
        return any(name in scope for scope in self._states) or any(
            name in scope for scope in self._functions
        )

    def _annotation_name_value(self, name: str) -> _AbstractValue:
        for scope in reversed(self._annotations):
            if name in scope:
                return scope[name]
        return _UNKNOWN_VALUE

    def _annotation_value(self, annotation: ast.AST | None) -> _AbstractValue:
        """Resolve only authoritative config model annotations.

        Merely containing a token such as ``config`` is not enough: wrappers,
        unrelated classes, and the runtime evaluator's colliding
        ``ConsensusConfig`` must not acquire root provenance.
        """

        if annotation is None:
            return _UNKNOWN_VALUE
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                parsed = ast.parse(annotation.value, mode="eval")
            except SyntaxError:
                return _UNKNOWN_VALUE
            return self._annotation_value(parsed.body)
        if isinstance(annotation, ast.Name):
            return self._annotation_name_value(annotation.id)
        if isinstance(annotation, ast.Attribute):
            owner = self._annotation_value(annotation.value)
            if _ANNOTATION_MODULE in owner.origins:
                section = _SECTION_ANNOTATIONS.get(annotation.attr)
                if section is not None:
                    return _origin_value(section)
                if annotation.attr == "OuroborosConfig":
                    return _origin_value(_CONFIG_ROOT)
            dotted = self._dotted_name(annotation)
            if dotted is not None:
                module, _, name = dotted.rpartition(".")
                if module in _CONFIG_ANNOTATION_MODULES:
                    section = _SECTION_ANNOTATIONS.get(name)
                    if section is not None:
                        return _origin_value(section)
                    if name == "OuroborosConfig":
                        return _origin_value(_CONFIG_ROOT)
            return _UNKNOWN_VALUE
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            return _join_values(
                self._annotation_value(annotation.left),
                self._annotation_value(annotation.right),
            )
        if isinstance(annotation, ast.Subscript):
            wrapper = _callable_name(annotation.value)
            if wrapper in {"Annotated", "Optional"}:
                item = annotation.slice
                if wrapper == "Annotated" and isinstance(item, ast.Tuple) and item.elts:
                    item = item.elts[0]
                return self._annotation_value(item)
        return _UNKNOWN_VALUE

    @staticmethod
    def _dotted_name(node: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parts.append(node.id)
        return ".".join(reversed(parts))

    def _local_functions(self, name: str) -> _FunctionSet:
        for scope in reversed(self._functions):
            if name in scope:
                return scope[name]
        return frozenset()

    def _function_value(self, node: ast.AST) -> _FunctionSet:
        value = self._expression_value(node)
        if value.callables:
            return frozenset(target.function for target in value.callables)
        if isinstance(node, ast.Name):
            return self._local_functions(node.id)
        if isinstance(node, ast.Lambda):
            return frozenset({node})
        if isinstance(node, ast.IfExp):
            truth = self._static_truth(node.test)
            if truth is not None:
                return self._function_value(node.body if truth else node.orelse)
            return self._function_value(node.body) | self._function_value(node.orelse)
        return frozenset()

    @staticmethod
    def _method_is_static(function: _FunctionNode) -> bool:
        return isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            _callable_name(decorator) == "staticmethod" for decorator in function.decorator_list
        )

    @staticmethod
    def _method_is_property(function: _FunctionNode) -> bool:
        return isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            _callable_name(decorator) == "property" for decorator in function.decorator_list
        )

    def _call_targets(
        self, node: ast.AST
    ) -> tuple[tuple[_FunctionNode, _AbstractValue | None], ...]:
        value = self._expression_value(node)
        targets = [(target.function, target.receiver) for target in value.callables]
        for function in self._source_index.methods(value.instance_classes, "__call__"):
            target = (function, None if self._method_is_static(function) else value)
            if target not in targets:
                targets.append(target)
        if not targets:
            targets.extend((function, None) for function in self._function_value(node))
        return tuple(targets)

    def _call_has_relevant_provenance(
        self,
        node: ast.Call,
        function: _FunctionNode,
        bound_receiver: _AbstractValue | None,
    ) -> bool:
        if isinstance(function, ast.Lambda):
            return True
        if any(isinstance(child, (ast.Global, ast.Nonlocal)) for child in ast.walk(function)):
            return True
        values = [
            self._expression_value(
                argument.value if isinstance(argument, ast.Starred) else argument
            )
            for argument in node.args
        ]
        values.extend(self._expression_value(keyword.value) for keyword in node.keywords)
        values.extend(self._default_argument_values(function).values())
        values.extend(self._closure_bindings.get(id(function), {}).values())
        tracked = TRACKED_SECTIONS | {_CONFIG_ROOT}
        if any(
            _contained_origins(value) & tracked
            or value.callables
            or any(identity in self._deferred_generators for identity in value.identity)
            for value in values
        ):
            return True
        return bound_receiver is not None and bool(_contained_origins(bound_receiver) & tracked)

    @staticmethod
    def _named_value(
        pairs: tuple[tuple[str, _AbstractValue], ...] | None, name: str
    ) -> _AbstractValue | None:
        if pairs is None:
            return None
        return dict(pairs).get(name)

    def _replace_name_value(self, name: str, value: _AbstractValue) -> None:
        for scope in reversed(self._states):
            if name in scope:
                scope[name] = value
                return
        self._states[-1][name] = value

    def _replace_identity(
        self,
        value: _AbstractValue,
        identity: frozenset[int],
        replacement: _AbstractValue,
        *,
        conservative: bool,
    ) -> _AbstractValue:
        if value.identity & identity:
            return _join_values(value, replacement) if conservative else replacement
        items = (
            tuple(
                self._replace_identity(item, identity, replacement, conservative=conservative)
                for item in value.items
            )
            if value.items is not None
            else None
        )
        entries = (
            tuple(
                (
                    key,
                    self._replace_identity(item, identity, replacement, conservative=conservative),
                )
                for key, item in value.entries
            )
            if value.entries is not None
            else None
        )
        attributes = (
            tuple(
                (
                    name,
                    self._replace_identity(item, identity, replacement, conservative=conservative),
                )
                for name, item in value.attributes
            )
            if value.attributes is not None
            else None
        )
        if items == value.items and entries == value.entries and attributes == value.attributes:
            return value
        return _AbstractValue(
            origins=value.origins,
            items=items,
            entries=entries,
            attributes=attributes,
            identity=value.identity,
            literal=value.literal,
            string_value=value.string_value,
            truth=value.truth,
            classes=value.classes,
            instance_classes=value.instance_classes,
            modules=value.modules,
            callables=value.callables,
            serialized_sections=value.serialized_sections,
            accessed_attributes=value.accessed_attributes,
        )

    def _replace_shared_value(
        self,
        owner: _AbstractValue,
        replacement: _AbstractValue,
        receiver: ast.AST,
    ) -> None:
        if owner.identity:
            for scope in self._states:
                for name, value in tuple(scope.items()):
                    scope[name] = self._replace_identity(
                        value,
                        owner.identity,
                        replacement,
                        conservative=len(owner.identity) > 1,
                    )
            # A joined owner may refer to one of several source objects, so
            # those source aliases need conservative updates. The receiver
            # itself, however, names whichever object was selected at runtime
            # and therefore always observes the mutation strongly.
            if isinstance(receiver, ast.Name):
                self._replace_name_value(receiver.id, replacement)
        elif isinstance(receiver, ast.Name):
            self._replace_name_value(receiver.id, replacement)

    @staticmethod
    def _mapping_replacement(
        owner: _AbstractValue,
        entries: Iterable[tuple[str, _AbstractValue]],
    ) -> _AbstractValue:
        normalized = tuple(entries)
        return _AbstractValue(
            origins=owner.origins,
            items=owner.items,
            entries=normalized,
            attributes=owner.attributes,
            identity=owner.identity,
            literal=owner.literal,
            string_value=owner.string_value,
            truth=False if not normalized else None,
            classes=owner.classes,
            instance_classes=owner.instance_classes,
            modules=owner.modules,
            callables=owner.callables,
            serialized_sections=owner.serialized_sections,
            accessed_attributes=owner.accessed_attributes,
        )

    @staticmethod
    def _sequence_replacement(
        owner: _AbstractValue,
        items: tuple[_AbstractValue, ...],
    ) -> _AbstractValue:
        return _AbstractValue(
            origins=owner.origins,
            items=items,
            entries=owner.entries,
            attributes=owner.attributes,
            identity=owner.identity,
            literal=owner.literal,
            string_value=owner.string_value,
            truth=False if not items else None,
            classes=owner.classes,
            instance_classes=owner.instance_classes,
            modules=owner.modules,
            callables=owner.callables,
            serialized_sections=owner.serialized_sections,
            accessed_attributes=owner.accessed_attributes,
        )

    @staticmethod
    def _attribute_replacement(
        owner: _AbstractValue,
        attributes: tuple[tuple[str, _AbstractValue], ...],
    ) -> _AbstractValue:
        return _AbstractValue(
            origins=owner.origins,
            items=owner.items,
            entries=owner.entries,
            attributes=attributes,
            identity=owner.identity,
            literal=owner.literal,
            string_value=owner.string_value,
            truth=owner.truth,
            classes=owner.classes,
            instance_classes=owner.instance_classes,
            modules=owner.modules,
            callables=owner.callables,
            serialized_sections=owner.serialized_sections,
            accessed_attributes=owner.accessed_attributes,
        )

    @staticmethod
    def _mapping_values(value: _AbstractValue) -> tuple[_AbstractValue, ...]:
        return tuple(item for _, item in value.entries or ())

    @staticmethod
    def _add_mapping_entry(
        entries: dict[str, _AbstractValue],
        key: str,
        value: _AbstractValue,
    ) -> None:
        """Apply one mapping write in evaluation order.

        A dynamic key may collide with every key written before it. A later
        literal key, however, deterministically overrides any earlier dynamic
        collision for that literal key.
        """

        if key == _DYNAMIC_KEY:
            for known in tuple(entries):
                if known != _DYNAMIC_KEY:
                    entries[known] = _join_values(entries[known], value)
            entries[_DYNAMIC_KEY] = _join_values(entries.get(_DYNAMIC_KEY, _UNKNOWN_VALUE), value)
            return
        entries[key] = value

    @classmethod
    def _overlay_mapping_entries(
        cls,
        base: Mapping[str, _AbstractValue],
        source: Mapping[str, _AbstractValue],
    ) -> dict[str, _AbstractValue]:
        """Overlay a normalized source mapping onto a normalized base."""

        result = dict(base)
        source_known = {key for key in source if key != _DYNAMIC_KEY}
        wildcard = source.get(_DYNAMIC_KEY)
        if wildcard is not None:
            for known in tuple(result):
                if known != _DYNAMIC_KEY and known not in source_known:
                    result[known] = _join_values(result[known], wildcard)
            result[_DYNAMIC_KEY] = _join_values(result.get(_DYNAMIC_KEY, _UNKNOWN_VALUE), wildcard)
        for key in source_known:
            result[key] = source[key]
        return result

    @classmethod
    def _mapping_source_entries(
        cls,
        source: _AbstractValue,
    ) -> tuple[tuple[str, _AbstractValue], ...]:
        if source.entries is not None:
            return source.entries
        if source.items is not None:
            entries: dict[str, _AbstractValue] = {}
            for pair in source.items:
                if pair.items is not None and len(pair.items) >= 2:
                    cls._add_mapping_entry(
                        entries, pair.items[0].literal or _DYNAMIC_KEY, pair.items[1]
                    )
            return tuple(sorted(entries.items()))
        return ((_DYNAMIC_KEY, _conservative_value(source)),)

    def _dict_method_value(self, node: ast.Call) -> _AbstractValue | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        method = node.func.attr
        if method not in {
            "clear",
            "copy",
            "get",
            "items",
            "keys",
            "pop",
            "setdefault",
            "update",
            "values",
        }:
            return None
        owner = self._expression_value(node.func.value)
        values = self._mapping_values(owner)
        if method == "copy":
            return _AbstractValue(
                origins=owner.origins,
                items=owner.items,
                entries=owner.entries,
                attributes=owner.attributes,
                identity=frozenset({id(node)}),
                literal=owner.literal,
                string_value=owner.string_value,
                truth=owner.truth,
            )
        if method == "values":
            return _AbstractValue(items=values) if owner.entries is not None else _UNKNOWN_VALUE
        if method == "items":
            if owner.entries is None:
                return _UNKNOWN_VALUE
            return _AbstractValue(
                items=tuple(_AbstractValue(items=(_UNKNOWN_VALUE, value)) for value in values)
            )
        if method == "keys":
            return (
                _AbstractValue(items=tuple(_UNKNOWN_VALUE for _ in owner.entries))
                if owner.entries is not None
                else _UNKNOWN_VALUE
            )

        if method == "clear":
            replacement = self._mapping_replacement(owner, ())
            self._replace_shared_value(owner, replacement, node.func.value)
            return _UNKNOWN_VALUE
        if method == "update":
            entries = dict(owner.entries or ())
            if node.args:
                source = self._expression_value(node.args[0])
                entries = self._overlay_mapping_entries(
                    entries, dict(self._mapping_source_entries(source))
                )
            for keyword in node.keywords:
                value = self._expression_value(keyword.value)
                if keyword.arg is None:
                    if value.entries is not None:
                        entries = self._overlay_mapping_entries(entries, dict(value.entries))
                    else:
                        entries = self._overlay_mapping_entries(
                            entries,
                            {_DYNAMIC_KEY: _conservative_value(value)},
                        )
                else:
                    entries[_key_token(ast.Constant(keyword.arg))] = value
            replacement = self._mapping_replacement(owner, sorted(entries.items()))
            self._replace_shared_value(owner, replacement, node.func.value)
            return _UNKNOWN_VALUE

        default = self._expression_value(node.args[1]) if len(node.args) >= 2 else _UNKNOWN_VALUE
        if not node.args:
            return default
        key = node.args[0]
        if not isinstance(key, ast.Constant):
            if method == "setdefault":
                return self._dynamic_setdefault_value(node)
            return _join_values(*values, default)
        token = _key_token(key)
        entries = dict(owner.entries or ())
        selected = entries.get(token)
        wildcard = entries.get(_DYNAMIC_KEY)
        if method == "get":
            if selected is not None:
                return (
                    _join_values(selected, default)
                    if _MAPPING_MISSING in selected.origins
                    else selected
                )
            return _join_values(wildcard, default) if wildcard is not None else default

        if method == "pop":
            if selected is not None and owner.entries is not None:
                replacement = self._mapping_replacement(
                    owner, ((name, value) for name, value in owner.entries if name != token)
                )
                self._replace_shared_value(owner, replacement, node.func.value)
                return selected
            return default

        # ``setdefault`` returns the existing value or inserts and returns the
        # default. A dynamic key may hit any existing entry or create a new
        # one, so join the default into every possible target and ``**``.
        if selected is not None and _MAPPING_MISSING not in selected.origins:
            return selected
        if owner.entries is not None:
            prior = selected if selected is not None else wildcard
            inserted = _join_values(prior, default) if prior is not None else default
            entries[token] = inserted
            replacement = self._mapping_replacement(owner, sorted(entries.items()))
            self._replace_shared_value(owner, replacement, node.func.value)
            return inserted
        return default

    def _dynamic_setdefault_value(self, node: ast.Call) -> _AbstractValue:
        assert isinstance(node.func, ast.Attribute)
        owner = self._expression_value(node.func.value)
        values = self._mapping_values(owner)
        default = self._expression_value(node.args[1]) if len(node.args) >= 2 else _UNKNOWN_VALUE
        if owner.entries is not None:
            entries = dict(owner.entries)
            entries[_DYNAMIC_KEY] = _join_values(entries.get(_DYNAMIC_KEY, _UNKNOWN_VALUE), default)
            replacement = self._mapping_replacement(owner, sorted(entries.items()))
            self._replace_shared_value(owner, replacement, node.func.value)
        return _join_values(*values, default)

    def _static_truth(self, node: ast.AST) -> bool | None:
        """Return truth only when Python runtime behavior is statically certain."""

        constant_value = _safe_constant_value(node)
        if constant_value is not _STATIC_UNKNOWN:
            return bool(constant_value)
        if isinstance(node, ast.Constant):
            return bool(node.value)
        if isinstance(node, ast.Dict):
            if not node.keys:
                return False
            return True if any(key is not None for key in node.keys) else None
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            if not node.elts:
                return False
            return True if any(not isinstance(item, ast.Starred) for item in node.elts) else None
        if isinstance(node, (ast.Name, ast.Attribute)):
            value = self._expression_value(node)
            return value.truth
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            operand = self._static_truth(node.operand)
            return None if operand is None else not operand
        if isinstance(node, ast.BoolOp):
            truths = [self._static_truth(value) for value in node.values]
            if isinstance(node.op, ast.And):
                if False in truths:
                    return False
                return True if all(value is True for value in truths) else None
            if True in truths:
                return True
            return False if all(value is False for value in truths) else None
        if isinstance(node, ast.IfExp):
            test = self._static_truth(node.test)
            if test is not None:
                return self._static_truth(node.body if test else node.orelse)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            left = self._expression_value(node.left)
            right = self._expression_value(node.comparators[0])
            if left.literal is not None and right.literal is not None:
                equal = left.literal == right.literal
                if isinstance(node.ops[0], ast.Eq):
                    return equal
                if isinstance(node.ops[0], ast.NotEq):
                    return not equal
        return None

    def _reachable_bool_values(self, node: ast.BoolOp) -> tuple[ast.expr, ...]:
        reachable: list[ast.expr] = []
        for value in node.values:
            reachable.append(value)
            truth = self._static_truth(value)
            if isinstance(node.op, ast.And) and truth is False:
                break
            if isinstance(node.op, ast.Or) and truth is True:
                break
        return tuple(reachable)

    def _expression_value(self, node: ast.AST) -> _AbstractValue:
        cached = self._expression_cache.get(id(node))
        if cached is not None:
            return cached
        if isinstance(node, ast.GeneratorExp):
            self.visit_GeneratorExp(node)
            return self._expression_cache[id(node)]
        if isinstance(node, ast.Constant):
            return _AbstractValue(
                literal=_key_token(node),
                string_value=node.value if isinstance(node.value, str) else None,
                truth=bool(node.value),
            )
        constant_value = _safe_constant_value(node)
        if isinstance(constant_value, str):
            return _AbstractValue(
                literal=_key_token(ast.Constant(constant_value)),
                string_value=constant_value,
                truth=bool(constant_value),
            )
        if isinstance(node, ast.Name):
            value = (
                _origin_value(_GETATTR_BUILTIN)
                if node.id == "getattr" and not self._name_is_bound("getattr")
                else _origin_value(f"<builtin-consumer:{node.id}>")
                if node.id in _TRACKED_BUILTIN_CONSUMERS and not self._name_is_bound(node.id)
                else self._name_value(node.id)
            )
            if value.callables:
                return value
            functions = self._local_functions(node.id)
            if functions:
                return _join_values(
                    value,
                    _AbstractValue(
                        callables=tuple(_CallableTarget(function) for function in functions)
                    ),
                )
            return value
        if isinstance(node, ast.Lambda):
            self._capture_function_closure(node)
            return _AbstractValue(callables=(_CallableTarget(node),))
        if isinstance(node, ast.Attribute):
            owner = self._expression_value(node.value)
            attribute = self._named_value(owner.attributes, node.attr)
            if attribute is not None:
                return attribute
            property_getters = tuple(
                function
                for function in self._source_index.methods(owner.instance_classes, node.attr)
                if self._method_is_property(function)
            )
            if property_getters:
                return _join_values(
                    *(
                        self._local_direct_call_value(function, (owner,))
                        for function in property_getters
                    )
                )
            if _OPERATOR_MODULE in owner.origins and node.attr == "attrgetter":
                return _origin_value(_ATTRGETTER_FACTORY)
            if _BUILTINS_MODULE in owner.origins and node.attr == "getattr":
                return _origin_value(_GETATTR_BUILTIN)
            if _BUILTINS_MODULE in owner.origins and node.attr in _TRACKED_BUILTIN_CONSUMERS:
                return _origin_value(f"<builtin-consumer:{node.attr}>")
            if node.attr == "deque" and "collections" in owner.modules:
                return _origin_value("<external-consumer:collections.deque>")
            resolved_modules = {
                child.name
                for module_name in owner.modules
                if (
                    child := self._source_index.resolve_module(
                        f"{module_name}.{node.attr}", self._module
                    )
                )
                is not None
            }
            resolved_classes = frozenset(
                class_node
                for module_name in owner.modules
                if (module := self._source_index.resolve_module(module_name, self._module))
                is not None
                for class_node in self._source_index.resolve_classes(module, node.attr)
            )
            imported_functions = frozenset(
                function
                for module_name in owner.modules
                if (module := self._source_index.resolve_module(module_name, self._module))
                is not None
                for function in self._source_index.resolve_functions(module, node.attr)
            )
            methods = self._source_index.methods(owner.classes, node.attr)
            if resolved_modules or resolved_classes or imported_functions or methods:
                callable_targets = [
                    *(_CallableTarget(function) for function in imported_functions),
                    *(
                        _CallableTarget(
                            function,
                            None if self._method_is_static(function) else owner,
                        )
                        for function in methods
                    ),
                ]
                return _AbstractValue(
                    modules=frozenset(resolved_modules),
                    classes=resolved_classes,
                    callables=tuple(callable_targets),
                )
            if node.attr in TRACKED_SECTIONS and _CONFIG_ROOT in owner.origins:
                return _origin_value(node.attr)
            if node.attr == "TYPE_CHECKING" and _TYPING_MODULE in owner.origins:
                return _type_checking_value()
            if _looks_like_config_name(node.attr) and (
                _CONFIG_ROOT in owner.origins
                or isinstance(node.value, ast.Name)
                and node.value.id in {"self", "cls"}
            ):
                return _origin_value(_CONFIG_ROOT)
            return _UNKNOWN_VALUE
        if isinstance(node, ast.Call):
            dict_method_value = self._dict_method_value(node)
            if dict_method_value is not None:
                self._expression_cache[id(node)] = dict_method_value
                return dict_method_value
            function_value = self._expression_value(node.func)
            if function_value.origins & _LAZY_BUILTIN_CONSUMER_ORIGINS:
                iterable_arguments = (
                    node.args[1:] if _callable_name(node.func) in {"filter", "map"} else node.args
                )
                return _join_values(
                    *(self._expression_value(argument) for argument in iterable_arguments)
                )
            if _ATTRGETTER_FACTORY in function_value.origins:
                return _AbstractValue(
                    accessed_attributes=frozenset(
                        argument.value
                        for argument in node.args
                        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                    )
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr == "model_dump":
                sections = self._expression_value(node.func.value).origins & TRACKED_SECTIONS
                if sections:
                    return _AbstractValue(serialized_sections=sections)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "vars"
                and not self._name_is_bound("vars")
                and node.args
            ):
                sections = self._expression_value(node.args[0]).origins & TRACKED_SECTIONS
                if sections:
                    return _AbstractValue(serialized_sections=sections)
            callable_name = _callable_name(node.func)
            if callable_name is not None and _CONFIG_FACTORY.search(callable_name):
                if isinstance(node.func, ast.Name):
                    return _origin_value(_CONFIG_ROOT)
                if isinstance(node.func, ast.Attribute) and (
                    _CONFIG_ROOT in self._expression_value(node.func.value).origins
                    or isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"self", "cls"}
                ):
                    return _origin_value(_CONFIG_ROOT)
            if _GETATTR_BUILTIN in function_value.origins and len(node.args) >= 2:
                field_name = self._expression_value(node.args[1]).string_value
                if (
                    field_name in TRACKED_SECTIONS
                    and _CONFIG_ROOT in self._expression_value(node.args[0]).origins
                ):
                    return _origin_value(field_name)
            values = [
                self._local_call_value(node, function, bound_receiver)
                for function, bound_receiver in self._call_targets(node.func)
                if self._call_has_relevant_provenance(node, function, bound_receiver)
            ]
            if values:
                value = _join_values(*values)
                self._expression_cache[id(node)] = value
                return value
            constructor_classes = self._expression_value(node.func).classes
            if constructor_classes or callable_name is not None and callable_name[:1].isupper():
                items: list[_AbstractValue] = []
                for argument in node.args:
                    if isinstance(argument, ast.Starred):
                        expanded = self._expression_value(argument.value)
                        if expanded.items is not None:
                            items.extend(expanded.items)
                        else:
                            items.append(_conservative_value(expanded))
                    else:
                        items.append(self._expression_value(argument))
                attributes: list[tuple[str, _AbstractValue]] = []
                for keyword in node.keywords:
                    value = self._expression_value(keyword.value)
                    attributes.append(
                        (keyword.arg, value)
                        if keyword.arg is not None
                        else ("**", _conservative_value(value))
                    )
                return _AbstractValue(
                    items=tuple(items),
                    attributes=tuple(attributes),
                    identity=frozenset({id(node)}),
                    classes=constructor_classes,
                    instance_classes=constructor_classes,
                )
            if isinstance(node.func, ast.Name) and node.func.id == "dict":
                entries: dict[str, _AbstractValue] = {}
                if node.args:
                    source = self._expression_value(node.args[0])
                    entries = self._overlay_mapping_entries(
                        entries, dict(self._mapping_source_entries(source))
                    )
                for keyword in node.keywords:
                    value = self._expression_value(keyword.value)
                    if keyword.arg is not None:
                        entries[_key_token(ast.Constant(keyword.arg))] = value
                    elif value.entries is not None:
                        entries = self._overlay_mapping_entries(entries, dict(value.entries))
                    else:
                        entries = self._overlay_mapping_entries(
                            entries,
                            {_DYNAMIC_KEY: _conservative_value(value)},
                        )
                return _AbstractValue(
                    entries=tuple(sorted(entries.items())), identity=frozenset({id(node)})
                )
            return _UNKNOWN_VALUE
        if isinstance(node, ast.Subscript):
            owner = self._expression_value(node.value)
            if _CONFIG_ROOT in owner.origins:
                return _origin_value(_CONFIG_ROOT)
            if isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, int) and owner.items is not None:
                    try:
                        return owner.items[node.slice.value]
                    except IndexError:
                        return _UNKNOWN_VALUE
                entries = dict(owner.entries or ())
                entry = entries.get(_key_token(node.slice))
                if entry is not None:
                    return entry
                return entries.get(_DYNAMIC_KEY, _UNKNOWN_VALUE)
            candidates = [*(owner.items or ()), *(value for _, value in owner.entries or ())]
            return _join_values(*candidates)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            sequence_items: list[_AbstractValue] = []
            for element in node.elts:
                value = self._expression_value(
                    element.value if isinstance(element, ast.Starred) else element
                )
                if isinstance(element, ast.Starred) and value.items is not None:
                    sequence_items.extend(value.items)
                else:
                    sequence_items.append(value)
            return _AbstractValue(
                items=tuple(sequence_items),
                identity=(frozenset({id(node)}) if isinstance(node, ast.List) else frozenset()),
                truth=self._static_truth(node),
            )
        if isinstance(node, ast.Dict):
            literal_entries: dict[str, _AbstractValue] = {}
            for key, item in zip(node.keys, node.values, strict=True):
                value = self._expression_value(item)
                if key is not None:
                    key_value = self._expression_value(key)
                    self._add_mapping_entry(
                        literal_entries, key_value.literal or _DYNAMIC_KEY, value
                    )
                elif value.entries is not None:
                    literal_entries = self._overlay_mapping_entries(
                        literal_entries, dict(value.entries)
                    )
                else:
                    literal_entries = self._overlay_mapping_entries(
                        literal_entries,
                        {_DYNAMIC_KEY: _conservative_value(value)},
                    )
            return _AbstractValue(
                entries=tuple(sorted(literal_entries.items())),
                identity=frozenset({id(node)}),
                truth=self._static_truth(node),
            )
        if isinstance(node, ast.IfExp):
            truth = self._static_truth(node.test)
            if truth is not None:
                return self._expression_value(node.body if truth else node.orelse)
            return _join_values(
                self._expression_value(node.body), self._expression_value(node.orelse)
            )
        if isinstance(node, ast.BoolOp):
            return _join_values(
                *(self._expression_value(value) for value in self._reachable_bool_values(node))
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._expression_value(node.left)
            right = self._expression_value(node.right)
            entries = self._overlay_mapping_entries(
                dict(left.entries or ()), dict(right.entries or ())
            )
            return _AbstractValue(
                entries=tuple(sorted(entries.items())),
                identity=frozenset({id(node)}),
            )
        if isinstance(node, ast.NamedExpr):
            return self._expression_value(node.value)
        return _UNKNOWN_VALUE

    def _record(self, section: str, name: str) -> None:
        field = ConfigField(section, name)
        if field in self._fields:
            self.reads.add(field)

    def _record_possible_exception(self, before: _BindingSnapshot) -> None:
        """Preserve a feasible exception edge without ending normal evaluation."""

        if self._exception_capture_depth == 0:
            return
        self._append_abrupt(
            self._flow_abrupts[-1],
            _AbruptPath(
                "raise",
                self._join_binding_snapshots(before, self._binding_snapshot()),
                exception_upper_bound="BaseException",
            ),
        )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        before = self._binding_snapshot()
        if isinstance(node.ctx, ast.Load):
            for section in self._expression_value(node.value).origins & TRACKED_SECTIONS:
                self._record(section, node.attr)
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            self._record_possible_exception(before)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        truth = self._static_truth(node.test)
        if truth is None:
            self.visit(node.body)
            self.visit(node.orelse)
        else:
            self.visit(node.body if truth else node.orelse)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        for value in self._reachable_bool_values(node):
            self.visit(value)

    def _consume_deferred_generator(self, node: ast.AST, *, mode: str = "full") -> None:
        value = self._expression_value(node)
        for identity in value.identity:
            deferred = self._deferred_generators.get(identity)
            if deferred is None:
                continue
            if isinstance(deferred.node, ast.GeneratorExp):
                self._consume_generator_expression(deferred.node, identity, mode=mode)
                continue
            self._visit_function_body(
                deferred.node,
                deferred.scoped,
                consume_generator=True,
                generator_consumer_mode=mode,
                generator_identity=identity,
            )

    def _consume_deferred_coroutine(self, node: ast.AST) -> None:
        value = self._expression_value(node)
        for identity in value.identity:
            deferred = self._deferred_coroutines.get(identity)
            if deferred is not None:
                self._visit_function_body(deferred.node, deferred.scoped)

    def visit_Call(self, node: ast.Call) -> None:
        before = self._binding_snapshot()
        accessor = self._expression_value(node.func)
        if accessor.origins & _EAGER_CONSUMER_ORIGINS:
            mode = (
                "one_turn"
                if "<builtin-consumer:next>" in accessor.origins
                else "any"
                if "<builtin-consumer:any>" in accessor.origins
                else "all"
                if "<builtin-consumer:all>" in accessor.origins
                else "full"
            )
            for argument in node.args:
                self._consume_deferred_generator(argument, mode=mode)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "send"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value is None
        ):
            self._consume_deferred_generator(node.func.value, mode="one_turn")
        # Evaluate once even when the call is a standalone mutating
        # ``setdefault`` expression.
        self._expression_value(node)
        if accessor.accessed_attributes:
            for argument in node.args:
                value = self._expression_value(
                    argument.value if isinstance(argument, ast.Starred) else argument
                )
                for section in value.origins & TRACKED_SECTIONS:
                    for name in accessor.accessed_attributes:
                        self._record(section, name.partition(".")[0])
        if _GETATTR_BUILTIN in accessor.origins and len(node.args) >= 2:
            field_name = self._expression_value(node.args[1]).string_value
            if field_name is not None:
                for section in self._expression_value(node.args[0]).origins & TRACKED_SECTIONS:
                    self._record(section, field_name)
        self.generic_visit(node)
        self._record_possible_exception(before)

    def visit_Await(self, node: ast.Await) -> None:
        """Awaiting ``anext`` consumes one turn of an async generator."""
        self._consume_deferred_coroutine(node.value)
        if isinstance(node.value, ast.Call):
            accessor = self._expression_value(node.value.func)
            if accessor.origins & _ASYNC_BUILTIN_CONSUMER_ORIGINS:
                for argument in node.value.args:
                    self._consume_deferred_generator(argument, mode="one_turn")
            elif (
                isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "asend"
                and node.value.args
                and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value is None
            ):
                self._consume_deferred_generator(node.value.func.value, mode="one_turn")
        self.generic_visit(node)

    def visit_Starred(self, node: ast.Starred) -> None:
        """Iterable unpacking eagerly consumes a deferred generator."""
        self._consume_deferred_generator(node.value)
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """``yield from`` eagerly consumes its delegated iterable."""
        self._consume_deferred_generator(node.value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        before = self._binding_snapshot()
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            for section in self._expression_value(node.value).serialized_sections:
                self._record(section, node.slice.value)
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            self._record_possible_exception(before)

    @staticmethod
    def _join_states(
        *states: Mapping[str, _AbstractValue],
    ) -> dict[str, _AbstractValue]:
        names = set().union(*(state.keys() for state in states))
        return {
            name: _join_values(*(state.get(name, _UNKNOWN_VALUE) for state in states))
            for name in names
        }

    @staticmethod
    def _join_function_bindings(
        *states: Mapping[str, _FunctionSet],
    ) -> dict[str, _FunctionSet]:
        names = set().union(*(state.keys() for state in states))
        return {
            name: frozenset().union(*(state.get(name, frozenset()) for state in states))
            for name in names
        }

    @staticmethod
    def _join_annotation_bindings(
        *states: Mapping[str, _AbstractValue],
    ) -> dict[str, _AbstractValue]:
        names = set().union(*(state.keys() for state in states))
        return {
            name: _join_values(*(state.get(name, _UNKNOWN_VALUE) for state in states))
            for name in names
        }

    def _binding_snapshot(self) -> _BindingSnapshot:
        return (
            dict(self._states[-1]),
            dict(self._functions[-1]),
            dict(self._annotations[-1]),
        )

    def _restore_bindings(self, snapshot: _BindingSnapshot) -> None:
        state, functions, annotations = snapshot
        self._states[-1] = dict(state)
        self._functions[-1] = dict(functions)
        self._annotations[-1] = dict(annotations)

    def _join_binding_snapshots(self, *snapshots: _BindingSnapshot) -> _BindingSnapshot:
        return (
            self._join_states(*(snapshot[0] for snapshot in snapshots)),
            self._join_function_bindings(*(snapshot[1] for snapshot in snapshots)),
            self._join_annotation_bindings(*(snapshot[2] for snapshot in snapshots)),
        )

    def _join_optional_bindings(
        self, snapshots: Iterable[_BindingSnapshot]
    ) -> _BindingSnapshot | None:
        paths = tuple(snapshots)
        return self._join_binding_snapshots(*paths) if paths else None

    def _append_abrupt(self, target: list[_AbruptPath], path: _AbruptPath) -> None:
        for index, existing in enumerate(target):
            if (
                existing.kind == path.kind
                and existing.return_value == path.return_value
                and existing.exception_type == path.exception_type
                and existing.exception_exclusions == path.exception_exclusions
                and existing.exception_upper_bound == path.exception_upper_bound
            ):
                target[index] = _AbruptPath(
                    path.kind,
                    self._join_binding_snapshots(existing.bindings, path.bindings),
                    path.return_value,
                    path.exception_type,
                    path.exception_exclusions,
                    path.exception_upper_bound,
                )
                return
        target.append(path)

    def _extend_abrupts(self, target: list[_AbruptPath], paths: Iterable[_AbruptPath]) -> None:
        for path in paths:
            self._append_abrupt(target, path)

    def _apply_flow_result(self, result: _FlowResult) -> None:
        self._extend_abrupts(self._flow_abrupts[-1], result.abrupt)
        self._path_reachable = result.fallthrough is not None
        if result.fallthrough is not None:
            self._restore_bindings(result.fallthrough)

    def _visit_binding_branch(
        self,
        statements: Iterable[ast.stmt],
        initial: _BindingSnapshot,
    ) -> _FlowResult:
        previous_reachable = self._path_reachable
        previous_cache = self._expression_cache
        self._restore_bindings(initial)
        self._path_reachable = True
        self._flow_abrupts.append([])
        self._expression_cache = {}
        try:
            for statement in statements:
                self.visit(statement)
                if not self._path_reachable:
                    break
            return _FlowResult(
                self._binding_snapshot() if self._path_reachable else None,
                tuple(self._flow_abrupts[-1]),
            )
        finally:
            self._flow_abrupts.pop()
            self._expression_cache = previous_cache
            self._path_reachable = previous_reachable

    def _binding_scope_index(self, name: str) -> int:
        """Resolve the lexical scope targeted by global/nonlocal assignment."""
        if name in self._global_names[-1]:
            return 0
        if name in self._nonlocal_names[-1]:
            for index in range(len(self._states) - 2, -1, -1):
                if name in self._states[index] or index == 0:
                    return index
        return len(self._states) - 1

    def _capture_function_closure(self, node: _FunctionNode) -> None:
        visible: dict[str, _AbstractValue] = {}
        if len(self._states) <= 1:
            return
        for scope in self._states:
            visible.update(scope)
        visible = {
            name: value
            for name, value in visible.items()
            if _contained_origins(value) & (TRACKED_SECTIONS | {_CONFIG_ROOT})
        }
        if not visible:
            return
        existing = self._closure_bindings.get(id(node))
        self._closure_bindings[id(node)] = (
            self._join_states(existing, visible) if existing is not None else visible
        )

    def visit_Global(self, node: ast.Global) -> None:
        self._global_names[-1].update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._nonlocal_names[-1].update(node.names)

    def _bind_target_value(self, target: ast.expr, value: _AbstractValue) -> None:
        if isinstance(target, ast.Name):
            self._states[self._binding_scope_index(target.id)][target.id] = value
        elif isinstance(target, ast.Starred):
            self._bind_target_value(target.value, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_target_value(element, _conservative_value(value))

    def _bind_annotation_target(self, target: ast.expr, value: _AbstractValue) -> None:
        if isinstance(target, ast.Name):
            self._annotations[self._binding_scope_index(target.id)][target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_annotation_target(element, _UNKNOWN_VALUE)

    def _bind_function_target(
        self,
        target: ast.expr,
        functions: _FunctionSet,
    ) -> None:
        if isinstance(target, ast.Name):
            self._functions[self._binding_scope_index(target.id)][target.id] = functions
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_function_target(element, frozenset())

    def _bind_destructured(self, target: ast.expr, value: _AbstractValue) -> None:
        if isinstance(target, ast.Name):
            self._states[self._binding_scope_index(target.id)][target.id] = value
            return
        if isinstance(target, ast.Starred):
            self._bind_target_value(target.value, value)
            return
        if not isinstance(target, (ast.Tuple, ast.List)):
            return
        if value.items is None:
            self._bind_target_value(target, _conservative_value(value))
            return

        starred = next(
            (
                index
                for index, element in enumerate(target.elts)
                if isinstance(element, ast.Starred)
            ),
            None,
        )
        if starred is None:
            if len(target.elts) == len(value.items):
                for child_target, child_value in zip(target.elts, value.items, strict=True):
                    self._bind_destructured(child_target, child_value)
            else:
                self._bind_target_value(target, _conservative_value(value))
            return

        suffix_length = len(target.elts) - starred - 1
        if len(value.items) < starred + suffix_length:
            self._bind_target_value(target, _conservative_value(value))
            return
        for child_target, child_value in zip(
            target.elts[:starred], value.items[:starred], strict=True
        ):
            self._bind_destructured(child_target, child_value)
        starred_items = value.items[starred : len(value.items) - suffix_length or None]
        self._bind_target_value(target.elts[starred], _AbstractValue(items=starred_items))
        if suffix_length:
            for child_target, child_value in zip(
                target.elts[-suffix_length:], value.items[-suffix_length:], strict=True
            ):
                self._bind_destructured(child_target, child_value)

    def _visit_store_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Attribute):
            self.visit(target.value)
        elif isinstance(target, ast.Subscript):
            self.visit(target.value)
            self.visit(target.slice)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._visit_store_target(element)
        elif isinstance(target, ast.Starred):
            self._visit_store_target(target.value)

    def _assign_store_target(self, target: ast.expr, value: _AbstractValue) -> None:
        if isinstance(target, ast.Attribute):
            owner = self._expression_value(target.value)
            if owner.attributes is None and not owner.identity:
                return
            attributes = dict(owner.attributes or ())
            attributes[target.attr] = value
            replacement = self._attribute_replacement(owner, tuple(sorted(attributes.items())))
            self._replace_shared_value(owner, replacement, target.value)
            return
        if not isinstance(target, ast.Subscript):
            return
        owner = self._expression_value(target.value)
        if owner.entries is None and not owner.identity:
            return
        if owner.items is not None:
            items = list(owner.items)
            if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, int):
                index = target.slice.value
                if -len(items) <= index < len(items):
                    items[index] = value
                else:
                    return
            else:
                items = [_join_values(item, value) for item in items]
            replacement = self._sequence_replacement(owner, tuple(items))
            self._replace_shared_value(owner, replacement, target.value)
            return
        entries = dict(owner.entries or ())
        if isinstance(target.slice, ast.Constant):
            entries[_key_token(target.slice)] = value
        else:
            self._add_mapping_entry(entries, _DYNAMIC_KEY, value)
        replacement = self._mapping_replacement(owner, sorted(entries.items()))
        self._replace_shared_value(owner, replacement, target.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = self._expression_value(node.value)
        annotation_value = self._annotation_value(node.value)
        function = self._function_value(node.value)
        for target in node.targets:
            self._visit_store_target(target)
            self._assign_store_target(target, value)
            self._bind_destructured(target, value)
            self._bind_annotation_target(target, annotation_value)
            self._bind_function_target(target, function)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        self._bind_function_target(node.target, frozenset())
        if node.value is not None:
            self.visit(node.value)
            self._visit_store_target(node.target)
            value = self._expression_value(node.value)
            self._assign_store_target(node.target, value)
            self._bind_destructured(node.target, value)
            self._bind_annotation_target(node.target, self._annotation_value(node.value))
            self._bind_function_target(node.target, self._function_value(node.value))

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_destructured(node.target, self._expression_value(node.value))
        self._bind_function_target(node.target, self._function_value(node.value))

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                index = self._binding_scope_index(target.id)
                self._states[index].pop(target.id, None)
                self._annotations[index].pop(target.id, None)
                self._functions[index].pop(target.id, None)
            else:
                self.visit(target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        owner = self._expression_value(node.target)
        self.visit(node.target)
        self.visit(node.value)
        if isinstance(node.op, ast.BitOr):
            right = self._expression_value(node.value)
            entries = dict(owner.entries or ())
            if right.entries is not None:
                entries = self._overlay_mapping_entries(entries, dict(right.entries))
            else:
                entries = self._overlay_mapping_entries(
                    entries,
                    {_DYNAMIC_KEY: _conservative_value(right)},
                )
            replacement = self._mapping_replacement(owner, sorted(entries.items()))
            self._replace_shared_value(owner, replacement, node.target)
            return
        if isinstance(node.target, ast.Name):
            index = self._binding_scope_index(node.target.id)
            self._states[index][node.target.id] = self._name_value(node.target.id)
            self._functions[index][node.target.id] = frozenset()

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        initial = self._binding_snapshot()
        truth = self._static_truth(node.test)
        if truth is not None:
            live_branch = node.body if truth else node.orelse
            dead_branch = node.orelse if truth else node.body
            if self._uses_type_checking(node.test):
                self._register_type_only_imports(dead_branch)
                initial = self._binding_snapshot()
            self._apply_flow_result(self._visit_binding_branch(live_branch, initial))
            return
        body_result = self._visit_binding_branch(node.body, initial)
        else_result = (
            self._visit_binding_branch(node.orelse, initial)
            if node.orelse
            else _FlowResult(initial)
        )
        self._apply_flow_result(
            _FlowResult(
                self._join_optional_bindings(
                    state
                    for state in (body_result.fallthrough, else_result.fallthrough)
                    if state is not None
                ),
                (*body_result.abrupt, *else_result.abrupt),
            )
        )

    @staticmethod
    def _iteration_values(value: _AbstractValue) -> tuple[_AbstractValue, ...]:
        if value.items is not None:
            return value.items
        if value.entries is not None:
            return tuple(_UNKNOWN_VALUE for _ in value.entries)
        return (_UNKNOWN_VALUE,)

    def _bind_iteration_target(self, target: ast.expr, value: _AbstractValue) -> bool:
        candidates = self._iteration_values(value)
        if not candidates:
            return False
        initial = dict(self._states[-1])
        candidate_states: list[dict[str, _AbstractValue]] = []
        for candidate in candidates:
            self._states[-1] = dict(initial)
            self._bind_destructured(target, candidate)
            self._bind_function_target(target, frozenset())
            self._bind_annotation_target(target, _UNKNOWN_VALUE)
            candidate_states.append(dict(self._states[-1]))
        self._states[-1] = self._join_states(*candidate_states)
        return True

    def visit_For(self, node: ast.For) -> None:
        if isinstance(node.iter, ast.GeneratorExp):
            self._visit_comprehension(node.iter)
        else:
            mode = "one_turn" if node.body and isinstance(node.body[0], ast.Break) else "full"
            self._consume_deferred_generator(node.iter, mode=mode)
        self.visit(node.iter)
        iterable = self._expression_value(node.iter)
        zero_iterations_possible = self._static_truth(node.iter) is not True
        entry = self._binding_snapshot()
        self._restore_bindings(entry)
        if not self._bind_iteration_target(node.target, iterable):
            completed = (
                self._visit_binding_branch(node.orelse, entry)
                if node.orelse
                else _FlowResult(entry)
            )
            self._apply_flow_result(completed)
            return
        header = entry
        while True:
            self._restore_bindings(header)
            self._bind_iteration_target(node.target, iterable)
            body_result = self._visit_binding_branch(node.body, self._binding_snapshot())
            back_edges = [path.bindings for path in body_result.abrupt if path.kind == "continue"]
            if body_result.fallthrough is not None:
                back_edges.append(body_result.fallthrough)
            joined = self._join_binding_snapshots(entry, *back_edges)
            if joined == header:
                break
            header = joined

        normal_paths = ([] if not zero_iterations_possible else [entry]) + back_edges
        normal_entry = self._join_optional_bindings(normal_paths)
        normal_result = (
            self._visit_binding_branch(node.orelse, normal_entry)
            if node.orelse and normal_entry is not None
            else _FlowResult(normal_entry)
        )
        break_paths = [path.bindings for path in body_result.abrupt if path.kind == "break"]
        outer_abrupt = tuple(
            path for path in body_result.abrupt if path.kind not in {"break", "continue"}
        )
        post_paths = [*break_paths]
        if normal_result.fallthrough is not None:
            post_paths.append(normal_result.fallthrough)
        self._apply_flow_result(
            _FlowResult(
                self._join_optional_bindings(post_paths),
                (*outer_abrupt, *normal_result.abrupt),
            )
        )

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_While(self, node: ast.While) -> None:
        entry = self._binding_snapshot()
        self.visit(node.test)
        truth = self._static_truth(node.test)
        if truth is False:
            tested_state = self._binding_snapshot()
            completed = (
                self._visit_binding_branch(node.orelse, tested_state)
                if node.orelse
                else _FlowResult(tested_state)
            )
            self._apply_flow_result(completed)
            return
        header = entry
        while True:
            self._restore_bindings(header)
            self.visit(node.test)
            tested_state = self._binding_snapshot()
            body_result = self._visit_binding_branch(node.body, tested_state)
            back_edges = [path.bindings for path in body_result.abrupt if path.kind == "continue"]
            if body_result.fallthrough is not None:
                back_edges.append(body_result.fallthrough)
            joined = self._join_binding_snapshots(entry, *back_edges)
            if joined == header:
                break
            header = joined

        normal_entry: _BindingSnapshot | None = None
        if truth is not True:
            self._restore_bindings(header)
            self.visit(node.test)
            normal_entry = self._binding_snapshot()
        normal_result = (
            self._visit_binding_branch(node.orelse, normal_entry)
            if node.orelse and normal_entry is not None
            else _FlowResult(normal_entry)
        )
        break_paths = [path.bindings for path in body_result.abrupt if path.kind == "break"]
        outer_abrupt = tuple(
            path for path in body_result.abrupt if path.kind not in {"break", "continue"}
        )
        post_paths = [*break_paths]
        if normal_result.fallthrough is not None:
            post_paths.append(normal_result.fallthrough)
        self._apply_flow_result(
            _FlowResult(
                self._join_optional_bindings(post_paths),
                (*outer_abrupt, *normal_result.abrupt),
            )
        )

    @classmethod
    def _exception_name(cls, node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Call):
            return cls._exception_name(node.func)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return cls._dotted_name(node)
        return None

    @classmethod
    def _handler_exception_names(cls, node: ast.AST | None) -> frozenset[str]:
        if node is None:
            return frozenset({"BaseException"})
        if isinstance(node, ast.Tuple):
            return frozenset().union(*(cls._handler_exception_names(item) for item in node.elts))
        name = cls._exception_name(node)
        return frozenset({name.rsplit(".", 1)[-1]}) if name is not None else frozenset()

    @staticmethod
    def _builtin_exception_type(name: str) -> type[BaseException] | None:
        candidate = getattr(builtins, name.rsplit(".", 1)[-1], None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            return candidate
        return None

    @classmethod
    def _known_exception_subclass(cls, child: str, parent: str) -> bool | None:
        child = child.rsplit(".", 1)[-1]
        parent = parent.rsplit(".", 1)[-1]
        if child == parent:
            return True
        if parent == "BaseException":
            return True
        child_type = cls._builtin_exception_type(child)
        parent_type = cls._builtin_exception_type(parent)
        if child_type is None or parent_type is None:
            return None
        return issubclass(child_type, parent_type)

    @classmethod
    def _exception_domain_intersection(cls, left: str, right: str) -> str | None:
        if cls._known_exception_subclass(left, right) is True:
            return left
        if cls._known_exception_subclass(right, left) is True:
            return right
        left_type = cls._builtin_exception_type(left)
        right_type = cls._builtin_exception_type(right)
        if left_type is not None and right_type is not None:
            return None
        return right

    @classmethod
    def _exception_domain_excluded(cls, domain: str, exclusions: frozenset[str]) -> bool:
        return any(
            cls._known_exception_subclass(domain, excluded) is True for excluded in exclusions
        )

    @classmethod
    def _partition_exception_path(
        cls, path: _AbruptPath, handler: ast.ExceptHandler
    ) -> tuple[tuple[_AbruptPath, ...], tuple[_AbruptPath, ...]]:
        """Split one raise into this handler's caught and still-unmatched domains."""

        names = cls._handler_exception_names(handler.type)
        if not names:
            return (path,), (path,)
        if path.exception_type is not None:
            raised = path.exception_type.rsplit(".", 1)[-1]
            upper_bound = path.exception_upper_bound or "BaseException"
            exact_caught_domains: list[str] = []
            exact_feasible_names: set[str] = set()
            exact_fully_consumed = False
            for name in names:
                relation = cls._known_exception_subclass(raised, name)
                domain = cls._exception_domain_intersection(upper_bound, name)
                if (
                    relation is False
                    or domain is None
                    or cls._exception_domain_excluded(domain, path.exception_exclusions)
                ):
                    continue
                exact_feasible_names.add(name)
                if relation is True or cls._known_exception_subclass(upper_bound, name) is True:
                    exact_fully_consumed = True
                if any(
                    cls._known_exception_subclass(domain, existing) is True
                    for existing in exact_caught_domains
                ):
                    continue
                exact_caught_domains = [
                    existing
                    for existing in exact_caught_domains
                    if cls._known_exception_subclass(existing, domain) is not True
                ]
                exact_caught_domains.append(domain)
            if not exact_caught_domains:
                return (), (path,)
            if exact_fully_consumed:
                return (path,), ()
            caught = tuple(
                _AbruptPath(
                    path.kind,
                    path.bindings,
                    path.return_value,
                    path.exception_type,
                    path.exception_exclusions,
                    domain,
                )
                for domain in exact_caught_domains
            )
            remaining_exclusions = path.exception_exclusions | exact_feasible_names
            remaining = _AbruptPath(
                path.kind,
                path.bindings,
                path.return_value,
                path.exception_type,
                remaining_exclusions,
                upper_bound,
            )
            return caught, (remaining,)

        upper_bound = path.exception_upper_bound or "BaseException"
        caught_domains: list[str] = []
        feasible_names: set[str] = set()
        fully_consumed = False
        for name in names:
            domain = cls._exception_domain_intersection(upper_bound, name)
            if domain is None or cls._exception_domain_excluded(domain, path.exception_exclusions):
                continue
            feasible_names.add(name)
            if cls._known_exception_subclass(upper_bound, name) is True:
                fully_consumed = True
            if any(
                cls._known_exception_subclass(domain, existing) is True
                for existing in caught_domains
            ):
                continue
            caught_domains = [
                existing
                for existing in caught_domains
                if cls._known_exception_subclass(existing, domain) is not True
            ]
            caught_domains.append(domain)
        if not caught_domains:
            return (), (path,)
        caught = tuple(
            _AbruptPath(
                "raise",
                path.bindings,
                exception_exclusions=path.exception_exclusions,
                exception_upper_bound=domain,
            )
            for domain in caught_domains
        )
        if fully_consumed:
            return caught, ()
        remaining_exclusions = path.exception_exclusions | feasible_names
        if cls._exception_domain_excluded(upper_bound, remaining_exclusions):
            return caught, ()
        remaining = _AbruptPath(
            "raise",
            path.bindings,
            exception_exclusions=remaining_exclusions,
            exception_upper_bound=upper_bound,
        )
        return caught, (remaining,)

    def _visit_try(self, node: ast.Try | ast.TryStar) -> None:
        self._exception_capture_depth += 1
        try:
            self._visit_try_paths(node)
        finally:
            self._exception_capture_depth -= 1

    def _visit_try_paths(self, node: ast.Try | ast.TryStar) -> None:
        entry = self._binding_snapshot()
        body_result = self._visit_binding_branch(node.body, entry)
        normal_result = (
            self._visit_binding_branch(node.orelse, body_result.fallthrough)
            if node.orelse and body_result.fallthrough is not None
            else _FlowResult(body_result.fallthrough)
        )
        handler_results: list[_FlowResult] = []
        remaining_raises = [path for path in body_result.abrupt if path.kind == "raise"]
        for handler in node.handlers:
            caught: list[_AbruptPath] = []
            still_unmatched: list[_AbruptPath] = []
            for path in remaining_raises:
                caught_paths, remaining_paths = self._partition_exception_path(path, handler)
                self._extend_abrupts(caught, caught_paths)
                self._extend_abrupts(still_unmatched, remaining_paths)
            remaining_raises = still_unmatched
            if not caught:
                continue
            handler_entries = [path.bindings for path in caught]
            if isinstance(node, ast.TryStar):
                handler_entries.extend(
                    state
                    for result in handler_results
                    for state in (
                        result.fallthrough,
                        *(path.bindings for path in result.abrupt),
                    )
                    if state is not None
                )
            self._restore_bindings(self._join_binding_snapshots(*handler_entries))
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name is not None:
                self._states[-1][handler.name] = _UNKNOWN_VALUE
                self._functions[-1][handler.name] = frozenset()
                self._annotations[-1][handler.name] = _UNKNOWN_VALUE
            self._caught_exception_stack.append(tuple(caught))
            try:
                handler_results.append(
                    self._visit_binding_branch(handler.body, self._binding_snapshot())
                )
            finally:
                self._caught_exception_stack.pop()

        fallthrough_paths = [
            state
            for state in (
                normal_result.fallthrough,
                *(result.fallthrough for result in handler_results),
            )
            if state is not None
        ]
        abrupt = [
            *(path for path in body_result.abrupt if path.kind != "raise"),
            *remaining_raises,
            *normal_result.abrupt,
            *(path for result in handler_results for path in result.abrupt),
        ]
        if not node.finalbody:
            self._apply_flow_result(
                _FlowResult(self._join_optional_bindings(fallthrough_paths), tuple(abrupt))
            )
            return

        final_fallthrough: _BindingSnapshot | None = None
        final_abrupt: list[_AbruptPath] = []
        if fallthrough_paths:
            completed_final = self._visit_binding_branch(
                node.finalbody, self._join_binding_snapshots(*fallthrough_paths)
            )
            final_fallthrough = completed_final.fallthrough
            final_abrupt.extend(completed_final.abrupt)
        for path in abrupt:
            abrupt_final = self._visit_binding_branch(node.finalbody, path.bindings)
            if abrupt_final.fallthrough is not None:
                final_abrupt.append(
                    _AbruptPath(
                        path.kind,
                        abrupt_final.fallthrough,
                        path.return_value,
                        path.exception_type,
                        path.exception_exclusions,
                        path.exception_upper_bound,
                    )
                )
            final_abrupt.extend(abrupt_final.abrupt)
        self._apply_flow_result(_FlowResult(final_fallthrough, tuple(final_abrupt)))

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_try(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._visit_try(node)

    def _visit_pattern_reads(self, pattern: ast.pattern) -> None:
        if isinstance(pattern, ast.MatchValue):
            self.visit(pattern.value)
        elif isinstance(pattern, ast.MatchSequence):
            for child in pattern.patterns:
                self._visit_pattern_reads(child)
        elif isinstance(pattern, ast.MatchMapping):
            for key in pattern.keys:
                self.visit(key)
            for child in pattern.patterns:
                self._visit_pattern_reads(child)
        elif isinstance(pattern, ast.MatchClass):
            self.visit(pattern.cls)
            for child in (*pattern.patterns, *pattern.kwd_patterns):
                self._visit_pattern_reads(child)
        elif isinstance(pattern, ast.MatchOr):
            for child in pattern.patterns:
                self._visit_pattern_reads(child)
        elif isinstance(pattern, ast.MatchAs) and pattern.pattern is not None:
            self._visit_pattern_reads(pattern.pattern)

    def _bind_pattern(self, pattern: ast.pattern, subject: _AbstractValue) -> None:
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None:
                self._bind_pattern(pattern.pattern, subject)
            if pattern.name is not None:
                self._states[-1][pattern.name] = subject
                self._functions[-1][pattern.name] = frozenset()
                self._annotations[-1][pattern.name] = _UNKNOWN_VALUE
            return
        if isinstance(pattern, ast.MatchStar):
            if pattern.name is not None:
                self._states[-1][pattern.name] = subject
                self._functions[-1][pattern.name] = frozenset()
                self._annotations[-1][pattern.name] = _UNKNOWN_VALUE
            return
        if isinstance(pattern, ast.MatchOr):
            initial = dict(self._states[-1])
            alternatives: list[dict[str, _AbstractValue]] = []
            for alternative in pattern.patterns:
                self._states[-1] = dict(initial)
                self._bind_pattern(alternative, subject)
                alternatives.append(dict(self._states[-1]))
            self._states[-1] = self._join_states(*alternatives)
            return
        if isinstance(pattern, ast.MatchSequence):
            self._bind_sequence_pattern(pattern, subject)
            return
        if isinstance(pattern, ast.MatchMapping):
            entries = dict(subject.entries or ())
            fallback = _conservative_value(subject)
            matched_tokens: set[str] = set()
            for key, child in zip(pattern.keys, pattern.patterns, strict=True):
                token = _key_token(key)
                matched_tokens.add(token)
                self._bind_pattern(child, entries.get(token, fallback))
            if pattern.rest is not None:
                remaining = tuple(
                    (key, value)
                    for key, value in subject.entries or ()
                    if key not in matched_tokens
                )
                self._states[-1][pattern.rest] = _AbstractValue(
                    entries=remaining, identity=frozenset({id(pattern)})
                )
                self._functions[-1][pattern.rest] = frozenset()
                self._annotations[-1][pattern.rest] = _UNKNOWN_VALUE
            return
        if isinstance(pattern, ast.MatchClass):
            fallback = _conservative_value(subject)
            for index, child in enumerate(pattern.patterns):
                value = (
                    subject.items[index]
                    if subject.items is not None and index < len(subject.items)
                    else fallback
                )
                self._bind_pattern(child, value)
            attributes = dict(subject.attributes or ())
            for name, child in zip(pattern.kwd_attrs, pattern.kwd_patterns, strict=True):
                self._bind_pattern(child, attributes.get(name, fallback))

    def _bind_sequence_pattern(self, pattern: ast.MatchSequence, subject: _AbstractValue) -> None:
        if subject.items is None:
            fallback = _conservative_value(subject)
            for child in pattern.patterns:
                self._bind_pattern(child, fallback)
            return
        starred = next(
            (
                index
                for index, child in enumerate(pattern.patterns)
                if isinstance(child, ast.MatchStar)
            ),
            None,
        )
        if starred is None and len(pattern.patterns) == len(subject.items):
            for child, value in zip(pattern.patterns, subject.items, strict=True):
                self._bind_pattern(child, value)
            return
        if starred is not None:
            suffix_length = len(pattern.patterns) - starred - 1
            if len(subject.items) >= starred + suffix_length:
                for child, value in zip(
                    pattern.patterns[:starred], subject.items[:starred], strict=True
                ):
                    self._bind_pattern(child, value)
                middle = subject.items[starred : len(subject.items) - suffix_length or None]
                self._bind_pattern(pattern.patterns[starred], _AbstractValue(items=middle))
                if suffix_length:
                    for child, value in zip(
                        pattern.patterns[-suffix_length:],
                        subject.items[-suffix_length:],
                        strict=True,
                    ):
                        self._bind_pattern(child, value)
                return
        fallback = _conservative_value(subject)
        for child in pattern.patterns:
            self._bind_pattern(child, fallback)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        subject = self._expression_value(node.subject)
        initial = self._binding_snapshot()
        branches: list[_FlowResult] = []
        unmatched = True
        for case in node.cases:
            self._restore_bindings(initial)
            self._visit_pattern_reads(case.pattern)
            self._bind_pattern(case.pattern, subject)
            if case.guard is not None:
                self.visit(case.guard)
                if self._static_truth(case.guard) is False:
                    continue
            branches.append(self._visit_binding_branch(case.body, self._binding_snapshot()))
            if (
                isinstance(case.pattern, ast.MatchAs)
                and case.pattern.pattern is None
                and (case.guard is None or self._static_truth(case.guard) is True)
            ):
                unmatched = False
                break
        if unmatched:
            branches.append(_FlowResult(initial))
        self._apply_flow_result(
            _FlowResult(
                self._join_optional_bindings(
                    result.fallthrough for result in branches if result.fallthrough is not None
                ),
                tuple(path for result in branches for path in result.abrupt),
            )
        )

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
    ) -> None:
        first, *remaining = node.generators
        self.visit(first.iter)
        first_value = self._expression_value(first.iter)
        self._states.append({})
        self._annotations.append({})
        self._functions.append({})
        self._global_names.append(set())
        self._nonlocal_names.append(set())
        try:
            if not self._bind_iteration_target(first.target, first_value):
                self._expression_cache[id(node)] = _UNKNOWN_VALUE
                return
            for condition in first.ifs:
                self.visit(condition)
                if self._static_truth(condition) is False:
                    self._expression_cache[id(node)] = _UNKNOWN_VALUE
                    return
            for generator in remaining:
                self.visit(generator.iter)
                if not self._bind_iteration_target(
                    generator.target, self._expression_value(generator.iter)
                ):
                    self._expression_cache[id(node)] = _UNKNOWN_VALUE
                    return
                for condition in generator.ifs:
                    self.visit(condition)
                    if self._static_truth(condition) is False:
                        self._expression_cache[id(node)] = _UNKNOWN_VALUE
                        return
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
                result = _AbstractValue(
                    entries=((_key_token(node.key), self._expression_value(node.value)),),
                    identity=frozenset({id(node)}),
                )
            else:
                self.visit(node.elt)
                result = _AbstractValue(items=(self._expression_value(node.elt),))
            self._expression_cache[id(node)] = result
        finally:
            self._nonlocal_names.pop()
            self._global_names.pop()
            self._functions.pop()
            self._annotations.pop()
            self._states.pop()

    def _consume_generator_expression(
        self, node: ast.GeneratorExp, identity: int, *, mode: str
    ) -> None:
        """Advance one generator-expression identity with runtime consumption semantics."""
        first, *remaining = node.generators
        candidates = self._iteration_values(self._expression_value(first.iter))
        start = self._deferred_generator_positions.get(identity, 0)
        consumed = 0
        for candidate in candidates[start:]:
            consumed += 1
            self._states.append({})
            self._annotations.append({})
            self._functions.append({})
            self._global_names.append(set())
            self._nonlocal_names.append(set())
            yielded = True
            try:
                self._bind_destructured(first.target, candidate)
                for condition in first.ifs:
                    self.visit(condition)
                    if self._static_truth(condition) is False:
                        yielded = False
                        break
                if yielded:
                    for generator in remaining:
                        self.visit(generator.iter)
                        if not self._bind_iteration_target(
                            generator.target, self._expression_value(generator.iter)
                        ):
                            yielded = False
                            break
                        for condition in generator.ifs:
                            self.visit(condition)
                            if self._static_truth(condition) is False:
                                yielded = False
                                break
                        if not yielded:
                            break
                if yielded:
                    self.visit(node.elt)
                    truth = self._static_truth(node.elt)
            finally:
                self._nonlocal_names.pop()
                self._global_names.pop()
                self._functions.pop()
                self._annotations.pop()
                self._states.pop()
            if yielded and (
                mode == "one_turn"
                or mode == "any"
                and truth is True
                or mode == "all"
                and truth is False
            ):
                break
        self._deferred_generator_positions[identity] = start + consumed

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        # Python eagerly evaluates only the outermost iterable at generator
        # construction.  The targets, filters, nested iterables and element
        # expression stay deferred until a consumer advances the generator.
        if id(node) in self._expression_cache:
            return
        self.visit(node.generators[0].iter)
        self._deferred_generators[id(node)] = _DeferredGenerator(node, {})
        self._expression_cache[id(node)] = _AbstractValue(identity=frozenset({id(node)}))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def _uses_type_checking(self, node: ast.AST) -> bool:
        return any(
            _TYPE_CHECKING_FALSE in self._expression_value(candidate).origins
            for candidate in ast.walk(node)
            if isinstance(candidate, (ast.Name, ast.Attribute))
        )

    def _register_type_only_imports(self, statements: Iterable[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, ast.Import):
                self._bind_import(statement, runtime=False)
            elif isinstance(statement, ast.ImportFrom):
                self._bind_import_from(statement, runtime=False)
            elif isinstance(statement, ast.If):
                self._register_type_only_imports(statement.body)
                self._register_type_only_imports(statement.orelse)

    @staticmethod
    def _imported_annotation(module: str | None, name: str) -> _AbstractValue:
        if module in _CONFIG_ANNOTATION_MODULES:
            section = _SECTION_ANNOTATIONS.get(name)
            if section is not None:
                return _origin_value(section)
            if name == "OuroborosConfig":
                return _origin_value(_CONFIG_ROOT)
            if name == "models":
                return _origin_value(_ANNOTATION_MODULE)
        return _UNKNOWN_VALUE

    def _bind_import(self, node: ast.Import, *, runtime: bool) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.partition(".")[0]
            annotation_value = (
                _origin_value(_ANNOTATION_MODULE)
                if alias.name in _CONFIG_ANNOTATION_MODULES
                else _UNKNOWN_VALUE
            )
            self._annotations[-1][bound] = annotation_value
            if runtime:
                self._functions[-1][bound] = frozenset()
                imported_name = alias.name if alias.asname else alias.name.partition(".")[0]
                module = self._source_index.resolve_module(imported_name, self._module)
                self._states[-1][bound] = (
                    _origin_value(_TYPING_MODULE)
                    if alias.name == "typing"
                    else _origin_value(_OPERATOR_MODULE)
                    if alias.name == "operator"
                    else _origin_value(_BUILTINS_MODULE)
                    if alias.name == "builtins"
                    else _AbstractValue(modules=frozenset({"collections"}))
                    if alias.name == "collections"
                    else _AbstractValue(
                        modules=frozenset({module.name}) if module is not None else frozenset()
                    )
                )

    def _bind_import_from(self, node: ast.ImportFrom, *, runtime: bool) -> None:
        module = self._source_index.resolve_module(node.module, self._module, node.level)
        for alias in node.names:
            if alias.name == "*":
                continue
            bound = alias.asname or alias.name
            self._annotations[-1][bound] = self._imported_annotation(node.module, alias.name)
            if runtime:
                functions = (
                    self._source_index.resolve_functions(module, alias.name)
                    if module is not None
                    else frozenset()
                )
                classes = (
                    self._source_index.resolve_classes(module, alias.name)
                    if module is not None
                    else frozenset()
                )
                imported_module = self._source_index.resolve_module(
                    f"{module.name}.{alias.name}" if module is not None else alias.name,
                    self._module,
                )
                self._functions[-1][bound] = functions
                self._states[-1][bound] = (
                    _type_checking_value()
                    if node.module == "typing" and alias.name == "TYPE_CHECKING"
                    else _origin_value(_ATTRGETTER_FACTORY)
                    if node.module == "operator" and alias.name == "attrgetter"
                    else _origin_value(_GETATTR_BUILTIN)
                    if node.module == "builtins" and alias.name == "getattr"
                    else _origin_value(f"<builtin-consumer:{alias.name}>")
                    if node.module == "builtins" and alias.name in _TRACKED_BUILTIN_CONSUMERS
                    else _origin_value("<external-consumer:collections.deque>")
                    if node.module == "collections" and alias.name == "deque"
                    else _AbstractValue(
                        classes=classes,
                        modules=(
                            frozenset({imported_module.name})
                            if imported_module is not None
                            else frozenset()
                        ),
                    )
                )

    def visit_Import(self, node: ast.Import) -> None:
        self._bind_import(node, runtime=True)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._bind_import_from(node, runtime=True)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._visit_store_target(item.optional_vars)
                self._bind_target_value(item.optional_vars, _UNKNOWN_VALUE)
                self._bind_function_target(item.optional_vars, frozenset())
                self._bind_annotation_target(item.optional_vars, _UNKNOWN_VALUE)
        self._apply_flow_result(self._visit_binding_branch(node.body, self._binding_snapshot()))

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)

    def _argument_value(self, argument: ast.arg) -> _AbstractValue:
        if _looks_like_config_name(argument.arg):
            return _origin_value(_CONFIG_ROOT)
        return self._annotation_value(argument.annotation)

    def _scoped_arguments(self, arguments: ast.arguments) -> dict[str, _AbstractValue]:
        scoped = {
            argument.arg: self._argument_value(argument)
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        }
        if arguments.vararg is not None:
            scoped[arguments.vararg.arg] = self._argument_value(arguments.vararg)
        if arguments.kwarg is not None:
            scoped[arguments.kwarg.arg] = self._argument_value(arguments.kwarg)
        return scoped

    def _default_argument_values(self, node: _FunctionNode) -> dict[str, _AbstractValue]:
        previous_cache = self._expression_cache
        self._expression_cache = {}
        try:
            positional = (*node.args.posonlyargs, *node.args.args)
            defaulted = positional[-len(node.args.defaults) :] if node.args.defaults else ()
            values = {
                argument.arg: self._expression_value(default)
                for argument, default in zip(
                    defaulted,
                    node.args.defaults,
                    strict=True,
                )
            }
            values.update(
                {
                    argument.arg: self._expression_value(default)
                    for argument, default in zip(
                        node.args.kwonlyargs, node.args.kw_defaults, strict=True
                    )
                    if default is not None
                }
            )
            return values
        finally:
            self._expression_cache = previous_cache

    def _scoped_function_arguments(self, node: _FunctionNode) -> dict[str, _AbstractValue]:
        scoped = self._scoped_arguments(node.args)
        for name, value in self._default_argument_values(node).items():
            scoped[name] = _join_values(scoped[name], value)
        return scoped

    @staticmethod
    def _declared_functions(
        statements: Iterable[ast.stmt],
    ) -> dict[str, _FunctionSet]:
        return {
            statement.name: frozenset({statement})
            for statement in statements
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def _bound_call_arguments(
        self,
        call: ast.Call,
        function: _FunctionNode,
        bound_receiver: _AbstractValue | None = None,
    ) -> dict[str, _AbstractValue]:
        arguments = function.args
        scoped = self._scoped_arguments(arguments)
        scoped.update(self._default_argument_values(function))
        positional_parameters = (*arguments.posonlyargs, *arguments.args)
        positional_states: list[
            tuple[int, dict[str, _AbstractValue], tuple[_AbstractValue, ...]]
        ] = [(0, {}, ())]

        def bind_positional(
            states: list[tuple[int, dict[str, _AbstractValue], tuple[_AbstractValue, ...]]],
            value: _AbstractValue,
        ) -> list[tuple[int, dict[str, _AbstractValue], tuple[_AbstractValue, ...]]]:
            bound: list[tuple[int, dict[str, _AbstractValue], tuple[_AbstractValue, ...]]] = []
            for index, bindings, extras in states:
                if index < len(positional_parameters):
                    updated = dict(bindings)
                    updated[positional_parameters[index].arg] = value
                    bound.append((index + 1, updated, extras))
                elif arguments.vararg is not None:
                    bound.append((index, bindings, (*extras, value)))
            return bound

        if bound_receiver is not None:
            positional_states = bind_positional(positional_states, bound_receiver)

        for argument in call.args:
            if isinstance(argument, ast.Starred):
                expanded = self._expression_value(argument.value)
                if expanded.items is not None:
                    for value in expanded.items:
                        positional_states = bind_positional(positional_states, value)
                else:
                    conservative = _conservative_value(expanded)
                    expanded_states: list[
                        tuple[
                            int,
                            dict[str, _AbstractValue],
                            tuple[_AbstractValue, ...],
                        ]
                    ] = []
                    for state in positional_states:
                        alternatives = [state]
                        current = [state]
                        while current and current[0][0] < len(positional_parameters):
                            current = bind_positional(current, conservative)
                            alternatives.extend(current)
                        if arguments.vararg is not None and current:
                            alternatives.extend(bind_positional(current, conservative))
                        expanded_states.extend(alternatives)
                    positional_states = expanded_states
            else:
                positional_states = bind_positional(
                    positional_states, self._expression_value(argument)
                )

        for parameter in positional_parameters:
            scoped[parameter.arg] = _join_values(
                *(
                    bindings.get(parameter.arg, scoped[parameter.arg])
                    for _, bindings, _ in positional_states
                )
            )
        if arguments.vararg is not None and positional_states:
            scoped[arguments.vararg.arg] = _AbstractValue(
                items=(
                    _join_values(*(item for _, _, extras in positional_states for item in extras)),
                )
            )

        named_parameters = {
            parameter.arg: parameter for parameter in (*arguments.args, *arguments.kwonlyargs)
        }
        extra_keywords: list[tuple[str, _AbstractValue]] = []
        for keyword in call.keywords:
            value = self._expression_value(keyword.value)
            if keyword.arg is None:
                if value.entries is not None:
                    entries = dict(value.entries)
                    wildcard = entries.get(_DYNAMIC_KEY)
                    for name in named_parameters:
                        selected = entries.get(_key_token(ast.Constant(name)))
                        if selected is not None:
                            scoped[name] = selected
                        elif wildcard is not None:
                            scoped[name] = _join_values(scoped[name], wildcard)
                    extra_keywords.extend(value.entries)
                elif arguments.kwarg is not None:
                    scoped[arguments.kwarg.arg] = _conservative_value(value)
                else:
                    conservative = _conservative_value(value)
                    for name in named_parameters:
                        scoped[name] = _join_values(scoped[name], conservative)
                continue
            if keyword.arg in named_parameters:
                scoped[keyword.arg] = value
            else:
                extra_keywords.append((_key_token(ast.Constant(keyword.arg)), value))
        if arguments.kwarg is not None and extra_keywords:
            scoped[arguments.kwarg.arg] = _AbstractValue(entries=tuple(extra_keywords))
        return scoped

    def _bound_direct_arguments(
        self,
        function: _FunctionNode,
        values: tuple[_AbstractValue, ...],
    ) -> dict[str, _AbstractValue]:
        """Bind already-evaluated positional values to a local callable."""

        arguments = function.args
        scoped = self._scoped_arguments(arguments)
        scoped.update(self._default_argument_values(function))
        positional = (*arguments.posonlyargs, *arguments.args)
        for parameter, value in zip(positional, values, strict=False):
            scoped[parameter.arg] = value
        if arguments.vararg is not None:
            scoped[arguments.vararg.arg] = _AbstractValue(items=values[len(positional) :])
        return scoped

    def _visit_function_body(
        self,
        node: _FunctionNode,
        scoped: dict[str, _AbstractValue],
        *,
        consume_generator: bool = False,
        generator_consumer_mode: str = "full",
        generator_identity: int | None = None,
    ) -> tuple[_AbstractValue, ...]:
        scoped = {**self._closure_bindings.get(id(node), {}), **scoped}
        self._states.append(scoped)
        self._annotations.append({})
        self._global_names.append(set())
        self._nonlocal_names.append(set())
        function_bindings: dict[str, _FunctionSet] = {
            argument.arg: frozenset()
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        if node.args.vararg is not None:
            function_bindings[node.args.vararg.arg] = frozenset()
        if node.args.kwarg is not None:
            function_bindings[node.args.kwarg.arg] = frozenset()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_bindings.update(self._declared_functions(node.body))
        self._functions.append(function_bindings)
        previous_cache = self._expression_cache
        previous_module = self._module
        self._module = self._source_index.owner(node) or self._module
        self._expression_cache = {}
        self._function_body_depth += 1
        if generator_identity is not None:
            self._generator_consumer_modes.append(generator_consumer_mode)
            self._generator_skip_yields.append(
                self._deferred_generator_positions.get(generator_identity, 0)
            )
            self._generator_advanced_yields.append(0)
        try:
            if isinstance(node, ast.Lambda):
                self.visit(node.body)
                return (self._expression_value(node.body),)
            if _is_generator_function(node) and not consume_generator:
                return ()
            result = self._visit_binding_branch(node.body, self._binding_snapshot())
            return tuple(
                path.return_value or _UNKNOWN_VALUE
                for path in result.abrupt
                if path.kind == "return"
            )
        finally:
            if generator_identity is not None:
                advanced = self._generator_advanced_yields.pop()
                self._deferred_generator_positions[generator_identity] = (
                    self._deferred_generator_positions.get(generator_identity, 0) + advanced
                )
                self._generator_skip_yields.pop()
                self._generator_consumer_modes.pop()
            self._function_body_depth -= 1
            self._expression_cache = previous_cache
            self._module = previous_module
            self._functions.pop()
            self._nonlocal_names.pop()
            self._global_names.pop()
            self._annotations.pop()
            self._states.pop()

    def _local_call_value(
        self,
        call: ast.Call,
        function: _FunctionNode,
        bound_receiver: _AbstractValue | None = None,
    ) -> _AbstractValue:
        scoped = self._bound_call_arguments(call, function, bound_receiver)
        if _is_generator_function(function):
            self._deferred_generators[id(call)] = _DeferredGenerator(function, scoped)
            return _AbstractValue(identity=frozenset({id(call)}))
        if isinstance(function, ast.AsyncFunctionDef):
            self._deferred_coroutines[id(call)] = _DeferredCoroutine(function, scoped)
            return _AbstractValue(identity=frozenset({id(call)}))
        function_id = id(function)
        if function_id in self._active_calls:
            return _UNKNOWN_VALUE
        self._active_calls.add(function_id)
        try:
            returned = self._visit_function_body(
                function,
                scoped,
            )
        finally:
            self._active_calls.remove(function_id)
        return _join_values(*returned)

    def _local_direct_call_value(
        self,
        function: _FunctionNode,
        values: tuple[_AbstractValue, ...],
    ) -> _AbstractValue:
        function_id = id(function)
        if function_id in self._active_calls:
            return _UNKNOWN_VALUE
        self._active_calls.add(function_id)
        try:
            returned = self._visit_function_body(
                function,
                self._bound_direct_arguments(function, values),
            )
        finally:
            self._active_calls.remove(function_id)
        return _join_values(*returned)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self.visit(node.value)
        value = self._expression_value(node.value) if node.value is not None else _UNKNOWN_VALUE
        self._append_abrupt(
            self._flow_abrupts[-1],
            _AbruptPath("return", self._binding_snapshot(), return_value=value),
        )
        self._path_reachable = False

    def visit_Yield(self, node: ast.Yield) -> None:
        if self._generator_skip_yields and self._generator_skip_yields[-1] > 0:
            self._generator_skip_yields[-1] -= 1
            return
        if node.value is not None:
            self.visit(node.value)
        if not self._generator_consumer_modes:
            return
        self._generator_advanced_yields[-1] += 1
        mode = self._generator_consumer_modes[-1]
        truth = self._static_truth(node.value) if node.value is not None else False
        if (
            mode == "one_turn"
            or mode == "any"
            and truth is True
            or mode == "all"
            and truth is False
        ):
            # One-turn APIs always suspend. ``any`` and ``all`` suspend once a
            # statically decisive yielded value short-circuits the consumer.
            self._path_reachable = False

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is not None:
            exception_name = self._exception_name(node.exc)
            leaf = exception_name.rsplit(".", 1)[-1] if exception_name else ""
            if isinstance(node.exc, ast.Call) and (
                leaf.endswith(("Error", "Exception"))
                or leaf
                in {
                    "BaseException",
                    "GeneratorExit",
                    "KeyboardInterrupt",
                    "StopAsyncIteration",
                    "StopIteration",
                    "SystemExit",
                }
            ):
                self.visit(node.exc.func)
                for argument in node.exc.args:
                    self.visit(argument)
                for keyword in node.exc.keywords:
                    self.visit(keyword.value)
            else:
                self.visit(node.exc)
        if node.cause is not None:
            self.visit(node.cause)
        bindings = self._binding_snapshot()
        if node.exc is None and self._caught_exception_stack:
            for caught in self._caught_exception_stack[-1]:
                self._append_abrupt(
                    self._flow_abrupts[-1],
                    _AbruptPath(
                        "raise",
                        bindings,
                        exception_type=caught.exception_type,
                        exception_exclusions=caught.exception_exclusions,
                        exception_upper_bound=caught.exception_upper_bound,
                    ),
                )
        else:
            exception_type = self._exception_name(node.exc)
            self._append_abrupt(
                self._flow_abrupts[-1],
                _AbruptPath(
                    "raise",
                    bindings,
                    exception_type=exception_type,
                    exception_upper_bound=("BaseException" if exception_type is None else None),
                ),
            )
        self._path_reachable = False

    def visit_Break(self, node: ast.Break) -> None:
        self._append_abrupt(
            self._flow_abrupts[-1],
            _AbruptPath("break", self._binding_snapshot()),
        )
        self._path_reachable = False

    def visit_Continue(self, node: ast.Continue) -> None:
        self._append_abrupt(
            self._flow_abrupts[-1],
            _AbruptPath("continue", self._binding_snapshot()),
        )
        self._path_reachable = False

    def _decorated_function_value(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> _AbstractValue:
        original = _AbstractValue(callables=(_CallableTarget(node),))
        value = original
        for decorator in reversed(node.decorator_list):
            targets = self._call_targets(decorator)
            if not targets:
                return original
            replacements = [
                self._local_direct_call_value(
                    function,
                    ((receiver,) if receiver is not None else ()) + (value,),
                )
                for function, receiver in targets
            ]
            if not replacements:
                return original
            replacement = _join_values(*replacements)
            if not replacement.callables and not replacement.instance_classes:
                if replacement != _UNKNOWN_VALUE:
                    return replacement
                return original
            value = replacement
        return value

    def _visit_function_definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self._capture_function_closure(node)
        value = self._decorated_function_value(node)
        functions = frozenset(target.function for target in value.callables)
        self._states[-1][node.name] = value
        self._functions[-1][node.name] = functions
        self._annotations[-1][node.name] = _UNKNOWN_VALUE

    def _visit_reachable_values(self, values: Iterable[_AbstractValue]) -> None:
        """Scan bodies that remain reachable from final scope bindings.

        Calls executed while walking a scope are already scanned at their call
        sites. Deferred scanning here covers exported definitions without
        letting a later overwrite or deletion keep an erased body alive.
        """
        visited: set[int] = set()
        pending = list(values)
        while pending:
            value = pending.pop()
            for target in value.callables:
                function_id = id(target.function)
                if function_id in visited:
                    continue
                visited.add(function_id)
                scoped = self._scoped_function_arguments(target.function)
                if target.receiver is not None:
                    scoped.update(self._bound_direct_arguments(target.function, (target.receiver,)))
                self._visit_function_body(target.function, scoped)
            for class_node in value.classes:
                class_id = id(class_node)
                if class_id in visited:
                    continue
                visited.add(class_id)
                pending.extend(self._class_member_values.get(class_id, ()))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_definition(node)

    def visit_Module(self, node: ast.Module) -> None:
        self._functions[-1].update(self._declared_functions(node.body))
        for statement in node.body:
            self.visit(statement)
        self._visit_reachable_values(self._states[-1].values())

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._states.append({})
        self._annotations.append({})
        self._functions.append(self._declared_functions(node.body))
        self._global_names.append(set())
        self._nonlocal_names.append(set())
        try:
            for statement in node.body:
                self.visit(statement)
            self._class_member_values[id(node)] = tuple(self._states[-1].values())
        finally:
            self._nonlocal_names.pop()
            self._global_names.pop()
            self._functions.pop()
            self._annotations.pop()
            self._states.pop()
        self._states[-1][node.name] = _AbstractValue(classes=frozenset({node}))
        self._annotations[-1][node.name] = _UNKNOWN_VALUE
        self._functions[-1][node.name] = frozenset()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)


def runtime_reads(source_root: Path, fields: frozenset[ConfigField]) -> frozenset[ConfigField]:
    """Return config fields loaded from their named sections in production Python."""

    reads: set[ConfigField] = set()
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(source_root.rglob("*.py"))
    }
    source_index = _SourceIndex(source_root, trees)
    for path, tree in trees.items():
        visitor = _RuntimeReadVisitor(fields, source_index, source_index.module_for_path(path))
        visitor.visit(tree)
        reads.update(visitor.reads)
    return frozenset(reads)


def _split_markdown_row(line: str) -> tuple[str, ...]:
    """Split a Markdown table row without treating escaped pipes as separators."""

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in line.strip().strip("|"):
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            current.append(character)
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    cells.append("".join(current).strip())
    return tuple(cells)


def parse_reference_rows(text: str) -> dict[ConfigField, ReferenceRow]:
    """Parse field rows from the two tracked config-reference sections."""

    section: str | None = None
    rows: dict[ConfigField, ReferenceRow] = {}
    for line in text.splitlines():
        heading = _SECTION_HEADING.match(line)
        if heading is not None:
            section = heading.group("section")
            continue
        if line.startswith("## "):
            section = None
            continue
        field_match = _FIELD_ROW.match(line)
        if section is None or field_match is None:
            continue
        cells = _split_markdown_row(line)
        if len(cells) != 4:
            raise ValueError(f"malformed config-reference row: {line}")
        field = ConfigField(section, field_match.group("field"))
        if field in rows:
            raise ValueError(f"duplicate config-reference row for {field.dotted}")
        rows[field] = ReferenceRow(default=cells[2], description=cells[3])
    return rows


def parse_inert_markers(text: str) -> dict[ConfigField, InertMarker]:
    """Parse JSON markers that make inert documentation machine-checkable."""

    markers: dict[ConfigField, InertMarker] = {}
    for match in _INERT_MARKER.finditer(text):
        payload = json.loads(match.group("payload"))
        if set(payload) != {"section", "field", "status", "effective_control"}:
            raise ValueError("config-field-contract marker has an invalid schema")
        if payload["status"] != "inert":
            raise ValueError("config-field-contract marker status must be inert")
        if payload["section"] not in TRACKED_SECTIONS:
            raise ValueError("config-field-contract marker has an unknown section")
        if not isinstance(payload["field"], str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", payload["field"]
        ):
            raise ValueError("config-field-contract marker has an invalid field name")
        field = ConfigField(payload["section"], payload["field"])
        effective_control = payload["effective_control"]
        if not isinstance(effective_control, str) or not effective_control.strip():
            raise ValueError(f"inert marker for {field.dotted} needs an effective control")
        if field in markers:
            raise ValueError(f"duplicate inert marker for {field.dotted}")
        markers[field] = InertMarker(field=field, effective_control=effective_control)
    return markers


def _literal_default(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("`") and normalized.endswith("`"):
        normalized = normalized[1:-1].strip()
    if normalized.startswith('"') and normalized.endswith('"'):
        normalized = normalized[1:-1]
    return normalized


def opus_default_violations(direct_model: str, consensus_model: str) -> tuple[str, ...]:
    """Validate the intentionally different direct and OpenRouter Opus ids."""

    direct_pattern = re.compile(r"claude-opus-(?P<major>[1-9][0-9]*)-(?P<minor>0|[1-9][0-9]*)")
    consensus_pattern = re.compile(
        r"openrouter/anthropic/claude-opus-"
        r"(?P<major>[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)"
    )
    direct_match = direct_pattern.fullmatch(direct_model)
    consensus_match = consensus_pattern.fullmatch(consensus_model)
    violations: list[str] = []

    if direct_model == consensus_model:
        violations.append("Opus defaults must use distinct direct and OpenRouter identifiers")
    if direct_match is None:
        violations.append(
            "evaluation.semantic_model: invalid direct Opus default "
            f"{direct_model!r}; expected 'claude-opus-<major>-<minor>'"
        )
    if consensus_match is None:
        violations.append(
            "consensus.advocate_model: invalid OpenRouter Opus default "
            f"{consensus_model!r}; expected "
            "'openrouter/anthropic/claude-opus-<major>.<minor>'"
        )
    if direct_match is not None and consensus_match is not None:
        direct_version = direct_match.group("major", "minor")
        consensus_version = consensus_match.group("major", "minor")
        if direct_version != consensus_version:
            expected_consensus = (
                f"openrouter/anthropic/claude-opus-{direct_version[0]}.{direct_version[1]}"
            )
            violations.append(
                "consensus.advocate_model: OpenRouter Opus default "
                f"{consensus_model!r} does not correspond to direct default {direct_model!r}; "
                f"expected {expected_consensus!r}"
            )
    return tuple(violations)


def audit_contract(
    *,
    fields: frozenset[ConfigField],
    reads: frozenset[ConfigField],
    rows: Mapping[ConfigField, ReferenceRow],
    markers: Mapping[ConfigField, InertMarker],
    allowlist: Mapping[ConfigField, str],
    documented_defaults: Mapping[ConfigField, str],
) -> ContractReport:
    """Classify every field and return precise bidirectional drift failures."""

    violations: list[str] = []
    marker_fields = frozenset(markers)
    allowlisted_fields = frozenset(allowlist)

    for stale in sorted((marker_fields | allowlisted_fields | frozenset(rows)) - fields):
        violations.append(f"{stale.dotted}: docs/allowlist names no schema field")

    for field, reason in sorted(allowlist.items()):
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            violations.append(f"{field.dotted}: schema-only allowlist needs a concrete rationale")
        if field in reads:
            violations.append(
                f"{field.dotted}: production-wired field remains schema-only allowlisted"
            )
        if field in marker_fields:
            violations.append(f"{field.dotted}: field is both documented inert and allowlisted")

    for field in sorted(fields):
        dispositions = (
            int(field in reads) + int(field in marker_fields) + int(field in allowlisted_fields)
        )
        if dispositions == 0:
            violations.append(
                f"{field.dotted}: no production read, inert documentation, or schema-only rationale"
            )
        elif dispositions > 1:
            violations.append(f"{field.dotted}: conflicting config-field dispositions")

        row = rows.get(field)
        if row is None:
            violations.append(f"{field.dotted}: missing docs/config-reference.md field row")
            continue
        visibly_inert = "currently inert" in row.description.lower()
        if field in reads and visibly_inert:
            violations.append(f"{field.dotted}: production-wired field is still documented inert")
        if field in marker_fields:
            marker = markers[field]
            if not visibly_inert:
                violations.append(
                    f"{field.dotted}: inert marker lacks visible 'Currently inert' text"
                )
            if "effective control:" not in row.description.lower():
                violations.append(f"{field.dotted}: inert docs do not label the effective control")
            if marker.effective_control not in row.description:
                violations.append(
                    f"{field.dotted}: visible docs omit marker effective control "
                    f"{marker.effective_control!r}"
                )
        elif visibly_inert:
            violations.append(f"{field.dotted}: visible inert text lacks a structured marker")

    for field, expected in sorted(documented_defaults.items()):
        if field not in fields:
            violations.append(f"{field.dotted}: default contract names no schema field")
            continue
        row = rows.get(field)
        if row is not None and _literal_default(row.default) != expected:
            violations.append(
                f"{field.dotted}: documented default {_literal_default(row.default)!r} "
                f"does not match {expected!r}"
            )

    return ContractReport(tuple(sorted(set(violations))), reads)


def audit_repository(repo_root: Path = REPO_ROOT) -> ContractReport:
    """Audit the checked-out repository contract."""

    fields = schema_fields()
    reference = (repo_root / REFERENCE_PATH).read_text(encoding="utf-8")
    report = audit_contract(
        fields=fields,
        reads=runtime_reads(repo_root / "src" / "ouroboros", fields),
        rows=parse_reference_rows(reference),
        markers=parse_inert_markers(reference),
        allowlist=SCHEMA_ONLY_ALLOWLIST,
        documented_defaults=DOCUMENTED_DEFAULTS,
    )
    return ContractReport(
        tuple(
            sorted(
                set(report.violations)
                | set(opus_default_violations(DEFAULT_OPUS_MODEL, DEFAULT_CONSENSUS_OPUS_MODEL))
            )
        ),
        report.runtime_reads,
    )


def main() -> int:
    try:
        report = audit_repository()
    except (OSError, SyntaxError, ValueError) as error:
        print(f"Config reference contract could not run: {error}", file=sys.stderr)
        return 2
    if report.violations:
        print("Config reference contract violations:")
        for violation in report.violations:
            print(f"- {violation}")
        return 1
    reads = ", ".join(field.dotted for field in sorted(report.runtime_reads))
    print(f"Config reference contract OK ({len(report.runtime_reads)} production reads: {reads})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
