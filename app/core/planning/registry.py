"""Activity Registry for loading and managing YAML activity definitions.

This module provides:
- ActivityRegistry: Loads and caches activity definitions from YAML files
- Activity type detection from keywords
- Strategy template loading

The registry loads activity definitions at startup and provides
fast lookups for plan creation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


class ActivityRegistry:
    """Registry for activity and strategy definitions.

    Loads YAML definitions from the activities/ directory and provides
    lookups by name, category, and detection keywords.

    Usage:
        registry = ActivityRegistry(Path("activities"))
        swap_def = registry.get_activity("swap")
        dca_def = registry.get_strategy("dca")

    Attributes:
        _activities: Dict of activity name -> definition
        _strategies: Dict of strategy name -> definition
        _keyword_index: Dict of keyword -> activity name
    """

    def __init__(self, activities_dir: Optional[Path] = None):
        """Initialize the registry.

        Args:
            activities_dir: Path to activities directory.
                           Defaults to sherpa/activities/
        """
        if activities_dir is None:
            # Default to sherpa/activities/ relative to this file
            activities_dir = Path(__file__).parent.parent.parent.parent / "activities"

        self._activities_dir = activities_dir
        self._activities: Dict[str, Dict[str, Any]] = {}
        self._strategies: Dict[str, Dict[str, Any]] = {}
        self._keyword_index: Dict[str, str] = {}

        self._load_definitions()

    def _load_definitions(self) -> None:
        """Load all activity and strategy definitions from YAML files."""
        if not self._activities_dir.exists():
            logger.warning(f"Activities directory not found: {self._activities_dir}")
            return

        # Load base activities
        for yaml_file in self._activities_dir.glob("*.yaml"):
            if yaml_file.name.startswith("_"):
                continue  # Skip private files

            try:
                with open(yaml_file, "r") as f:
                    definition = yaml.safe_load(f)

                if definition and "name" in definition:
                    name = definition["name"]
                    self._activities[name] = definition
                    self._index_keywords(name, definition)
                    logger.debug(f"Loaded activity: {name}")

            except Exception as e:
                logger.error(f"Failed to load activity {yaml_file}: {e}")

        # Load strategies
        strategies_dir = self._activities_dir / "strategies"
        if strategies_dir.exists():
            for yaml_file in strategies_dir.glob("*.yaml"):
                if yaml_file.name.startswith("_"):
                    continue

                try:
                    with open(yaml_file, "r") as f:
                        definition = yaml.safe_load(f)

                    if definition and "name" in definition:
                        name = definition["name"]

                        # Resolve extends (inherit from base activity)
                        if "extends" in definition:
                            base_name = definition["extends"]
                            if base_name in self._activities:
                                definition = self._merge_definitions(
                                    self._activities[base_name], definition
                                )

                        self._strategies[name] = definition
                        self._index_keywords(name, definition)
                        logger.debug(f"Loaded strategy: {name}")

                except Exception as e:
                    logger.error(f"Failed to load strategy {yaml_file}: {e}")

        logger.info(
            f"Loaded {len(self._activities)} activities, "
            f"{len(self._strategies)} strategies"
        )

    def _index_keywords(self, name: str, definition: Dict[str, Any]) -> None:
        """Index detection keywords for an activity/strategy."""
        keywords = definition.get("detection_keywords", [])
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Don't overwrite existing keywords (first definition wins)
            if keyword_lower not in self._keyword_index:
                self._keyword_index[keyword_lower] = name

    def _merge_definitions(
        self, base: Dict[str, Any], child: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge child definition with base, with child taking precedence."""
        result = base.copy()

        for key, value in child.items():
            if key == "guardrails" and "inherit" in value and value.get("inherit"):
                # Merge guardrails with overrides
                base_guardrails = result.get("guardrails", {}).copy()
                overrides = value.get("overrides", {})
                base_guardrails.update(overrides)
                result["guardrails"] = base_guardrails
            elif key == "extends":
                continue  # Don't include extends in merged definition
            elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # Deep merge dicts
                result[key] = {**result[key], **value}
            else:
                result[key] = value

        return result

    def get_activity(self, name: str) -> Optional[Dict[str, Any]]:
        """Get an activity definition by name.

        Args:
            name: Activity name (e.g., 'swap', 'bridge')

        Returns:
            Activity definition dict, or None if not found
        """
        return self._activities.get(name.lower())

    def get_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a strategy definition by name.

        Args:
            name: Strategy name (e.g., 'dca', 'momentum')

        Returns:
            Strategy definition dict, or None if not found
        """
        return self._strategies.get(name.lower())

    def detect_activity(self, text: str) -> Optional[str]:
        """Detect activity type from text using keywords.

        Args:
            text: User input text

        Returns:
            Activity name if detected, None otherwise
        """
        text_lower = text.lower()

        # Check for keyword matches
        for keyword, activity_name in self._keyword_index.items():
            if keyword in text_lower:
                return activity_name

        return None

    def list_activities(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered activities."""
        return self._activities.copy()

    def list_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered strategies."""
        return self._strategies.copy()

    def get_guardrails(
        self, activity_name: str, strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get guardrails for an activity, optionally overridden by strategy.

        Args:
            activity_name: Base activity name
            strategy_name: Optional strategy name for overrides

        Returns:
            Merged guardrails dict
        """
        guardrails = {}

        # Get base activity guardrails
        activity = self.get_activity(activity_name)
        if activity:
            guardrails = activity.get("guardrails", {}).copy()

        # Apply strategy overrides
        if strategy_name:
            strategy = self.get_strategy(strategy_name)
            if strategy:
                strategy_guardrails = strategy.get("guardrails", {})
                if strategy_guardrails.get("inherit"):
                    overrides = strategy_guardrails.get("overrides", {})
                    guardrails.update(overrides)
                else:
                    guardrails = strategy_guardrails

        return guardrails

    def get_input_schema(self, name: str) -> Dict[str, Any]:
        """Get the input schema for an activity or strategy.

        Args:
            name: Activity or strategy name

        Returns:
            Input schema dict
        """
        definition = self.get_strategy(name) or self.get_activity(name)
        if not definition:
            return {}

        inputs = definition.get("inputs", {})

        # For strategies, also include strategy_config.parameters
        if "strategy_config" in definition:
            params = definition["strategy_config"].get("parameters", {})
            inputs = {**inputs, **params}

        return inputs

    def reload(self) -> None:
        """Reload all definitions from disk."""
        self._activities.clear()
        self._strategies.clear()
        self._keyword_index.clear()
        self._load_definitions()


# Singleton instance
_registry: Optional[ActivityRegistry] = None


def get_activity_registry() -> ActivityRegistry:
    """Get the singleton ActivityRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ActivityRegistry()
    return _registry
