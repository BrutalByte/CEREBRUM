"""
UEToolkitClient â€” HTTP wrapper for the ue-llm-toolkit plugin (Phase 94).

Provides a sync, gracefully-degrading interface to the ue-llm-toolkit HTTP
server (default localhost:3000).  All methods return False/None on failure so
callers never need try/except â€” the toolkit is treated as optional.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Type

logger = logging.getLogger("cerebrum.ue_toolkit")

_TIMEOUT = 10  # seconds per HTTP call


class UEToolkitClient:
    """Sync HTTP client for the ue-llm-toolkit plugin running in the UE5 editor."""

    def __init__(self, base_url: str = "http://localhost:3000") -> None:
        self.base_url = base_url.rstrip("/")
        self._available: Optional[bool] = None  # None = not yet probed

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/mcp/status")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
        return bool(self._available)

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def call(self, tool: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/mcp/tool/{tool}"
        body = json.dumps(params).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                result = json.loads(resp.read().decode())
                if not result.get("success", True):
                    logger.warning("UEToolkit %s failed: %s", tool, result.get("error"))
                    return None
                return result.get("data", result)
        except urllib.error.HTTPError as e:
            logger.warning("UEToolkit %s HTTP %d: %s", tool, e.code, e.reason)
        except urllib.error.URLError as e:
            logger.warning("UEToolkit unreachable (%s) â€” %s", tool, e.reason)
        except Exception as e:
            logger.warning("UEToolkit %s error: %s", tool, e)
        return None

    # ------------------------------------------------------------------
    # Widget operations
    # ------------------------------------------------------------------

    def create_widget(self, content_path: str, parent_class: str = "UserWidget") -> bool:
        result = self.call("blueprint_modify", {
            "action": "create",
            "blueprint_path": content_path,
            "parent_class": parent_class,
        })
        return result is not None

    def add_widget_element(
        self,
        widget_path: str,
        element_type: str,
        element_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        result = self.call("widget_editor", {
            "action": "add_element",
            "widget_path": widget_path,
            "element_type": element_type,
            "element_name": element_name,
            "properties": properties or {},
        })
        return result is not None

    def set_widget_property(
        self,
        widget_path: str,
        element_name: str,
        property_name: str,
        value: Any,
    ) -> bool:
        result = self.call("widget_editor", {
            "action": "set_property",
            "widget_path": widget_path,
            "element_name": element_name,
            "property_name": property_name,
            "value": value,
        })
        return result is not None

    # ------------------------------------------------------------------
    # Blueprint operations
    # ------------------------------------------------------------------

    def compile_blueprint(self, blueprint_path: str) -> bool:
        result = self.call("blueprint_modify", {
            "action": "compile_blueprint",
            "blueprint_path": blueprint_path,
        })
        return result is not None

    def save_all(self) -> bool:
        result = self.call("asset_save_all", {})
        return result is not None

    # ------------------------------------------------------------------
    # Script execution
    # ------------------------------------------------------------------

    def run_python(self, code: str) -> Optional[str]:
        result = self.call("execute_script", {"code": code, "language": "python"})
        if result is None:
            return None
        return result.get("output", "")

    # ------------------------------------------------------------------
    # Actor / level operations
    # ------------------------------------------------------------------

    def spawn_actor(
        self,
        class_path: str,
        location: Optional[list] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        result = self.call("spawn_actor", {
            "class_path": class_path,
            "location": location or [0.0, 0.0, 0.0],
            "properties": properties or {},
        })
        if result is None:
            return None
        return result.get("actor_path")

    def set_actor_property(self, actor_path: str, prop: str, value: Any) -> bool:
        result = self.call("set_property", {
            "object_path": actor_path,
            "property_name": prop,
            "property_value": value,
        })
        return result is not None
