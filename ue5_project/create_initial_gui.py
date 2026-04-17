"""
create_initial_gui.py — Phase 94: Build WBP_CerebrumHUD via ue-llm-toolkit.

Run this script ONCE with both the CEREBRUM REST server and the UE5 editor
open (plugin installed, HTTP server on localhost:3000).

    python ue5_project/create_initial_gui.py [--toolkit-url http://localhost:3000]

The script creates /Game/UI/WBP_CerebrumHUD with five panels, compiles, and
saves it.  Subsequent runs are idempotent — the widget_editor tool skips
elements that already exist.
"""
import argparse
import sys
import os

# Allow importing api/ from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.ue_toolkit_client import UEToolkitClient

WIDGET_PATH = "/Game/UI/WBP_CerebrumHUD"


def build_hud(client: UEToolkitClient) -> None:
    print(f"[1/8] Creating widget blueprint at {WIDGET_PATH}...")
    client.create_widget(WIDGET_PATH, parent_class="UserWidget")

    # ── MetabolicPanel (top-left) ──────────────────────────────────────────
    print("[2/8] Adding MetabolicPanel...")
    client.add_widget_element(WIDGET_PATH, "CanvasPanel", "MetabolicPanel", {
        "anchor_min": [0.0, 0.0], "anchor_max": [0.22, 0.38],
        "offset": [8.0, 8.0, -8.0, -8.0],
    })
    for i, (scalar, color) in enumerate([
        ("Reinforcement", {"r": 0.2, "g": 0.9, "b": 0.2, "a": 1.0}),
        ("Arousal",       {"r": 1.0, "g": 0.6, "b": 0.0, "a": 1.0}),
        ("Novelty",       {"r": 0.3, "g": 0.7, "b": 1.0, "a": 1.0}),
        ("Cohesion",      {"r": 0.8, "g": 0.3, "b": 0.9, "a": 1.0}),
        ("Persistence",   {"r": 0.9, "g": 0.9, "b": 0.2, "a": 1.0}),
    ]):
        top = 28 + i * 30
        client.add_widget_element(WIDGET_PATH, "TextBlock", f"Lbl_{scalar}", {
            "parent": "MetabolicPanel",
            "text": scalar, "font_size": 11,
            "position": [4.0, float(top)], "size": [80.0, 18.0],
        })
        client.add_widget_element(WIDGET_PATH, "ProgressBar", f"Bar_{scalar}", {
            "parent": "MetabolicPanel",
            "fill_color": color,
            "position": [86.0, float(top + 2)], "size": [120.0, 14.0],
            "percent": 0.5,
        })

    # ── QueryConsole (bottom-left) ─────────────────────────────────────────
    print("[3/8] Adding QueryConsole...")
    client.add_widget_element(WIDGET_PATH, "CanvasPanel", "QueryConsole", {
        "anchor_min": [0.0, 0.65], "anchor_max": [0.38, 1.0],
        "offset": [8.0, 8.0, -8.0, -8.0],
    })
    client.add_widget_element(WIDGET_PATH, "EditableTextBox", "QueryInput", {
        "parent": "QueryConsole",
        "hint_text": "Query the knowledge graph...",
        "position": [4.0, 4.0], "size": [260.0, 28.0],
    })
    client.add_widget_element(WIDGET_PATH, "Button", "QueryBtn", {
        "parent": "QueryConsole",
        "position": [268.0, 4.0], "size": [60.0, 28.0],
    })
    client.add_widget_element(WIDGET_PATH, "TextBlock", "QueryBtnLabel", {
        "parent": "QueryBtn", "text": "Query", "font_size": 12,
    })
    client.add_widget_element(WIDGET_PATH, "ScrollBox", "ResultsScroll", {
        "parent": "QueryConsole",
        "position": [4.0, 36.0], "size": [324.0, 150.0],
    })

    # ── LoopStatusPanel (top-right) ────────────────────────────────────────
    print("[4/8] Adding LoopStatusPanel...")
    client.add_widget_element(WIDGET_PATH, "CanvasPanel", "LoopStatusPanel", {
        "anchor_min": [0.78, 0.0], "anchor_max": [1.0, 0.32],
        "offset": [8.0, 8.0, -8.0, -8.0],
    })
    for row, (name, default) in enumerate([
        ("Cycles",        "0"),
        ("Approval Rate", "—"),
        ("Edges Added",   "0"),
        ("Loop Status",   "Idle"),
    ]):
        client.add_widget_element(WIDGET_PATH, "TextBlock", f"Loop_{name.replace(' ', '_')}_Lbl", {
            "parent": "LoopStatusPanel",
            "text": f"{name}:", "font_size": 11,
            "position": [4.0, float(8 + row * 24)], "size": [100.0, 18.0],
        })
        client.add_widget_element(WIDGET_PATH, "TextBlock", f"Loop_{name.replace(' ', '_')}_Val", {
            "parent": "LoopStatusPanel",
            "text": default, "font_size": 11,
            "position": [106.0, float(8 + row * 24)], "size": [80.0, 18.0],
        })
    # Circuit breaker warning (hidden by default)
    client.add_widget_element(WIDGET_PATH, "Border", "CircuitWarning", {
        "parent": "LoopStatusPanel",
        "visibility": "Hidden",
        "background_color": {"r": 0.8, "g": 0.0, "b": 0.0, "a": 0.85},
        "position": [4.0, 108.0], "size": [184.0, 28.0],
    })
    client.add_widget_element(WIDGET_PATH, "TextBlock", "CircuitWarningTxt", {
        "parent": "CircuitWarning",
        "text": "⚠ Circuit Breaker Tripped",
        "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
        "font_size": 12,
    })

    # ── ActiveInferencePanel (bottom-right) ────────────────────────────────
    print("[5/8] Adding ActiveInferencePanel...")
    client.add_widget_element(WIDGET_PATH, "CanvasPanel", "ActiveInferencePanel", {
        "anchor_min": [0.78, 0.65], "anchor_max": [1.0, 1.0],
        "offset": [8.0, 8.0, -8.0, -8.0],
    })
    for row, (name, default) in enumerate([
        ("Soliton Index", "—"),
        ("Last Seeds",    "—"),
        ("Pulse Count",   "0"),
        ("Reason",        "—"),
    ]):
        client.add_widget_element(WIDGET_PATH, "TextBlock", f"AI_{name.replace(' ', '_')}_Lbl", {
            "parent": "ActiveInferencePanel",
            "text": f"{name}:", "font_size": 11,
            "position": [4.0, float(8 + row * 24)], "size": [90.0, 18.0],
        })
        client.add_widget_element(WIDGET_PATH, "TextBlock", f"AI_{name.replace(' ', '_')}_Val", {
            "parent": "ActiveInferencePanel",
            "text": default, "font_size": 11,
            "position": [96.0, float(8 + row * 24)], "size": [110.0, 18.0],
        })

    # ── GraphStatsPanel (top-center) ───────────────────────────────────────
    print("[6/8] Adding GraphStatsPanel...")
    client.add_widget_element(WIDGET_PATH, "HorizontalBox", "GraphStatsPanel", {
        "anchor_min": [0.3, 0.0], "anchor_max": [0.7, 0.06],
        "offset": [0.0, 4.0, 0.0, -4.0],
    })
    for stat in ["Nodes", "Edges", "Communities"]:
        client.add_widget_element(WIDGET_PATH, "TextBlock", f"Stat_{stat}", {
            "parent": "GraphStatsPanel",
            "text": f"{stat}: —", "font_size": 13,
            "padding": [16.0, 0.0, 16.0, 0.0],
        })

    # ── Compile + save ─────────────────────────────────────────────────────
    print("[7/8] Compiling blueprint...")
    ok = client.compile_blueprint(WIDGET_PATH)
    if not ok:
        print("  WARNING: compile reported failure — check UE5 output log.")

    print("[8/8] Saving all assets...")
    client.save_all()

    print("\nDone. WBP_CerebrumHUD created at", WIDGET_PATH)
    print("Open the UE5 editor → Content Browser → UI → WBP_CerebrumHUD to verify.")
    print("Place a BP_CerebrumBrain actor in your level and set HUDWidgetClass = WBP_CerebrumHUD.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CEREBRUM HUD widget via ue-llm-toolkit")
    parser.add_argument("--toolkit-url", default="http://localhost:3000",
                        help="URL of the ue-llm-toolkit HTTP server")
    args = parser.parse_args()

    client = UEToolkitClient(args.toolkit_url)
    print(f"Checking ue-llm-toolkit at {args.toolkit_url}...")
    if not client.is_available():
        print("ERROR: ue-llm-toolkit is not reachable.")
        print("  1. Make sure UE5 editor is open with the UELLMToolkit plugin enabled.")
        print("  2. Verify: GET http://localhost:3000/mcp/status returns 200.")
        sys.exit(1)

    print("Toolkit available. Building initial GUI...\n")
    build_hud(client)


if __name__ == "__main__":
    main()
