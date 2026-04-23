"""Context growth chart view — shows what fills the context at each iteration."""

from __future__ import annotations

import json
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Collapsible, Static
from textual_plotext import PlotextPlot

from ctxrot import pricing
from ctxrot.tokenizer import count_tokens
from ctxrot.tui.tree_view import RlmTreeView

# Distinct colors (RGB tuples)
COLOR_INPUT = (70, 130, 180)  # steel blue
COLOR_REASONING = (255, 165, 0)  # orange
COLOR_TOOL = (50, 205, 50)  # lime green
COLOR_OUTPUT = (220, 60, 60)  # red

# Rich markup colors (hex equivalents of the plotext RGB tuples above)
RICH_INPUT = "#4682B4"  # steel blue
RICH_REASONING = "#FFA500"  # orange
RICH_TOOL = "#32CD32"  # lime green
RICH_OUTPUT = "#DC3C3C"  # red
RICH_FREE = "dim"

# Context grid layout
GRID_COLS = 20
GRID_TOTAL_CELLS = 100  # 5 rows × 20 cols
CELL_USED = "\u26c1"  # ⛁
CELL_FREE = "\u26f6"  # ⛶


ROLE_COLORS = {
    "system": "dim",
    "user": "cyan",
    "assistant": "green",
    "tool": "yellow",
}


class IterationContent(Container):
    """Composite widget for a single iteration's content with collapsible messages."""

    DEFAULT_CSS = """
    IterationContent {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }
    IterationContent .iter-header {
        text-style: bold;
        height: auto;
        color: white;
        margin-bottom: 1;
    }
    IterationContent Collapsible {
        height: auto;
        padding: 0 0 0 1;
        margin-bottom: 0;
    }
    IterationContent .msg-content {
        height: auto;
        padding: 0 0 0 2;
    }
    """

    def __init__(
        self,
        row: dict,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        markup: bool = True,
    ) -> None:
        super().__init__(
            name=name, id=id, classes=classes, disabled=disabled, markup=markup
        )
        self._row = row

    def compose(self) -> ComposeResult:
        seq = self._row["seq"]
        prompt_tokens = int(self._row["prompt_tokens"] or 0)
        compl_tokens = int(self._row["completion_tokens"] or 0)

        yield Static(
            f"{'─' * 60}\n"
            f"  Iteration #{seq}  ({prompt_tokens:,} prompt tokens,"
            f" {compl_tokens:,} completion tokens)\n"
            f"{'─' * 60}",
            classes="iter-header",
        )

        # Render messages as collapsibles
        messages_json = self._row.get("messages_json")
        if messages_json:
            try:
                messages = json.loads(messages_json)
                for msg in messages:
                    role = msg.get("role", "?")
                    content = self._extract_content(msg)
                    token_count = count_tokens(content)
                    preview = content[:80].replace("\n", " ").strip()
                    color = ROLE_COLORS.get(role, "white")
                    title = (
                        f"[{color}][{role}][/{color}]"
                        f" ({token_count:,} tokens) {preview}..."
                    )
                    yield Collapsible(
                        Static(content, classes="msg-content"),
                        title=title,
                        collapsed=True,
                    )
            except (json.JSONDecodeError, TypeError):
                raw = messages_json
                yield Collapsible(
                    Static(raw, classes="msg-content"),
                    title=f"[raw] ({count_tokens(raw):,} tokens)",
                    collapsed=True,
                )

        # Render completion as collapsible
        completion = self._row.get("completion")
        if completion:
            preview = completion[:80].replace("\n", " ").strip()
            color = ROLE_COLORS.get("assistant", "green")
            title = (
                f"[{color}]Completion[/{color}]"
                f" ({count_tokens(completion):,} tokens) {preview}..."
            )
            yield Collapsible(
                Static(completion, classes="msg-content"),
                title=title,
                collapsed=True,
            )

    @staticmethod
    def _extract_content(msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", str(part)))
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        return str(content)


class GrowthView(Container):
    """Stacked bar chart + stats table showing context composition per iteration."""

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        markup: bool = True,
    ) -> None:
        super().__init__(
            name=name, id=id, classes=classes, disabled=disabled, markup=markup
        )
        self._last_growth_count: int = -1
        self._last_session_id: str | None = None

    DEFAULT_CSS = """
    GrowthView {
        layout: vertical;
        height: 1fr;
    }
    GrowthView PlotextPlot {
        height: 1fr;
        min-height: 14;
    }
    GrowthView #stats-scroll {
        display: none;
        height: 1fr;
    }
    GrowthView #content-scroll {
        display: none;
        height: 1fr;
    }
    GrowthView #rlm-tree {
        display: none;
        height: 1fr;
        min-height: 14;
    }
    GrowthView .stats-panel {
        height: auto;
        padding: 0 1;
    }
    GrowthView .content-panel {
        height: auto;
        padding: 0 1;
    }
    GrowthView #context-bar {
        height: 1;
        padding: 0 1;
        text-style: bold;
    }
    """

    def compose(self) -> ComposeResult:
        yield PlotextPlot(id="growth-chart")
        yield RlmTreeView(id="rlm-tree")
        with VerticalScroll(id="stats-scroll"):
            yield Static(
                "Waiting for data...",
                id="stats-panel",
                classes="stats-panel",
            )
        with VerticalScroll(id="content-scroll"):
            yield Static(
                "No content data. Use store_content=True in CtxRotCallback.",
                id="content-placeholder",
                classes="content-panel",
            )
        yield Static("", id="context-bar")

    def _show_panel(self, panel: str) -> None:
        """Show one of: chart, tree, stats, content. Hide the others."""
        chart = self.query_one("#growth-chart")
        tree = self.query_one("#rlm-tree")
        stats = self.query_one("#stats-scroll")
        content = self.query_one("#content-scroll")
        chart.display = panel == "chart"
        tree.display = panel == "tree"
        stats.display = panel == "stats"
        content.display = panel == "content"
        self._last_growth_count = -1

    def _visible_panel(self) -> str:
        """Return which panel is currently visible."""
        if self.query_one("#stats-scroll").display:
            return "stats"
        if self.query_one("#content-scroll").display:
            return "content"
        if self.query_one("#rlm-tree").display:
            return "tree"
        return "chart"

    def _default_panel(self) -> str:
        """Return the default panel based on session mode."""
        # If tree is the natural default (RLM session), return to it
        # This is detected by whether the tree has data loaded
        tree = self.query_one("#rlm-tree", RlmTreeView)
        if tree._last_count > 0:
            return "tree"
        return "chart"

    def action_toggle_stats(self) -> None:
        """Toggle between chart/tree view and scrollable stats view."""
        if self._visible_panel() == "stats":
            self._show_panel(self._default_panel())
        else:
            self._show_panel("stats")

    def action_toggle_content(self) -> None:
        """Toggle between current view and scrollable content view."""
        if self._visible_panel() == "content":
            self._show_panel(self._default_panel())
        else:
            self._show_panel("content")

    def update_from_data(self, data: dict) -> None:
        session_id = data["session_id"]
        session_mode = data.get("session_mode")
        visible = self._visible_panel()

        # Reset caches when session changes
        if session_id != self._last_session_id:
            self._last_session_id = session_id
            self._last_growth_count = -1
            # Clear old per-iteration content widgets
            scroll = self.query_one("#content-scroll", VerticalScroll)
            scroll.remove_children()

        # For RLM sessions, default to tree instead of chart
        # For non-RLM sessions, ensure we're not stuck on the tree panel
        if session_mode == "rlm" and visible == "chart":
            self._show_panel("tree")
            visible = "tree"
        elif session_mode != "rlm" and visible == "tree":
            self._show_panel("chart")
            visible = "chart"

        growth = data["growth"]
        summary = data["summary"]
        model = data["model"]

        # Update the always-visible context bar (with rot indicator if available)
        rot = data.get("rot")
        self._update_context_bar(summary, model, rot)

        # Only update the currently visible panel, and only if data changed
        growth_count = len(growth)

        if visible == "tree":
            tree_data = data.get("tree_data", [])
            if tree_data:
                tools_by_iter = data.get("tools_by_iter", {})
                tree_widget = self.query_one("#rlm-tree", RlmTreeView)
                tree_widget.update_tree(tree_data, tools_by_iter, session_id)
        elif visible == "chart":
            if growth_count != self._last_growth_count:
                self._last_growth_count = growth_count
                tools_by_iter = data["tools_by_iter"]
                layers = _compute_layers(growth, tools_by_iter)
                self._update_chart(growth, layers, tools_by_iter, model)
        elif visible == "stats":
            if growth_count != self._last_growth_count:
                self._last_growth_count = growth_count
                tools_by_iter = data["tools_by_iter"]
                tool_impact = data["tool_impact"]
                layers = _compute_layers(growth, tools_by_iter)
                self._update_stats(
                    growth,
                    layers,
                    tools_by_iter,
                    tool_impact,
                    summary,
                    session_id,
                    model,
                    rot,
                )
        elif visible == "content":
            self._update_content_from_data(data.get("content_rows", []))

    def _update_chart(
        self,
        growth: list[dict],
        layers: list[dict],
        tools_by_iter: dict[int, list[dict]],
        model: str,
    ) -> None:
        chart = self.query_one("#growth-chart", PlotextPlot)
        plt = chart.plt
        plt.clear_figure()
        plt.theme("dark")

        if not growth:
            plt.title("No data yet")
            chart.refresh()
            return

        seqs = [g["seq"] for g in growth]

        layer_input = [entry["input"] for entry in layers]
        layer_reasoning = [entry["reasoning"] for entry in layers]
        layer_tools = [entry["tools"] for entry in layers]
        layer_output = [entry["output"] for entry in layers]

        stacked_kwargs: dict[str, Any] = {
            "labels": ["input", "reasoning", "tools", "output"],
            "color": [COLOR_INPUT, COLOR_REASONING, COLOR_TOOL, COLOR_OUTPUT],
        }
        plt.stacked_bar(
            seqs,
            [layer_input, layer_reasoning, layer_tools, layer_output],
            **stacked_kwargs,
        )

        # Annotate tool calls on each iteration
        for i, g in enumerate(growth):
            seq = g["seq"]
            tools_here = tools_by_iter.get(seq, [])
            if tools_here:
                names = ", ".join(t["tool_name"] for t in tools_here)
                bar_top = (
                    layer_input[i]
                    + layer_reasoning[i]
                    + layer_tools[i]
                    + layer_output[i]
                )
                plt.text(names, seq, bar_top, alignment="center", color="green+")

        # Context window limit
        ctx_window = pricing.get_context_window(model)
        max_total = max(
            entry["input"] + entry["reasoning"] + entry["tools"] + entry["output"]
            for entry in layers
        )
        if max_total > ctx_window * 0.5:
            plt.hline(ctx_window, color="red+")

        plt.title(f"Context Composition — {model}")
        plt.xlabel("Iteration")
        plt.ylabel("Tokens")

        chart.refresh()

    def _update_stats(
        self,
        growth: list[dict],
        layers: list[dict],
        tools_by_iter: dict[int, list[dict]],
        tool_impact: list[dict],
        summary: dict,
        session_id: str,
        model: str,
        rot: dict | None = None,
    ) -> None:
        widget = self.query_one("#stats-panel", Static)
        if not growth:
            widget.update("Waiting for data...")
            return

        lines: list[str] = []

        # Context usage grid at the top
        grid_text = self._render_grid(layers, model)
        if grid_text:
            lines.append(grid_text)
            lines.append("")

        # Timeline view — shows how context is built iteration by iteration
        last_idx = len(growth) - 1
        for i, g in enumerate(growth):
            seq = g["seq"]
            prompt = g["prompt_tokens"]
            compl = g["completion_tokens"]

            # Build the prompt composition string
            if i == 0:
                composition = f"[{prompt:,} input]"
            else:
                prev = growth[i - 1]
                prev_prompt = prev["prompt_tokens"]
                prev_compl = prev["completion_tokens"]
                prev_tools = tools_by_iter.get(prev["seq"], [])

                prompt_delta = prompt - prev_prompt

                if prev_tools:
                    tool_d = max(0, prompt_delta - prev_compl)
                    reasoning_d = prompt_delta - tool_d
                else:
                    reasoning_d = prompt_delta
                    tool_d = 0

                if prompt < prev_prompt:
                    # DSPy compressed context
                    composition = f"[compressed → {prompt:,}]"
                elif tool_d > 0:
                    composition = (
                        f"[{prev_prompt:,} + {reasoning_d:,} reason"
                        f" + {tool_d:,} tool = {prompt:,}]"
                    )
                else:
                    composition = (
                        f"[{prev_prompt:,} + {reasoning_d:,} reason" f" = {prompt:,}]"
                    )

            suffix = " (final answer)" if i == last_idx else ""
            lines.append(f"  #{seq}  {composition} → {compl:,} out{suffix}")

            # Show tool calls that happened after this iteration
            tools_here = tools_by_iter.get(seq, [])
            if tools_here:
                for t in tools_here:
                    # Compute this tool's context cost from next iteration's delta
                    if i < last_idx:
                        next_g = growth[i + 1]
                        delta = next_g["prompt_tokens"] - prompt
                        total_tool_d = max(0, delta - compl)
                        per_tool = total_tool_d // max(1, len(tools_here))
                        lines.append(f"        ↳ {t['tool_name']}  +{per_tool:,} tok")
                    else:
                        lines.append(f"        ↳ {t['tool_name']}")

        # Tool summary by name — show raw output vs actual context cost
        if tool_impact:
            # Compute actual context cost per tool from prompt deltas
            ctx_cost_per_tool = _tool_context_cost(growth, tools_by_iter)

            lines.append("")
            lines.append("  Tools:")
            for t in tool_impact:
                name = t["tool_name"]
                raw = t["avg_tokens"]
                ctx = ctx_cost_per_tool.get(name)
                if ctx is not None:
                    lines.append(
                        f"    {name:<20s}  {t['call_count']} calls  "
                        f"~{raw:,} tok raw → ~{ctx:,} tok in context"
                    )
                else:
                    lines.append(
                        f"    {name:<20s}  {t['call_count']} calls  "
                        f"~{raw:,} tok/call"
                    )

        # Rot analysis section
        if rot:
            lines.append("")
            lines.append("[bold]Context Rot Analysis[/bold]")
            lines.append(
                "  [dim]Detects when an agent starts looping or degrading"
                " as context fills up.[/dim]"
            )

            rs = rot.get("summary", {})
            if not rot.get("has_content"):
                lines.append(
                    "  [dim]No content data. Use store_content=True"
                    " to enable repetition detection.[/dim]"
                )
            else:
                if rs.get("repetition_detected"):
                    lines.append(
                        f"  [bold red]REPETITION DETECTED"
                        f" at iteration #{rs['onset_iteration']}"
                        f"  (max overlap: {rs['max_repetition']:.2f})[/bold red]"
                    )
                else:
                    lines.append(
                        f"  [green]Repetition: LOW"
                        f" (max overlap: {rs.get('max_repetition', 0):.2f})[/green]"
                    )

                lines.append("")
                lines.append("  Per-iteration repetition:")
                lines.append(
                    "  [dim]  ngram  — word trigram overlap vs previous"
                    " completion (0=unique, 1=identical)[/dim]"
                )
                lines.append(
                    "  [dim]  seqsim — character-level similarity vs"
                    " previous (rapidfuzz ratio)[/dim]"
                )
                lines.append(
                    "  [dim]  cumul  — max trigram overlap vs ANY prior"
                    " completion (catches loops)[/dim]"
                )
                lines.append(
                    "  [dim]  <<<    — flagged when ngram > 0.4"
                    " (repetition onset)[/dim]"
                )
                lines.append("  [dim]  seq   ngram    seqsim   cumul[/dim]")
                for r in rot.get("repetition") or []:
                    flag = (
                        " [bold red]<<<[/bold red]" if r["ngram_jaccard"] > 0.4 else ""
                    )
                    lines.append(
                        f"   #{r['seq']:>2d}    {r['ngram_jaccard']:.2f}"
                        f"     {r['sequence_similarity']:.2f}"
                        f"     {r['cumulative_max']:.2f}{flag}"
                    )

            eff = rot.get("efficiency")
            if eff:
                lines.append("")
                lines.append("  Efficiency (completion / prompt):")
                lines.append(
                    "  [dim]  ratio — completion_tokens / prompt_tokens."
                    " Naturally declines as context grows;[/dim]"
                )
                lines.append(
                    "  [dim]  a sudden drop may indicate the model is"
                    " struggling with a large context.[/dim]"
                )
                lines.append("  [dim]  seq   ratio     tokens[/dim]")
                for e in eff:
                    lines.append(
                        f"   #{e['seq']:>2d}    {e['efficiency_ratio']:.4f}"
                        f"    {e['completion_tokens']:,} / {e['prompt_tokens']:,}"
                    )

        widget.update("\n".join(lines))

    def _update_context_bar(
        self, summary: dict, model: str, rot: dict | None = None
    ) -> None:
        """Update the always-visible context summary bar."""
        if not summary:
            return
        ctx_window = pricing.get_context_window(model)
        max_prompt = summary["max_prompt_tokens"]
        pct = (max_prompt / ctx_window * 100) if ctx_window > 0 else 0
        dur = summary["total_duration_ms"] / 1000
        cost_str = f"${summary['total_cost']:.4f}" if summary["total_cost"] else "n/a"

        # Rot indicator
        rot_str = ""
        if rot and rot.get("summary"):
            rs = rot["summary"]
            if rs.get("repetition_detected"):
                rot_str = f" · [bold red]REP at #{rs['onset_iteration']}[/bold red]"
            elif rot.get("has_content"):
                rot_str = " · [green]REP: low[/green]"

        # Warn if model is not in litellm's registry
        unknown_str = ""
        if not pricing.is_model_known(model):
            unknown_str = " · [bold yellow]⚠ unknown model (ctx default 200K)[/bold yellow]"

        ctx_bar = self.query_one("#context-bar", Static)
        ctx_bar.update(
            f"Context: {pct:.1f}% of {_fmt(ctx_window)} · "
            f"{summary['total_calls']} iters · "
            f"{_fmt(summary['total_prompt'])} in /"
            f" {_fmt(summary['total_completion'])} out · "
            f"cache {summary['cache_hit_pct']}% · {cost_str} · {dur:.1f}s"
            f"{rot_str}{unknown_str}"
        )

    @staticmethod
    def _render_grid(layers: list[dict], model: str) -> str:
        """Render a Claude Code /context-style grid and return as Rich markup string."""
        if not layers:
            return ""

        last = layers[-1]
        ctx_window = pricing.get_context_window(model)

        cats = [
            ("Input", last["input"], RICH_INPUT),
            ("Reasoning", last["reasoning"], RICH_REASONING),
            ("Tools", last["tools"], RICH_TOOL),
            ("Output", last["output"], RICH_OUTPUT),
        ]
        used_total = sum(tokens for _, tokens, _ in cats)
        free_tokens = max(0, ctx_window - used_total)

        # Allocate cells proportionally
        total_cells = GRID_TOTAL_CELLS
        cell_counts = []
        allocated = 0
        for _, tokens, _ in cats:
            n = round(tokens / ctx_window * total_cells) if ctx_window > 0 else 0
            cell_counts.append(n)
            allocated += n
        free_cells = max(0, total_cells - allocated)

        # Build flat cell list
        cells: list[tuple[str, str]] = []
        for i, (_, _, color) in enumerate(cats):
            cells.extend([(CELL_USED, color)] * cell_counts[i])
        cells.extend([(CELL_FREE, RICH_FREE)] * free_cells)
        cells = cells[:total_cells]
        while len(cells) < total_cells:
            cells.append((CELL_FREE, RICH_FREE))

        # Build grid lines
        grid_lines: list[str] = []
        for row_start in range(0, total_cells, GRID_COLS):
            row_cells = cells[row_start : row_start + GRID_COLS]
            line = " ".join(f"[{color}]{char}[/]" for char, color in row_cells)
            grid_lines.append(line)

        # Build right-side legend
        pct = used_total / ctx_window * 100 if ctx_window > 0 else 0
        unknown_warn = ""
        if not pricing.is_model_known(model):
            unknown_warn = " [bold yellow]⚠ unknown[/bold yellow]"
        info_lines: list[str] = [
            f"  {model} \u00b7 {_fmt(used_total)}/{_fmt(ctx_window)} tokens{unknown_warn}",
            f"  ({pct:.1f}%)",
            "",
            "  [bold]Estimated usage by category[/bold]",
        ]
        for (name, tokens, color), _count in zip(cats, cell_counts, strict=False):
            cat_pct = tokens / ctx_window * 100 if ctx_window > 0 else 0
            info_lines.append(
                f"  [{color}]{CELL_USED}[/] {name}: "
                f"{_fmt(tokens)} tokens ({cat_pct:.1f}%)"
            )
        free_pct = free_tokens / ctx_window * 100 if ctx_window > 0 else 0
        info_lines.append(
            f"  [{RICH_FREE}]{CELL_FREE}[/] Free space: "
            f"{_fmt(free_tokens)} ({free_pct:.1f}%)"
        )

        # Combine grid and legend side by side
        max_lines = max(len(grid_lines), len(info_lines))
        while len(grid_lines) < max_lines:
            grid_lines.append("")
        while len(info_lines) < max_lines:
            info_lines.append("")

        title = "[bold]Context Usage[/bold]\n"
        body_lines = []
        for gl, il in zip(grid_lines, info_lines, strict=False):
            body_lines.append(f"{gl}   {il}")

        return title + "\n".join(body_lines)

    def _update_content_from_data(self, content_rows: list[dict]) -> None:
        """Render full prompt messages and completions as per-iteration widgets."""
        scroll = self.query_one("#content-scroll", VerticalScroll)

        if not content_rows:
            # Show placeholder if not already present
            if not scroll.query("#content-placeholder"):
                scroll.mount(
                    Static(
                        "No content data." " Use store_content=True in CtxRotCallback.",
                        id="content-placeholder",
                        classes="content-panel",
                    )
                )
            return

        # Remove placeholder if present
        for ph in scroll.query("#content-placeholder"):
            ph.remove()

        # Only mount widgets for new iterations (existing ones are immutable)
        existing_ids = {w.id for w in scroll.query(IterationContent)}
        for row in content_rows:
            widget_id = f"content-iter-{row['seq']}"
            if widget_id in existing_ids:
                continue
            scroll.mount(IterationContent(row, id=widget_id, classes="content-panel"))


def _compute_layers(
    growth: list[dict], tools_by_iter: dict[int, list[dict]]
) -> list[dict]:
    """Compute stacked layers using delta-based decomposition.

    Instead of assuming all previous completions are fed back (which DSPy's ReAct
    doesn't do — it reformats/compresses), we use the actual prompt delta between
    iterations to derive what was added:

    - delta = prompt[n] - prompt[n-1]  (actual growth observed)
    - If tools were called after iter n-1: tool_added = sum(output_tokens_est)
    - reasoning_added = delta - tool_added  (remainder = completion fed back + formatting)

    This keeps the stacked bars consistent with the real prompt_tokens values.
    """
    if not growth:
        return []

    base_input = growth[0]["prompt_tokens"]
    last_idx = len(growth) - 1
    layers = []
    acc_reasoning = 0
    acc_tools = 0

    for i, g in enumerate(growth):
        if i == 0:
            reasoning_added = 0
            tool_added = 0
        else:
            prev = growth[i - 1]
            prev_seq = prev["seq"]
            delta = g["prompt_tokens"] - prev["prompt_tokens"]

            # Tool tokens added between prev iteration and this one
            tools_called = tools_by_iter.get(prev_seq, [])
            tool_added = sum(t["output_tokens_est"] for t in tools_called)

            # Reasoning = whatever else was added (prev completion fed back + DSPy formatting)
            reasoning_added = max(0, delta - tool_added)

            # If delta < 0 (DSPy compressed), don't grow either layer
            if delta < 0:
                reasoning_added = 0
                tool_added = 0

        acc_reasoning += reasoning_added
        acc_tools += tool_added

        layers.append(
            {
                "input": base_input,
                "reasoning": acc_reasoning,
                "tools": acc_tools,
                "output": g["completion_tokens"] if i == last_idx else 0,
            }
        )

    return layers


def _tool_context_cost(
    growth: list[dict], tools_by_iter: dict[int, list[dict]]
) -> dict[str, int]:
    """Compute avg context cost per tool from prompt deltas.

    Uses proportional attribution: the actual prompt delta is split between
    completion-fed-back and tool outputs according to their expected sizes.
    This handles compression/reformatting gracefully (ratio < 1 shrinks both
    proportionally instead of zeroing out tool cost).
    """
    # tool_name -> list of per-call context costs
    costs: dict[str, list[int]] = {}

    for i in range(len(growth) - 1):
        current = growth[i]
        seq = current["seq"]
        tools_called = tools_by_iter.get(seq, [])
        if not tools_called:
            continue

        next_entry = growth[i + 1]
        delta = max(0, next_entry["prompt_tokens"] - current["prompt_tokens"])
        prev_compl = current["completion_tokens"]
        total_tool_raw = sum(t["output_tokens_est"] for t in tools_called)

        # Expected growth = completion fed back + tool outputs
        expected = prev_compl + total_tool_raw
        if expected <= 0:
            continue

        # Ratio of actual vs expected growth (compression or expansion factor)
        ratio = delta / expected

        # Each tool's share = its raw output * ratio
        for t in tools_called:
            ctx = round(t["output_tokens_est"] * ratio)
            costs.setdefault(t["tool_name"], []).append(ctx)

    return {name: round(sum(vals) / len(vals)) for name, vals in costs.items()}


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
