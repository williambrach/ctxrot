"""RLM execution tree view — shows the iterative reasoning graph."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Tree


class RlmTreeView(Tree):
    """Interactive tree visualization for RLM execution sessions."""

    DEFAULT_CSS = """
    RlmTreeView {
        height: 1fr;
        min-height: 14;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("RLM Execution", **kwargs)
        self._last_count: int = -1
        self._last_session_id: str | None = None

    def update_tree(
        self,
        tree_data: list[dict],
        tools_by_iter: dict[int, list[dict]],
        session_id: str,
    ) -> None:
        """Rebuild the tree from RLM call data."""
        if (
            len(tree_data) == self._last_count
            and session_id == self._last_session_id
        ):
            return
        self._last_count = len(tree_data)
        self._last_session_id = session_id

        self.clear()
        self.root.expand()

        if not tree_data:
            self.root.add_leaf(Text("No data yet", style="dim"))
            return

        # Group calls by iteration
        iterations: dict[int, list[dict]] = {}
        for call in tree_data:
            it = call.get("iteration") or 0
            iterations.setdefault(it, []).append(call)

        for iter_num in sorted(iterations.keys()):
            calls = iterations[iter_num]
            action_call = None
            sub_queries = []

            for c in calls:
                if c["call_type"] in ("action", "extract"):
                    action_call = c
                elif c["call_type"] == "sub_query":
                    sub_queries.append(c)

            if action_call is None:
                # Ungrouped calls (shouldn't happen with new data)
                for c in calls:
                    self.root.add_leaf(_format_call_label(c))
                continue

            ct = action_call["call_type"]

            # Build iteration node
            label = _format_action_label(iter_num, action_call, ct)
            iter_node = self.root.add(label, expand=True)

            # Code preview from completion
            completion = action_call.get("completion")
            if completion:
                preview = _extract_code_preview(completion, ct)
                if preview:
                    iter_node.add_leaf(Text(preview, style="dim"))

            # Tool calls after this action
            tools_here = tools_by_iter.get(action_call["seq"], [])
            for t in tools_here:
                tool_label = _format_tool_label(t)
                iter_node.add_leaf(tool_label)

            # Sub-query nodes
            for i, sq in enumerate(sub_queries, 1):
                sq_label = _format_sub_query_label(i, sq)
                sq_node = iter_node.add(sq_label)
                if sq.get("completion"):
                    preview = sq["completion"][:80].replace("\n", " ").strip()
                    sq_node.add_leaf(Text(preview, style="dim"))


def _format_action_label(iter_num: int, call: dict, call_type: str) -> Text:
    """Format a rich label for an action/extract iteration node."""
    tokens = f"{call['prompt_tokens']:,} in / {call['completion_tokens']:,} out"
    duration = f"{call['duration_ms'] / 1000:.1f}s" if call["duration_ms"] else ""
    cost = f"${call['cost']:.4f}" if call.get("cost") else ""

    label = Text()
    if call_type == "extract":
        label.append(" [extract]  ", style="bold green")
    else:
        label.append(f" Iter {iter_num}  ", style="bold")
    label.append(tokens, style="")
    if duration:
        label.append(f"   {duration}", style="cyan")
    if cost:
        label.append(f"   {cost}", style="yellow")
    return label


def _format_sub_query_label(index: int, call: dict) -> Text:
    """Format a rich label for a sub-query node."""
    tokens = f"{call['prompt_tokens']:,} in / {call['completion_tokens']:,} out"
    duration = f"{call['duration_ms'] / 1000:.1f}s" if call["duration_ms"] else ""

    label = Text()
    label.append(f" sub_query #{index}  ", style="bold yellow")
    label.append(tokens, style="")
    if duration:
        label.append(f"   {duration}", style="cyan")
    return label


def _format_tool_label(tool: dict) -> Text:
    """Format a rich label for a tool call node."""
    label = Text()
    label.append(f" {tool['tool_name']}  ", style="bold green")
    label.append(f"+{tool['output_tokens_est']:,} tok", style="")
    if tool.get("duration_ms"):
        label.append(f"   {tool['duration_ms'] / 1000:.1f}s", style="cyan")
    return label


def _format_call_label(call: dict) -> Text:
    """Fallback label for ungrouped calls."""
    tokens = f"{call['prompt_tokens']:,} in / {call['completion_tokens']:,} out"
    label = Text()
    label.append(f" #{call['seq']}  ", style="bold")
    label.append(tokens, style="")
    return label


def _extract_code_preview(completion: str, call_type: str) -> str:
    """Extract a short preview from the completion text."""
    if not completion:
        return ""
    if call_type == "extract":
        # For extract, show the answer
        text = completion.strip()
        # Strip DSPy markers
        for marker in ("[[ ## answer ## ]]", "[[ ## completed ## ]]"):
            text = text.replace(marker, "").strip()
        if text:
            return f"answer: {text[:80]}"
        return ""
    # For action, show first meaningful line of code
    lines = completion.strip().splitlines()
    for line in lines:
        stripped = line.strip()
        # Skip DSPy markers and empty lines
        if not stripped or stripped.startswith("[[ ##"):
            continue
        # Skip markdown code fences
        if stripped.startswith("```"):
            continue
        return f"code: {stripped[:80]}"
    return ""
