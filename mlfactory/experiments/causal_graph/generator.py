"""Deterministic symbolic CausalGraph task generator and renderers.

The graph is authoritative.  Rendered text and canonical traces are derived
from it; no language model is involved in generation or verification.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPS = ("COPY", "NOT", "AND", "OR")
WORLD_IDS = ("greenhouse", "factory", "security", "household", "warehouse", "computer")
TRAIN_WORLDS = WORLD_IDS[:4]
HELDOUT_WORLDS = WORLD_IDS[4:]

# Each vocabulary describes the same Boolean predicate with paired true/false
# surface words.  Names are deliberately fixed but selected deterministically.
WORLD_VOCAB: dict[str, dict[str, Any]] = {
    "greenhouse": {"places": ["greenhouse", "nursery"], "entities": [
        ("circulation pump", "running", "stopped"), ("intake valve", "open", "closed"),
        ("warning lamp", "lit", "dark"), ("safety relay", "enabled", "disabled"),
        ("exhaust fan", "running", "stopped"), ("humidity sensor", "active", "inactive"),
        ("shade motor", "running", "stopped"), ("irrigation line", "open", "closed"),
    ]},
    "factory": {"places": ["factory floor", "assembly plant"], "entities": [
        ("drive motor", "running", "stopped"), ("intake gate", "open", "closed"),
        ("warning beacon", "lit", "dark"), ("safety relay", "enabled", "disabled"),
        ("conveyor", "running", "stopped"), ("pressure sensor", "active", "inactive"),
        ("loading arm", "extended", "retracted"), ("coolant valve", "open", "closed"),
    ]},
    "security": {"places": ["security building", "office entrance"], "entities": [
        ("access gate", "open", "closed"), ("entry sensor", "active", "inactive"),
        ("warning alarm", "sounding", "silent"), ("security relay", "enabled", "disabled"),
        ("door lock", "locked", "unlocked"), ("camera", "active", "inactive"),
        ("badge reader", "powered", "unpowered"), ("patrol light", "lit", "dark"),
    ]},
    "household": {"places": ["house", "apartment"], "entities": [
        ("water pump", "running", "stopped"), ("kitchen switch", "on", "off"),
        ("warning light", "lit", "dark"), ("main breaker", "enabled", "disabled"),
        ("heater", "running", "stopped"), ("thermostat", "active", "inactive"),
        ("door latch", "locked", "unlocked"), ("hall lamp", "lit", "dark"),
    ]},
    "warehouse": {"places": ["warehouse", "loading depot"], "entities": [
        ("loading motor", "running", "stopped"), ("dock gate", "open", "closed"),
        ("status indicator", "lit", "dark"), ("safety controller", "enabled", "disabled"),
        ("forklift", "moving", "parked"), ("barcode scanner", "active", "inactive"),
        ("cargo lock", "locked", "unlocked"), ("ventilation fan", "running", "stopped"),
    ]},
    "computer": {"places": ["computer system", "server room"], "entities": [
        ("database service", "running", "stopped"), ("firewall port", "open", "closed"),
        ("warning monitor", "lit", "dark"), ("system flag", "enabled", "disabled"),
        ("backup job", "running", "stopped"), ("health check", "active", "inactive"),
        ("cache service", "running", "stopped"), ("network link", "up", "down"),
    ]},
}

OP_TEMPLATES = {
    "COPY": [
        "The {child} is {true} exactly when the {parent} is {parent_true}.",
        "Whenever the {parent} is {parent_true}, the {child} is {true}; otherwise it is {false}.",
        "The state of the {child} follows the state of the {parent}.",
    ],
    "NOT": [
        "The {child} is {true} exactly when the {parent} is {parent_false}.",
        "The {child} is the opposite of the {parent}: it is {true} when the {parent} is {parent_false}.",
        "Whenever the {parent} is {parent_true}, the {child} is {false}; whenever it is {parent_false}, the child is {true}.",
    ],
    "AND": [
        "The {child} is {true} exactly when both the {parent} is {parent_true} and the {second} is {second_true}; otherwise it is {false}.",
        "The {child} is {true} if and only if the {parent} is {parent_true} and the {second} is {second_true}.",
        "The {child} is {false} exactly when at least one of the {parent} is {parent_false} or the {second} is {second_false}; otherwise it is {true}.",
    ],
    "OR": [
        "The {child} is {true} exactly when either the {parent} is {parent_true} or the {second} is {second_true}; otherwise it is {false}.",
        "The {child} is {true} if and only if at least one of these is true: the {parent} is {parent_true}, or the {second} is {second_true}.",
        "The {child} is {false} exactly when both the {parent} is {parent_false} and the {second} is {second_false}; otherwise it is {true}.",
    ],
}

QUERY_TEMPLATES = [
    "After all conditions propagate, is the {node} {true}?",
    "Once every rule has taken effect, is the {node} {true}?",
    "What is the final state of the {node}: is it {true}?",
]


def _stable_id(seed: int, prefix: str = "cg") -> str:
    return f"{prefix}-{hashlib.sha256(str(seed).encode()).hexdigest()[:16]}"


def _state_phrase(label: dict[str, str], value: bool) -> str:
    return label["true"] if value else label["false"]


def _validate_request(depth: int, relevant_nodes: int, binary_gate_count: int, negation_count: int) -> None:
    if depth < 1 or depth > 64:
        raise ValueError("depth must be in [1, 64]")
    if relevant_nodes < depth + 1:
        raise ValueError("relevant_nodes must be at least depth + 1")
    if binary_gate_count < 0 or binary_gate_count > depth:
        raise ValueError("binary_gate_count must be in [0, depth]")
    if negation_count < 0 or negation_count > depth - binary_gate_count:
        raise ValueError("negation_count exceeds available unary spine nodes")
    if relevant_nodes > depth + 1 + binary_gate_count:
        raise ValueError(
            "this MVP graph shape supports at most depth + 1 + binary_gate_count relevant nodes"
        )


def _node_label(world: str, index: int) -> dict[str, str]:
    vocab = WORLD_VOCAB[world]["entities"]
    name, true, false = vocab[index % len(vocab)]
    cycle = index // len(vocab)
    return {"name": f"{name}{f' {cycle + 1}' if cycle else ''}", "true": true, "false": false}


def generate_task(
    seed: int,
    *,
    depth: int,
    relevant_nodes: int | None = None,
    distractor_nodes: int = 0,
    binary_gate_count: int | None = None,
    negation_count: int | None = None,
    source_update_count: int = 0,
    world_id: str = "greenhouse",
    render_template_id: int | None = None,
) -> dict[str, Any]:
    """Generate one deterministic, non-degenerate CausalGraph task.

    The MVP graph is a depth-controlled spine.  Extra relevant nodes are
    source nodes attached at binary gates, which keeps fan-in <= 2 while
    making every requested relevant node query-relevant.
    """
    if world_id not in WORLD_VOCAB:
        raise ValueError(f"unknown world_id: {world_id}")
    rng = random.Random(seed)
    binary_gate_count = depth // 4 if binary_gate_count is None else int(binary_gate_count)
    binary_gate_count = min(max(binary_gate_count, 0), depth)
    negation_count = max(0, depth // 5) if negation_count is None else int(negation_count)
    extra = min(max(0, int(relevant_nodes or (depth + 1 + min(binary_gate_count, 2)))), binary_gate_count)
    relevant_nodes = depth + 1 + extra
    _validate_request(depth, relevant_nodes, binary_gate_count, negation_count)
    if distractor_nodes < 0 or distractor_nodes > 64:
        raise ValueError("distractor_nodes must be in [0, 64]")
    if source_update_count < 0 or source_update_count > 32:
        raise ValueError("source_update_count must be in [0, 32]")

    # Choose gate and NOT positions deterministically.  Gate positions are
    # one-indexed spine nodes; the final query may be a gate.
    positions = list(range(1, depth + 1))
    rng.shuffle(positions)
    gate_positions = set(sorted(positions[:binary_gate_count]))
    unary_positions = [p for p in positions if p not in gate_positions]
    rng.shuffle(unary_positions)
    not_positions = set(unary_positions[:negation_count])

    labels: dict[str, dict[str, str]] = {}
    nodes: dict[str, dict[str, Any]] = {}
    source_ids = ["source_0"] + [f"source_extra_{i}" for i in range(extra)]
    for i, node_id in enumerate(source_ids):
        labels[node_id] = _node_label(world_id, i)
        nodes[node_id] = {"id": node_id, "kind": "source", "label": labels[node_id]}
    spine_ids: list[str] = ["source_0"]
    for level in range(1, depth + 1):
        node_id = "query" if level == depth else f"derived_{level:02d}"
        labels[node_id] = _node_label(world_id, len(labels))
        if level in gate_positions:
            # Each extra source is used once; remaining gates reuse source_0.
            extra_index = sum(1 for p in gate_positions if p <= level) - 1
            second = source_ids[1 + extra_index] if extra_index < extra else "source_0"
            op = "AND" if rng.random() < 0.5 else "OR"
            parents = [spine_ids[-1], second]
        else:
            op = "NOT" if level in not_positions else "COPY"
            parents = [spine_ids[-1]]
        nodes[node_id] = {"id": node_id, "kind": "derived", "op": op, "parents": parents, "label": labels[node_id]}
        spine_ids.append(node_id)

    # Distractors are independent source facts.  They are not included in the
    # relevant ancestry and therefore cannot influence the verifier answer.
    distractor_ids: list[str] = []
    for i in range(int(distractor_nodes)):
        node_id = f"distractor_{i:02d}"
        labels[node_id] = _node_label(world_id, len(labels))
        nodes[node_id] = {"id": node_id, "kind": "distractor", "label": labels[node_id]}
        distractor_ids.append(node_id)

    topological_order = source_ids + [f"derived_{level:02d}" if level < depth else "query" for level in range(1, depth + 1)] + distractor_ids
    initial_values = {node_id: bool(rng.getrandbits(1)) for node_id in source_ids + distractor_ids}
    updates: list[dict[str, Any]] = []
    updateable = source_ids[:]
    for index in range(int(source_update_count)):
        node_id = rng.choice(updateable)
        action = rng.choice(("SET TRUE", "SET FALSE", "TOGGLE"))
        updates.append({"step": index + 1, "node": node_id, "action": action})

    states = dict(initial_values)
    for node_id in topological_order:
        node = nodes[node_id]
        if node["kind"] != "derived":
            continue
        parent_values = [states[parent] for parent in node["parents"]]
        op = node["op"]
        states[node_id] = parent_values[0] if op == "COPY" else not parent_values[0] if op == "NOT" else all(parent_values) if op == "AND" else any(parent_values)
    for update in updates:
        node_id = update["node"]
        if update["action"] == "SET TRUE":
            states[node_id] = True
        elif update["action"] == "SET FALSE":
            states[node_id] = False
        else:
            states[node_id] = not states[node_id]
        # Re-propagate the derived spine after every explicit source update.
        for node_id2 in topological_order:
            node2 = nodes[node_id2]
            if node2["kind"] != "derived":
                continue
            pv = [states[parent] for parent in node2["parents"]]
            states[node_id2] = pv[0] if node2["op"] == "COPY" else not pv[0] if node2["op"] == "NOT" else all(pv) if node2["op"] == "AND" else any(pv)

    task = {
        "id": _stable_id(seed), "seed": int(seed), "graph": {"nodes": nodes},
        "topological_order": topological_order, "source_values_initial": initial_values,
        "source_updates": updates, "query_node": "query", "canonical_node_states": states,
        "canonical_answer": "YES" if states["query"] else "NO", "depth": depth,
        "relevant_nodes": relevant_nodes, "distractor_nodes": int(distractor_nodes),
        "binary_gate_count": sum(1 for n in nodes.values() if n.get("op") in ("AND", "OR")),
        "negation_count": sum(1 for n in nodes.values() if n.get("op") == "NOT"),
        "source_update_count": len(updates), "world_id": world_id,
        "render_template_id": int(render_template_id if render_template_id is not None else rng.randrange(3)),
        "render_template_ids": [],
    }
    task["rendered_prompt"] = render_task(task)
    # Renderer is part of the deterministic task, so regeneration must include
    # no hidden state.
    return task


def verify_task(task: dict[str, Any]) -> dict[str, Any]:
    """Independently recompute graph state and validate generator invariants."""
    graph = task["graph"]["nodes"]
    order = task["topological_order"]
    if len(order) != len(graph) or set(order) != set(graph):
        raise ValueError("topological_order does not contain every graph node exactly once")
    position = {node_id: i for i, node_id in enumerate(order)}
    for node_id, node in graph.items():
        if node["kind"] == "derived":
            if len(node["parents"]) not in (1, 2):
                raise ValueError(f"invalid fan-in for {node_id}")
            if any(position[parent] >= position[node_id] for parent in node["parents"]):
                raise ValueError(f"graph is not acyclic/topological at {node_id}")
            if node.get("op") not in OPS:
                raise ValueError(f"unknown op for {node_id}")
    states = dict(task["source_values_initial"])
    for node_id in order:
        node = graph[node_id]
        if node["kind"] != "derived":
            continue
        pv = [states[parent] for parent in node["parents"]]
        states[node_id] = pv[0] if node["op"] == "COPY" else not pv[0] if node["op"] == "NOT" else all(pv) if node["op"] == "AND" else any(pv)
    for update in task["source_updates"]:
        if update["action"] == "SET TRUE":
            states[update["node"]] = True
        elif update["action"] == "SET FALSE":
            states[update["node"]] = False
        elif update["action"] == "TOGGLE":
            states[update["node"]] = not states[update["node"]]
        else:
            raise ValueError(f"unknown update {update['action']}")
        for node_id in order:
            node = graph[node_id]
            if node["kind"] == "derived":
                pv = [states[parent] for parent in node["parents"]]
                states[node_id] = pv[0] if node["op"] == "COPY" else not pv[0] if node["op"] == "NOT" else all(pv) if node["op"] == "AND" else any(pv)
    if states != task["canonical_node_states"]:
        raise ValueError("stored canonical states disagree with independent evaluation")
    if task["canonical_answer"] != ("YES" if states[task["query_node"]] else "NO"):
        raise ValueError("stored canonical answer disagrees with query state")
    ancestors = _ancestors(graph, task["query_node"])
    if any(node_id in graph and graph[node_id]["kind"] == "distractor" for node_id in ancestors):
        raise ValueError("distractor influences query")
    relevant = set(graph) - {n for n in graph if graph[n]["kind"] == "distractor"}
    if task["relevant_nodes"] != len(relevant) or ancestors | {task["query_node"]} != relevant:
        raise ValueError("relevant_nodes metadata or query ancestry mismatch")
    depths: dict[str, int] = {}
    for node_id in order:
        node = graph[node_id]
        if node["kind"] == "source":
            depths[node_id] = 0
        elif node["kind"] == "distractor":
            depths[node_id] = 0
        else:
            depths[node_id] = 1 + max(depths[parent] for parent in node["parents"])
    if depths[task["query_node"]] != int(task["depth"]):
        raise ValueError("depth metadata does not equal longest source-to-query path")
    return {"valid": True, "answer": task["canonical_answer"], "node_count": len(graph), "query_ancestors": len(ancestors), "depth": depths[task["query_node"]]}


def _ancestors(graph: dict[str, dict[str, Any]], node_id: str) -> set[str]:
    result: set[str] = set()
    stack = list(graph[node_id].get("parents", []))
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(graph[current].get("parents", []))
    return result


def _rule_text(node: dict[str, Any], graph: dict[str, dict[str, Any]], template_id: int) -> str:
    op = node["op"]
    parent, *rest = node["parents"]
    child_label, parent_label = node["label"], graph[parent]["label"]
    values = {"child": child_label["name"], "parent": parent_label["name"],
              "second": graph[rest[0]]["label"]["name"] if rest else "", "true": child_label["true"],
              "false": child_label["false"], "parent_true": parent_label["true"], "parent_false": parent_label["false"]}
    if rest:
        second_label = graph[rest[0]]["label"]
        values.update({"second_true": second_label["true"], "second_false": second_label["false"]})
    return OP_TEMPLATES[op][template_id % len(OP_TEMPLATES[op])].format(**values)


def render_task(task: dict[str, Any]) -> str:
    """Render the graph into an unambiguous natural-language prompt."""
    rng = random.Random(task["seed"] + 99173)
    world = WORLD_VOCAB[task["world_id"]]
    graph = task["graph"]["nodes"]
    order = task["topological_order"]
    lines = [f"In a {world['places'][task['render_template_id'] % len(world['places'])]}:", ""]
    for node_id in order:
        node = graph[node_id]
        if node["kind"] in ("source", "distractor"):
            value = task["source_values_initial"][node_id]
            lines.append(f"The {node['label']['name']} is {_state_phrase(node['label'], value)}.")
    lines.append("")
    lines.append("Rules:")
    rule_nodes = [graph[n] for n in order if graph[n]["kind"] == "derived"]
    rng.shuffle(rule_nodes)
    template_ids: list[int] = []
    for node in rule_nodes:
        template_id = rng.randrange(3)
        template_ids.append(template_id)
        lines.append(f"- {_rule_text(node, graph, template_id)}")
    lines.append("")
    for update in task["source_updates"]:
        label = graph[update["node"]]["label"]
        lines.append(f"Later, the {label['name']} is {update['action'].lower()}.")
    if task["source_updates"]:
        lines.append("")
    query_label = graph[task["query_node"]]["label"]
    lines.append(QUERY_TEMPLATES[task["render_template_id"] % len(QUERY_TEMPLATES)].format(node=query_label["name"], true=query_label["true"]))
    lines.append("Reply with a concise derivation and end with exactly FINAL: YES or FINAL: NO.")
    task["render_template_ids"] = template_ids
    return "\n".join(lines)


def canonical_trace(task: dict[str, Any]) -> str:
    """Create a concise, query-relevant symbolic derivation target."""
    graph = task["graph"]["nodes"]
    states = task["canonical_node_states"]
    lines: list[str] = []
    for node_id in task["topological_order"]:
        node = graph[node_id]
        if node["kind"] == "source" and node_id in _ancestors(graph, task["query_node"]):
            lines.append(f"The {node['label']['name']} is {_state_phrase(node['label'], states[node_id])}.")
        elif node["kind"] == "derived" and node_id in _ancestors(graph, task["query_node"]):
            parent_names = " and ".join(graph[p]["label"]["name"] for p in node["parents"])
            lines.append(f"The {node['label']['name']} is {_state_phrase(node['label'], states[node_id])} because of {parent_names}.")
    answer = states[task["query_node"]]
    lines.append(f"The {graph[task['query_node']]['label']['name']} is {_state_phrase(graph[task['query_node']]['label'], answer)}.")
    lines.append(f"FINAL: {'YES' if answer else 'NO'}")
    return "\n".join(lines)


def regenerate_matches(task: dict[str, Any]) -> bool:
    """Check the stable identity and rendered prompt for deterministic replay."""
    replay = generate_task(
        task["seed"], depth=task["depth"], relevant_nodes=task["relevant_nodes"],
        distractor_nodes=task["distractor_nodes"], binary_gate_count=task["binary_gate_count"],
        negation_count=task["negation_count"], source_update_count=task["source_update_count"],
        world_id=task["world_id"], render_template_id=task["render_template_id"],
    )
    return replay["id"] == task["id"] and replay["rendered_prompt"] == task["rendered_prompt"] and replay["canonical_answer"] == task["canonical_answer"]


def write_jsonl(path: str | Path, tasks: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
