"""
Builds a queryable lineage DAG from model records and answers the one
question no existing tool answers today: given a compromised model,
what is everything downstream of it.
"""
import json
import time
import networkx as nx


def load_graph(path):
    """Builds a directed graph, edges point parent -> child, from a
    lineage JSON file (either true_lineage.json or tracked_lineage.json)."""
    with open(path) as f:
        records = json.load(f)
    return graph_from_records(records)


def graph_from_records(records):
    """Same construction, from an in-memory dict of records. Used by the
    sweep, which generates thousands of graphs and never touches disk."""
    g = nx.DiGraph()
    for rid, rec in records.items():
        g.add_node(rid, **{k: v for k, v in rec.items() if k != "parent_ids"})
    for rid, rec in records.items():
        for parent_id in rec["parent_ids"]:
            if parent_id in records:
                g.add_edge(parent_id, rid)
    return g


def blast_radius(graph, root_id):
    """Every model reachable forward from root_id, root included.
    This is the query: 'if root_id turns out to be compromised, what's
    affected downstream, right now, with what we have on record.'"""
    if root_id not in graph:
        return set()
    start = time.perf_counter()
    affected = nx.descendants(graph, root_id)
    affected.add(root_id)
    elapsed = time.perf_counter() - start
    return affected, elapsed


if __name__ == "__main__":
    g = load_graph("data/tracked_lineage.json")
    patient_zeros = [n for n, d in g.nodes(data=True) if d.get("is_patient_zero")]
    print(f"Loaded tracked graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print(f"Patient zero models: {patient_zeros}")
    for pz in patient_zeros:
        affected, elapsed = blast_radius(g, pz)
        print(f"\nBlast radius of {pz}: {len(affected)} models "
              f"(query took {elapsed * 1000:.3f} ms)")
        print(f"  {sorted(affected)}")
