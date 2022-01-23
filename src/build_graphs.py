from torch import LongTensor
from typing import List, Tuple, Set, Optional
from pathlib import Path
import torch
from collections import defaultdict
import numpy as np
import math

MY_GRAPH = ("mygraph.csv", ",")


def make_undirected(edge_index: LongTensor) -> LongTensor:
    assert edge_index.shape[0] == 2
    from_node, to_node = edge_index
    return torch.cat([edge_index, torch.stack([to_node, from_node])], dim=1)


def build_edge_indices(
    file: str,
    split_symbol: str,
    get_angry=False,
    directed=True,
    target_num_nodes: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[LongTensor, LongTensor, int]:
    """
    get_angry: throw a tantrum when there are edges that are both positive and negative when viewed as undirected
    """
    pos_edge_index: set[Tuple[int, int]] = set()
    neg_edge_index: set[Tuple[int, int]] = set()
    nodes = set()
    with open(Path("data") / Path(file)) as f:
        for line in f.read().splitlines():
            # check formatting:
            if len(line) == 0 or line[0].isspace() or line[0] == "#" or line[0] == "%":
                continue
            edge = line.split(split_symbol)
            if edge[0] == "" or edge[1] == "" or edge[2] == "":
                print(f"weird line found: {line}")
                continue

            # building the graph:
            from_node = int(edge[0])
            to_node = int(edge[1])
            smaller_node = min(from_node, to_node)
            bigger_node = max(from_node, to_node)
            if smaller_node == bigger_node:
                # do not add self edge
                # print(f"Found self-edge: {smaller_node}")
                continue
                # raise Exception(f"Self-Edge: {smaller_node}")
            sign = int(float(edge[2]))
            nodes.add(from_node)
            nodes.add(to_node)
            if sign > 0:
                pos_edge_index.add((smaller_node, bigger_node))
            else:
                neg_edge_index.add((smaller_node, bigger_node))

    double_occurences = pos_edge_index & neg_edge_index
    pos_edge_index = pos_edge_index - double_occurences
    neg_edge_index = neg_edge_index - double_occurences
    if get_angry and len(double_occurences) != 0:
        raise Exception(
            f"({len(double_occurences)=}) nodes have both positive and negative link: {double_occurences} "
        )

    # prune graph from low-degree nodes:
    if target_num_nodes is not None:
        nodes, pos_edge_index, neg_edge_index = filter_out_nodes(
            nodes,
            list(pos_edge_index),
            list(neg_edge_index),
            target_num_nodes=target_num_nodes,
            seed=seed,
        )

    # clean up nodes so that we use values 0..max:
    nodes = sorted(list(nodes))
    node_dict = {node: i for i, node in enumerate(nodes)}
    pos_edge_index_1 = list(
        map(lambda edge: (node_dict[edge[0]], node_dict[edge[1]]), pos_edge_index)
    )
    neg_edge_index_1 = list(
        map(lambda edge: (node_dict[edge[0]], node_dict[edge[1]]), neg_edge_index)
    )

    if not directed:
        pos_edge_index_1.extend(
            list(map(lambda edge: (node_dict[edge[1]], node_dict[edge[0]]), pos_edge_index))
        )
        neg_edge_index_1.extend(
            list(map(lambda edge: (node_dict[edge[1]], node_dict[edge[0]]), neg_edge_index))
        )

    return (
        LongTensor(pos_edge_index_1).T,
        LongTensor(neg_edge_index_1).T,
        len(nodes),
    )


def filter_out_nodes(
    nodes: Set[int],
    pos_edge_index: List[Tuple[int, int]],
    neg_edge_index: List[Tuple[int, int]],
    target_num_nodes: int,
    seed: Optional[int],
) -> Tuple[Set[int], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    The input edge indices are interpreted as undirected!
    Returns a new positive and negative edge_index and also the new set of nodes
    """
    pos_adj_lists = defaultdict(set)
    neg_adj_lists = defaultdict(set)
    for x, y in pos_edge_index:
        pos_adj_lists[x].add(y)
        pos_adj_lists[y].add(x)
    for x, y in neg_edge_index:
        neg_adj_lists[x].add(y)
        neg_adj_lists[y].add(x)

    degrees = {node: len(pos_adj_lists[node]) + len(neg_adj_lists[node]) for node in nodes}

    probabilities = np.asarray([deg for deg in degrees.values()])
    probabilities = probabilities / np.sum(probabilities)

    np.random.seed(seed)
    final_nodes = set(
        np.random.choice(
            list(degrees.keys()), size=target_num_nodes, replace=False, p=probabilities
        )
    )

    new_pos_edge_index: Set[Tuple[int, int]] = set()
    new_neg_edge_index: Set[Tuple[int, int]] = set()

    for x, y in pos_edge_index:
        if x in final_nodes and y in final_nodes:
            new_pos_edge_index.add((x, y))

    for x, y in neg_edge_index:
        if x in final_nodes and y in final_nodes:
            new_neg_edge_index.add((x, y))

    return final_nodes, new_pos_edge_index, new_neg_edge_index
