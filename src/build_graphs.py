from torch import LongTensor
from typing import List, Tuple
from pathlib import Path
import torch

MY_GRAPH = ("mygraph.csv", ",")

BITCOIN_ALPHA_THEIRS = ("bitcoinalpha_theirs.csv", ",")
BITCOIN_OTC_THEIRS = ("bitcoinotc_theirs.csv", ",")
BITCOIN_ALPHA = ("soc-sign-bitcoinalpha.csv", ",")
BITCOIN_OTC = ("soc-sign-bitcoinotc.csv", ",")
SLASHDOT = ("soc-sign-Slashdot090221.txt", "\t")
EPINIONS = ("soc-sign-epinions.txt", "\t")
SLASHDOT_ALTERNATIVE = ("slashdot_zoo.matrix", " ")

ALL_FILES = [
    BITCOIN_ALPHA,
    BITCOIN_OTC,
    SLASHDOT,
    EPINIONS
]


def build_edge_indices(file: str, split_symbol: str, get_angry = False) -> Tuple[LongTensor, LongTensor, int]:
    """
    get_angry: throw a tantrum when there are edges that are both positive and negative when viewed as undirected
    """
    pos_edge_index: set[Tuple[int, int]] = set()
    neg_edge_index: set[Tuple[int, int]] = set()
    nodes = set()
    with open(Path("data") / Path(file)) as f:
        for line in f.read().splitlines():
            if len(line) == 0 or line[0].isspace() or line[0] == "#" or line[0] == "%":
                continue
            edge = line.split(split_symbol)
            if edge[0] == "" or edge[1] == "" or edge[2] == "":
                print(f"weird line found: {line}")
                continue
            from_node = int(edge[0])
            to_node = int(edge[1])
            smaller_node = min(from_node, to_node)
            bigger_node = max(from_node, to_node)
            if smaller_node == bigger_node:
                raise Exception(f"Self-Edge: {smaller_node}")
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
        raise Exception(f"({len(double_occurences)=}) nodes have both positive and negative link: {double_occurences} ")
    nodes = sorted(list(nodes))
    node_dict = {node: i for i, node in enumerate(nodes)}
    pos_edge_index_1 = list(map(lambda edge: (node_dict[edge[0]], node_dict[edge[1]]), pos_edge_index))
    pos_edge_index_2 = list(map(lambda edge: (node_dict[edge[1]], node_dict[edge[0]]), pos_edge_index))
    neg_edge_index_1 = list(map(lambda edge: (node_dict[edge[0]], node_dict[edge[1]]), neg_edge_index))
    neg_edge_index_2 = list(map(lambda edge: (node_dict[edge[1]], node_dict[edge[0]]), neg_edge_index))

    return LongTensor(pos_edge_index_1 + pos_edge_index_2).T, LongTensor(neg_edge_index_1 + neg_edge_index_2).T, len(nodes)

def main():
    edge_indices = build_edge_indices(*SLASHDOT_ALTERNATIVE, get_angry=False)
    pass

if  __name__ == "__main__":
    main()
