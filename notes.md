# Notes
- their TruncatedSVD (SSE?) uses by default 30 n_iter, but torch_geometric sets it to 128
- we remove edges that are positive in one way and negative in the other
- Considering undirected graphs: We were not sure whether we need to construct the edge index with both entries (e.g. (0,1) and (1,0)) for the MessagePassing class of torch geometric
- Investigating the original code of the paper we discovered an additional linear layer before the final embeddings are returned. This is missing in both the paper and the recent implementation in torch_geometric