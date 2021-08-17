"""Implementation of IVDetect."""


import json
import pickle as pkl
from collections import defaultdict
from glob import glob
from pathlib import Path

import dgl
import dgl.function as fn
import networkx as nx
import pandas as pd
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.dl as dl
import sastvd.helpers.glove as svdg
import sastvd.helpers.joern as svdj
import sastvd.helpers.tokenise as svdt
import sastvd.ivdetect.pyramidpooling as ivdp
import sastvd.ivdetect.treelstm as ivdts
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from torch.nn.utils.rnn import pad_sequence


def feature_extraction(filepath):
    """Extract relevant components of IVDetect Code Representation.

    DEBUGGING:
    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/180189.c"
    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/182480.c"

    PRINTING:
    svdj.plot_graph_node_edge_df(nodes, svdj.rdg(edges, "ast"), [24], 0)
    svdj.plot_graph_node_edge_df(nodes, svdj.rdg(edges, "reftype"))
    pd.options.display.max_colwidth = 500
    print(subseq.to_markdown(mode="github", index=0))
    print(nametypes.to_markdown(mode="github", index=0))
    print(uedge.to_markdown(mode="github", index=0))

    4/5 COMPARISON:
    Theirs: 31, 22, 13, 10, 6, 29, 25, 23
    Ours  : 40, 30, 19, 14, 7, 38, 33, 31
    Pred  : 40,   , 19, 14, 7, 38, 33, 31
    """
    cache_name = "_".join(str(filepath).split("/")[-3:])
    cachefp = svd.get_dir(svd.cache_dir() / "ivdetect_feat_ext") / Path(cache_name).stem
    try:
        with open(cachefp, "rb") as f:
            return pkl.load(f)
    except:
        pass

    try:
        nodes, edges = svdj.get_node_edges(filepath)
    except:
        return None

    # 1. Generate tokenised subtoken sequences
    subseq = (
        nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    subseq = subseq[["lineNumber", "code", "local_type"]].copy()
    subseq.code = subseq.local_type + " " + subseq.code
    subseq = subseq.drop(columns="local_type")
    subseq = subseq[~subseq.eq("").any(1)]
    subseq = subseq[subseq.code != " "]
    subseq.lineNumber = subseq.lineNumber.astype(int)
    subseq = subseq.sort_values("lineNumber")
    subseq.code = subseq.code.apply(svdt.tokenise)
    subseq = subseq.set_index("lineNumber").to_dict()["code"]

    # 2. Line to AST
    ast_edges = svdj.rdg(edges, "ast")
    ast_nodes = svdj.drop_lone_nodes(nodes, ast_edges)
    ast_nodes = ast_nodes[ast_nodes.lineNumber != ""]
    ast_nodes.lineNumber = ast_nodes.lineNumber.astype(int)
    ast_nodes["lineidx"] = ast_nodes.groupby("lineNumber").cumcount().values
    ast_edges = ast_edges[ast_edges.line_out == ast_edges.line_in]
    ast_dict = pd.Series(ast_nodes.lineidx.values, index=ast_nodes.id).to_dict()
    ast_edges.innode = ast_edges.innode.map(ast_dict)
    ast_edges.outnode = ast_edges.outnode.map(ast_dict)
    ast_edges = ast_edges.groupby("line_in").agg({"innode": list, "outnode": list})
    ast_nodes.code = ast_nodes.code.fillna("").apply(svdt.tokenise)
    nodes_per_line = (
        ast_nodes.groupby("lineNumber").agg({"lineidx": list}).to_dict()["lineidx"]
    )
    ast_nodes = ast_nodes.groupby("lineNumber").agg({"code": list})
    ast = ast_edges.join(ast_nodes, how="inner")
    ast["ast"] = ast.apply(lambda x: [x.outnode, x.innode, x.code], axis=1)
    ast = ast.to_dict()["ast"]

    # If it is a lone node (nodeid doesn't appear in edges) or it is a node with no
    # incoming connections (parent node), then add an edge from that node to the node
    # with id = 0 (unless it is zero itself).
    # DEBUG:
    # import sastvd.helpers.graphs as svdgr
    # svdgr.simple_nx_plot(ast[20][0], ast[20][1], ast[20][2])
    for k, v in ast.items():
        allnodes = nodes_per_line[k]
        outnodes = v[0]
        innodes = v[1]
        lonenodes = [i for i in allnodes if i not in outnodes + innodes]
        parentnodes = [i for i in outnodes if i not in innodes]
        for n in set(lonenodes + parentnodes) - set([0]):
            outnodes.append(0)
            innodes.append(n)
        ast[k] = [outnodes, innodes, v[2]]

    # 3. Variable names and types
    reftype_edges = svdj.rdg(edges, "reftype")
    reftype_nodes = svdj.drop_lone_nodes(nodes, reftype_edges)
    reftype_nx = nx.Graph()
    reftype_nx.add_edges_from(reftype_edges[["innode", "outnode"]].to_numpy())
    reftype_cc = list(nx.connected_components(reftype_nx))
    varnametypes = list()
    for cc in reftype_cc:
        cc_nodes = reftype_nodes[reftype_nodes.id.isin(cc)]
        var_type = cc_nodes[cc_nodes["_label"] == "TYPE"].name.item()
        for idrow in cc_nodes[cc_nodes["_label"] == "IDENTIFIER"].itertuples():
            varnametypes += [[idrow.lineNumber, var_type, idrow.name]]
    nametypes = pd.DataFrame(varnametypes, columns=["lineNumber", "type", "name"])
    nametypes = nametypes.drop_duplicates().sort_values("lineNumber")
    nametypes.type = nametypes.type.apply(svdt.tokenise)
    nametypes.name = nametypes.name.apply(svdt.tokenise)
    nametypes["nametype"] = nametypes.type + " " + nametypes.name
    nametypes = nametypes.groupby("lineNumber").agg({"nametype": lambda x: " ".join(x)})
    nametypes = nametypes.to_dict()["nametype"]

    # 4/5. Data dependency / Control dependency context
    # Group nodes into statements
    nodesline = nodes[nodes.lineNumber != ""].copy()
    nodesline.lineNumber = nodesline.lineNumber.astype(int)
    nodesline = (
        nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    edgesline = edges.copy()
    edgesline.innode = edgesline.line_in
    edgesline.outnode = edgesline.line_out
    nodesline.id = nodesline.lineNumber
    edgesline = svdj.rdg(edgesline, "pdg")
    nodesline = svdj.drop_lone_nodes(nodesline, edgesline)
    # Drop duplicate edges
    edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])
    # REACHING DEF to DDG
    edgesline["etype"] = edgesline.apply(
        lambda x: "DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
    )
    edgesline = edgesline[edgesline.innode.apply(lambda x: isinstance(x, float))]
    edgesline = edgesline[edgesline.outnode.apply(lambda x: isinstance(x, float))]
    edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
    edgesline_reverse.columns = ["outnode", "innode", "etype"]
    uedge = pd.concat([edgesline, edgesline_reverse])
    uedge = uedge[uedge.innode != uedge.outnode]
    uedge = uedge.groupby(["innode", "etype"]).agg({"outnode": set})
    uedge = uedge.reset_index()
    if len(uedge) > 0:
        uedge = uedge.pivot("innode", "etype", "outnode")
        if "DDG" not in uedge.columns:
            uedge["DDG"] = None
        if "CDG" not in uedge.columns:
            uedge["CDG"] = None
        uedge = uedge.reset_index()[["innode", "CDG", "DDG"]]
        uedge.columns = ["lineNumber", "control", "data"]
        uedge.control = uedge.control.apply(
            lambda x: list(x) if isinstance(x, set) else []
        )
        uedge.data = uedge.data.apply(lambda x: list(x) if isinstance(x, set) else [])
        data = uedge.set_index("lineNumber").to_dict()["data"]
        control = uedge.set_index("lineNumber").to_dict()["control"]
    else:
        data = {}
        control = {}

    # Generate PDG
    pdg_nodes = nodesline.copy()
    pdg_nodes = pdg_nodes[["id"]].sort_values("id")
    pdg_nodes["subseq"] = pdg_nodes.id.map(subseq).fillna("")
    pdg_nodes["ast"] = pdg_nodes.id.map(ast).fillna("")
    pdg_nodes["nametypes"] = pdg_nodes.id.map(nametypes).fillna("")
    pdg_nodes["data"] = pdg_nodes.id.map(data)
    pdg_nodes["control"] = pdg_nodes.id.map(control)
    pdg_edges = edgesline.copy()
    pdg_nodes = pdg_nodes.reset_index(drop=True).reset_index()
    pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
    pdg_edges.innode = pdg_edges.innode.map(pdg_dict)
    pdg_edges.outnode = pdg_edges.outnode.map(pdg_dict)
    pdg_edges = pdg_edges.dropna()
    pdg_edges = (pdg_edges.outnode.tolist(), pdg_edges.innode.tolist())

    # Cache
    with open(cachefp, "wb") as f:
        pkl.dump([pdg_nodes, pdg_edges], f)
    return pdg_nodes, pdg_edges


class GruWrapper(nn.Module):
    """Get last state from GRU."""

    def __init__(
        self, input_size, hidden_size, num_layers, dropout, bidirectional=False
    ):
        """Initilisation."""
        super(GruWrapper, self).__init__()
        self.gru = dl.DynamicRNN(
            nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True,
            )
        )

    def forward(self, x, x_lens, return_sequence=False):
        """Forward pass."""
        # Load data from disk on CPU
        out, hidden = self.gru(x, x_lens)
        if return_sequence:
            return out, hidden
        out = out[range(out.shape[0]), x_lens - 1, :]
        return out, hidden


class IVDetect(nn.Module):
    """IVDetect model."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        """Initilisation."""
        super(IVDetect, self).__init__()
        self.gru = GruWrapper(input_size, hidden_size, num_layers, dropout=0)
        self.gru2 = GruWrapper(input_size, hidden_size, num_layers, dropout=0)
        self.gru3 = GruWrapper(input_size, hidden_size, num_layers, dropout=0)
        self.gru4 = GruWrapper(input_size, hidden_size, num_layers, dropout=0)
        self.bigru = GruWrapper(
            hidden_size, hidden_size, num_layers, dropout=0, bidirectional=True
        )
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.treelstm = ivdts.TreeLSTM(input_size, hidden_size, dropout=0)
        self.gcn1 = GraphConv(hidden_size * 2, hidden_size * 2)
        self.gcn2 = GraphConv(hidden_size * 2, 2)
        self.h_size = hidden_size
        self.pool = ivdp.SpatialPyramidPooling([16])
        self.fc1 = nn.Linear(256, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, 2, bias=True)
        self.att = nn.MultiheadAttention(
            hidden_size * 2, 1, dropout=0.0, batch_first=True
        )
        self.dropout = dropout

    def forward(self, g, dataset, e_weights=[]):
        """Forward pass.

        DEBUG:
        import sastvd.helpers.graphs as svdgr
        from importlib import reload
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = BigVulGraphDataset(partition="train", sample=10)
        g = dgl.batch([dataset[0], dataset[1]]).to(dev)

        input_size = 200
        hidden_size = 200
        num_layers = 2

        reload(ivdts)
        model = IVDetect(200, 200, 2).to(dev)
        ret = model(g, dataset)

        """
        # Load data from disk on CPU
        nodes = list(
            zip(
                g.ndata["_SAMPLE"].detach().cpu().int().numpy(),
                g.ndata["_LINE"].detach().cpu().int().numpy(),
            )
        )
        data = dict()
        asts = []
        for sampleid in set([n[0] for n in nodes]):
            datasetitem = dataset.item(sampleid)
            for row in datasetitem["df"].to_dict(orient="records"):
                data[(sampleid, row["id"])] = row
            asts += datasetitem["asts"]
        asts = [i for i in asts if i is not None]
        asts = dgl.batch(asts).to(self.dev)

        feat = defaultdict(list)
        for n in nodes:
            f1 = torch.Tensor(data[n]["subseq"])
            f1 = f1 if f1.shape[0] > 0 else torch.zeros(1, 200)
            f1_lens = len(f1)
            feat["f1"].append(f1)
            feat["f1_lens"].append(f1_lens)

            f3 = torch.Tensor(data[n]["nametypes"])
            f3 = f3 if f3.shape[0] > 0 else torch.zeros(1, 200)
            f3_lens = len(f3)
            feat["f3"].append(f3)
            feat["f3_lens"].append(f3_lens)

        # Pass through GRU / TreeLSTM
        F1, _ = self.gru(
            pad_sequence(feat["f1"], batch_first=True).to(self.dev),
            torch.Tensor(feat["f1_lens"]).long(),
        )
        F2 = self.treelstm(asts)
        F3, _ = self.gru2(
            pad_sequence(feat["f3"], batch_first=True).to(self.dev),
            torch.Tensor(feat["f3_lens"]).long(),
        )
        F4, _ = self.gru3(
            pad_sequence(feat["f1"], batch_first=True).to(self.dev),
            torch.Tensor(feat["f1_lens"]).long(),
        )
        F5, _ = self.gru4(
            pad_sequence(feat["f1"], batch_first=True).to(self.dev),
            torch.Tensor(feat["f1_lens"]).long(),
        )

        # Fill null values (e.g. line has no AST representation / datacontrol deps)
        # BUG: POTENTIAL MEMORY LEAK
        F2 = torch.stack(
            [F2[i] if i in F2 else torch.zeros(self.h_size).to(self.dev) for i in nodes]
        )

        # Group together feature vectors for every statement, including data/control dep
        # BUG: POTENTIAL MEMORY LEAK
        batched_feat_vecs = []
        node_dict = {k: v for v, k in enumerate(nodes)}
        for idx, n in enumerate(nodes):
            sampleid, _ = n
            statement_feat_vecs = []
            statement_feat_vecs.append(F1[idx])
            statement_feat_vecs.append(F2[idx])
            statement_feat_vecs.append(F3[idx])

            # if isinstance(data[n]["data"], list):
            #     for d in data[n]["data"][:5]:
            #         f4_idx = node_dict[(sampleid, d)]
            #         statement_feat_vecs.append(F4[f4_idx])

            # if isinstance(data[n]["control"], list):
            #     for d in data[n]["control"][:5]:
            #         f5_idx = node_dict[(sampleid, d)]
            #         statement_feat_vecs.append(F5[f5_idx])

            batched_feat_vecs.append(torch.stack(statement_feat_vecs))

        batched_feat_lens = torch.Tensor([i.shape[0] for i in batched_feat_vecs]).long()
        batched_feat_vecs = pad_sequence(batched_feat_vecs, batch_first=True)

        # BiGru Aggregation
        bigru_out, hidden = self.bigru(batched_feat_vecs, batched_feat_lens, True)

        # Add attention based on hidden state TODO: How is "hidden" incorporated?
        # hidden = torch.cat([hidden[-2], hidden[-1]], dim=1).unsqueeze(1)
        _, Wi = self.att(
            bigru_out, bigru_out, bigru_out
        )  # TODO: Add hidden to the outs
        Fi_prime = torch.bmm(Wi, bigru_out)

        # :TODO: SUM OUTPUTS -> PAPER EQUATION 1
        Fi_prime = Fi_prime.sum(dim=1)

        # Assign node features to graph
        g.ndata["_FEAT"] = Fi_prime

        # Pool graph outputs
        h = self.gcn1(g, g.ndata["_FEAT"])
        h = F.relu(h)

        # Dropout
        h = F.dropout(h, self.dropout)

        # Unbatch and pool
        g.ndata["h"] = h

        # Edge masking
        if len(e_weights) > 0:
            g.edata["ew"] = e_weights
            g.update_all(fn.u_mul_e("h", "ew", "m"), fn.mean("m", "h"))

        method_rep_matrices = [i.ndata["h"] for i in dgl.unbatch(g)]

        # Pool and classify
        out = [
            self.pool(h.unsqueeze(0).unsqueeze(0)).squeeze()
            for h in method_rep_matrices
        ]
        out = torch.stack(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out


class BigVulGraphDataset:
    """Represent BigVul as graph dataset."""

    def __init__(self, partition="train", sample=-1):
        """Init class."""
        # Get finished samples
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(svd.processed_dir() / "bigvul/before/*nodes*"))
        ]
        self.df = svdd.bigvul()
        self.df = self.df[self.df.label == partition]
        self.df = self.df[self.df.id.isin(self.finished)]

        # Balance training set
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
            self.df = pd.concat([vul, nonvul])

        # Small sample (for debugging):
        if sample > 0:
            self.df = self.df.sample(sample, random_state=0)

        # Filter out samples with no lineNumber from Joern output
        self.df["valid"] = svd.dfmp(
            self.df, BigVulGraphDataset.check_validity, "id", desc="Validate Samples: "
        )
        self.df = self.df[self.df.valid]

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

        # Load Glove vectors.
        glove_path = svd.processed_dir() / "bigvul/glove_False/vectors.txt"
        self.emb_dict, _ = svdg.glove_dict(glove_path)

    def itempath(_id):
        """Get itempath path from item id."""
        return svd.processed_dir() / f"bigvul/before/{_id}.c"

    def check_validity(_id):
        """Check whether sample with id=_id has node/edges.

        Example:
        _id = 1320
        with open(str(svd.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
            nodes = json.load(f)
        """
        valid = 0
        with open(str(BigVulGraphDataset.itempath(_id)) + ".nodes.json", "r") as f:
            nodes = json.load(f)
            lineNums = set()
            for n in nodes:
                if "lineNumber" in n.keys():
                    lineNums.add(n["lineNumber"])
                    if len(lineNums) > 1:
                        valid = 1
                        break
            if valid == 0:
                return False
        with open(str(BigVulGraphDataset.itempath(_id)) + ".edges.json", "r") as f:
            edges = json.load(f)
            edge_set = set([i[2] for i in edges])
            if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                return False
            return True

    def get_vuln_indices(self, _id):
        """Obtain vulnerable lines from sample ID."""
        df = self.df[self.df.id == _id]
        removed = df.removed.item()
        return dict([(i, 1) for i in removed])

    def _feat_ext_itempath(_id):
        """Run feature extraction with itempath."""
        feature_extraction(BigVulGraphDataset.itempath(_id))

    def cache_features(self):
        """Save features to disk as cache."""
        svd.dfmp(
            self.df,
            BigVulGraphDataset._feat_ext_itempath,
            "id",
            ordr=False,
            desc="Cache features: ",
        )

    def __getitem__(self, idx):
        """Override getitem."""
        _id = self.idx2id[idx]
        n, e = feature_extraction(BigVulGraphDataset.itempath(_id))
        n["vuln"] = n.id.map(self.get_vuln_indices(_id)).fillna(0)
        g = dgl.graph(e)
        g.ndata["_LINE"] = torch.Tensor(n["id"].astype(int).to_numpy())
        g.ndata["_VULN"] = torch.Tensor(n["vuln"].astype(int).to_numpy())
        g.ndata["_SAMPLE"] = torch.Tensor([_id] * len(n))
        g = dgl.add_self_loop(g)
        return g

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def item(self, _id):
        """Get item data."""
        n, _ = feature_extraction(BigVulGraphDataset.itempath(_id))
        n.subseq = n.subseq.apply(lambda x: svdg.get_embeddings(x, self.emb_dict, 200))
        n.nametypes = n.nametypes.apply(
            lambda x: svdg.get_embeddings(x, self.emb_dict, 200)
        )

        asts = []

        def ast_dgl(row, lineid):
            if len(row) == 0:
                return None
            outnode, innode, ndata = row
            g = dgl.graph((outnode, innode))
            g.ndata["_FEAT"] = torch.Tensor(
                svdg.get_embeddings_list(ndata, self.emb_dict, 200)
            )
            g.ndata["_ID"] = torch.Tensor([_id] * g.number_of_nodes())
            g.ndata["_LINE"] = torch.Tensor([lineid] * g.number_of_nodes())
            return g

        for row in n.itertuples():
            asts.append(ast_dgl(row.ast, row.id))

        return {"df": n, "asts": asts}

    def stats(self):
        """Print dataset stats."""
        print(self.df.groupby(["label", "vul"]).count()[["id"]])
