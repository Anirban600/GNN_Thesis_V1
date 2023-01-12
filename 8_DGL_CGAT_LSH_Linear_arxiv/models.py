import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import dgl
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import numpy as np


class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, "both", bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        graph_margin=0.1,
        class_margin=0.1,
        top_k=3,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._graph_margin = graph_margin
        self._class_margin = class_margin
        self._top_k = top_k
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, label):
        
        def adjacency_message(edges):
            
            """
            Compute binary message on edges.
            Compares whether source and destination nodes
            have the same or different labels.
            """

            l_src = edges.src['l']
            l_dst = edges.dst['l']

            if l_src.ndim > 1: adj = torch.all(l_src == l_dst, dim=1)
            else: adj = (l_src == l_dst)

            return {'adj': adj.detach()}
        

        def class_loss(nodes):
            
            """
            Loss function on class boundaries.
            
            Enforces high attention to adjacent nodes with the same label
            and lower attention to adjacent nodes with different labels.
            """

            m = nodes.mailbox['m']

            w = m[:, :, :-1]
            adj = m[:, :, -1].unsqueeze(-1).bool() # This 3D matrix will contain if i is a neigh of j
            same_class = w.masked_fill(adj == 0, np.nan).unsqueeze(2)
            diff_class = w.masked_fill(adj == 1, np.nan).unsqueeze(1)
            difference = (diff_class + self._class_margin - same_class).clamp(0)
            loss = torch.nansum(torch.nansum(difference, 1), 1)
            return {'boundary_loss': loss}

        
        def graph_loss(nodes):
            
            """
            Loss function on graph structure.
            
            Enforces high attention to adjacent nodes and 
            lower attention to distant nodes via negative sampling.
            """

            msg = nodes.mailbox['m']

            pw = msg[:, :, :, 0, :].unsqueeze(1)
            nw = msg[:, :, :, 1, :].unsqueeze(2)

            loss = (nw + self._graph_margin - pw).clamp(0)
            loss = loss.sum(1).sum(1)

            return {'graph_loss': loss}


        def construct_negative_graph(graph, k=1):
            """
            Reshuffle the edges of a graph.
            Parameters:
            - - - - -
            graph: DGL graph object
                input graph to reshuffle
            k: int
                number of edge pairs to generate
                if original graph has E edges, new graph will 
                have <= k*E edges
            
            Returns:
            - - - -
            neg_graph: DGL graph object
                reshuffled graph
            """

            src, dst = graph.edges()

            neg_src = src.repeat_interleave(k)
            neg_dst = torch.stack([dst[torch.randperm(graph.num_edges())]
                                for i in range(k)], dim=0)
            neg_dst = neg_dst.view(-1)

            neg_graph = dgl.graph((neg_src, neg_dst),
                                num_nodes=graph.number_of_nodes())

            return neg_graph


        def topk_reduce_func(nodes):
    
            """
            Aggregate attention-weighted messages over the top-K 
            attention-valued destination nodes
            """

            K = self._top_k

            m = nodes.mailbox['m']
            [m,_] = torch.sort(m, dim=1, descending=True)
            m = m[:,:K,:,:].sum(1)

            return {'ft': m}
        
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            # compute graph structure loss
            Lg = torch.tensor(0)
            if self._graph_margin is not None:
                neg_graph = [construct_negative_graph(i, k=1) for i in dgl.unbatch(graph)]
                neg_graph = dgl.batch(neg_graph)

                neg_graph.srcdata.update({'ft': feat_src, 'el': el})
                neg_graph.dstdata.update({'er': er})
                neg_graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                ne = self.leaky_relu(neg_graph.edata.pop('e'))

                combined = torch.stack([e, ne]).transpose(0, 1).transpose(1, 2)
                graph.edata['combined'] = combined
                graph.update_all(fn.copy_e('combined', 'm'), graph_loss)
                Lg = graph.ndata['graph_loss'].sum() / (graph.num_nodes() * self._num_heads)
            
            # compute class boundary loss
            Lb = torch.tensor(0)
            if self._class_margin is not None:
                graph.ndata['l'] = label
                graph.apply_edges(adjacency_message)
                adj = graph.edata.pop('adj').float()

                combined = torch.cat([e.squeeze(2), adj.unsqueeze(-1)], dim=1)
                graph.edata['combined'] = combined
                graph.update_all(fn.copy_e('combined', 'm'), class_loss)
                Lb = graph.ndata['boundary_loss'].sum() / (graph.num_nodes() * self._num_heads)

                # remove edge data to release memory
                graph.edata.pop('combined');

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)

            return rst, Lg, Lb


class GAT(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        n_heads,
        graph_margin,
        class_margin,
        top_k,
        activation,
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        use_attn_dst=True,
        use_symmetric_norm=False,
        stacking=False,
        types=None,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads
        self.graph_margin = graph_margin
        self.class_margin = class_margin
        self.top_k = top_k
        self.stacking = stacking
        self.types = types

        self.convs = nn.ModuleList()
        self.stacks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            if not i: in_hidden = in_feats
            elif self.stacking and self.types[i] == "cat": in_hidden = n_heads * n_hidden * 2
            else: in_hidden = n_heads * n_hidden
            out_hidden = n_hidden if i < n_layers - 1 else n_heads * n_hidden
            num_heads = n_heads if i < n_layers - 1 else 1
 
            self.convs.append(
                GATConv(
                    in_hidden,
                    out_hidden,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    graph_margin=graph_margin,
                    class_margin=class_margin,
                    top_k=top_k,
                    use_attn_dst=use_attn_dst,
                    use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                )
            )
            if self.stacking:
                x = 2 if self.types[i + 1] == "cat" else 1
                self.norms.append(nn.BatchNorm1d(num_heads * out_hidden * x))
            else:
                self.norms.append(nn.BatchNorm1d(num_heads * out_hidden))
            
            self.stacks.append(
                GATConv(
                    in_hidden,
                    out_hidden,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    graph_margin=graph_margin,
                    class_margin=class_margin,
                    top_k=top_k,
                    use_attn_dst=use_attn_dst,
                    use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                )
            )


        if self.stacking and self.types[-1] == "cat": in_hidden = n_heads * n_hidden * 2
        else: in_hidden = n_heads * n_hidden
        self.linear = nn.Linear(in_hidden, n_classes)
        self.bias_last = ElementWiseLinear(in_hidden, weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat, feat_graph=None, labels=None):
        h = feat
        h = self.input_drop(h)
        
        Lg_all = Lb_all = 0

        if self.stacking:
            for i in range(self.n_layers):
                conv = self.convs[i](graph, h)
                conv = conv.flatten(1)
                aug = self.stacks[i](feat_graph, h)
                aug = aug.flatten(1)

                if self.types[i + 1] == "avg": h = (conv + aug) / 2
                else: h = torch.cat((conv, aug), 1)

                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)
        
        else:
            for i in range(self.n_layers):
                conv, lg, lb = self.convs[i](graph, h, labels)

                h = conv
                Lg_all += lg
                Lb_all += lb

                h = h.flatten(1)
                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        h = self.bias_last(h)
        h_final = self.linear(h)

        return h_final, h, Lg_all, Lb_all
