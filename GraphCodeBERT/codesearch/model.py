# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import scipy
import torch.nn as nn
import torch
import dgl

from transformers import RobertaConfig, RobertaModel
def build_graphs_from_batched_adjacency_tensor(batched_adj_tensor, device):
    graphs = []
    for adj_tensor in batched_adj_tensor:
        # Extract edges from the adjacency matrix
        src, dst = torch.nonzero(adj_tensor, as_tuple=True)
        # Create a graph for the current adjacency matrix
        graph = dgl.graph((src, dst)).to(device)
        graphs.append(graph)
    return graphs


def get_reinitialized_roberta(config=None, positional_size = 0, additional_hidden_space=0):
    # Create a configuration object for RobertaModel
    config = RobertaConfig(
        vocab_size=50265,  # Number of words in the token vocabulary
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=1,
        hidden_size=768 + additional_hidden_space - positional_size,
        intermediate_size=3072,  # Typically 4 times the hidden_size
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
    ) if config is None else config
    # Instantiate a new Roberta model using the configuration
    return RobertaModel(config)


import torch
from torch import Tensor
# from torch_scatter import scatter


def add_random_walk_pe_batch(adj_batch: torch.Tensor, walk_length: int) -> torch.Tensor:
    """
    Adds the random walk positional encoding to a batch of adjacency matrices.

    Args:
        adj_batch (torch.Tensor): The batch of adjacency matrices (batch_size x N x N).
        walk_length (int): The number of random walk steps.

    Returns:
        torch.Tensor: The positional encoding matrix for each graph in the batch (batch_size x N x W),
                where W is the walk length and N is the number of nodes.
    """
    batch_size, N, _ = adj_batch.shape
    device = adj_batch.device

    # Normalizing each adjacency matrix in the batch to make them transition matrices
    row_sums = adj_batch.sum(dim=2, keepdim=True).clamp(min=1)  # Avoid division by zero
    transition_matrices = adj_batch / row_sums

    # Initialize the positional encoding list with the identity matrix for each graph in the batch
    pe_list = [torch.eye(N, device=device).unsqueeze(0).repeat(batch_size, 1, 1)]

    # Perform random walks
    out = transition_matrices
    for _ in range(1, walk_length):
        pe_list.append(out)
        out = torch.bmm(out, transition_matrices)  # Batch matrix multiplication for the next step in the walk

    # Stack along new dimension creating the positional encoding matrix for each graph
    pe = torch.stack(pe_list, dim=-1)

    return pe


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.bert = encoder
        self.bert.pooler = None
        self.linear_layer = torch.nn.Linear(768, 50265)
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None):
        if code_inputs is not None:
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.bert.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            encoded= self.bert(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]
            logit = self.linear_layer(encoded)
            return logit


class ModelPositional(nn.Module):
    def __init__(self, encoder, tokenizer, positional_size, extra_positional_size, positional_encoding_type='random_walk'):
        super(ModelPositional, self).__init__()
        assert positional_encoding_type in ['random_walk', 'laplacian']
        self.bert = encoder
        self.bert.pooler = None
        self.linear_layer = torch.nn.Linear(768+extra_positional_size, 50265)
        self.custom_word_embeddings = nn.Embedding(len(tokenizer), 768 - positional_size + extra_positional_size)
        self.positional_size = positional_size
        self.positional_encoding_type = positional_encoding_type

    def forward(self, code_inputs=None, attn_mask=None, position_idx=None):
        if code_inputs is not None:
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            inputs_embeddings = self.custom_word_embeddings(code_inputs)
            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

            # positional_encodings = add_random_walk_pe_batch(attn_mask, walk_length=5)
            adj_matrix = attn_mask.clone().detach()
            for a in range(adj_matrix.shape[1]):
                adj_matrix[:, a, a] = 1
            graphs = build_graphs_from_batched_adjacency_tensor(adj_matrix, inputs_embeddings.device)
            if self.positional_encoding_type == 'laplacian':
                try:
                    positional_encodings = torch.stack([dgl.lap_pe(g, k=self.positional_size, padding=True) for g in graphs]).to(inputs_embeddings.device)
                except:
                    # For some "pathalogical" graphs, the laplacian matrix may not be diagonalizable.
                    # In this case, we fall back to random walk positional encoding. See more here:
                    # https://github.com/scipy/scipy/issues/9185
                    print("Laplacian matrix is not diagonalizable. Falling back to random walk positional encoding.")
                    positional_encodings = torch.stack(
                        [dgl.random_walk_pe(g, k=self.positional_size) for g in graphs]).to(inputs_embeddings.device)
            else:
                positional_encodings = torch.stack([dgl.random_walk_pe(g, k=self.positional_size) for g in graphs]).to(inputs_embeddings.device)

            inputs_embeddings = torch.cat([inputs_embeddings, positional_encodings], dim=-1)
            encoded = self.bert(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0]
            logit = self.linear_layer(encoded)
            return logit


class ModelMidAttention(nn.Module):

    def __init__(self, config):
        super(ModelMidAttention, self).__init__()
        # create a copy of the config called config1
        config1 = config
        config1.num_hidden_layers = 6
        self.bert1 = RobertaModel(config1)
        self.bert2 = RobertaModel(config1)
        self.bert1.pooler = None
        self.bert2.pooler = None
        self.linear_layer = torch.nn.Linear(768, 50265)

    def forward(self, code_inputs=None, attn_mask=None, position_idx=None):
        if code_inputs is not None:
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            inputs_embeddings = self.bert1.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
            encoded = self.bert1(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0]
            updated_positions = torch.arange(0,position_idx.shape[1]).repeat(position_idx.shape[0], 1).to(position_idx.device)
            encoded = self.bert2(inputs_embeds=encoded, position_ids=updated_positions)[0]
            logit = self.linear_layer(encoded)
            return logit

      
        
 
