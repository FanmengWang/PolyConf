# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import tree
import math
import torch
import yaml
import math
import copy
import pickle
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from typing import Dict, Any, List
from unicore.modules import TransformerEncoder, TransformerDecoder
from functools import partial
import scipy.stats as stats
from scipy.spatial import distance_matrix
import functools as fn
from data import utils as du
from data import se3_diffuser
from .ipa_pytorch import IpaScore
from openfold.utils import rigid_utils as ru
from openfold.utils.rigid_utils import Rotation, Rigid
from utils.utils import get_mar_diff_model

logger = logging.getLogger(__name__)

@register_model("polyconf_phase2")
class PolyConfPhase2Model(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", 
            type=int, 
            metavar="L", 
            help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--dropout", 
            type=float, 
            metavar="D", 
            help="dropout probability"
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", 
            type=int, 
            help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--post-ln", 
            type=bool, 
            help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--whole-pyg-encoder-type",
            type=str,
            default="gin",
            choices=["gin", "gcn"],
        )
        parser.add_argument(
            "--whole-pyg-encoder-depth",
            type=int,
            metavar="H",
        )
        parser.add_argument(
            "--whole-pyg-encoder-embed-dim",
            type=int,
            metavar="H",
        )
        parser.add_argument(
            "--whole-pyg-encoder-output-dim",
            type=int,
            metavar="H",
        )
        parser.add_argument(
            "--whole-pyg-encoder-dropout",
            type=float,
            metavar="H",
        )
        parser.add_argument(
            "--whole-pyg-encoder-pool",
            type=str,
        )
        parser.add_argument(
            "--mar-encoder-embed-dim",
            type=int,
            metavar="H",
        )
        parser.add_argument(
            "--mar-encoder-depth",
            type=int,
            metavar="H",
        )
        parser.add_argument(
            "--mar-encoder-num_heads",
            type=int,
            metavar="H",
        )
        parser.add_argument(
            "--mar-decoder-embed-dim",
            type=int,
            metavar="H",
        )
        parser.add_argument(
            "--mar-decoder-depth",
            type=int,
            metavar="H",
        )
        parser.add_argument(
            "--mar-decoder-num_heads",
            type=int,
            metavar="H",
        )
        parser.add_argument(
            "--mar-encoder-dropout", 
            type=float, 
            metavar="D", 
            help="dropout probability"
        )
        parser.add_argument(
            "--mar-encoder-emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--mar-encoder-attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--mar-encoder-activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--mar-encoder-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--mar-decoder-dropout", 
            type=float, 
            metavar="D", 
            help="dropout probability"
        )
        parser.add_argument(
            "--mar-decoder-emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--mar-decoder-attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--mar-decoder-activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--mar-decoder-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--mar-max-seq-len",
            type=int,
            metavar="D",
        )
        parser.add_argument(
            "--mar-label-drop-prob",
            type=float,
            metavar="D",
        )
        parser.add_argument(
            "--mar_mask_ratio_lower_bound",
            type=float,
            metavar="D",
        )
        parser.add_argument(
            "--mar-num-ar-steps",
            type=int,
            metavar="D",
        )
        parser.add_argument(
            "--mar-cfg",
            type=float,
            default=1.0,
            metavar="D",
        )
        parser.add_argument(
            "--mar-cfg-schedule",
            type=str,
            default="linear",
            choices=["linear", "constant"],
        )
        parser.add_argument(
            "--mar-diff-in-node-features", 
            type=int, 
            default=45
        )
        parser.add_argument(
            "--mar-diff-in-edge-features", 
            type=int, 
            default=4
        )
        parser.add_argument(
            "--mar-diff-batch-mul", 
            type=int, 
            default=4
        )
        parser.add_argument(
            "--mar-diff-ns", 
            type=int, 
            default=32, 
            help="Number of hidden features per node of order 0"
        )
        parser.add_argument(
            "--mar-diff-nv", 
            type=int, 
            default=8, 
            help="Number of hidden features per node of orser >0"
        )
        parser.add_argument(
            "--mar-diff-sigma-embed-dim", 
            type=int, 
            default=32, 
            help="Dimension of sinusoidal embedding of sigma"
        )
        parser.add_argument(
            "--mar-diff-sigma-min", 
            type=float, 
            default=0.01*3.14, 
            help="Minimum sigma used for training"
        )
        parser.add_argument(
            "--mar-diff-sigma-max", 
            type=float, 
            default=3.14, 
            help="Maximum sigma used for training"
        )
        parser.add_argument(
            "--mar-diff-num-conv-layers", 
            type=int, 
            default=4, 
            help="Number of interaction layers"
        )
        parser.add_argument(
            "--mar-diff-max-radius", 
            type=float, 
            default=5.0, 
            help="Radius cutoff for geometric graph"
        )
        parser.add_argument(
            "--mar-diff-radius-embed-dim", 
            type=int, 
            default=50, 
            help="Dimension of embedding of distances"
        )
        parser.add_argument(
            "--mar-diff-scale-by-sigma", 
            action='store_true', 
            default=True, 
            help="Whether to normalise the score"
        )
        parser.add_argument(
            "--mar-diff-no-residual", 
            action='store_true', 
            default=False, 
            help="If set, it removes residual connection"
        )
        parser.add_argument(
            "--mar-diff-no-batch-norm", 
            action='store_true', 
            default=False, 
            help="If set, it removes the batch norm"
        )
        parser.add_argument(
            "--mar-diff-use-second-order-repr", 
            action='store_true', 
            default=False, 
            help="Whether to use only up to first order representations or also second"
        )
        parser.add_argument(
            "--mar-diff-pre-mmff", 
            action='store_true', 
            default=False, 
            help='Whether to run MMFF on the local structure conformer'
        )
        parser.add_argument(
            "--mar-diff-post-mmff", 
            action='store_true', 
            default=False, 
            help='Whether to run MMFF on the final generated structures'
        )
        parser.add_argument(
            "--mar-diff-no-random", 
            action='store_true', 
            default=False, 
            help='Whether avoid randomising the torsions of the seed conformer'
        )
        parser.add_argument(
            "--mar-diff-no-model", 
            action='store_true', 
            default=False, 
            help='Whether to return seed conformer without running model'
        )
        parser.add_argument(
            "--mar-diff-single-conf", 
            action='store_true', 
            default=False, 
            help='Whether to start from a single local structure'
        )
        parser.add_argument(
            "--mar-diff-inference-steps", 
            type=int, 
            default=20, 
            help='Number of denoising steps'
        )
        parser.add_argument(
            "--mar-diff-confs-per-mol", 
            type=int, 
            default=1, 
            help='If set for every molecule this number of conformers is generated'
        )
        parser.add_argument(
            "--mar-diff-ode", 
            action='store_true', 
            default=False, 
            help='Whether to run the probability flow ODE instead of the SDE'
        )
        parser.add_argument(
            "--mar-diff-likelihood", 
            choices=['full', 'hutch'], 
            default=None, 
            help='Technique to compute likelihood'
        )
        parser.add_argument(
            "--mar-diff-dump_pymol", 
            type=str, 
            default=None, 
            help="Whether to save .pdb file with denoising dynamics"
        )
        parser.add_argument(
            "--mar-diff-water", 
            action='store_true', 
            default=False, 
            help='Whether to compute xTB energy in water'
        )
        parser.add_argument(
            "--mar-diff-batch-size", 
            type=int, 
            default=32, 
            help='Number of conformers generated in parallel'
        )
        parser.add_argument(
            "--mar-diff-xtb", 
            type=str, 
            default=None, 
            help='If set, it indicates path to local xtb main directory'
        )
        parser.add_argument(
            "--mar-diff-no-energy", 
            action='store_true', 
            default=False, 
            help='If set skips computation of likelihood, energy etc'
        )
        parser.add_argument(
            "--mar-diff-pg-weight-log-0", 
            type=float, 
            default=None
        )
        parser.add_argument(
            "--mar-diff-pg-weight-log-1", 
            type=float, 
            default=None
        )
        parser.add_argument(
            "--mar-diff-pg-repulsive-weight-log-0", 
            type=float, 
            default=None
        )
        parser.add_argument(
            "--mar-diff-pg-repulsive-weight-log-1", 
            type=float, 
            default=None
        )
        parser.add_argument(
            "--mar-diff-pg-langevin-weight-log-0", 
            type=float, 
            default=None
        )
        parser.add_argument(
            "--mar-diff-pg-langevin-weight-log-1", 
            type=float, 
            default=None
        )
        parser.add_argument(
            "--mar-diff-pg-kernel-size-log-0", 
            type=float, 
            default=None
        )
        parser.add_argument(
            "--mar-diff-pg-kernel-size-log-1", 
            type=float, 
            default=None
        )
        parser.add_argument(
            "--mar-diff-pg-invariant", 
            type=bool, 
            default=False
        )
        
        parser.add_argument(
            "--mar-norm-layer",
            default=partial(nn.LayerNorm, eps=1e-6),
        )
        
       
    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self._num_updates = None
        self.dictionary = dictionary
        
        # --------------------------------------------------------------------------
        # Whole pyg encoder basic specifics
        if args.whole_pyg_encoder_type == 'gin':
            from .ginet_molclr import GINet
            self.whole_pyg_encoder = GINet(num_layer=args.whole_pyg_encoder_depth, emb_dim=args.whole_pyg_encoder_embed_dim, feat_dim=args.whole_pyg_encoder_output_dim, 
                                           drop_ratio=args.whole_pyg_encoder_dropout, pool=args.whole_pyg_encoder_pool)
        elif args.whole_pyg_encoder_type == 'gcn':
            from .gcn_molclr import GCN
            self.whole_pyg_encoder = GCN(num_layer=args.whole_pyg_encoder_depth, emb_dim=args.whole_pyg_encoder_embed_dim, feat_dim=args.whole_pyg_encoder_output_dim, 
                                           drop_ratio=args.whole_pyg_encoder_dropout, pool=args.whole_pyg_encoder_pool)
            
        # --------------------------------------------------------------------------
        # Repeat unit encoder basic specifics
        K = 128
        self.padding_idx = dictionary.pad()
        n_edge_type = len(dictionary) * len(dictionary)
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        
        self.repeat_unit_encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )
        
        # --------------------------------------------------------------------------
        # MAR basic specifics
        self.mar_fake_latent = nn.Parameter(torch.zeros(1, args.mar_encoder_embed_dim))
        self.mar_mask_ratio_generator = stats.truncnorm((args.mar_mask_ratio_lower_bound - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        
        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.mar_token_embed_dim = args.encoder_embed_dim + (args.whole_pyg_encoder_output_dim // 2)
        self.mar_z_proj = nn.Linear(self.mar_token_embed_dim, args.mar_encoder_embed_dim, bias=True)
        self.mar_class_embedding_proj = nn.Linear(768, args.mar_encoder_embed_dim, bias=True)
        self.mar_z_proj_ln = nn.LayerNorm(args.mar_encoder_embed_dim, eps=1e-6)
        self.mar_encoder_pos_embed_learned = RotaryPositionEmbedding(args.mar_encoder_embed_dim)
        
        self.mar_encoder = TransformerEncoder(
            encoder_layers=args.mar_encoder_depth,
            embed_dim=args.mar_encoder_embed_dim,
            ffn_embed_dim=(args.mar_encoder_embed_dim * 4),
            attention_heads=args.mar_encoder_num_heads,
            emb_dropout=args.mar_encoder_emb_dropout,
            dropout=args.mar_encoder_dropout,
            attention_dropout=args.mar_encoder_attention_dropout,
            activation_dropout=args.mar_encoder_activation_dropout,
            activation_fn=args.mar_encoder_activation_fn,
            max_seq_len=args.mar_max_seq_len,
        )
        self.mar_encoder_norm = args.mar_norm_layer(args.mar_encoder_embed_dim)
        
        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.mar_decoder_embed = nn.Linear(args.mar_encoder_embed_dim, args.mar_decoder_embed_dim, bias=True)
        self.mar_mask_token = nn.Parameter(torch.zeros(1, 1, args.mar_decoder_embed_dim))
        self.mar_decoder_pos_embed_learned = RotaryPositionEmbedding(args.mar_decoder_embed_dim)
        
        self.mar_decoder = TransformerDecoder(
            decoder_layers=args.mar_decoder_depth,
            embed_dim=args.mar_decoder_embed_dim,
            ffn_embed_dim=(args.mar_decoder_embed_dim * 4),
            attention_heads=args.mar_decoder_num_heads,
            emb_dropout=args.mar_decoder_emb_dropout,
            dropout=args.mar_decoder_dropout,
            attention_dropout=args.mar_decoder_attention_dropout,
            activation_dropout=args.mar_decoder_activation_dropout,
            activation_fn=args.mar_decoder_activation_fn,
            max_seq_len=args.mar_max_seq_len,
        )
        self.mar_decoder_norm = args.mar_norm_layer(args.mar_decoder_embed_dim)
        self.mar_diffusion_pos_embed_learned = RotaryPositionEmbedding(args.mar_decoder_embed_dim)
        
        # --------------------------------------------------------------------------
        # MAR diffusion specifics
        self.args.mar_diff_plus_hidden_dim = args.mar_decoder_embed_dim
        self.mar_diff_model = get_mar_diff_model(self.args)
        
        # --------------------------------------------------------------------------
        # Frame diffusion specifics
        with open(f'polyconf/config/base.yaml', 'r') as f:
            self.frame_conf = EasyDict(yaml.safe_load(f))
        self.frame_embedding_layer = Embedder(self.frame_conf.model, args.mar_encoder_embed_dim)
        self.frame_diffuser = se3_diffuser.SE3Diffuser(self.frame_conf.diffuser)
        self.frame_score_model = IpaScore(self.frame_conf.model, self.frame_diffuser)
        
        self.apply(init_bert_params)


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)


    def forward_mae_encoder(self, x, mask, padding_mask, class_embedding, buffer_size):
        x = self.mar_z_proj(x)
        class_embedding = self.mar_class_embedding_proj(class_embedding)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, buffer_size, embed_dim, device=x.device).to(x.dtype), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(bsz, buffer_size, device=x.device).to(mask.dtype), mask], dim=1)
        padding_mask_with_buffer = torch.cat([torch.zeros(bsz, buffer_size, device=x.device).to(padding_mask.dtype), padding_mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.args.mar_label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.mar_fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        x = self.mar_encoder_pos_embed_learned.rotate(x)
        x = self.mar_z_proj_ln(x)

        x = self.mar_encoder(x, padding_mask=torch.bitwise_or(mask_with_buffer.bool(), padding_mask_with_buffer.bool()))  # Don't consider mask or padding positions
        x = self.mar_encoder_norm(x)

        return x
    
                  
    def forward(
        self,
        psmi,
        psmi_rep,
        whole_pyg,
        key_point_index,
        repeat_unit_smi, 
        repeat_unit_actual_num,
        repeat_unit_torsion_pygs,
        src_tokens,
        src_edge_type,
        src_coord,
        src_distance,
        input_coord,
        tgt_coord,
        rigids_0,
        rigids_t,
        t,
        gt_trans_score,
        gt_rot_score,
        trans_score_scaling,
        rot_score_scaling,
        **kwargs
    ):  
        
        bsz = src_coord.shape[0]
        repeat_unit_num = src_coord.shape[1] 
        repeat_unit_atom_num = src_coord.shape[2]
        key_point_index = [x + 1 for x in key_point_index]
        
        # Whole pyg encoder
        whole_pyg_batch = whole_pyg.to(src_tokens.device)        
        garph_attr = self.whole_pyg_encoder(whole_pyg_batch).detach()
        graph_global_att = garph_attr.unsqueeze(1) # (bsz, d) -> (bsz, 1, d)
                
        # Repeat unit encoder
        assert src_tokens.shape[0] == bsz and src_tokens.shape[1] == repeat_unit_atom_num
        assert src_edge_type.shape[0] == bsz and src_edge_type.shape[1] == repeat_unit_atom_num and src_edge_type.shape[1] == repeat_unit_atom_num
        src_tokens = src_tokens.unsqueeze(1).repeat(1, repeat_unit_num, 1).reshape(-1, repeat_unit_atom_num)  # (bsz, n) -> (bsz, 1, n) -> (bsz, r_n, n) -> (bsz * r_n, n)
        src_edge_type = src_edge_type.unsqueeze(1).repeat(1, repeat_unit_num, 1, 1).reshape(-1, repeat_unit_atom_num, repeat_unit_atom_num)  # (bsz, n, n) -> (bsz, 1, n, n) -> (bsz, r_n, n, n) -> (bsz * r_n, n, n)
        src_coord = src_coord.reshape(-1, repeat_unit_atom_num, 3)  # (bsz, r_n, n, 3) -> (bsz * r_n, n, 3)     
        src_distance = src_distance.reshape(-1, repeat_unit_atom_num, repeat_unit_atom_num)  # (bsz, r_n, n, n) -> (bsz * r_n, n, n) 
        
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
                
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type) 
        encoder_rep, _ , _ , _, _ = self.repeat_unit_encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias) # (bsz * r_n, n, d)
        encoder_rep = encoder_rep.detach()

        # MAR encoder
        mar_x = torch.cat([encoder_rep[:, 0, :].reshape(bsz, -1, encoder_rep.shape[-1]), graph_global_att.repeat(1, repeat_unit_num, 1)], dim=-1)  # (bsz, r_n, d)      
        mar_mask = torch.zeros(bsz, repeat_unit_num, device=mar_x.device) # (bsz, r_n)            
        mar_class_embedding = psmi_rep.to(mar_x.dtype) # (bsz, 768)
        mar_buffer_size = max(math.floor(min(repeat_unit_actual_num) / 4), 1)                
        mar_padding_mask = torch.zeros((bsz, mar_x.shape[1]), dtype=torch.bool).to(src_tokens.device)   
        for idx, num in enumerate(repeat_unit_actual_num):
            mar_padding_mask[idx, num:] = True
        mar_x = self.forward_mae_encoder(mar_x, mar_mask, mar_padding_mask, mar_class_embedding, mar_buffer_size).detach()  # (bsz, mar_buffer_size + r_n, d)
         
        init_node_embed, init_edge_embed = self.frame_embedding_layer(
            seq_idx=torch.arange(1, repeat_unit_num + 1).unsqueeze(0).repeat(bsz, 1).to(src_tokens.device) ,
            t=t,
            mar_encoder_output=mar_x[:, mar_buffer_size:]
        )
        bb_mask = (~mar_padding_mask).type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]
        
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]
        
        frame_model_out = self.frame_score_model(node_embed, edge_embed, bb_mask, edge_mask, rigids_t, t, input_coord, key_point_index)

        diffuse_mask = bb_mask
        loss_mask = bb_mask * diffuse_mask
        batch_loss_mask = torch.any(bb_mask, dim=-1)
        
        pred_rot_score = frame_model_out['rot_score'] * diffuse_mask[..., None]
    
        # Rotation loss
        if self.frame_conf.experiment.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-6)

            pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-6)

            # Separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis)**2 * loss_mask[..., None]
            axis_loss = torch.sum(
                axis_loss, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)

            # Separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle)**2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            angle_loss *= self.frame_conf.experiment.rot_loss_weight
            angle_loss *= t > self.frame_conf.experiment.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        
        else:
            rot_mse = (gt_rot_score - pred_rot_score)**2 * loss_mask[..., None]
            rot_loss = torch.sum(
                rot_mse / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            rot_loss *= self.frame_conf.experiment.rot_loss_weight
            rot_loss *= t > self.frame_conf.experiment.rot_loss_t_threshold
        rot_loss *= int(self.frame_conf.diffuser.diffuse_rot)

        # Pairwise distance loss
        def rot_apply(frames, mats): 
            rots = frames._rots.get_rot_mats()
            mats = mats.unsqueeze(-1)  # (bsz, r_n, n, 3) -> (bsz, r_n, n, 3, 1)
            rots = rots.unsqueeze(-3)  # (bsz, r_n, 3, 3) -> (bsz, r_n, 1, 3, 3)
            rotated_mats = torch.matmul(rots, mats)  
            return rotated_mats.squeeze(-1)
        
        assert torch.equal(tgt_coord[:, :-1, key_point_index[3]], tgt_coord[:, 1:, key_point_index[0]]) 
        
        rotated_coord = rot_apply(frame_model_out['final_rigids'], input_coord)            
        trans_coord = torch.cumsum(
            torch.cat((  
            torch.zeros((bsz, 1, 3)).to(src_tokens.device),   
            (rotated_coord[:, :-1, key_point_index[3]] - rotated_coord[:, 1:, key_point_index[0]]).squeeze(-2)
            ), dim=1),
            dim=1  
        )  
        pred_coord = rotated_coord + trans_coord.unsqueeze(-2) 
        assert torch.allclose(pred_coord[:, :-1, key_point_index[3]], pred_coord[:, 1:, key_point_index[0]], atol=1e-3) 
        
        indices_to_keep = [i for i in range(1, tgt_coord.shape[2]-1) if (i != key_point_index[1] and i != key_point_index[3])]
        pred_coord = pred_coord[:, :, indices_to_keep]
        tgt_coord = tgt_coord[:, :, indices_to_keep]
        
        gt_flat_atoms = tgt_coord.reshape([bsz, -1, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_coord.reshape([bsz, -1, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)
        
        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, tgt_coord.shape[2]))
        flat_loss_mask = flat_loss_mask.reshape([bsz, -1])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, tgt_coord.shape[2]))
        flat_res_mask = flat_res_mask.reshape([bsz, -1])
        
        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]
        
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask  = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - repeat_unit_num)
        dist_mat_loss *= self.frame_conf.experiment.dist_mat_loss_weight
        dist_mat_loss *= t < self.frame_conf.experiment.dist_mat_loss_t_filter
        dist_mat_loss *= self.frame_conf.experiment.aux_loss_weight
        
        # final loss
        final_loss = (rot_loss + dist_mat_loss)

        def normalize_loss(x):
            return x.sum() /  (batch_loss_mask.sum() + 1e-10)
        
        return normalize_loss(final_loss), normalize_loss(rot_loss), normalize_loss(dist_mat_loss)
    
              
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates


    def get_num_updates(self):
        return self._num_updates


class RotaryPositionEmbedding:  
    def __init__(self, d_model, base=10000):  
        self.d_model = d_model  
        self.base = base  

    def rotate(self, x):  
        # x shape: (batch_size, seq_len, d_model)  
        seq_len = x.size(1)  

        # calculate position
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)  # (seq_len, 1)  
        frequencies = 1 / (self.base ** (torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device) / self.d_model))  
        angles = positions * frequencies.unsqueeze(0)  # (seq_len, d_model // 2)  

        # calculate sin and cos value  
        sin = angles.sin()  
        cos = angles.cos()  

        # extent sin and cos
        sin_cos = torch.zeros(seq_len, self.d_model, device=x.device)  
        sin_cos[:, 0::2] = sin  
        sin_cos[:, 1::2] = cos  
        sin_cos = sin_cos.unsqueeze(0)  # (1, seq_len, d_model)  
        
        # rotation
        x_rotated = (x * sin_cos) + (torch.cat([x[..., 1::2], x[..., ::2]], dim=-1) * sin_cos)  

        return x_rotated
    
    
class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


def get_index_embedding(indices, embed_size, max_len=2056):
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class Embedder(nn.Module):

    def __init__(self, model_conf, mar_encoder_embed_dim):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size
        edge_in = t_embed_size * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims + mar_encoder_embed_dim, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            seq_idx,
            t,
            mar_encoder_output,
        ):
        num_batch, num_res = seq_idx.shape
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        pair_feats.append(self.index_embedder(rel_seq_offset))

        node_feats.append(mar_encoder_output)

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        return node_embed, edge_embed
    
    
@register_model_architecture("polyconf_phase2", "polyconf_phase2")
def base_architecture(args):
    # Repeat unit encoder basic specifics
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    # Whole pyg encoder basic specifics
    args.whole_pyg_encoder_depth = getattr(args, "whole_pyg_encoder_depth", 5)
    args.whole_pyg_encoder_embed_dim = getattr(args, "whole_pyg_encoder_embed_dim", 300)
    args.whole_pyg_encoder_output_dim = getattr(args, "whole_pyg_encoder_output_dim", 512)
    args.whole_pyg_encoder_dropout = getattr(args, "whole_pyg_encoder_dropout", 0)
    args.whole_pyg_encoder_pool = getattr(args, "whole_pyg_encoder_pool", "mean")
    # MAR basic specifics
    args.mar_encoder_embed_dim = getattr(args, "mar_encoder_embed_dim", 768)
    args.mar_encoder_depth = getattr(args, "mar_encoder_depth", 6)
    args.mar_encoder_num_heads = getattr(args, "mar_encoder_num_heads", 12)
    args.mar_decoder_embed_dim = getattr(args, "mar_decoder_embed_dim", 768)
    args.mar_decoder_depth = getattr(args, "mar_decoder_depth", 6)
    args.mar_decoder_num_heads = getattr(args, "mar_decoder_num_heads", 12)
    args.mar_encoder_dropout = getattr(args, "mar_encoder_dropout", 0.1)
    args.mar_encoder_emb_dropout = getattr(args, "mar_encoder_emb_dropout", 0.1)
    args.mar_encoder_attention_dropout = getattr(args, "mar_encoder_attention_dropout", 0.1)
    args.mar_encoder_activation_dropout = getattr(args, "mar_encoder_activation_dropout", 0.0)
    args.mar_encoder_activation_fn = getattr(args, "mar_encoder_activation_fn", "gelu")
    args.mar_decoder_dropout = getattr(args, "mar_decoder_dropout", 0.1)
    args.mar_decoder_emb_dropout = getattr(args, "mar_decoder_emb_dropout", 0.1)
    args.mar_decoder_attention_dropout = getattr(args, "mar_decoder_attention_dropout", 0.1)
    args.mar_decoder_activation_dropout = getattr(args, "mar_decoder_activation_dropout", 0.0)
    args.mar_decoder_activation_fn = getattr(args, "mar_decoder_activation_fn", "gelu")
    args.mar_max_seq_len = getattr(args, "mar_max_seq_len", 1024)
    args.mar_label_drop_prob = getattr(args, "mar_label_drop_prob", 0.1)
    args.mar_mask_ratio_lower_bound = getattr(args, "mar_mask_ratio_lower_bound", 0.7)
