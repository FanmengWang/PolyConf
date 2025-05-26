# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import tree
import math
import torch
import yaml
import math
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
import functools as fn
import scipy.stats as stats
from scipy.spatial import distance_matrix
from utils.utils import get_mar_diff_model
import diffusion.torus as torus
from diffusion.sampling import *
from data import utils as du
from data import se3_diffuser
from .ipa_pytorch import IpaScore
from openfold.utils import rigid_utils as ru
from openfold.utils.rigid_utils import Rotation, Rigid


logger = logging.getLogger(__name__)


@register_model("polyconf_inference")
class PolyConfInferenceModel(BaseUnicoreModel):
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


    def sample_orders(self, bsz, seq_len, repeat_unit_actual_num):
        orders = []
        for idx in range(bsz):
            order = np.array(list(range(seq_len)))
            np.random.shuffle(order[:repeat_unit_actual_num[idx]])
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders


    def random_masking(self, x, orders, repeat_unit_actual_num):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mar_mask_ratio_generator.rvs(1)[0]
        mask = torch.zeros(bsz, seq_len, device=x.device)
        for idx in range(bsz):
            num_masked_tokens = int(np.ceil(int(repeat_unit_actual_num[idx]) * mask_rate))
            mask[idx] = torch.scatter(mask[idx], dim=-1, index=orders[idx][:num_masked_tokens],
                                src=torch.ones(seq_len, device=x.device))
        return mask


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
    
    
    def forward_mae_decoder(self, x, mask, padding_mask, buffer_size):
        x = self.mar_decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), buffer_size, device=x.device).to(mask.dtype), mask], dim=1)
        padding_mask_with_buffer = torch.cat([torch.zeros(x.size(0), buffer_size, device=x.device).to(padding_mask.dtype), padding_mask], dim=1)
        
        # pad mask tokens
        mask_tokens = self.mar_mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        ## recover the informatiom form unmask and unpadding positions 
        x_after_pad[(~torch.bitwise_or(mask_with_buffer.bool(), padding_mask_with_buffer.bool())).nonzero(as_tuple=True)] = x[(~torch.bitwise_or(mask_with_buffer.bool(), padding_mask_with_buffer.bool())).nonzero(as_tuple=True)].reshape(-1, x.shape[2])

        # decoder position embedding
        x = self.mar_decoder_pos_embed_learned.rotate(x_after_pad)

        # apply Transformer blocks
        x = self.mar_decoder(x, padding_mask=padding_mask_with_buffer) # Don't consider padding positions
        x = self.mar_decoder_norm(x)

        x = x[:, buffer_size:]
        x = self.mar_diffusion_pos_embed_learned.rotate(x)
        return x
    
    
    def mask_by_order(self, mask_lens, order, bsz, seq_len):  
        masking = torch.zeros(bsz, seq_len).cuda()  
        for i in range(bsz):  
            mask_len = mask_lens[i]
            masking[i] = torch.scatter(masking[i], dim=-1, index=order[i, :mask_len.long()], src=torch.ones(seq_len).cuda())
        return masking.bool() 


    def embed_func(self, mol, numConfs):
        AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=5)
        return mol


    def sample_confs(self, smi_idx, raw_smi, n_confs, smi, mar_z):
        mol, data = get_seed(smi)
        if not mol:
            print('Failed to get seed', smi)
            return None

        n_rotable_bonds = int(data.edge_mask.sum())

        conformers, pdb = embed_seeds(mol, data, n_confs, single_conf=self.args.mar_diff_single_conf,
                                        pdb=self.args.mar_diff_dump_pymol, embed_func=self.embed_func, mmff=self.args.mar_diff_pre_mmff)
        if not conformers:
            print("Failed to embed", smi)
            return None

        if not self.args.mar_diff_no_random and n_rotable_bonds > 0.5:
            conformers = perturb_seeds(conformers, pdb)

        if not self.args.mar_diff_no_model and n_rotable_bonds > 0.5:
            conformers = sample(conformers, self.mar_diff_model, self.args.mar_diff_sigma_max, self.args.mar_diff_sigma_min, self.args.mar_diff_inference_steps,
                                self.args.mar_diff_batch_size, self.args.mar_diff_ode, self.args.mar_diff_likelihood, pdb,
                                pg_weight_log_0=self.args.mar_diff_pg_weight_log_0, 
                                pg_weight_log_1=self.args.mar_diff_pg_weight_log_1,
                                pg_repulsive_weight_log_0=self.args.mar_diff_pg_repulsive_weight_log_0,
                                pg_repulsive_weight_log_1=self.args.mar_diff_pg_repulsive_weight_log_1,
                                pg_kernel_size_log_0=self.args.mar_diff_pg_kernel_size_log_0,
                                pg_kernel_size_log_1=self.args.mar_diff_pg_kernel_size_log_1,
                                pg_langevin_weight_log_0=self.args.mar_diff_pg_langevin_weight_log_0,
                                pg_langevin_weight_log_1=self.args.mar_diff_pg_langevin_weight_log_1,
                                pg_invariant=self.args.mar_diff_pg_invariant, 
                                mol=mol,
                                mar_z=mar_z)

        if self.args.mar_diff_dump_pymol:
            if not os.path.isdir(self.args.mar_diff_dump_pymol):
                os.mkdir(self.args.mar_diff_dump_pymol)
            pdb.write(f'{self.args.mar_diff_dump_pymol}/{raw_smi}_{smi_idx}.pdb', limit_parts=5)

        mols = [pyg_to_mol(mol, conf, self.args.mar_diff_post_mmff, rmsd=not self.args.mar_diff_no_energy) for conf in conformers]
        if self.args.mar_diff_likelihood:
            if n_rotable_bonds < 0.5:
                print(f"Skipping mol {smi} with 0 rotable bonds")
                return None
        for mol, data in zip(mols, conformers):
            populate_likelihood(mol, data, water=self.args.mar_diff_water, xtb=self.args.mar_diff_xtb)

        if self.args.mar_diff_xtb:
            mols = [mol for mol in mols if mol.xtb_energy]
        return mols


    def encoder_mar_diff_output(self, mol, device):
        mol = Chem.RemoveHs(mol)
        
        # original input
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coordinates = mol.GetConformer().GetPositions()
        assert 'H' not in atoms
        
        # coord normalize
        coordinates = torch.from_numpy((coordinates - coordinates.mean(axis=0, keepdims=True)).astype(np.float32))
        
        # extract encoder input
        src_tokens = torch.from_numpy(self.dictionary.vec_index(atoms)).long()
        src_tokens = torch.cat([torch.full_like(src_tokens[0], self.dictionary.bos()).unsqueeze(0), src_tokens, torch.full_like(src_tokens[0], self.dictionary.eos()).unsqueeze(0)], dim=0).unsqueeze(0).to(device)
        src_edge_type = src_tokens.view(-1, 1) * len(self.dictionary) + src_tokens.view(1, -1).unsqueeze(0).to(device)
        src_coord = torch.cat([torch.full_like(coordinates[0], 0.0).unsqueeze(0), coordinates, torch.full_like(coordinates[0], 0.0).unsqueeze(0)], dim=0).unsqueeze(0)
        src_distance = torch.from_numpy(distance_matrix(src_coord.view(-1, 3).numpy(), src_coord.view(-1, 3).numpy()).astype(np.float32)).unsqueeze(0).to(device)
        
        # encoder
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
        encoder_rep, _ , _ , _, _ = self.repeat_unit_encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        
        return encoder_rep[:, 0, :].unsqueeze(1)
        
        
    def forward(
        self,
        psmi,
        psmi_rep,
        whole_pyg,
        repeat_unit_smi, 
        repeat_unit_actual_num,
        **kwargs
    ):  
        bsz = 1
        repeat_unit_num = int(repeat_unit_actual_num[0]) 
        device = repeat_unit_actual_num.device
        
        # Whole pyg encoder
        whole_pyg_batch = whole_pyg.to(device)        
        garph_attr = self.whole_pyg_encoder(whole_pyg_batch)
        graph_global_att = garph_attr.unsqueeze(1) # (bsz, d) -> (bsz, 1, d)
        
        try:           
            # init and sample generation orders
            mar_mask = torch.ones(bsz, repeat_unit_num).to(device)   
            mar_tokens = torch.cat([torch.zeros(bsz, repeat_unit_num, self.args.encoder_embed_dim).to(device), graph_global_att.repeat(1, repeat_unit_num, 1)], dim=-1)  # (bsz, r_n, d)      
            mar_orders = self.sample_orders(bsz, repeat_unit_num, repeat_unit_actual_num)
            
            if self.args.mar_num_ar_steps > repeat_unit_num:
                self.args.mar_num_ar_steps = repeat_unit_num
            mar_indices = list(range(self.args.mar_num_ar_steps))
            mar_generated_mols = np.empty(repeat_unit_num, dtype=object)  
            
            # generate latents
            for step in mar_indices:
                mar_cur_tokens = mar_tokens.clone()
                
                # padding identify
                mar_padding_mask = torch.zeros((bsz, repeat_unit_num), dtype=torch.bool).to(device)   
                for idx, num in enumerate(repeat_unit_actual_num):
                    mar_padding_mask[idx, num:] = True
                
                # class embedding
                mar_class_embedding = psmi_rep.float()
                
                # mar encoder
                mar_buffer_size = max(math.floor(min(repeat_unit_actual_num) / 4), 1)     
                mar_x = self.forward_mae_encoder(mar_tokens, mar_mask, mar_padding_mask, mar_class_embedding, mar_buffer_size) # (bsz, mar_buffer_size + r_n, d)
                
                # mar decoder
                mar_z = self.forward_mae_decoder(mar_x, mar_mask, mar_padding_mask, mar_buffer_size) # (bsz, r_n, d) 
                
                # mask ratio for the next round, following MaskGIT and MAGE.
                mar_mask_ratio = np.cos(math.pi / 2. * (step + 1) / self.args.mar_num_ar_steps)
                mar_mask_len = torch.floor(repeat_unit_actual_num * mar_mask_ratio).unsqueeze(1)
                
                # masks out at least one for the next iteration
                mar_mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                    torch.minimum(torch.sum(mar_mask[:bsz], dim=-1, keepdims=True) - torch.sum(mar_padding_mask[:bsz], dim=-1, keepdims=True) - 1, mar_mask_len))
                
                # get masking for next iteration and locations to be predicted in this iteration
                mar_mask_next = self.mask_by_order(mar_mask_len, mar_orders, bsz, repeat_unit_num)
                
                if step >= self.args.mar_num_ar_steps - 1:
                    mar_mask_to_pred = mar_mask[:bsz].bool() & ~mar_padding_mask[:bsz].bool()
                else:
                    mar_mask_to_pred = torch.logical_xor(mar_mask[:bsz].bool(), mar_mask_next.bool())
                
                mar_mask = mar_mask_next
                
                # sample token latents for this step                
                mar_z = mar_z[mar_mask_to_pred]
                                
                # diffusiom sampling
                mar_diff_encoder_rep_list = []
                mar_diff_mol_list = []
                for idx in range(mar_z.shape[0]):
                    mols = self.sample_confs(idx, repeat_unit_smi[0], self.args.mar_diff_confs_per_mol, repeat_unit_smi[0], mar_z[idx].unsqueeze(0).repeat(self.args.mar_diff_confs_per_mol, 1))
                    mar_diff_encoder_rep_list.append(self.encoder_mar_diff_output(mols[0], device))
                    mar_diff_mol_list.append(mols[0])
                mar_diff_encoder_rep = torch.cat(mar_diff_encoder_rep_list, dim=1)
                
                mar_sampled_token_latent = torch.cat([mar_diff_encoder_rep, graph_global_att.repeat(1, mar_diff_encoder_rep.shape[1], 1)], dim=-1)                
                mar_cur_tokens[mar_mask_to_pred] = mar_sampled_token_latent.reshape(-1, mar_sampled_token_latent.shape[2])
                mar_tokens = mar_cur_tokens.clone()
                
                mar_generated_mols[mar_mask_to_pred[0].cpu().numpy()] = mar_diff_mol_list
                
            # MAR encoder    
            mar_mask = torch.zeros(bsz, repeat_unit_num, device=mar_x.device) # (bsz, r_n)            
            mar_class_embedding = psmi_rep.to(mar_x.dtype) # (bsz, 768)
            mar_buffer_size = max(math.floor(min(repeat_unit_actual_num) / 4), 1)                
            mar_padding_mask = torch.zeros((bsz, repeat_unit_num), dtype=torch.bool).to(device)   
            for idx, num in enumerate(repeat_unit_actual_num):
                mar_padding_mask[idx, num:] = True
            mar_x = self.forward_mae_encoder(mar_tokens, mar_mask, mar_padding_mask, mar_class_embedding, mar_buffer_size) # (bsz, mar_buffer_size + r_n, d)
            
            input_coord = torch.from_numpy(self.extract_confs(mar_generated_mols)).to(device)  
            key_point_list = self.process_monomer(psmi[0])
            init_feats = self.sample_init_feats(repeat_unit_num, input_coord, key_point_list, device)
            all_rigids = self.inference(init_feats, 
                        mar_x[:, mar_buffer_size:],
                        num_t=self.frame_conf.inference.diffusion.num_t, 
                        min_t=self.frame_conf.inference.diffusion.min_t, 
                        noise_scale=self.frame_conf.inference.diffusion.noise_scale,
                        device=device)
            
            return {psmi[0]: {'mols': mar_generated_mols, 'rigids': all_rigids}}
        
        except:
            print(f'******************************************')
            return {psmi[0]: 'fail'}
    
            
    def rot2trans(self, rigids, input_coord, key_point_index, device):
        frames = Rigid.from_tensor_7(deepcopy(rigids))
        rots = frames._rots.get_rot_mats().to(device)
        rotated_coord = torch.matmul(rots.unsqueeze(-3), input_coord.unsqueeze(-1)).squeeze(-1)    
        frames._trans = torch.cumsum(
            torch.cat((  
            torch.zeros((1, 3)).to(device),   
            (rotated_coord[:-1, key_point_index[3]] - rotated_coord[1:, key_point_index[0]])
            ), dim=0),
            dim=0  
        )
        return frames.to_tensor_7()
    
    
    def process_monomer(self, monomer_smiles):
        smi = monomer_smiles
        mol = Chem.MolFromSmiles(smi)
        
        assert 'H' not in [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        key_point_list = []
        for atom in mol.GetAtoms():
            atom_neighbors = atom.GetNeighbors()
            for atom_neighbor in atom_neighbors:
                if atom_neighbor.GetSymbol() == '*':
                    key_point_list.append(atom.GetIdx())
                    key_point_list.append(atom_neighbor.GetIdx())
        
        assert len(key_point_list) == 4, 'Unvalid PSMILES'

        star_0 = key_point_list[1]
        star_1 = key_point_list[3]
        neighbor_0 = key_point_list[0]
        neighbor_1 = key_point_list[2]
        
        atom = mol.GetAtomWithIdx(star_0)
        atom.SetAtomicNum(mol.GetAtomWithIdx(neighbor_1).GetAtomicNum())

        atom = mol.GetAtomWithIdx(star_1)
        atom.SetAtomicNum(mol.GetAtomWithIdx(neighbor_0).GetAtomicNum())

        processed_smi = ""
        replacement = [str(mol.GetAtomWithIdx(neighbor_1).GetSymbol()), str(mol.GetAtomWithIdx(neighbor_0).GetSymbol())]
        count = 0
        i = 0
        while i != len(smi):
            if smi[i:i+3] == "[*]":
                processed_smi += replacement[count]
                count += 1
                i += 3
            else:
                processed_smi += smi[i]
                i += 1

        pre_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        processed_mol = Chem.MolFromSmiles(processed_smi)
        post_atoms = [atom.GetSymbol() for atom in processed_mol.GetAtoms()]
        
        assert pre_atoms == post_atoms, 'Unmatch Order'
        return key_point_list
    
    
    def extract_confs(self, mols):
        confs_dict = {}
        for idx, mol in enumerate(mols):
            mol = Chem.RemoveHs(mol)
            confs_dict[idx] = mol.GetConformer().GetPositions().astype(np.float32)
        confs = np.array([confs_dict[idx] for idx in confs_dict.keys()])
        return confs
    
    
    def sample_init_feats(self, sample_length, input_coord, key_point_list, device):
        # Process motif features.
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)

        # Initialize data
        ref_sample = self.frame_diffuser.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )
                
        ref_sample['rigids_t'] = self.rot2trans(ref_sample['rigids_t'], input_coord, key_point_list, device)
        
        res_idx = torch.arange(1, sample_length+1)
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2)),
            'sc_ca_t': np.zeros((sample_length, 3)),
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(device), init_feats)
        
        return init_feats
    
    
    def inference(self, data_init, cond_info, num_t=None, min_t=None, center=True, self_condition=True, noise_scale=1.0, device=None):
        sample_feats = copy.deepcopy(data_init)
        if sample_feats['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones(
                (sample_feats['rigids_t'].shape[0],)).to(device)
        
        if num_t is None:
            num_t = self.frame_conf.data.num_t
        
        if min_t is None:
            min_t = self.frame_conf.data.min_t
        
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1/num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))]
        
        with torch.no_grad():
            for t in reverse_steps:
                if t > min_t:
                    sample_feats = self.set_t_feats(sample_feats, t, t_placeholder)
                    
                    bb_mask = sample_feats['res_mask'].type(torch.float32)  # [B, N]
                    edge_mask = bb_mask[..., None] * bb_mask[..., None, :]
                    
                    init_node_embed, init_edge_embed = self.frame_embedding_layer(
                        seq_idx=sample_feats['seq_idx'],
                        t=sample_feats['t'],
                        mar_encoder_output=cond_info,
                    )

                    edge_embed = init_edge_embed * edge_mask[..., None]
                    node_embed = init_node_embed * bb_mask[..., None]
                
                    frame_model_out = self.frame_score_model(node_embed, edge_embed, bb_mask, edge_mask, sample_feats['rigids_t'], sample_feats['t'])
                    
                    rot_score = frame_model_out['rot_score']
                    trans_score = frame_model_out['trans_score']
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                    
                    rigids_t = self.frame_diffuser.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                        rot_score=du.move_to_np(rot_score),
                        trans_score=du.move_to_np(trans_score),
                        diffuse_mask=du.move_to_np(diffuse_mask),
                        t=t,
                        dt=dt,
                        center=center,
                        noise_scale=noise_scale,
                    )
        
                else:
                    bb_mask = sample_feats['res_mask'].type(torch.float32)  # [B, N]
                    edge_mask = bb_mask[..., None] * bb_mask[..., None, :]
                    
                    init_node_embed, init_edge_embed = self.frame_embedding_layer(
                        seq_idx=sample_feats['seq_idx'],
                        t=sample_feats['t'],
                        mar_encoder_output=cond_info,
                    )

                    edge_embed = init_edge_embed * edge_mask[..., None]
                    node_embed = init_node_embed * bb_mask[..., None]
                    
                    frame_model_out = self.frame_score_model(node_embed, edge_embed, bb_mask, edge_mask, sample_feats['rigids_t'], sample_feats['t'])
                    rigids_t = ru.Rigid.from_tensor_7(frame_model_out['rigids'])
                
                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
                all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))
        
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_rigids = flip(all_rigids)
        return all_rigids[:, 0]
    
    
    def set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.frame_diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats    
    
    
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
    
    
@register_model_architecture("polyconf_inference", "polyconf_inference")
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