# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from pymol import cmd
from tqdm import tqdm
from rdkit import Chem  
from rdkit import RDLogger
from rdkit.Chem import AllChem  

RDLogger.DisableLog('rdApp.*')


def cal_energy(mol_path):
    mol = Chem.MolFromMolFile(mol_path, removeHs=False)
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)  
    mmff_forcefield = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)  
    mmff_energy = (mmff_forcefield.CalcEnergy()) /  mol.GetNumAtoms() 
    return mmff_energy


if __name__ == '__main__':
    gen_path = f'./generated_confs'
    refer_path = f'./dataset/true_confs'
    
    s_matr_scores = []
    s_matp_scores = []
    e_matr_scores = []
    e_matp_scores = []
    
    for data_idx in tqdm(range(2030)):
        rmsd_mat = -1 * np.ones([5, 10], dtype=float)
        energy_mat = -1 * np.ones([5, 10], dtype=float)
        for i in range(10):
            for j in range(5):
                gen_mol_path = os.path.join(gen_path, f'data_{data_idx}', f'generated_conf_{i}.sdf')
                true_mol_path = os.path.join(refer_path, f'data_{data_idx}', f'true_conf_{j}.sdf')
                cmd.reinitialize()
                cmd.feedback("disable", "matrix", "everything")
                cmd.load(true_mol_path, 'true_mol')
                cmd.load(gen_mol_path, 'gen_mol')
                rmsd = cmd.align('true_mol', 'gen_mol')[0]
                rmsd_mat[j,i] = rmsd
                energy_mat[j,i] = (cal_energy(gen_mol_path)-cal_energy(true_mol_path)) * 0.0015936

        rmsd_ref_min = rmsd_mat.min(-1)    
        rmsd_gen_min = rmsd_mat.min(0)
        s_matr_scores.append(rmsd_ref_min.mean())
        s_matp_scores.append(rmsd_gen_min.mean())

        energy_ref_min = energy_mat.min(-1)    
        energy_gen_min = energy_mat.min(0)
        e_matr_scores.append(energy_ref_min.mean())
        e_matp_scores.append(energy_gen_min.mean())

    s_matr_scores = np.array(s_matr_scores)
    s_matp_scores = np.array(s_matp_scores)
    print('S-MAT-R_mean: %.4f | S-MAT-R_median: %.4f' % (np.mean(s_matr_scores), np.median(s_matr_scores)))
    print('S-MAT-P_mean: %.4f | S-MAT-P_median: %.4f' % (np.mean(s_matp_scores), np.median(s_matp_scores)))

    e_matr_scores = np.array(e_matr_scores)
    e_matp_scores = np.array(e_matp_scores)
    print('E-MAT-R_mean: %.4f | E-MAT-R_median: %.4f' % (np.mean(e_matr_scores), np.median(e_matr_scores)))
    print('E-MAT-P_mean: %.4f | E-MAT-P_median: %.4f' % (np.mean(e_matp_scores), np.median(e_matp_scores)))
