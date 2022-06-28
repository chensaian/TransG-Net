import csv
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

ids_miles_file = "./data/id_smiles.csv"
ids_des_file = "./data/ms_des.csv"
ms_file = "./data/ms_spectrum.csv"

des_dic = {}
ms = []
with open(ms_file) as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        ms.append(row)

with open(ids_des_file) as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        des_dic[row[0]] = [row[1:], row[1], ms[i]]

with open(ids_miles_file) as f:
    reader = csv.reader(f)
    data_list = []
    error_ids = []
    error_smiles = []
    for i, row in tqdm(enumerate(reader), total=65177):
        try:
            name, smile = row
            mol = Chem.MolFromSmiles(smile)
            atoms = mol.GetAtoms()
            atoms_list = []
            for i in range(len(atoms)):
                atoms_list.append(int(atoms[i].GetAtomicNum()))
            indices = torch.tensor(atoms_list)
            atoms_attr = torch.nn.functional.one_hot(
                indices - 1, 16).float()  
            bonds = mol.GetBonds()
            bonds_list = []
            index = torch.zeros([2, 2*len(bonds)]).long()
            for i in range(len(bonds)):
                if bonds[i].GetBondType() == rdchem.BondType.SINGLE:
                    bonds_list.append(0)
                elif bonds[i].GetBondType() == rdchem.BondType.DOUBLE:
                    bonds_list.append(1)
                elif bonds[i].GetBondType() == rdchem.BondType.TRIPLE:
                    bonds_list.append(2)
                else:
                    bonds_list.append(3)
                index[0][i] = bonds[i].GetBeginAtomIdx()
                index[1][i] = bonds[i].GetEndAtomIdx()
                index[0][i+len(bonds)] = bonds[i].GetEndAtomIdx()
                index[1][i+len(bonds)] = bonds[i].GetBeginAtomIdx()
            

            bonds_list+=bonds_list
            bonds_attr = torch.nn.functional.one_hot(
                torch.tensor(bonds_list), 4).float()

            des = torch.tensor([float(i) for i in des_dic[name][0]]).reshape(
                (1, -1)).float()
            y = torch.tensor([float(des_dic[name][1])]).float()
            ms_spec = torch.tensor([float(i) for i in des_dic[name][2]]).reshape(
                (1, -1)).float()
        except Exception as e:
            error_ids.append(name)
            error_smiles.append(smile)
            continue
        data = Data(x=atoms_attr, edge_attr=bonds_attr, edge_index=index,
                    id=name, smile=smile, y=y, ms_spec=ms_spec)
        data_list.append(data)


torch.save(data_list, "./data/ms_dataset.pth")
print("error id:{}".format(error_ids))
print("error smile:{}".format(error_smiles))
