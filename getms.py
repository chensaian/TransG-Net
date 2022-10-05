import json
import time
import requests
import csv
#import torch
import numpy as np
from rdkit.Chem import Descriptors
from rdkit import Chem
import os
import random
import xml
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse
from tqdm import tqdm

# 第一步下载数据，和获取数值型数据


def get_page(url):
    try:
        header = {
            "user-agent": "Chrome/89.0.4381.102"
        }
        response = requests.get(url, headers=header)
        if response.status_code == 200:
            return response.text
        return None
    except Exception:
        return None


def get_XML(file_ids):
    for file_id in file_ids:
        path = "./ms_data/msxml/{}.xml".format(file_id)
        if os.path.exists(path):
            print("{}.xml 文件已存在。".format(file_id))
            continue
        JSONfile = get_page(
            "https://hmdb.ca/metabolites/{}.xml".format(file_id))
        if JSONfile != "" or JSONfile != None:
            with open(path, "a") as f:
                f.write(JSONfile)
            print("{}.xml写入完毕！".format(file_id))
        else:
            print("{}.xml无法写入！！！".format(file_id))
        time.sleep(1)


def get_smile(file_name):
    DOMTree = parse(r'./ms_data/msxml/{}'.format(file_name))
    booklist = DOMTree.documentElement
    books = booklist.getElementsByTagName('smiles')[0]
    smile = books.childNodes[0].data
    return smile


def get_descriptors(ids, smiles):
    # descriptors = ["MolLogP", "ExactMolWt", "HeavyAtomMolWt", "NHOHCount", "NOCount", "NumHAcceptors", "NumHDonors",
    #                "NumHeteroatoms", "NumRotatableBonds", "NumValenceElectrons", "RingCount", "TPSA"]
    file = "./ms_data/ms_des.csv"
    print("======开始写入描述符======")
    n = len(smiles)
    for i in tqdm(range(len(smiles))):
        descriptor = []
        try:
            with open(file, "a") as f:
                m = Chem.MolFromSmiles(smiles[i])
                descriptor.append(ids[i])
                # descriptor.append(smiles[i])
                descriptor.append(Descriptors.MolLogP(m))
                descriptor.append(Descriptors.ExactMolWt(m))
                descriptor.append(Descriptors.HeavyAtomMolWt(m))
                descriptor.append(Descriptors.NHOHCount(m))
                descriptor.append(Descriptors.NOCount(m))
                descriptor.append(Descriptors.NumHAcceptors(m))
                descriptor.append(Descriptors.NumHDonors(m)
                                  )  # H受体和给体包含分子的电子分布信息
                descriptor.append(Descriptors.NumHeteroatoms(m))
                descriptor.append(
                    Descriptors.NumRotatableBonds(m))  # 包含拓扑结构的相关信息
                descriptor.append(Descriptors.NumValenceElectrons(m))  # 电子信息
                descriptor.append(Descriptors.RingCount(m))  # 3D结构和电子信息
                descriptor.append(Descriptors.TPSA(m))  # 极化信息 电子信息
                des = ",".join([str(item) for item in descriptor]) + "\n"
                # des[0] = 0
                f.write(des)
        except Exception as e:
            continue
    print("======描述符写入完毕======")
    return 


if __name__ == '__main__':
    # ms_root = "./ms_data/hmdb_predicted_cms_peak_lists"
    # files = os.listdir(ms_root)
    # filess = [file[:11] for file in files]
    # # 获取了唯一id
    # files = list(set(filess))
    # get_XML(files)
    # getDescriptors(smiles)

    xml_root = "./ms_data/msxml"
    files = os.listdir(xml_root)
    ids = []
    smiles = []
    id_smiles_file = "./ms_data/id_smiles.csv"
    for file in tqdm(files):
        with open(id_smiles_file, "a") as f:
            try:
                name, smile = file[:11], get_smile(file)
            except Exception as e:
                continue
            ids.append(name)
            smiles.append(smile)
            f.write("{},{}\n".format(name, smile))
    print("smiles采集完毕")
    get_descriptors(ids, smiles)
