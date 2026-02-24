##this file for substrate class prediction
import random
import torch
import os
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from utils.dataset import *
from sklearn.model_selection import train_test_split
# Preprocessing
from utils.protein_init import *
from utils.ligand_init import *
# Model
from model_fitting.model_pair import net
#from model_fitting.model_pair1028 import net
import argparse
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_score,average_precision_score, recall_score
from sklearn import metrics
import numpy as np
import torch.optim as optim
from rdkit import Chem
from build_vocab import WordVocab

def inchi_to_smiles(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol:
        smiles = Chem.MolToSmiles(mol)
        return smiles
    else:
        return "Invalid InChI"


def evaluate_reg(y_true, y_pred_proba,best_threshold =None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred_proba]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    print("best_threshold is",best_threshold)
    bias = -np.log((1 - best_threshold) / best_threshold)

    y_pred = [1 if pred >= best_threshold else 0 for pred in y_pred_proba]

    #y_pred = (y_pred_proba >= 0.5).astype(int)
    # Accuracy
    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    recall = recall_score(y_true, y_pred)
    #accuracy = accuracy_score(y_true, y_pred)
    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_true, y_pred)
    # ROC-AUC score (requires predicted probabilities for positive class)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    # Precision (for binary classification, with class '1' as the positive class by default)
    precision = precision_score(y_true, y_pred, average='macro')

    return {'F1':best_f1,'acc': accuracy,
        'pre':precision,
        'mcc': mcc,
        'roc_auc': roc_auc,
            'auprc':auprc,
            'recall':recall}

def virtual_screening(model, data_loader, device):
    reg_preds = []
    reg_truths = []
    model.eval()
    count = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            count += 1
            data = data.to(device)
            reg_pred, sp_loss, o_loss, cl_loss,att_weight = model(
                mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, total_fea=data.total_fea, bond_x=data.mol_edge_attr, atom_edge_index=data.mol_edge_index,
                # Protein
                residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo, residue_edge_index=data.prot_edge_index,
                residue_edge_weight=data.prot_edge_weight,
                # Mol-Protein Interaction batch
                mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch)
            # interaction_keys = list(zip(data.prot_key, data.mol_key))
            reg_pred = torch.sigmoid(reg_pred)
            reg_pred = reg_pred.squeeze().cpu().detach().numpy().reshape(-1).tolist()
            reg_preds += reg_pred
            reg_y = data.reg_y.squeeze().cpu().detach().numpy().reshape(-1).tolist()
            reg_truths += reg_y

        if len(reg_truths) > 0:
            eval_reg_result = evaluate_reg(np.array(reg_truths), np.array(reg_preds))
    return reg_preds, reg_truths, eval_reg_result

class Trainer(object):
    def __init__(self, model, lrate, min_lrate, num_epochs,warmup_iters=2000, lr_decay_iters=None, schedule_lr=True, evaluate_metric='auprc',result_path='', device='cuda'):

        self.model = model
        self.model.to(device)
        #self.optimizer = self.model.optimizer(learning_rate=lrate,betas=(0.9,0.999), eps=1e-8)#self.model.optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, betas=(0.9, 0.999), eps=1e-6)
        self.class_loss = torch.nn.BCEWithLogitsLoss() #torch.nn.BCELoss()  # missing_mse_loss
        self.num_epochs = num_epochs
        self.result_path = result_path

        self.device = device
        self.evaluate_metric = evaluate_metric

    def train_epoch(self, train_loader, val_loader,test_loader):
        iter_num = 0
        best_result = 0.0
        for epoch in range(1, self.num_epochs + 1):
            print("epoch:", epoch)
            self.model.train()

            for data in train_loader:
                self.optimizer.zero_grad()
                data = data.to(self.device)

                pred, sp_loss, o_loss, cl_loss, _= self.model(mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, total_fea=data.total_fea, bond_x=data.mol_edge_attr,atom_edge_index=data.mol_edge_index,residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,residue_edge_index=data.prot_edge_index, residue_edge_weight=data.prot_edge_weight,mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch)

                binary_pred = pred.squeeze()
                reg_y = data.reg_y.squeeze()
                class_loss = self.class_loss(binary_pred, reg_y) #* 0.9

                loss_val = class_loss

                loss_val.backward()
                self.optimizer.step()
                iter_num += 1

            print("starting to evaluate -------------------------------------------")
            _,_,val_result = virtual_screening(self.model, val_loader, self.device)
            val_result = {k: round(v, 4) for k, v in val_result.items()}


            if val_result[self.evaluate_metric] > best_result:
                print(f'Validation {self.evaluate_metric} increased ({best_result:.4f} --> {val_result[self.evaluate_metric]:.4f})', 'best_precision:', val_result["pre"],'roc_auc:', val_result["roc_auc"] ,'MCC:', val_result["mcc"], 'Saving model ...')
                best_result = val_result[self.evaluate_metric]
                torch.save(self.model.state_dict(), os.path.join(self.result_path, 'model_pair.pt'))
            else:
                print(f'current {self.evaluate_metric}: ', val_result[self.evaluate_metric], ' No improvement since', best_result)

print("starting to parse parameters")
parser = argparse.ArgumentParser()
### Seed and device
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--result_path', type=str,default="./BYO_RESULT",help='path to save results')
parser.add_argument('--epochs', type=int, default=80, help='')

# optimizer params - only change this for PDBBind v2016
parser.add_argument('--lrate',type=float,default=5e-5,help='learning rate') # change to 1e-5 for LargeScaleInteractionDataset
parser.add_argument('--eps',type=float,default=1e-8, help='higher = closer to SGD') # change to 1e-5 for PDBv2016 ,1e-8
# batch size
parser.add_argument('--batch_size',type=int,default=32)#16
args = parser.parse_args()

with open('../config1.json','r') as f:
    config = json.load(f)
# overwrite
config['optimizer']['lrate'] = args.lrate
config['optimizer']['eps'] = args.eps
device = torch.device(args.device)

model_path = os.path.join(args.result_path,'save_model_seed{}'.format(args.seed))
if not os.path.exists(model_path):
    os.makedirs(model_path)

# seed initialize
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

## import files
sequences = ["MEKEDQEKTGKLTLVLALATFLAAFGSSFQYGYNVAAVNSPSEFMQQFYNDTYYDRNKENIESFTLTLLWSLTVSMFPFGGFIGSLMVGFLVNNLGRKGALLFNNIFSILPAILMGCSKIAKSFEIIIASRLLVGICAGISSNVVPMYLGELAPKNLRGALGVVPQLFITVGILVAQLFGLRSVLASEEGWPILLGLTGVPAGLQLLLLPFFPESPRYLLIQKKNESAAEKALQTLRGWKDVDMEMEEIRKEDEAEKAAGFISVWKLFRMQSLRWQLISTIVLMAGQQLSGVNAIYYYADQIYLSAGVKSNDVQYVTAGTGAVNVFMTMVTVFVVELWGRRNLLLIGFSTCLTACIVLTVALALQNTISWMPYVSIVCVIVYVIGHAVGPSPIPALFITEIFLQSSRPSAYMIGGSVHWLSNFIVGLIFPFIQVGLGPYSFIIFAIICLLTTIYIFMVVPETKGRTFVEINQIFAKKNKVSDVYPEKEEKELNDLPPATREQ",
            "MSNKQETKILGMPPFVVDFLMGGVSAAVSKTAAAPIERIKLLVQNQDEMIKAGRLDRRYNGIIDCFRRTTADEGLMALWRGNTANVIRYFPTQALNFAFRDKFKAMFGYKKDKDGYAKWMAGNLASGGAAGATSLLFVYSLDYARTRLANDAKSAKGGGARQFNGLIDVYRKTLASDGIAGLYRGFGPSVAGIVVYRGLYFGMYDSIKPVVLVGPLANNFLASFLLGWCVTTGAGIASYPLDTVRRRMMMTSGEAVKYKSSIDAFRQIIAKEGVKSLFKGAGANILRGVAGAGVLSIYDQLQILLFGKAFKGGSG"]

data_path = "../data/data/"
data_train = pd.read_pickle(data_path + "transporter_substrate_pairs/" +  "df_UID_MID_train.pkl")
data_test = pd.read_pickle(data_path + "transporter_substrate_pairs/" + "df_UID_MID_test.pkl")
data_train = data_train.loc[~data_train["Sequence"].isin(sequences)]
#print('111111',len(data_train),len(data_test)) #26701 6282

data_train.reset_index(inplace = True, drop = True)
data_test.reset_index(inplace = True, drop = True)

sequence = [data_train['Sequence'][ind] for ind in data_train.index]
Inchi = [data_train["molecule ID"][ind] for ind in data_train.index]
Smiles = [inchi_to_smiles(Inchi_data) for Inchi_data in Inchi]
Label = [float(data_train["outcome"][ind]) for ind in data_train.index]

ecfp_train = {}
for ind in data_train.index:
    ecfp_train[Smiles[ind]] = np.array([list(data_train["ChemBERTa"][ind])])

data_dic = {"Pro_seq": sequence,"Smile":Smiles, "label": Label}
df_train = pd.DataFrame(data_dic)


train_df, valid_df = train_test_split(df_train, test_size=0.1,random_state= args.seed)
valid_df = valid_df.reset_index(drop=True)
train_df = train_df.reset_index(drop=True)

print("starting to init substrate and protein")
protein_dict = protein_init(sequence)
torch.save(protein_dict, data_path + "pair_protein_train.pt")
ligand_dict = ligand_init(Smiles)
torch.save(ligand_dict, data_path + "pair_subs_train.pt")
#protein_dict = torch.load(data_path + "pair_protein_train.pt")
#ligand_dict = torch.load(data_path + "pair_subs_train.pt")

torch.cuda.empty_cache()
train_dataset = PairDataset(train_df,ligand_dict, protein_dict,ecfp_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])#follow_batch描述节点信息 用于确保 mini-batch 中的节点特征和目标的顺序与原始图中的节点顺序匹配

valid_dataset = PairDataset(valid_df,ligand_dict,  protein_dict,ecfp_train)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])

print('Computing training data degrees for PNA')
prot_deg = compute_pna_degrees(train_dataset)
#prot_deg = torch.load("train_prot_degree.pt", map_location="cpu")
model = net(prot_deg,mol_in_channels=config['params']['mol_in_channels'],  prot_in_channels=config['params']['prot_in_channels'],
            prot_evo_channels=config['params']['prot_evo_channels'], hidden_channels=config['params']['hidden_channels'], pre_layers=config['params']['pre_layers'],
            post_layers=config['params']['post_layers'],aggregators=config['params']['aggregators'],scalers=config['params']['scalers'],total_layer=config['params']['total_layer'],
            K = config['params']['K'],heads=config['params']['heads'], dropout=config['params']['dropout'],dropout_attn_score=config['params']['dropout_attn_score'],
            device=device).to(device)


sequence = [data_test['Sequence'][ind] for ind in data_test.index]
Inchi = [data_test["molecule ID"][ind] for ind in data_test.index]
Smiles = [inchi_to_smiles(Inchi_data) for Inchi_data in Inchi]
Label = [float(data_test["outcome"][ind]) for ind in data_test.index]

ecfp_test = {}
for ind in data_test.index:
    ecfp_test[Smiles[ind]] = np.array([list(data_test["ChemBERTa"][ind])])

data_dic = {"Pro_seq": sequence,"Smile":Smiles, "label": Label}
df_test = pd.DataFrame(data_dic)


print("starting to init protein")
protein_dict_test = protein_init(sequence)
torch.save(protein_dict, data_path + "pair_protein_test.pt")
ligand_dict_test = ligand_init(Smiles)
torch.save(ligand_dict, data_path + "pair_subs_test.pt")
#protein_dict_test = torch.load(data_path + "pair_protein_test.pt")
#ligand_dict_test = torch.load(data_path + "pair_subs_test.pt")

torch.cuda.empty_cache()
test_dataset = PairDataset(df_test,ligand_dict_test, protein_dict_test,ecfp_test)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=None, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])#follow_batch描述节点信息 用于确保 mini-batch 中的节点特征和目标的顺序与原始图中的节点顺序匹配


#print('start training model'+'-'*50)
engine = Trainer(model=model, lrate=args.lrate,num_epochs=args.epochs,warmup_iters=0,min_lrate=0,lr_decay_iters=0,schedule_lr="false",  result_path=args.result_path, device=device)
engine.train_epoch(train_loader, valid_loader, test_loader)
print('finished training model')

#############strating to test the model
#print('loading best checkpoint and predicting test data'+'-'*50)
#model.load_state_dict(torch.load(os.path.join(args.result_path, 'model_pair.pt')))

reg_preds, reg_truths, eval_reg_result= virtual_screening(model, test_loader, device=args.device)
df = pd.DataFrame({"reg_truth": reg_truths,"reg_pred": reg_preds})
df.to_excel("prediction_results.xlsx", index=False)
print('acc:', eval_reg_result["acc"],  "roc_auc:",eval_reg_result["roc_auc"], "F1:", eval_reg_result["F1"],'Auprc:',eval_reg_result["auprc"])



