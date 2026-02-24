##this file for substrate class prediction
import random
import torch
import os
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from utils.dataset import *  # data
from sklearn.model_selection import train_test_split
# Preprocessing
from utils.protein_init import *
# Model
from model_fitting.model_substrates import net
import argparse
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_score,recall_score,    f1_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from openTSNE import TSNE as openTSNE


def virtual_screening_with_embedding(model, data_loader, device):
    reg_preds  = []
    reg_truths = []
    reg_scores = []
    embeddings = []   # ← 存 embedding

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            data = data.to(device)

            reg_pred, sp_loss, o_loss, cl_loss = model(
                residue_x=data.prot_node_aa,
                residue_evo_x=data.prot_node_evo,
                residue_edge_index=data.prot_edge_index,
                residue_edge_weight=data.prot_edge_weight,
                prot_batch=data.prot_node_aa_batch
            )

            # ✅ 提取 classifier1 之后的 embedding（128维）
            emb = model.get_embedding(
                residue_x=data.prot_node_aa,
                residue_evo_x=data.prot_node_evo,
                residue_edge_index=data.prot_edge_index,
                residue_edge_weight=data.prot_edge_weight,
                prot_batch=data.prot_node_aa_batch
            )
            embeddings.append(emb.cpu().numpy())

            #prediction = torch.argmax(reg_pred, dim=-1)
            #reg_preds  += prediction.squeeze().cpu().numpy().reshape(-1).tolist()
            prob = torch.softmax(reg_pred, dim=-1)
            reg_probs = prob.cpu().numpy()
            reg_preds += np.argmax(reg_probs, axis=1).tolist()
            reg_scores += reg_probs.tolist()

            reg_truths += data.reg_y.squeeze().cpu().numpy().reshape(-1).tolist()

    embeddings = np.concatenate(embeddings, axis=0)
    eval_result = evaluate_reg(np.array(reg_truths), np.array(reg_preds))
    return reg_scores,reg_preds, reg_truths, eval_result, embeddings


def evaluate_reg(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    # Precision, Recall, F1 score (macro-averaged)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return {'acc': accuracy,
        'pre':precision,
        'recall': recall,
        'f1': f1}

def virtual_screening(model, data_loader, device):
    reg_preds = []
    reg_truths = []
    model.eval()
    count = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            count += 1
            data = data.to(device)
            reg_pred, sp_loss, o_loss, cl_loss = model(
                    residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                    residue_edge_index=data.prot_edge_index, residue_edge_weight=data.prot_edge_weight,
                    # Mol-Protein Interaction batch
                    prot_batch=data.prot_node_aa_batch)
            # interaction_keys = list(zip(data.prot_key, data.mol_key))

            prediction = torch.argmax(reg_pred, dim=-1)

            reg_pred = prediction.squeeze().cpu().detach().numpy().reshape(-1).tolist()
            reg_preds += reg_pred
            reg_y = data.reg_y.squeeze().cpu().detach().numpy().reshape(-1).tolist()
            reg_truths += reg_y

        if len(reg_truths) > 0:
            eval_reg_result = evaluate_reg(np.array(reg_truths), np.array(reg_preds))
    return reg_preds, reg_truths, eval_reg_result


class Trainer(object):
    def __init__(self, model, lrate, min_lrate, num_epochs,warmup_iters=2000, lr_decay_iters=None, schedule_lr=True, evaluate_metric='acc',result_path='', device='cuda'):

        self.model = model
        self.model.to(device)
        #self.optimizer = self.model.optimizer(learning_rate=lrate,betas=(0.9,0.999), eps=1e-8)#self.model.optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, betas=(0.9, 0.999), eps=1e-8)
        self.class_loss = torch.nn.CrossEntropyLoss()  # missing_mse_loss
        self.num_epochs = num_epochs
        self.result_path = result_path

        self.device = device
        self.evaluate_metric = evaluate_metric

    def train_epoch(self, train_loader, val_loader):
        iter_num = 0
        best_result = 0.0
        for epoch in range(1, self.num_epochs + 1):
            print("epoch:", epoch)
            self.model.train()

            for data in train_loader:
                self.optimizer.zero_grad()

                data = data.to(self.device)
                reg_pred, sp_loss, o_loss, cl_loss= self.model(
                    residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                    residue_edge_index=data.prot_edge_index, residue_edge_weight=data.prot_edge_weight,
                    prot_batch=data.prot_node_aa_batch)
                ## Loss compute
                loss_val = torch.tensor(0.).to(self.device)

                loss_val += cl_loss * 0.1
                reg_pred = reg_pred.squeeze()
                reg_y = data.reg_y.squeeze()
                class_loss = self.class_loss(reg_pred, reg_y) * 0.9
                loss_val += class_loss

                loss_val.backward()
                self.optimizer.step()
                self.model.temperature_clamp()
                iter_num += 1

            print("starting to evaluate -------------------------------------------")
            _,_,val_result = virtual_screening(self.model, val_loader, self.device)
            val_result = {k: round(v, 4) for k, v in val_result.items()}
            #print("Test performance of Epoch:", epoch, 'acc:', val_result["acc"], 'precision:',val_result["pre"], "recall:", val_result["recall"], "f1:", val_result["f1"])

            if val_result[self.evaluate_metric] > best_result:
                print(f'Validation acc increased ({best_result:.4f} --> {val_result["acc"]:.4f})', 'best_precision:', val_result["pre"],'best_recall:', val_result["recall"] ,'best_f1:', val_result["f1"], 'Saving model ...')
                best_result = val_result[self.evaluate_metric]
                torch.save(self.model.state_dict(), os.path.join(self.result_path, 'model_calssification.pt'))#.format(epoch)
            else:
                print('current acc: ', val_result["acc"], ' No improvement since best_mse', best_result)

    def get_lr(self, iter):
        # 1) linear warmup for warmup_iters steps
        if iter < self.warmup_iters:
            return self.lrate * iter / self.warmup_iters
        # 2) if iter > lr_decay_iters, return min learning rate
        if iter > self.lr_decay_iters:
            return self.min_lrate
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1

        return self.min_lrate + coeff * (self.lrate - self.min_lrate)



parser = argparse.ArgumentParser()
### Seed and device
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--result_path', type=str,default="./BYO_RESULT",help='path to save results')
parser.add_argument('--epochs', type=int, default=80, help='')

# optimizer params - only change this for PDBBind v2016
parser.add_argument('--lrate',type=float,default=1e-4,help='learning rate')
parser.add_argument('--eps',type=float,default=1e-8, help='higher = closer to SGD')
# batch size
parser.add_argument('--batch_size',type=int,default=32)#16
args = parser.parse_args()
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
data_path = "../data/data/"
data_train = pd.read_pickle(data_path + "substrate_classes/" +  "df_train.pkl")
data_test = pd.read_pickle(data_path + "substrate_classes/" + "df_test.pkl")

data_train.reset_index(inplace = True, drop = True)
data_test.reset_index(inplace = True, drop = True)

categories = ["anion", "cation", "sugar", "amino acid" , "protein", "electron", "other"]
c_to_y ={"anion" : 0, "cation" : 1, "sugar": 2, "amino acid" : 3, "protein" : 4, "electron" : 5, "other" : 6}

data_train["outcome"] = np.nan
for ind in data_train.index:
    c = np.array(categories)[np.array(data_train[categories].loc[ind]) == 1][0]
    data_train.loc[ind, 'outcome'] = c_to_y[c]

data_test["outcome"] = np.nan
for ind in data_test.index:
    c = np.array(categories)[np.array(data_test[categories].loc[ind]) == 1][0]
    data_test.loc[ind, 'outcome'] = c_to_y[c]

sequence = [data_train['Sequence'][ind] for ind in data_train.index]
Label = [float(data_train["outcome"][ind]) for ind in data_train.index]
data_dic = {"Pro_seq": sequence, "label": Label}
df_train = pd.DataFrame(data_dic)

train_df, valid_df = train_test_split(df_train, test_size=0.1,random_state= args.seed)
valid_df = valid_df.reset_index(drop=True)
train_df = train_df.reset_index(drop=True)

protein_dict = protein_init(sequence)
torch.save(protein_dict, data_path + "transport_protein.pt")
#protein_dict = torch.load(data_path + "transport_protein.pt")

torch.cuda.empty_cache()
train_dataset = ProteinMoleculeDataset(train_df, protein_dict)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])#follow_batch描述节点信息 用于确保 mini-batch 中的节点特征和目标的顺序与原始图中的节点顺序匹配

valid_dataset = ProteinMoleculeDataset(valid_df,  protein_dict)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])

print('Computing training data degrees for PNA')
prot_deg = compute_pna_degrees(train_dataset)
torch.save(prot_deg, "train_prot_degree_classification.pt")
#prot_deg = torch.load("train_prot_degree_classification.pt", map_location="cpu")

model = net( 7, prot_deg,device=device).to(device) #

print('start training model'+'-'*50)
engine = Trainer(model=model, lrate=args.lrate,num_epochs=args.epochs,warmup_iters=0,min_lrate=0,lr_decay_iters=0,schedule_lr="false", evaluate_metric= 'acc',  result_path=args.result_path, device=device)
engine.train_epoch(train_loader, valid_loader)
print('finished training model')

print("starting to test the model")

print('loading best checkpoint and predicting test data'+'-'*50)
#model.load_state_dict(torch.load(os.path.join(args.result_path, 'model_calssification.pt')))

sequence = [data_test['Sequence'][ind] for ind in data_test.index]
Label = [float(data_test["outcome"][ind]) for ind in data_test.index]
data_dic = {"Pro_seq": sequence, "label": Label}
df_test = pd.DataFrame(data_dic)

protein_dict = protein_init(sequence)
torch.save(protein_dict, data_path + "transport_protein_test.pt")
#protein_dict = torch.load(data_path + "transport_protein_test.pt")

torch.cuda.empty_cache()
test_dataset = ProteinMoleculeDataset(df_test, protein_dict)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=None, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])#follow_batch描述节点信息 用于确保 mini-batch 中的节点特征和目标的顺序与原始图中的节点顺序匹配

reg_scores,reg_preds, reg_truths, eval_result, embeddings = virtual_screening_with_embedding(model, test_loader, device=args.device)

np.save("embeddings.npy", embeddings)
print(f"Embedding shape: {embeddings.shape}")  # (N, 128)

result = {'pred':reg_preds,'prob':reg_scores,'truth':reg_truths}
df_result = pd.DataFrame(result)
df_result.to_csv('prediction_result.csv', index=False)

