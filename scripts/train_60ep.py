import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

ROOT     = "/home/nvidia/24PHD1237"
NPZ_PATH = f"{ROOT}/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz"
EXP_ROOT = f"{ROOT}/EEGMMIDB/experiments"
RUN_NAME = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_60EP_FINAL"
RUN_DIR  = os.path.join(EXP_ROOT, RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)
print("RUN_DIR:", RUN_DIR)

SEED_LIST = [1, 2, 3, 4, 5]
CFG = {
    "epochs": 60,
    "batch_size": 256,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "emb_dim": 128,
    "lambda_supcon": 0.5,
    "lambda_state": 0.5,
    "lambda_orth": 0.1,
    "temperature": 0.07,
    "kmeans_k": 3,
    "train_runs": ["r01","r02"],
    "test_runs":  ["r03","r04","r05","r06","r07","r08","r09","r10","r11","r12","r13","r14"],
    "patience": 10
}

print("Loading dataset...")
data  = np.load(NPZ_PATH, allow_pickle=True)
X     = data["X"]
Y     = data["y"] if "y" in data.files else data["Y"]
runs  = data["session"] if "session" in data.files else data["runs"]

def canon_run(x):
    x = str(x).lower().strip().replace("session","").replace("_","").replace("-","")
    if x.startswith("r") and x[1:].isdigit():   return "r"+x[1:].zfill(2)
    if x.startswith("run") and x[3:].isdigit(): return "r"+str(int(x[3:])).zfill(2)
    return x

runs = np.array([canon_run(r) for r in runs], dtype=object)
train_idx = np.array([i for i,r in enumerate(runs) if r in CFG["train_runs"]])
test_idx  = np.array([i for i,r in enumerate(runs) if r in CFG["test_runs"]])

X_train_np = X[train_idx]; Y_train_np = Y[train_idx]
X_test_np  = X[test_idx];  Y_test_np  = Y[test_idx]
runs_test  = runs[test_idx]

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
Y_train = torch.tensor(Y_train_np, dtype=torch.long)
Y_test  = torch.tensor(Y_test_np,  dtype=torch.long)
state_all   = np.array([0 if r in CFG["train_runs"] else 1 for r in runs], dtype=np.int64)
state_train = torch.tensor(state_all[train_idx], dtype=torch.long)
state_test  = torch.tensor(state_all[test_idx],  dtype=torch.long)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

class EEGDisentangleDS(Dataset):
    def __init__(self,X,Y,S): self.X=X; self.Y=Y; self.S=S
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.Y[i],self.S[i]

class EEGBackbone(nn.Module):
    def __init__(self,Cin=64,emb_in=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(Cin,64,kernel_size=7,padding=3), nn.BatchNorm1d(64), nn.ELU(),
            nn.Conv1d(64,128,kernel_size=5,padding=2), nn.BatchNorm1d(128), nn.ELU(),
            nn.Conv1d(128,256,kernel_size=3,padding=1), nn.BatchNorm1d(256), nn.ELU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self,x): return self.net(x).squeeze(-1)

class EEGDisentangle(nn.Module):
    def __init__(self,Cin=64,emb_in=256,emb_dim=128):
        super().__init__()
        self.backbone   = EEGBackbone(Cin,emb_in)
        self.id_head    = nn.Sequential(nn.Linear(256,emb_dim), nn.LayerNorm(emb_dim))
        self.state_head = nn.Sequential(nn.Linear(emb_dim,64), nn.ReLU(), nn.Linear(64,2))
        self.cs_head    = nn.Sequential(nn.Linear(256,emb_dim), nn.LayerNorm(emb_dim))
    def forward(self,x):
        f    = self.backbone(x)
        z_id = F.normalize(self.id_head(f), dim=1)
        z_cs = F.normalize(self.cs_head(f), dim=1)
        return z_id, z_cs

class ArcFaceLayer(nn.Module):
    def __init__(self,emb_dim,num_classes,s=30.0,m=0.50):
        super().__init__()
        self.s=s; self.m=m
        self.W = nn.Parameter(torch.FloatTensor(num_classes,emb_dim))
        nn.init.xavier_uniform_(self.W)
    def forward(self,x,labels):
        cosine = F.linear(F.normalize(x), F.normalize(self.W))
        theta  = torch.acos(cosine.clamp(-1+1e-7,1-1e-7))
        one_hot= torch.zeros_like(cosine).scatter_(1,labels.view(-1,1),1)
        output = torch.cos(theta + self.m*one_hot)*self.s
        return output

class SupConLoss(nn.Module):
    def __init__(self,temperature=0.07): super().__init__(); self.T=temperature
    def forward(self,features,labels):
        device=features.device; B=features.shape[0]
        f=F.normalize(features.squeeze(1),dim=1)
        sim=torch.matmul(f,f.T)/self.T
        mask=(labels.unsqueeze(0)==labels.unsqueeze(1)).float().to(device)
        mask.fill_diagonal_(0)
        exp_sim=torch.exp(sim-sim.max(dim=1,keepdim=True)[0])
        log_prob=sim-torch.log(exp_sim.sum(dim=1,keepdim=True)+1e-12)
        mean_log=((mask*log_prob).sum(dim=1))/(mask.sum(dim=1)+1e-12)
        return -mean_log.mean()

def extract_embeddings(model,X,Y,S):
    model.eval(); loader=DataLoader(EEGDisentangleDS(X,Y,S),batch_size=512,shuffle=False)
    embs=[]
    with torch.no_grad():
        for xb,yb,sb in loader:
            z_id,_=model(xb.to(DEVICE)); embs.append(z_id.cpu().numpy())
    E=np.concatenate(embs,axis=0)
    return E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)

def compute_auc_eer(scores,labels):
    fpr,tpr,_=roc_curve(labels,scores,pos_label=1)
    A=auc(fpr,tpr); fnr=1.0-tpr; k=np.nanargmin(np.abs(fpr-fnr))
    return float(A),(fpr[k]+fnr[k])/2.0

def evaluate_kmeans(E_train,E_test):
    K=CFG["kmeans_k"]; subjects=sorted(np.unique(Y_train_np).tolist())
    pvecs=[]; powner=[]
    for sid in subjects:
        idx=np.where(Y_train_np==sid)[0]; Xs=E_train[idx]
        if len(Xs)<K:
            c=Xs.mean(axis=0); c/=(np.linalg.norm(c)+1e-12)
            pvecs.append(c.astype(np.float32)); powner.append(int(sid)); continue
        km=KMeans(n_clusters=K,random_state=0,n_init=10).fit(Xs)
        for c in km.cluster_centers_:
            c/=(np.linalg.norm(c)+1e-12); pvecs.append(c.astype(np.float32)); powner.append(int(sid))
    pvecs=np.stack(pvecs); powner=np.array(powner,dtype=np.int64)
    rows=[]
    for run_name in CFG["test_runs"]:
        pidx=np.where(runs_test==run_name)[0]; P=E_test[pidx]; yr=Y_test_np[pidx]
        sm=P@pvecs.T; scores=[]; labels=[]
        for i in range(len(P)):
            yt=yr[i]; mg=(powner==yt); mi=(powner!=yt)
            scores.append(float(sm[i,mg].max())); labels.append(1)
            scores.extend(sm[i,mi].tolist()); labels.extend([0]*mi.sum())
        scores=np.array(scores,dtype=np.float64); labels=np.array(labels,dtype=np.int32)
        A,E=compute_auc_eer(scores,labels); rows.append({"run":run_name,"auc":A,"eer":E})
    return {"avg_auc":float(np.mean([r["auc"] for r in rows])),
            "avg_eer":float(np.mean([r["eer"] for r in rows])),"per_run":rows}

import csv
all_seed_rows=[]

for seed in SEED_LIST:
    print(f"\n{'='*70}\nSTARTING SEED {seed}\n{'='*70}")
    seed_dir=os.path.join(RUN_DIR,f"seed_{seed}"); os.makedirs(seed_dir,exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    tr_idx,val_idx=train_test_split(np.arange(len(X_train)),test_size=0.20,
                                     stratify=Y_train.numpy(),random_state=seed)
    X_tr=X_train[tr_idx]; Y_tr=Y_train[tr_idx]; S_tr=state_train[tr_idx]
    X_vl=X_train[val_idx]; Y_vl=Y_train[val_idx]; S_vl=state_train[val_idx]

    tr_loader=DataLoader(EEGDisentangleDS(X_tr,Y_tr,S_tr),batch_size=CFG["batch_size"],shuffle=True,drop_last=True)
    vl_loader=DataLoader(EEGDisentangleDS(X_vl,Y_vl,S_vl),batch_size=CFG["batch_size"],shuffle=False)

    num_classes=int(len(torch.unique(Y_tr)))
    model=EEGDisentangle(64,256,128).to(DEVICE)
    arc=ArcFaceLayer(128,num_classes).to(DEVICE)
    supcon=SupConLoss(CFG["temperature"])
    opt=torch.optim.Adam(list(model.parameters())+list(arc.parameters()),lr=CFG["lr"],weight_decay=CFG["weight_decay"])

    epoch_log=[]; best_val=float("inf"); pat=0
    log_file=open(os.path.join(seed_dir,"train_log.csv"),"w",newline="")
    writer=csv.writer(log_file)
    writer.writerow(["epoch","tr_loss","val_loss","tr_acc","val_acc","tr_arc","tr_sup","tr_orth","tr_state"])

    for ep in range(1, CFG["epochs"]+1):
        t0=time.time(); model.train(); arc.train()
        tl=ta=tc=tt=tarc=tsup=torth=tstate=0.0
        for xb,yb,sb in tr_loader:
            xb,yb,sb=xb.to(DEVICE),yb.to(DEVICE),sb.to(DEVICE)
            opt.zero_grad()
            z_id,z_cs=model(xb)
            al=nn.CrossEntropyLoss()(arc(z_id,yb),yb)
            sl=supcon(F.normalize(z_id.unsqueeze(1),dim=-1),yb)
            stl=nn.CrossEntropyLoss()(model.state_head(z_id),sb)
            dot=(F.normalize(z_id,dim=1)*F.normalize(z_cs,dim=1)).sum(dim=1)
            ol=(dot**2).mean()
            loss=al+CFG["lambda_supcon"]*sl+CFG["lambda_state"]*stl+CFG["lambda_orth"]*ol
            loss.backward(); opt.step()
            tl+=loss.item(); tarc+=al.item(); tsup+=sl.item(); torth+=ol.item(); tstate+=stl.item()
            preds=arc(z_id,yb).argmax(dim=1); tc+=(preds==yb).sum().item(); tt+=len(yb)
        nb=len(tr_loader)
        tl/=nb; tarc/=nb; tsup/=nb; torth/=nb; tstate/=nb; ta=tc/tt

        model.eval(); vl=0.0; vc=0; vt=0
        with torch.no_grad():
            for xb,yb,sb in vl_loader:
                xb,yb,sb=xb.to(DEVICE),yb.to(DEVICE),sb.to(DEVICE)
                z_id,z_cs=model(xb)
                al2=nn.CrossEntropyLoss()(arc(z_id,yb),yb)
                sl2=supcon(F.normalize(z_id.unsqueeze(1),dim=-1),yb)
                stl2=nn.CrossEntropyLoss()(model.state_head(z_id),sb)
                dot2=(F.normalize(z_id,dim=1)*F.normalize(z_cs,dim=1)).sum(dim=1)
                ol2=(dot2**2).mean()
                vl+=(al2+CFG["lambda_supcon"]*sl2+CFG["lambda_state"]*stl2+CFG["lambda_orth"]*ol2).item()
                preds=arc(z_id,yb).argmax(dim=1); vc+=(preds==yb).sum().item(); vt+=len(yb)
        vl/=len(vl_loader); va=vc/vt
        elapsed=time.time()-t0

        print(f"[seed={seed}] Ep {ep:02d}/{CFG['epochs']} | tr_loss={tl:.4f} val_loss={vl:.4f} | tr_acc={ta:.4f} val_acc={va:.4f} | arc={tarc:.4f} sup={tsup:.4f} orth={torth:.4f} | time={elapsed:.1f}s")
        writer.writerow([ep,tl,vl,ta,va,tarc,tsup,torth,tstate])
        log_file.flush()

        if vl<best_val:
            best_val=vl; pat=0
            torch.save({"epoch":ep,"model_state":model.state_dict(),"arc_state":arc.state_dict(),"val_loss":vl},
                       os.path.join(seed_dir,"model_best.pt"))
            print(f"  ✅ Best model saved (val_loss={vl:.4f})")
        else:
            pat+=1
            if pat>=CFG["patience"]:
                print(f"  ⏹ Early stopping at epoch {ep}"); break

    log_file.close()
    E_train=extract_embeddings(model,X_train,Y_train,state_train)
    E_test =extract_embeddings(model,X_test, Y_test, state_test)
    result =evaluate_kmeans(E_train,E_test)
    print(f"[seed={seed}] AVG AUC={result['avg_auc']:.4f} | AVG EER={result['avg_eer']*100:.2f}%")
    with open(os.path.join(seed_dir,"summary.json"),"w") as f:
        json.dump({"seed":seed,"avg_auc":result["avg_auc"],"avg_eer":result["avg_eer"],"per_run":result["per_run"]},f,indent=2)
    all_seed_rows.append({"seed":seed,"avg_auc":result["avg_auc"],"avg_eer":result["avg_eer"]})

df=pd.DataFrame(all_seed_rows)
print("\n===== FINAL MULTI-SEED SUMMARY =====")
print(df.to_string())
print(f"\nMean AUC : {df['avg_auc'].mean():.4f} ± {df['avg_auc'].std():.4f}")
print(f"Mean EER : {df['avg_eer'].mean()*100:.2f}% ± {df['avg_eer'].std()*100:.2f}%")
df.to_csv(os.path.join(RUN_DIR,"multi_seed_summary.csv"),index=False)
print("✅ ALL DONE. Results saved to:", RUN_DIR)
