"""Train from independently generated labeled strain scenes."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from fiber_net.network import JointModel,loss_function
class Frames(Dataset):
    def __init__(self,path):
        with np.load(path) as data:
            self.data={k:torch.from_numpy(data[k]) for k in ('y','coef','contact','support','force')}
    def __len__(self): return len(self.data['y'])
    def __getitem__(self,i): return {k:v[i] for k,v in self.data.items()}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/synthetic');p.add_argument('--out',default='runs/joint')
    p.add_argument('--epochs',type=int,default=35);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    tr=Frames(Path(a.data)/'train.npz');va=Frames(Path(a.data)/'val.npz')
    model=JointModel(tr.data['y'].shape[1]);opt=torch.optim.Adam(model.parameters(),lr=.002)
    loader=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=0)
    vloader=DataLoader(va,batch_size=a.batch_size,num_workers=0)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    best=float('inf');history=[];start=time.time()
    for epoch in range(a.epochs):
        model.train();total=0.
        for batch in loader:
            opt.zero_grad();pred=model(batch['y']);loss,_=loss_function(pred,batch);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=loss.item()*len(batch['y'])
        model.eval();valid=0.
        with torch.no_grad():
            for batch in vloader: valid+=loss_function(model(batch['y']),batch)[0].item()*len(batch['y'])
        row=dict(epoch=epoch+1,train_loss=total/len(tr),val_loss=valid/len(va),elapsed_s=time.time()-start)
        history.append(row)
        if row['val_loss']<best:
            best=row['val_loss'];torch.save(dict(state_dict=model.state_dict(),n=model.n,channels=model.channels,
                epoch=epoch+1,val_loss=best,seed=a.seed),out/'best.pt')
        if epoch in (14,24):
            for group in opt.param_groups: group['lr']*=.4
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8')
        print(json.dumps(row),flush=True)
    (out/'training_config.json').write_text(json.dumps(vars(a)|{'torch':torch.__version__,'device':'cpu'},indent=2),encoding='utf8')
if __name__=='__main__': main()
