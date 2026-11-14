"""CNN joint inference + fixed differentiable physical decoder."""
import torch
from torch import nn
from .physics import geometry

class JointModel(nn.Module):
    def __init__(self,n=1536,channels=24):
        super().__init__()
        self.n=n; self.channels=channels
        s,x,slope,intercept=geometry(n)
        design=torch.tensor(x,dtype=torch.float32)
        self.register_buffer('design',design)
        self.register_buffer('pinv',torch.linalg.pinv(design))
        self.register_buffer('slope',torch.tensor(slope,dtype=torch.float32))
        self.register_buffer('intercept',torch.tensor(intercept,dtype=torch.float32))
        static=torch.stack((design[:,0],design[:,1],torch.tensor(s/s[-1],dtype=torch.float32),self.slope))
        self.register_buffer('static',static)
        layers=[]
        for i,dilation in enumerate([1,2,4,8,16]):
            layers.extend([nn.Conv1d(6 if i==0 else channels,channels,5,padding=2*dilation,dilation=dilation),nn.SiLU()])
        self.encoder=nn.Sequential(*layers)
        self.support_head=nn.Conv1d(channels,1,1)
        self.force_head=nn.Conv1d(channels,1,1)
        # Learned correction to physics projection; zero-init keeps the initial estimate interpretable.
        self.shape_head=nn.Sequential(nn.Linear(channels*3,48),nn.SiLU(),nn.Linear(48,3))
        nn.init.zeros_(self.shape_head[-1].weight); nn.init.zeros_(self.shape_head[-1].bias)
        nn.init.zeros_(self.support_head.weight); nn.init.zeros_(self.support_head.bias)

    def forward(self,y):
        initial=y@self.pinv.T
        initial_bend=initial@self.design.T
        residual=y-initial_bend
        features=self.encoder(torch.cat((y[:,None],residual[:,None],self.static[None].expand(len(y),-1,-1)),dim=1))
        support=torch.tanh(self.support_head(features)[:,0])
        force=4*torch.sigmoid(self.force_head(features)[:,0])
        contact=support*(self.slope[None]*force+self.intercept[None])
        pooled=torch.cat((features.mean(-1),(features*self.design[:,0]).mean(-1),
                           (features*self.design[:,1]).mean(-1)),dim=1)
        coef=(y-contact)@self.pinv.T + .1*self.shape_head(pooled)
        bending=coef@self.design.T
        return dict(coef=coef,support=support,force=force,contact=contact,bending=bending,reconstruction=bending+contact)

def loss_function(out,batch):
    active=batch['support'].abs()
    weights=1+15*active
    contact=((out['contact']-batch['contact'])**2*weights).mean()
    support=((out['support']-batch['support'])**2*weights).mean()
    force=(((out['force']-batch['force'])**2)*active**4).sum()/(active**4).sum().clamp_min(1)
    shape=((out['coef']-batch['coef'])**2).mean()
    reconstruction=((out['reconstruction']-batch['y'])**2).mean()
    loss=5*contact+2*support+.15*force+4*shape+.5*reconstruction
    return loss,dict(contact=contact, support=support,force=force,shape=shape,reconstruction=reconstruction)
