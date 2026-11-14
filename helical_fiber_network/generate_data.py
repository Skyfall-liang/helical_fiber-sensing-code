import argparse
from fiber_net.data import save_dataset
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='data/synthetic');p.add_argument('--train',type=int,default=4096)
    p.add_argument('--val',type=int,default=512);p.add_argument('--test',type=int,default=768);p.add_argument('--points',type=int,default=1536)
    a=p.parse_args();save_dataset(a.out,a.train,a.val,a.test,a.points);print('Generated:',a.out)
