import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

def data(n=50000,seed=27):
 r=np.random.default_rng(seed);g=r.binomial(1,.22,n);y=r.binomial(1,.28,n);intensity=r.normal(y*1.5-g*.55,1,n);texture=r.normal(y*.8,1,n);return np.c_[intensity,texture],y,g
def dice(y,p):
 return float(2*np.sum(y*p)/(np.sum(y)+np.sum(p)+1e-9))
def evaluate(seed=27):
 x,y,g=data(seed=seed);cut=35000;out={}
 for name,balanced in {"standard":False,"group_balanced":True}.items():
  weight=None
  if balanced:
   counts=np.bincount(g[:cut]);weight=np.array([len(g[:cut])/(2*counts[v]) for v in g[:cut]])
  m=LogisticRegression(max_iter=500).fit(x[:cut],y[:cut],sample_weight=weight);p=m.predict_proba(x[cut:])[:,1]>=.5;yt=y[cut:];gt=g[cut:];d=[dice(yt[gt==v],p[gt==v]) for v in [0,1]];out[name]={"overall_dice":dice(yt,p),"group_0_dice":d[0],"group_1_dice":d[1],"dice_gap":abs(d[0]-d[1])}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
