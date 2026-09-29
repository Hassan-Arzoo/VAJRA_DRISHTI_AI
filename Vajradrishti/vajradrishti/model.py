"""Random forest baseline; labels and training examples are synthetic."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .features import FEATURE_NAMES

HORIZONS=(5,10,15,30,60)

def training_data(seed=42,n=240):
    rng=np.random.default_rng(seed)
    scale=np.array([25,20,12,.2,.3,.2,18,.2,15,.08,.2,5,15,4,3,.5,.5,.5,.5])
    offset=np.array([25,40,15,.2,.55,.7,265,.1,12,.05,.2,27,78,8,1005,.5,.5,.5,.5])
    x=rng.normal(size=(n,len(FEATURE_NAMES)))*scale+offset
    x[:,15:]=rng.integers(0,2,size=(n,4))
    core=(x[:,0]/65+x[:,1]/75+x[:,4]+x[:,8]/20+x[:,12]/140+x[:,3])/5
    targets=[]
    for base,gain in ((-2.8,5.2),(-2.4,4.7)):
        cols=[]
        for h in HORIZONS:
            p=1/(1+np.exp(-(base+gain*core*np.exp(-(h-5)/95)+.3*x[:,10])))
            cols.append(rng.binomial(1,np.clip(p,.01,.99)))
        targets.append(np.array(cols).T)
    return x,targets

def train_model(seed=42,events=240,trees=160,max_depth=8):
    x,ys=training_data(seed,events); fitted={"lightning":[],"thunderstorm":[]}
    for j in range(5):
        for name,y in zip(fitted,ys):
            model=RandomForestClassifier(n_estimators=trees,max_depth=max_depth,min_samples_leaf=3,class_weight="balanced",random_state=seed+j)
            model.fit(x,y[:,j]); fitted[name].append(model)
    return fitted

def predict(models,features):
    row=np.array([[features[n] for n in FEATURE_NAMES]]); result={}
    for target,mods in models.items():
        ps=[]
        for m in mods:
            classes=list(m.classes_); ps.append(float(m.predict_proba(row)[0][classes.index(1)]) if 1 in classes else 0.)
        result[target]={f"{h}_min":round(float(p),4) for h,p in zip(HORIZONS,np.minimum.accumulate(ps))}
    return result
