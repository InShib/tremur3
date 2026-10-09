from flask import Flask, render_template, jsonify, request
from pathlib import Path
from collections import Counter
import numpy as np, joblib, json, traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "Tim-Tremor"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH, META_PATH = MODEL_DIR / "model.joblib", MODEL_DIR / "metadata.json"
app = Flask(__name__)
LABELS = {0:"Label 0", 1:"Label 1", 2:"Label 2", 3:"Label 3"}

def extract_features(w):
    w = np.asarray(w, dtype=float)
    if w.shape != (128, 3):
        raise ValueError("A window must contain 128 samples with X, Y and Z axes.")
    out=[]
    for a in range(3):
        s=w[:,a]; c=s-s.mean()
        power=np.abs(np.fft.rfft(c))**2
        hz=np.fft.rfftfreq(len(c), d=1/50)
        ps=power.sum()+1e-12
        dom=float(hz[1+np.argmax(power[1:])]) if len(power)>1 else 0.0
        out.extend([s.mean(),s.std(),s.min(),s.max(),np.sqrt(np.mean(s*s)),
                    np.mean(np.abs(np.diff(s))),np.percentile(s,75)-np.percentile(s,25),
                    dom,float((hz*power).sum()/ps),
                    float(power[(hz>=3)&(hz<=12)].sum()/ps),np.sqrt(np.mean(c*c))])
    mag=np.sqrt((w*w).sum(axis=1)); c=mag-mag.mean()
    p=np.abs(np.fft.rfft(c))**2; hz=np.fft.rfftfreq(len(c),d=1/50); ps=p.sum()+1e-12
    out.extend([mag.mean(),mag.std(),np.sqrt(np.mean(c*c)),np.mean(np.abs(np.diff(mag))),
                float(hz[1+np.argmax(p[1:])]) if len(p)>1 else 0.0,
                float((hz*p).sum()/ps),float(p[(hz>=3)&(hz<=12)].sum()/ps)])
    return np.nan_to_num(np.asarray(out,dtype=float),nan=0,posinf=0,neginf=0)

def load_data():
    Xs=[]; ys=[]; groups=[]; windows=[]
    for xp in sorted(DATA_DIR.glob("*-X.npy"), key=lambda p:int(p.name.split("-")[0])):
        yp=xp.with_name(xp.name.replace("-X.npy","-Y.npy"))
        if not yp.exists(): continue
        try:
            xx=np.load(xp,allow_pickle=False); yy=np.load(yp,allow_pickle=False).reshape(-1)
            if xx.ndim!=3 or xx.shape[1:]!=(128,3) or len(xx)!=len(yy): continue
            for w,y in zip(xx,yy):
                label=int(y)
                if label not in LABELS: continue
                Xs.append(extract_features(w)); ys.append(label); groups.append(xp.stem.replace("-X","")); windows.append(w.astype(float))
        except Exception: continue
    if not Xs: raise RuntimeError("No valid matching *-X.npy and *-Y.npy files found in data/Tim-Tremor.")
    return np.vstack(Xs),np.asarray(ys),np.asarray(groups),windows

MODEL=None; META={}; XDATA=YDATA=GROUPS=WINDOWS=None; INIT_ERROR=None
try:
    XDATA,YDATA,GROUPS,WINDOWS=load_data()
    if MODEL_PATH.exists() and META_PATH.exists():
        try: MODEL=joblib.load(MODEL_PATH); META=json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception: MODEL=None
    if MODEL is None:
        split=GroupShuffleSplit(n_splits=1,test_size=.22,random_state=42)
        train,test=next(split.split(XDATA,YDATA,GROUPS))
        model=make_pipeline(SimpleImputer(strategy="median"),RandomForestClassifier(
            n_estimators=220,max_depth=12,min_samples_leaf=2,class_weight="balanced_subsample",random_state=42,n_jobs=-1))
        model.fit(XDATA[train],YDATA[train]); pred=model.predict(XDATA[test])
        accuracy=float(accuracy_score(YDATA[test],pred))
        macro=float(f1_score(YDATA[test],pred,average="macro",zero_division=0))
        model.fit(XDATA,YDATA); MODEL=model; joblib.dump(MODEL,MODEL_PATH)
        META={"samples":int(len(YDATA)),"segments":int(len(set(GROUPS.tolist()))),"features":int(XDATA.shape[1]),
              "label_counts":{str(k):int(v) for k,v in Counter(YDATA.tolist()).items()},
              "holdout_accuracy":accuracy,"holdout_macro_f1":macro,
              "validation":"Single group-held-out split by segment; exploratory only."}
        META_PATH.write_text(json.dumps(META,indent=2),encoding="utf-8")
except Exception as e:
    INIT_ERROR=str(e); traceback.print_exc()

def describe(w):
    w=np.asarray(w,float); mag=np.sqrt((w*w).sum(axis=1)); c=mag-mag.mean()
    p=np.abs(np.fft.rfft(c))**2; hz=np.fft.rfftfreq(len(c),d=1/50); valid=(hz>=.5)&(hz<=20)
    dom=float(hz[valid][np.argmax(p[valid])]) if valid.any() else 0.0
    return {"dominant_frequency_hz":round(dom,2),"rms_variation":round(float(np.sqrt(np.mean(c*c))),4),
            "magnitude_spread":round(float(mag.std()),4),"axis_x_std":round(float(w[:,0].std()),4),
            "axis_y_std":round(float(w[:,1].std()),4),"axis_z_std":round(float(w[:,2].std()),4),
            "mean_magnitude":round(float(mag.mean()),4)}

def prediction(w):
    f=extract_features(w).reshape(1,-1); label=int(MODEL.predict(f)[0])
    probs=MODEL.predict_proba(f)[0]
    return {"label":label,"label_name":LABELS[label],
            "probabilities":{str(int(c)):round(float(p),4) for c,p in zip(MODEL.classes_,probs)},
            "features":describe(w)}

@app.route("/")
def home(): return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({"ready":MODEL is not None,"error":INIT_ERROR,"samples":META.get("samples",0),
                    "segments":META.get("segments",0),"features":META.get("features",0),
                    "label_counts":META.get("label_counts",{}),"accuracy":META.get("holdout_accuracy"),
                    "macro_f1":META.get("holdout_macro_f1")})

@app.route("/api/test",methods=["POST"])
def test():
    if MODEL is None: return jsonify({"error":INIT_ERROR or "Model not ready"}),500
    try:
        payload=request.get_json(silent=True) or {}
        if payload.get("window") is not None:
            w=np.asarray(payload["window"],dtype=float)
            if w.shape!=(128,3): return jsonify({"error":"Expected 128 rows of [x,y,z] values."}),400
            result=prediction(w); result.update({"source":"hardware","window":w.tolist()})
        else:
            idx=int(payload.get("index",0)) % len(WINDOWS); w=WINDOWS[idx]
            result=prediction(w); result.update({"source":"dataset_demo","sample_index":idx+1,"total_samples":len(WINDOWS),"window":np.asarray(w).round(5).tolist()})
        return jsonify(result)
    except Exception as e: return jsonify({"error":str(e)}),400

if __name__=="__main__":
    print("TremorSense One-Click starting...")
    if MODEL is None: print("Model setup issue:",INIT_ERROR)
    else: print(f"Loaded {META['samples']} windows from {META['segments']} segments.")
    app.run(host="127.0.0.1",port=5001,debug=False)
