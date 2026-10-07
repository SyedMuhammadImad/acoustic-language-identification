"""Portable coursework feature extraction and speaker-held-out experiments."""
import argparse,csv,hashlib,json,re
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix

LABELS={'eng':'English','en':'English','english':'English','ur':'Urdu','urdu':'Urdu',
        'mix':'Mixed','mixed':'Mixed','ue':'Mixed','ar':'Arabic','arabic':'Arabic'}
def matrix(x):
 x=np.asarray(x,dtype=float)
 if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():raise ValueError('Expected a nonempty finite feature matrix')
 return x
class MultinomialLogisticRegression:
 """Stable full-batch softmax regression with a separate unregularized bias."""
 def __init__(self,learning_rate=.05,epochs=1500,l2=1e-4):
  if not np.isfinite(learning_rate) or learning_rate<=0 or not isinstance(epochs,int) or epochs<=0 or not np.isfinite(l2) or l2<0:raise ValueError('Invalid optimizer settings')
  self.lr,self.epochs,self.l2=learning_rate,epochs,l2;self.weights=None;self.loss_history=[]
 @staticmethod
 def loss_gradient(x,y,w,l2=0):
  z=x@w;z-=z.max(axis=1,keepdims=True)
  logsum=np.log(np.exp(z).sum(axis=1));prob=np.exp(z-logsum[:,None]);target=np.eye(w.shape[1])[y]
  regular=w.copy();regular[-1]=0
  return float(np.mean(logsum-z[np.arange(len(y)),y])+l2*np.sum(regular**2)/2),x.T@(prob-target)/len(y)+l2*regular
 def fit(self,x,y):
  self.weights=None
  x=matrix(x);y=np.asarray(y)
  if y.ndim!=1 or len(y)!=len(x):raise ValueError('Invalid target shape')
  if y.dtype.kind in 'fi' and not np.isfinite(y).all():raise ValueError('Invalid target labels')
  self.classes_,target=np.unique(y,return_inverse=True)
  if len(self.classes_)<2:raise ValueError('At least two classes required')
  self.n_features_=x.shape[1];x=np.c_[x,np.ones(len(x))]
  self.weights=np.zeros((x.shape[1],len(self.classes_)));self.loss_history=[]
  for _ in range(self.epochs):
   loss,gradient=self.loss_gradient(x,target,self.weights,self.l2)
   if not np.isfinite(loss) or not np.isfinite(gradient).all():raise ValueError('Optimizer diverged')
   self.loss_history.append(loss);self.weights-=self.lr*gradient
  return self
 def predict_proba(self,x):
  if self.weights is None:raise ValueError('Model is not fitted')
  x=matrix(x)
  if x.shape[1]!=self.n_features_:raise ValueError('Feature count differs from training')
  with np.errstate(over='ignore',invalid='ignore'):z=np.c_[x,np.ones(len(x))]@self.weights
  if not np.isfinite(z).all():raise ValueError('Prediction overflow')
  z-=z.max(axis=1,keepdims=True)
  e=np.exp(z);return e/e.sum(axis=1,keepdims=True)
 def predict(self,x):
  probabilities=self.predict_proba(x)
  return self.classes_[probabilities.argmax(axis=1)]

def discover(root,manifest=None):
 root=Path(root).resolve()
 if not root.is_dir():raise ValueError('Dataset root must be a directory')
 records=[];seen={}
 if manifest:
  with Path(manifest).open(newline='',encoding='utf-8-sig') as f:
   rows=list(csv.DictReader(f))
  if not rows or any(not {'path','label','speaker'}<=row.keys() for row in rows):raise ValueError('Manifest needs path,label,speaker columns')
 else:
  rows=[]
  for path in sorted(root.rglob('*.wav')):
   match=re.fullmatch(r'(.+?)[ _-](eng|english|en|urdu|ur|mix|mixed|ue|arabic|ar)',path.parent.name.lower())
   if not match:raise ValueError('Unrecognized folder; provide an explicit CSV manifest')
   rows.append({'path':str(path.relative_to(root)),'label':LABELS[match[2]],'speaker':match[1]})
 for row in rows:
  rel=Path(row['path'])
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Manifest paths must be relative to data root')
  path=root/rel
  if not path.resolve().is_relative_to(root) or any((root/Path(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1)):raise ValueError('Dataset links or traversal are unsupported')
  if not path.is_file() or path.suffix.lower()!='.wav' or path.stat().st_size>32*1024*1024:raise ValueError('Expected a WAV file under 32 MB')
  label,speaker=row['label'].strip(),row['speaker'].strip()
  if not label or not speaker:raise ValueError('Empty label or speaker')
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if digest in seen:
   if seen[digest]!=(label,speaker):raise ValueError('Duplicate recording has conflicting labels or speaker')
   continue
  seen[digest]=(label,speaker);records.append((path,label,speaker,digest))
 if len(records)<6:raise ValueError('At least six distinct recordings required')
 return records

def features(path):
 import librosa,soundfile as sf
 info=sf.info(path)
 if info.duration<.15 or info.duration>60 or info.channels>2:raise ValueError('Audio must be 0.15–60 seconds and mono/stereo')
 audio,sr=librosa.load(path,sr=16000,mono=True,duration=30)
 if not np.isfinite(audio).all() or np.max(np.abs(audio))<1e-7:raise ValueError('Silent or invalid audio')
 mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=13,n_fft=512,hop_length=256).T
 d1=librosa.feature.delta(mfcc.T,width=9,mode='nearest').T
 d2=librosa.feature.delta(mfcc.T,width=9,order=2,mode='nearest').T
 flat=np.concatenate([v for a in (mfcc,d1,d2) for v in (a.mean(axis=0),a.std(axis=0))])
 if not np.isfinite(flat).all():raise ValueError('Invalid audio features')
 return flat.astype(np.float32),mfcc.astype(np.float32)

def load_features(records,cache=None):
 # Cache identity covers content, row order, labels, grouping and feature settings.
 key=hashlib.sha256(json.dumps([(r[1],r[2],r[3]) for r in records],ensure_ascii=False).encode()+b'mfcc13-16k-nfft512-hop256-first30s-v1').hexdigest()
 if cache:Path(cache).mkdir(parents=True,exist_ok=True)
 flat=[];sequences=[]
 for i,(path,_label,_speaker,digest) in enumerate(records):
  file=Path(cache)/(digest+'.npz') if cache else None
  if file and file.exists():
   with np.load(file,allow_pickle=False) as data:a,b=data['flat'],data['sequence']
  else:
   a,b=features(path)
   if file:np.savez_compressed(file,flat=a,sequence=b)
  if a.shape!=(78,) or b.ndim!=2 or b.shape[1]!=13 or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid feature cache')
  flat.append(a);sequences.append(b)
 return np.stack(flat),sequences,key

def measure(target,prediction,classes):
 p,r,f,_=precision_recall_fscore_support(target,prediction,labels=list(range(classes)),average='macro',zero_division=0)
 return {'accuracy':float(accuracy_score(target,prediction)),'macro_precision':float(p),'macro_recall':float(r),'macro_f1':float(f),
         'confusion_matrix':confusion_matrix(target,prediction,labels=list(range(classes))).tolist()}
class LanguageDetector:
 """Keep feature preprocessing and class decoding with the fitted classical model."""
 def __init__(self,kind='scratch'):
  if kind not in ('scratch','sklearn'):raise ValueError('Unknown model kind')
  self.kind=kind;self.model=None
 def fit(self,x,y):
  x=matrix(x)
  if x.shape[1]!=13:raise ValueError('Expected 13 mean MFCCs')
  self.scale=StandardScaler().fit(x)
  self.model=MultinomialLogisticRegression() if self.kind=='scratch' else LogisticRegression(max_iter=2000)
  self.model.fit(self.scale.transform(x),y)
  return self
 def predict_file(self,path):
  if self.model is None:raise ValueError('Detector is not fitted')
  flat,_=features(path)
  return self.model.predict(self.scale.transform(flat[None,:13]))[0]
def speaker_folds(y,groups):
 speakers=np.unique(groups)
 if len(speakers)<3:raise ValueError('At least three speakers required for separate train/validation/test speakers')
 for i,speaker in enumerate(speakers):
  test=np.flatnonzero(groups==speaker);val=np.flatnonzero(groups==speakers[(i+1)%len(speakers)])
  train=np.flatnonzero((groups!=speaker)&(groups!=speakers[(i+1)%len(speakers)]))
  if set(y[train])!=set(y) or set(y[val])!=set(y) or set(y[test])!=set(y):raise ValueError('Every speaker split must contain all classes')
  assert set(groups[train]).isdisjoint(groups[val]) and set(groups[train]).isdisjoint(groups[test])
  yield train,val,test
def classical(flat,y,groups):
 x=flat[:,:13];folds=[];predictions={name:np.full(len(y),-1) for name in ('scratch','sklearn')}
 for i,(train,_val,test) in enumerate(speaker_folds(y,groups)):
  # No tuned hyperparameters; both non-test speaker groups are training data.
  train=np.flatnonzero(groups!=groups[test[0]])
  scale=StandardScaler().fit(x[train]);a,b=scale.transform(x[train]),scale.transform(x[test])
  models={'scratch':MultinomialLogisticRegression().fit(a,y[train]),'sklearn':LogisticRegression(max_iter=2000).fit(a,y[train])}
  folds.append({'fold':i+1,'train':len(train),'test':len(test),'models':{}})
  for name,model in models.items():
   p=model.predict(b);predictions[name][test]=p;folds[-1]['models'][name]=measure(y[test],p,len(np.unique(y)))
 train,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
 scale=StandardScaler().fit(x[train]);clip={}
 for name,model in {'scratch':MultinomialLogisticRegression(),'sklearn':LogisticRegression(max_iter=2000)}.items():
  model.fit(scale.transform(x[train]),y[train]);clip[name]=measure(y[test],model.predict(scale.transform(x[test])),len(np.unique(y)))
 return {'speaker_held_out':{'folds':folds,'pooled_models':{name:measure(y,p,len(np.unique(y))) for name,p in predictions.items()}},
         'clip_random_split':{'train':len(train),'test':len(test),'models':clip,'warning':'Speakers occur in both sides; this is not unseen-speaker performance.'}}

def main(kind):
 parser=argparse.ArgumentParser(description='Media-free repository; local WAV recordings required for training.')
 parser.add_argument('--data-dir',required=True,type=Path);parser.add_argument('--manifest',type=Path)
 parser.add_argument('--cache-dir',type=Path);parser.add_argument('--output',required=True,type=Path)
 parser.add_argument('--epochs',type=int,default=30)
 args=parser.parse_args()
 if not 1<=args.epochs<=200:parser.error('epochs must be 1–200')
 records=discover(args.data_dir,args.manifest)
 flat,sequences,key=load_features(records,args.cache_dir)
 encoder=LabelEncoder().fit([r[1] for r in records]);y=encoder.transform([r[1] for r in records]);groups=np.array([r[2] for r in records])
 if kind=='scratch':result=classical(flat,y,groups)
 else:
  from neural_models import experiment
  result=experiment(flat,sequences,y,groups,args.epochs)
 result.update(samples=len(records),speakers=len(np.unique(groups)),labels=encoder.classes_.tolist(),dataset_feature_sha256=key,
               class_counts={label:int(np.sum(y==i)) for i,label in enumerate(encoder.classes_)},
               preprocessing='Mono 16kHz, first 30 seconds, MFCC13 n_fft512/hop256; train-only scaling')
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps({'samples':len(records),'speakers':len(np.unique(groups)),'labels':encoder.classes_.tolist(),'output':str(args.output)}))
