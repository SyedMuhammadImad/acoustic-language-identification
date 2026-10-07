"""Three original architecture families, with train-only scaling and masked pooling."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from sklearn.preprocessing import StandardScaler
from audio import speaker_folds,measure

class MLP(nn.Module):
 def __init__(self,classes):
  super().__init__()
  self.net=nn.Sequential(nn.Linear(78,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Dropout(.3),
   nn.Linear(256,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(.3),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,classes))
 def forward(self,x,lengths=None):return self.net(x)
class Recurrent(nn.Module):
 def __init__(self,kind,classes):
  super().__init__();self.recurrent=(nn.LSTM if kind=='LSTM' else nn.GRU)(13,64,batch_first=True)
  self.head=nn.Sequential(nn.Dropout(.3),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,classes))
 def forward(self,x,lengths):
  if x.ndim!=3 or len(lengths)!=len(x) or (lengths<=0).any() or (lengths>x.shape[1]).any():raise ValueError('Invalid sequence lengths')
  packed=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
  output,_=self.recurrent(packed);output,_=pad_packed_sequence(output,batch_first=True)
  mask=torch.arange(output.shape[1],device=x.device)[None,:]<lengths.to(x.device)[:,None]
  pooled=(output*mask[:,:,None]).sum(dim=1)/lengths.to(x.device)[:,None]
  return self.head(pooled)

def batch(inputs,indices,sequence):
 if not sequence:return torch.tensor(np.stack([inputs[i] for i in indices]),dtype=torch.float32),None
 values=[torch.tensor(inputs[i],dtype=torch.float32) for i in indices]
 lengths=torch.tensor([len(v) for v in values],dtype=torch.long)
 return pad_sequence(values,batch_first=True),lengths
def fit(model,inputs,y,train,val,sequence,epochs,seed):
 rng=np.random.default_rng(seed);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
 loss_fn=nn.CrossEntropyLoss();best=float('inf');snapshot=None;wait=0;history=[]
 for epoch in range(epochs):
  model.train();order=rng.permutation(train);loss_total=0;seen=0
  # BatchNorm requires at least two training examples; merge a singleton tail.
  chunks=list(np.array_split(order,max(1,int(np.ceil(len(order)/32)))))
  for ids in chunks:
   a,lengths=batch(inputs,ids,sequence);target=torch.tensor(y[ids],dtype=torch.long)
   optimizer.zero_grad();loss=loss_fn(model(a,lengths),target)
   if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
   loss_total+=float(loss.detach())*len(ids);seen+=len(ids)
  model.eval()
  with torch.inference_mode():
   a,lengths=batch(inputs,val,sequence);validation=float(loss_fn(model(a,lengths),torch.tensor(y[val],dtype=torch.long)))
  history.append({'epoch':epoch+1,'train_loss':loss_total/seen,'validation_loss':validation})
  if validation<best-1e-5:best=validation;snapshot=copy.deepcopy(model.state_dict());wait=0
  else:wait+=1
  if wait>=6:break
 if snapshot is None:raise ValueError('Training produced no valid checkpoint')
 model.load_state_dict(snapshot);model.eval()
 return history
def predict(model,inputs,indices,sequence):
 result=[]
 with torch.inference_mode():
  for start in range(0,len(indices),32):
   a,lengths=batch(inputs,indices[start:start+32],sequence)
   result.extend(model(a,lengths).argmax(dim=1).tolist())
 return np.asarray(result)
def experiment(flat,sequences,y,groups,epochs):
 torch.set_num_threads(2);classes=len(np.unique(y));folds=[];pooled={name:np.full(len(y),-1) for name in ('MLP','LSTM','GRU')}
 for fold,(train,val,test) in enumerate(speaker_folds(y,groups)):
  flat_scale=StandardScaler().fit(flat[train]);flat_inputs=flat_scale.transform(flat).astype(np.float32)
  sequence_scale=StandardScaler().fit(np.concatenate([sequences[i] for i in train],axis=0))
  seq_inputs=[sequence_scale.transform(a).astype(np.float32) for a in sequences]
  entry={'fold':fold+1,'train':len(train),'validation':len(val),'test':len(test),'models':{}}
  for name in ('MLP','LSTM','GRU'):
   seed=42+fold;torch.manual_seed(seed)
   model=MLP(classes) if name=='MLP' else Recurrent(name,classes)
   inputs=flat_inputs if name=='MLP' else seq_inputs
   history=fit(model,inputs,y,train,val,name!='MLP',epochs,seed)
   p=predict(model,inputs,test,name!='MLP');pooled[name][test]=p
   entry['models'][name]={**measure(y[test],p,classes),'epochs_run':len(history),'best_validation_loss':min(h['validation_loss'] for h in history),'history':history}
   print('Completed speaker fold',fold+1,name,flush=True)
  folds.append(entry)
 return {'speaker_held_out':{'folds':folds,'pooled_models':{name:measure(y,p,classes) for name,p in pooled.items()}},
         'selection':'Validation speaker loss selects checkpoint; held-out test speaker is evaluated once afterwards.',
         'implementation':'PyTorch reimplementation of original MLP/LSTM/GRU architecture families; dynamic class count and valid-length pooling.'}
