import numpy as np,pytest
from audio import MultinomialLogisticRegression as Softmax,speaker_folds,discover,features,LanguageDetector

def test_gradient_matches_finite_differences():
 rng=np.random.default_rng(42);x=np.c_[rng.normal(size=(8,3)),np.ones(8)];y=np.array([0,1,2,0,1,2,0,2]);w=rng.normal(size=(4,3))*.2
 loss,gradient=Softmax.loss_gradient(x,y,w,.03)
 assert np.isfinite(loss)
 for index in np.ndindex(w.shape):
  a,b=w.copy(),w.copy();a[index]+=1e-6;b[index]-=1e-6
  numeric=(Softmax.loss_gradient(x,y,a,.03)[0]-Softmax.loss_gradient(x,y,b,.03)[0])/2e-6
  assert abs(numeric-gradient[index])<1e-7

def test_noncontiguous_labels_and_probability_normalization():
 x=np.array([[-3,-3],[-2,-3],[3,-3],[2,-3],[0,3],[0,2]],dtype=float);y=np.array([10,10,40,40,90,90])
 model=Softmax(epochs=500).fit(x,y)
 assert np.array_equal(model.predict(x),y)
 assert np.allclose(model.predict_proba(x).sum(axis=1),1)
 assert model.loss_history[-1]<model.loss_history[0]
 np.testing.assert_allclose(model.predict_proba(np.array([[1e6,-1e6]] )).sum(axis=1),1)

def test_invalid_training_prediction_and_configuration():
 for kwargs in ({'epochs':0},{'epochs':1.5},{'learning_rate':np.inf},{'learning_rate':0},{'l2':-1}):
  with pytest.raises(ValueError):Softmax(**kwargs)
 for x,y in (([],[]),([[np.nan]],[0]),([[1],[2]],[0]),([[1],[2]],[1,1]),([[1],[2]],[np.nan,2])):
  with pytest.raises(ValueError):Softmax().fit(x,y)
 with pytest.raises(ValueError):Softmax().predict([[1]])
 model=Softmax(epochs=3).fit([[1],[2]],[0,1])
 with pytest.raises(ValueError):model.predict([[1,2]])

def test_speaker_splits_exclude_test_and_validation_speakers():
 groups=np.repeat(['A','B','C'],6);y=np.tile([0,1,2,0,1,2],3);seen=[]
 for train,val,test in speaker_folds(y,groups):
  assert set(groups[train]).isdisjoint(groups[test]) and set(groups[val]).isdisjoint(groups[test]) and set(groups[train]).isdisjoint(groups[val])
  seen.extend(test.tolist())
 assert sorted(seen)==list(range(len(y)))
 with pytest.raises(ValueError):list(speaker_folds(y[:12],groups[:12]))

def test_manifest_traversal_rejected(tmp_path):
 file=tmp_path/'manifest.csv';file.write_text('path,label,speaker\n../outside.wav,English,A\n')
 with pytest.raises(ValueError):discover(tmp_path,file)

def test_real_feature_extraction_and_silence_rejection(tmp_path):
 import soundfile as sf
 samples=np.sin(2*np.pi*300*np.arange(8000)/16000)*.1
 file=tmp_path/'tone.wav';sf.write(file,samples,16000)
 flat,sequence=features(file);assert flat.shape==(78,) and sequence.shape[1]==13 and np.isfinite(flat).all()
 detector=LanguageDetector().fit(np.stack([flat[:13]-5,flat[:13],flat[:13]+5]),['Low','Middle','High'])
 assert detector.predict_file(file)=='Middle'
 with pytest.raises(ValueError):LanguageDetector().predict_file(file)
 sf.write(file,np.zeros(8000),16000)
 with pytest.raises(ValueError):features(file)

def test_masked_recurrence_is_invariant_to_padding():
 # This check runs only in the neural repository.
 import importlib.util
 if importlib.util.find_spec('neural_models') is None:pytest.skip('Neural-model check applies to the neural repository')
 import torch
 from neural_models import Recurrent,MLP
 torch.manual_seed(42)
 for kind in ('LSTM','GRU'):
  model=Recurrent(kind,3).eval();short=torch.randn(1,5,13);padded=torch.cat([short,torch.full((1,7,13),1e6)],dim=1)
  with torch.inference_mode():assert torch.allclose(model(short,torch.tensor([5])),model(padded,torch.tensor([5])),atol=1e-6)
  with pytest.raises(ValueError):model(short,torch.tensor([6]))
 assert MLP(3).eval()(torch.zeros(2,78)).shape==(2,3)
