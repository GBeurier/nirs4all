"""Fresh-process Torch preparation before framework seeding and native calls."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _fresh(code: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", JAX_SKIP_CUDA_CONSTRAINTS_CHECK="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    completed = subprocess.run([sys.executable, "-B", "-c", code, *arguments], cwd=ROOT, env=environment,
                               capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


@pytest.mark.torch
@pytest.mark.parametrize("cv", ["no_cv", "cv", "serialized", "configs", "wrapper", "json"])
def test_cold_public_fit_refit_and_archive_keep_independent_torch_oracle(tmp_path, cv):
    _fresh("""
import resource,sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import pytest
from tests.integration.api import test_named_torch_nd as case
assert 'torch._dynamo' not in sys.modules
assert 'triton._C.libtriton' not in sys.modules
patch=pytest.MonkeyPatch()
case.native_only.__wrapped__(patch)
if sys.argv[2] in {'serialized','configs','wrapper','json'}:
    def run_serialized(cohort,root,*,cv=True):
        import nirs4all
        model={'class':'nirs4all.operators.models.multimodal.MultimodalRegressor','params':{
            'transformers':dict.fromkeys(cohort.sources,'passthrough'),'fusion':'intermediate',
            'model':{'class':'nirs4all.pipeline.dagml.torch_estimator.DagMLTorchEstimator','params':{
                'factory_path':'tests.fixtures.named_torch_nd.joint_factory','factory_params':{'hidden_units':3},
                'device':'cpu','task_type':'regression','epochs':2,'batch_size':4,'patience':2,'lr':0.01}}}}
        pipeline=[{'model':model}]
        if sys.argv[2]=='configs':
            from nirs4all.pipeline import PipelineConfigs
            pipeline=PipelineConfigs(pipeline)
        elif sys.argv[2]=='wrapper':
            pipeline={'steps':pipeline,'metadata':{'framework':'pytorch'}}
        elif sys.argv[2]=='json':
            import json
            path=Path(sys.argv[1])/'pipeline.json'
            path.write_text(json.dumps({'steps':pipeline}),encoding='utf-8')
            pipeline=path
        return nirs4all.run(pipeline,cohort,engine='dag-ml',refit=True,random_state=31,
            cpu_threads=1,gpu_devices=[],save_artifacts=True,save_charts=False,workspace_path=root,verbose=0)
    patch.setattr(case,'_run',run_serialized)
try:
    representation='signal_1d' if sys.argv[2]=='configs' else 'tabular_numeric'
    case.test_actual_fixed_io_shapes_train_jointly_and_replay_without_fit(representation,sys.argv[2]=='cv',patch,Path(sys.argv[1]))
finally:
    patch.undo()
""", str(tmp_path), cv)


@pytest.mark.torch
def test_runtime_preparation_preserves_rng_and_never_calls_user_factory_or_optimizer():
    _fresh("""
import torch
from nirs4all.pipeline.dagml.torch_estimator import DagMLTorchEstimator,prepare_torch_runtime
from nirs4all.operators.models.multimodal import MultimodalRegressor
def forbidden(*args,**kwargs):raise AssertionError('preparation constructed a factory or optimizer')
torch.optim.Adam=torch.optim.SGD=forbidden
model=DagMLTorchEstimator(factory_path='unused.forbidden',optimizer={'type':'SGD','lr':0.01})
params=model.get_params(deep=False).copy()
before=torch.get_rng_state().clone()
pipeline=[{'_or_':[[{'model':MultimodalRegressor({'nir':'passthrough'},model=model)}]]}]
prepare_torch_runtime(pipeline)
assert torch.equal(before,torch.get_rng_state())
assert model.get_params(deep=False)==params
assert not hasattr(model,'model_')
""")


def test_non_torch_declarations_keep_torch_optimizer_imports_lazy():
    _fresh("""
import sys
from sklearn.linear_model import Ridge
from nirs4all.pipeline.dagml.torch_estimator import prepare_torch_runtime
assert 'torch._dynamo' not in sys.modules
prepare_torch_runtime([{'model':Ridge()}])
from nirs4all.pipeline import PipelineConfigs
prepare_torch_runtime(PipelineConfigs([{'model':{'class':'sklearn.linear_model.Ridge','params':{'alpha':1.0}}}]))
prepare_torch_runtime([{'model':{'class':'sklearn.linear_model.Ridge','params':{
    'alpha':{'_range_':[1,3]},'metadata':{'framework':'pytorch'}}},
    'metadata':{'framework':'pytorch','type':'function','func':'unused.forbidden'},
    'train_params':{'framework':'pytorch'}}])
prepare_torch_runtime([{'metadata':{'class':'torch.nn.Linear','framework':'pytorch'}}])
assert 'torch._dynamo' not in sys.modules
""")


@pytest.mark.torch
def test_decorated_function_descriptors_and_model_choices_do_not_construct():
    _fresh("""
import sys,torch
from nirs4all.pipeline import PipelineConfigs
from nirs4all.pipeline.dagml.torch_estimator import prepare_torch_runtime
def factory(*args,**kwargs):raise AssertionError('runtime preparation invoked user factory')
factory.framework='pytorch'
class ForbiddenModule(torch.nn.Module):
    def __init__(self):raise AssertionError('runtime preparation constructed user module')
configs=PipelineConfigs([{'model':{'_or_':[{'function':'__main__.factory'},'__main__.ForbiddenModule']}}])
before=torch.get_rng_state().clone()
assert 'torch._dynamo' not in sys.modules
prepare_torch_runtime([{'model':{'type':'function','framework':'tensorflow','func':factory}}])
assert 'torch._dynamo' not in sys.modules
prepare_torch_runtime(configs)
prepare_torch_runtime([{'model':{'type':'function','framework':'pytorch','func':{'func':factory}}}])
assert 'torch._dynamo' in sys.modules
assert torch.equal(before,torch.get_rng_state())
""")


@pytest.mark.parametrize("torch_model", [False, True])
@pytest.mark.parametrize("seed", [19, None])
def test_serialized_constructors_keep_post_seed_order_and_default_rng(torch_model, seed):
    _fresh("""
import random,sys
import numpy as np
import pytest,torch
from sklearn.base import BaseEstimator
from nirs4all.pipeline.dagml.run_backend import run_via_dagml
use_torch=sys.argv[1]=='1';seed=None if sys.argv[2]=='None' else int(sys.argv[2])
observed=[]
class SeededConstructor(torch.nn.Module if use_torch else BaseEstimator):
    def __init__(self):
        super().__init__()
        observed.append((np.random.random(),random.random(),torch.rand(3)))
torch.manual_seed(73);np.random.seed(73);random.seed(73)
expected_seed=73 if seed is None else seed
generator=torch.Generator().manual_seed(expected_seed)
expected_torch=torch.rand(3,generator=generator)
expected_numpy=np.random.RandomState(expected_seed).random_sample()
expected_python=random.Random(expected_seed).random()
class FinishedConstruction(Exception):pass
def stop(pipeline):raise FinishedConstruction
patch=pytest.MonkeyPatch()
patch.setattr('nirs4all.pipeline.dagml.methods_multimodal.methods_model_in_pipeline',stop)
try:
    try:run_via_dagml([{'model':{'class':'__main__.SeededConstructor'}}],None,random_state=seed,verbose=0)
    except FinishedConstruction:pass
    else:raise AssertionError('constructor boundary was not reached')
finally:patch.undo()
assert len(observed)==1,observed
assert observed[0][0]==expected_numpy
assert observed[0][1]==expected_python
assert torch.equal(observed[0][2],expected_torch)
""", str(int(torch_model)), str(seed))


@pytest.mark.torch
def test_cpu_optimizer_preparation_without_optional_triton():
    _fresh("""
import importlib.abc,importlib.util,sys
original=importlib.util.find_spec
def find_spec(name,*args,**kwargs):
    return None if name=='triton' or name.startswith('triton.') else original(name,*args,**kwargs)
class NoTriton(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='triton' or fullname.startswith('triton.'):
            raise ModuleNotFoundError('optional Triton is absent',name=fullname)
importlib.util.find_spec=find_spec
sys.meta_path.insert(0,NoTriton())
from nirs4all.pipeline.dagml.torch_estimator import DagMLTorchEstimator,prepare_torch_runtime
prepare_torch_runtime([DagMLTorchEstimator(factory_path='unused.factory')])
assert 'triton' not in sys.modules
import torch
model=torch.nn.Linear(2,1)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
optimizer.zero_grad();model(torch.ones(2,2)).sum().backward();optimizer.step()
""")
