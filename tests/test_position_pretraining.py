"""Checks for position learning, frozen FFN, reload, and shuffled metrics."""
import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import Task
from sparse_model import SparseClauseTransformer, SparseModelConfig
from routing_model import convert_model, load_routing_model
from sparse_experiment import evaluate
from routing_experiment import parameter_hash

@pytest.mark.parametrize('mode', ['none','learned_abs','sinusoidal','relative'])
@pytest.mark.parametrize('routing', ['dense','row_tied'])
def test_stage_updates_and_reload(tmp_path, mode, routing):
    torch.manual_seed(23)
    task = Task('C5', 8)
    model = SparseClauseTransformer(SparseModelConfig(task.vocab_size,8,hidden=16,position_mode=mode))
    c = task.generate(16,1,torch.Generator().manual_seed(15))['clauses']
    q0 = model.Q43.detach().clone()
    f0 = parameter_hash(model.ff.named_parameters())
    pos0 = None if model.position_embedding is None else model.position_embedding.detach().clone()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01)
    loss = torch.nn.functional.cross_entropy(model.forward_queries(c[:,:2],torch.tensor([1])).flatten(0,2),c[:,2:].flatten())
    loss.backward(); opt.step()
    assert parameter_hash(model.ff.named_parameters()) != f0
    assert torch.equal(model.Q43,q0)
    if mode in ('learned_abs','relative'): assert not torch.equal(model.position_embedding,pos0)
    if mode == 'sinusoidal': assert torch.equal(model.position_embedding,pos0)
    pos1 = None if model.position_embedding is None else model.position_embedding.detach().clone()
    target = convert_model(model,routing,mode)
    if pos1 is not None: torch.testing.assert_close(target.position_embedding,pos1,rtol=0,atol=0)
    f1 = parameter_hash(target.ff.named_parameters())
    opt = torch.optim.Adam([p for p in target.parameters() if p.requires_grad],lr=.01)
    loss = torch.nn.functional.cross_entropy(target.forward_queries(c[:,:2],torch.tensor([1]))[:,:,4].flatten(0,1),c[:,2:,4].flatten())
    loss.backward(); opt.step()
    assert parameter_hash(target.ff.named_parameters()) == f1
    if mode in ('learned_abs','relative'): assert not torch.equal(target.position_embedding,pos1)
    path = tmp_path/'model.pt'
    torch.save({'model':target.state_dict(),'model_config':target.config_dict(), 'routing_mode':routing,'stage':'attention'},path)
    loaded,_ = load_routing_model(path)
    torch.testing.assert_close(loaded(c[:,:2]),target(c[:,:2]),rtol=0,atol=0)

def test_shuffled_oracle_and_no_position_invariance():
    task=Task('C5',8)
    model=SparseClauseTransformer(SparseModelConfig(task.vocab_size,8,hidden=16))
    with torch.no_grad(): model.Q43.normal_(); model.Q44.normal_()
    normal=evaluate(model,task,[4],samples=12,rollout_samples=4,seed=91)
    shuffled=evaluate(model,task,[4],samples=12,rollout_samples=4,seed=91,predicate_shuffle=True)
    assert normal[0]['teacher_forced']['counts']==shuffled[0]['teacher_forced']['counts']
    assert normal[0]['rollout']['counts']==shuffled[0]['rollout']['counts']
    oracle=evaluate(model,task,[4],samples=12,rollout_samples=0,seed=91,oracle=True,predicate_shuffle=True)[0]
    assert oracle['attention']['target_rule_mass']==pytest.approx(.5)
    assert oracle['attention']['current_state_mass']==pytest.approx(.5)
    assert oracle['panel_sha256']!=normal[0]['panel_sha256']
