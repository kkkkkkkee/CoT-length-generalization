import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from attention_inference import transform, InferenceAttention, METHODS
from data import Task,permute_predicates
from sparse_model import SparseClauseTransformer,SparseModelConfig
from evaluate import greedy_rollout

@pytest.mark.parametrize('method',METHODS,ids=lambda m:m['name'])
def test_causal_normalized_and_uses_model_rule_preference(method):
    task=Task('C5',8);c=task.generate(3,4,torch.Generator().manual_seed(13))['clauses'][:,:-1]
    q=torch.arange(4,8);a=torch.rand(3,1,4,8)
    future=torch.arange(8)[None,None,None,:]>q[None,None,:,None]
    a.masked_fill_(future,0);a/=a.sum(-1,keepdim=True)
    z=transform(a,c,q,method['kind'],method['beta'])
    assert (z>=0).all() and torch.isfinite(z).all()
    assert not z.masked_select(future).any()
    torch.testing.assert_close(z.sum(-1),torch.ones_like(z[...,0]))
    if method['kind']=='grouped':
        torch.testing.assert_close(z[...,:4].sum(-1),torch.full_like(z[...,0],.5))
        torch.testing.assert_close(z.gather(-1,q[None,None,:,None].expand(3,1,-1,1)),torch.full((3,1,4,1),.5))
        assert torch.equal(z[...,:4].argmax(-1),a[...,:4].argmax(-1))
    # Change variable IDs/values without changing clause types: no oracle identity used.
    c2=c.clone();c2[c2>0]=1
    torch.testing.assert_close(z,transform(a,c2,q,method['kind'],method['beta']))


def test_grouped_permutation_equivariance_and_no_predicate_fallback():
    c=Task('C5',8).generate(2,4,torch.Generator().manual_seed(14))['clauses'][:,:-1]
    a=torch.ones(2,1,1,8)/8;q=torch.tensor([7]);a[...,0]=.3;a/=a.sum(-1,keepdim=True)
    order=torch.tensor([2,0,3,1,4,5,6,7])
    z=transform(a,c,q,'grouped',2)
    torch.testing.assert_close(z[...,order],transform(a[...,order],c[:,order],q,'grouped',2))
    c.zero_();torch.testing.assert_close(a,transform(a,c,q,'grouped',2))


def test_true_greedy_matches_tf_exact_trajectory_and_baseline():
    torch.manual_seed(7);task=Task('C5',8)
    model=SparseClauseTransformer(SparseModelConfig(task.vocab_size,8,hidden=12))
    c=task.generate(3,4,torch.Generator().manual_seed(19))['clauses']
    original=InferenceAttention(model)
    torch.testing.assert_close(greedy_rollout(model,c[:,:5],4),greedy_rollout(original,c[:,:5],4))
    for kind in ['sharpen','grouped']:
        wrapped=InferenceAttention(model,kind,2)
        tf=wrapped.forward_queries(c[:,:-1],torch.arange(4,8)).argmax(-1)
        roll=greedy_rollout(wrapped,c[:,:5],4)
        assert torch.equal(tf.eq(c[:,5:]).all((1,2)),roll.eq(c[:,5:]).all((1,2)))


def test_existing_evaluator_interface_and_oracle_rejection():
    from sparse_experiment import evaluate
    task=Task('C5',8)
    model=SparseClauseTransformer(SparseModelConfig(task.vocab_size,8,hidden=12))
    wrapped=InferenceAttention(model,'grouped',2)
    rows=evaluate(wrapped,task,[4],samples=3,rollout_samples=2,predicate_shuffle=True)
    assert rows[0]['rollout']['samples']==2
    with pytest.raises(ValueError,match='forbidden'):
        evaluate(wrapped,task,[4],samples=2,rollout_samples=0,oracle=True)
