import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from attention_causal_probe import intervene, matched_random, diagnostics, conditions

def panel():
    g=torch.Generator().manual_seed(71)
    a=torch.rand((3,1,4,8),generator=g)
    q=torch.arange(4,8);r=torch.arange(4)
    a.masked_fill_(torch.arange(8)[None,None,None,:]>q[None,None,:,None],0)
    a/=a.sum(-1,keepdim=True)
    return a,r,q

@pytest.mark.parametrize('spec',[c for c in conditions() if c['kind']!='random'],ids=lambda c:c['name'])
def test_preserves_conservation_and_intended_factors(spec):
    a,r,q=panel();before=a.clone();z=intervene(a,r,q,spec)
    assert torch.equal(a,before)
    assert torch.isfinite(z).all() and (z>=0).all()
    torch.testing.assert_close(z.sum(-1),torch.ones_like(z[...,0]),atol=1e-6,rtol=0)
    assert not z.masked_select(torch.arange(8)[None,None,None,:]>q[None,None,:,None]).any()
    old=diagnostics(a,a,r,q);new=diagnostics(a,z,r,q)
    k=spec['kind']
    if k in ('original','sham'):assert torch.equal(a,z)
    if k=='ratio':torch.testing.assert_close(old['mass'],new['mass'],atol=1e-7,rtol=0)
    if k=='boost':
        torch.testing.assert_close(old['rule_share'],new['rule_share'],atol=1e-7,rtol=0)
        assert (new['mass']>=old['mass']-1e-7).all()
    if k in ('grid','ratio','boost_balance'):
        torch.testing.assert_close(new['rule_share'],torch.full_like(new['rule_share'],spec['ratio']),atol=1e-7,rtol=0)
    if k=='grid':torch.testing.assert_close(new['mass'],torch.full_like(new['mass'],spec['mass']),atol=1e-7,rtol=0)
    if k=='leak_rule':
        torch.testing.assert_close(new['state_weight'],old['state_weight'],atol=0,rtol=0)
        torch.testing.assert_close(new['rule_weight'],old['rule_weight']*(1-spec['fraction']),atol=1e-7,rtol=0)
        torch.testing.assert_close(new['l1'],2*old['rule_weight']*spec['fraction'],atol=1e-7,rtol=0)

@pytest.mark.parametrize('spec',[c for c in conditions()[2:11]],ids=lambda c:c['name'])
def test_random_exact_amplitude_and_batch_invariance(spec):
    a,r,q=panel();target=diagnostics(a,intervene(a,r,q,spec),r,q)['l1']
    z=matched_random(a,q,target,seed=31)
    torch.testing.assert_close((z.double()-a.double()).abs().sum(-1),target,atol=2e-7,rtol=0)
    torch.testing.assert_close(z.sum(-1),a.sum(-1),atol=2e-7,rtol=0)
    parts=torch.cat([matched_random(a[i:i+1],q,target[i:i+1],seed=31,offset=i) for i in range(3)])
    assert torch.equal(z,parts)
    assert (z>=0).all()
    assert not z.masked_select(torch.arange(8)[None,None,None,:]>q[None,None,:,None]).any()

def test_zero_mass_fallback_and_extreme_control():
    a,r,q=panel();a.zero_();a[...,0]=1
    z=intervene(a,r,q,dict(kind='grid',mass=.99,ratio=.5))
    assert torch.isfinite(z).all() and (z>=0).all()
    torch.testing.assert_close(z.sum(-1),torch.ones_like(z[...,0]),atol=1e-7,rtol=0)
    for delta in (0.,2.):
        c=matched_random(a,q,torch.full_like(a[...,0],delta),seed=41)
        torch.testing.assert_close((c-a).abs().sum(-1),torch.full_like(a[...,0],delta))
