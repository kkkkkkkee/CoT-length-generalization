"""Exact attention geometry for row-tied routing, not a correctness theorem.

For a correct full prefix with distinct variables, L predicates and t earlier
answers before the current answer, matching scores are d43_i,d44_i and every
nonmatching key of each type has score b43_i,b44_i. Therefore

  m = (exp d43 + exp d44) /
      (exp d43 + exp d44 + (L-1) exp b43 + t exp b44),
  r = exp d43 / (exp d43 + exp d44).

At the worst position t=L-1, m>=tau iff
  L <= 1 + (1-tau)/tau * (exp d43+exp d44)/(exp b43+exp b44).

The resulting capacity concerns an attention-mass threshold only. It does not
guarantee accurate values, variables or free generation. The equations do not
apply to malformed/generated prefixes with different key patterns or repeats.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from data import Task
from routing_model import RoutingClauseTransformer, load_routing_model
from sparse_model import SparseModelConfig


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def coefficients(model):
    """Read row-tied effective scores; reject genuinely dense nonmatching rows."""
    output = {}
    n = model.config.nvars
    eye = torch.eye(n, dtype=torch.bool)
    for suffix in ["43", "44"]:
        q = getattr(model, f"Q{suffix}").detach().cpu().double()
        diagonal = q.diag()
        if n > 1:
            background = q.masked_fill(eye, 0).sum(-1) / (n-1)
            residual = (q-background[:, None]).masked_select(~eye)
            if not torch.allclose(residual, torch.zeros_like(residual), atol=1e-10, rtol=0):
                raise ValueError("Nonmatching scores are not row-tied; this exact formula does not apply")
        else:
            background = torch.zeros_like(diagonal)
        output[f"d{suffix}"] = diagonal
        output[f"b{suffix}"] = background
    return output


def geometry(scores, length, prior_answer_counts):
    """Return m,r arrays [query_variable,position] using stable log sums."""
    if length < 1:
        raise ValueError("length must be positive")
    t = torch.as_tensor(prior_answer_counts, dtype=torch.float64).reshape(-1)
    if torch.any(t < 0) or torch.any(t > length-1) or torch.any(t != t.floor()):
        raise ValueError("prior-answer counts must be integers in [0,L-1]")
    log_signal = torch.logsumexp(torch.stack([scores['d43'],scores['d44']]), dim=0)
    rule_background = scores['b43'][:,None] + (math.log(length-1) if length > 1 else -math.inf)
    answer_background = scores['b44'][:,None] + torch.where(t > 0, t.log(), -torch.inf)[None]
    log_background = torch.logaddexp(rule_background, answer_background)
    m = torch.exp(log_signal[:,None] - torch.logaddexp(log_signal[:,None],log_background))
    r = torch.exp(scores['d43'] - log_signal)[:,None].expand_as(m)
    return m,r


def log_capacity(scores, threshold=.95):
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    log_signal = torch.logsumexp(torch.stack([scores['d43'],scores['d44']]), dim=0)
    log_background = torch.logsumexp(torch.stack([scores['b43'],scores['b44']]), dim=0)
    log_excess = math.log1p(-threshold)-math.log(threshold)+log_signal-log_background
    return torch.logaddexp(torch.zeros_like(log_excess),log_excess)


def finite_exp(value):
    return math.exp(value) if value < math.log(torch.finfo(torch.float64).max) else None


def feasible_integer_lengths(logs, nvars):
    """Cap in log space without rounding exp(log(max_length)) below the cap."""
    maximum = nvars - 1
    if maximum < 1:
        raise ValueError('At least two variables are required for a positive chain length')
    bound = math.log(maximum)
    lengths = torch.exp(logs.clamp_max(bound)).floor().clamp_min(1)
    return torch.where(logs >= bound, torch.full_like(lengths, maximum), lengths)


def summarize_capacity(logs, nvars):
    # Quantiles are explicitly taken in log-capacity space, then exponentiated.
    levels = {'min':0.,'q05':.05,'median':.5,'q95':.95,'max':1.}
    summary = {}
    for name,q in levels.items():
        log_value = torch.quantile(logs,q).item()
        summary[f'{name}_log_capacity'] = log_value
        summary[f'{name}_capacity'] = finite_exp(log_value)
    feasible = feasible_integer_lengths(logs, nvars)
    summary['median_feasible_integer_length'] = feasible.median().item()
    summary['minimum_feasible_integer_length'] = feasible.min().item()
    summary['variable_vocabulary_maximum_length'] = nvars-1
    summary['quantile_definition'] = 'linear interpolation in log-capacity, then exponentiation'
    return summary


def validate_formula():
    """Independent sampled-model check, including the last-answer convention."""
    errors=[]
    for mode in ['row_tied','diagonal']:
        torch.manual_seed(84211)
        task=Task('S5',8)
        model=RoutingClauseTransformer(SparseModelConfig(task.vocab_size,8,hidden=12),mode).double()
        with torch.no_grad():
            model.diagonal43.normal_(2.,.5)
            model.diagonal44.normal_(1.8,.5)
            if model.background43 is not None:
                model.background43.normal_(-.3,.4)
                model.background44.normal_(.1,.4)
        scores=coefficients(model)
        for length in [1,2,4,7]:
            sample=task.generate(13,length,torch.Generator().manual_seed(84213+length))
            c=sample['clauses'][:,:-1]
            queries=torch.arange(length,2*length)
            with torch.no_grad():
                _,a=model.forward_queries(c,queries,return_attention=True)
            rules=torch.arange(length)[None,:,None].expand(len(c),-1,-1)
            selfkeys=queries[None,:,None].expand(len(c),-1,-1)
            ap=a[:,0].gather(-1,rules).squeeze(-1)
            ay=a[:,0].gather(-1,selfkeys).squeeze(-1)
            m,r=geometry(scores,length,torch.arange(length))
            variables=c[:,length:2*length,3]-1
            positions=torch.arange(length)[None].expand(len(c),-1)
            expected_m=m[variables,positions]
            expected_r=r[variables,positions]
            torch.testing.assert_close(ap+ay,expected_m,atol=2e-12,rtol=2e-12)
            torch.testing.assert_close(ap/(ap+ay),expected_r,atol=2e-12,rtol=2e-12)
            errors.append(float((ap+ay-expected_m).abs().max()))
        logs=log_capacity(scores)
        caps=logs.exp()
        # Substitute the real-valued boundary directly in the scalar formula.
        log_signal=torch.logsumexp(torch.stack([scores['d43'],scores['d44']]),dim=0)
        log_back=torch.logsumexp(torch.stack([scores['b43'],scores['b44']]),dim=0)
        boundary=torch.exp(log_signal-torch.logaddexp(log_signal,(caps-1).log()+log_back))
        torch.testing.assert_close(boundary,torch.full_like(boundary,.95),atol=2e-12,rtol=2e-12)
    huge={k:torch.tensor([1000.,2000.],dtype=torch.float64) if k.startswith('d') else torch.zeros(2,dtype=torch.float64)
          for k in ['d43','d44','b43','b44']}
    assert torch.isfinite(log_capacity(huge)).all()
    feasible = feasible_integer_lengths(torch.tensor([math.log(2.5), math.log(127), 1000.], dtype=torch.float64), 128)
    torch.testing.assert_close(feasible, torch.tensor([2., 127., 127.], dtype=torch.float64))
    return {'passed':True,'models':['row_tied','diagonal'],'sampled_lengths':[1,2,4,7],
            'maximum_attention_mass_error':max(errors),'checks':['exact sampled attention mass','exact ratio','threshold boundary','overflow-safe log capacity','feasible vocabulary cap']}


def ranks(values):
    order=sorted(range(len(values)),key=values.__getitem__)
    result=[0.]*len(values)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:
            j+=1
        for k in range(i,j): result[order[k]]=(i+j-1)/2
        i=j
    return result


def correlation(x,y):
    if len(x)<3 or len(set(x))<2 or len(set(y))<2:return None
    a=torch.tensor([x,y],dtype=torch.float64)
    return torch.corrcoef(a)[0,1].item()


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields or ['run'])
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=ROOT/'runs/routing-recursive-2026-11-09')
    p.add_argument('--output',type=Path,default=ROOT/'figures/routing-recursive-2026-11-09/capacity')
    p.add_argument('--checkpoints',type=Path,nargs='*')
    p.add_argument('--threshold',type=float,default=.95)
    args=p.parse_args()
    torch.set_num_threads(2)
    validation=validate_formula()
    plan=read_json(args.runs/'campaign-plan.json') or {}
    specs={spec['name']:spec for spec in plan.get('runs',[])}
    paths=args.checkpoints if args.checkpoints else [args.runs/name/'checkpoint-final.pt' for name in specs]
    summaries=[];variables=[];performance=[];skipped=[]
    for checkpoint in paths:
        name=checkpoint.parent.name
        complete=read_json(checkpoint.parent/'complete.json')
        if not checkpoint.exists() or not complete or not complete.get('complete'):
            skipped.append({'run':name,'reason':'not complete'});continue
        model,state=load_routing_model(checkpoint,'cpu')
        if model.routing_mode not in {'row_tied','diagonal'}:
            skipped.append({'run':name,'reason':'dense routing is outside exact row-tied geometry'});continue
        scores=coefficients(model);logs=log_capacity(scores,args.threshold)
        config=read_json(checkpoint.parent/'config.json') or {}
        spec=specs.get(name,{})
        summary={'run':name,'routing_mode':model.routing_mode,'model_seed':config.get('seed'),
            'regime':spec.get('regime',config.get('label_source')),'stage':spec.get('stage',0),
            'training_length':config.get('length'),'threshold':args.threshold,
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            **summarize_capacity(logs,model.config.nvars)}
        summaries.append(summary)
        feasible = feasible_integer_lengths(logs, model.config.nvars)
        for i,logval in enumerate(logs.tolist()):
            variables.append({'run':name,'query_variable_token':i+1,'log_capacity':logval,
                'capacity':finite_exp(logval),'feasible_integer_length':int(feasible[i]),
                **{key:value[i].item() for key,value in scores.items()}})
        evaluation=read_json(checkpoint.parent/'evaluation.json') or {}
        if evaluation.get('checkpoint_sha256') != summary['checkpoint_sha256']:
            raise RuntimeError(f'Final evaluation hash does not match {checkpoint}')
        for row in evaluation.get('evaluations',[]):
            length=row['length']
            mass,ratio=geometry(scores,length,torch.arange(length))
            performance.append({**summary,'evaluation_length':length,
                'predicted_population_mean_relevant_mass':mass.mean().item(),
                'predicted_population_final_step_mean_relevant_mass':mass[:,-1].mean().item(),
                'predicted_population_rule_share_mean':ratio.mean().item(),
                'fraction_variables_meeting_threshold_at_last_step':(logs>=math.log(length)).double().mean().item(),
                'actual_panel_relevant_mass':row.get('attention',{}).get('relevant_mass'),
                'teacher_forced_value_accuracy':row['teacher_forced']['metrics']['value_accuracy'],
                'teacher_forced_clause_chain_accuracy':row['teacher_forced']['metrics']['clause_chain_accuracy'],
                'rollout_final_value_accuracy':row.get('rollout',{}).get('metrics',{}).get('final_value_accuracy')})
    grouped=defaultdict(list)
    for row in performance:grouped[(row['routing_mode'],row['evaluation_length'])].append(row)
    associations=[]
    for (mode,length),members in sorted(grouped.items()):
        x=[row['median_log_capacity'] for row in members]
        y=[row['teacher_forced_clause_chain_accuracy'] for row in members]
        associations.append({'routing_mode':mode,'evaluation_length':length,'n_checkpoints':len(members),
            'pearson_log_median_capacity_vs_chain_accuracy':correlation(x,y),
            'spearman_log_median_capacity_vs_chain_accuracy':correlation(ranks(x),ranks(y)),
            'warning':'Exploratory association across related checkpoints/stages and regimes; not independent replicates or causal evidence. No p-values.'})
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'capacity_by_run.csv',summaries)
    write_csv(args.output/'capacity_by_variable.csv',variables)
    write_csv(args.output/'capacity_and_performance.csv',performance)
    result={'formula_validation':validation,'threshold':args.threshold,'summaries':summaries,
        'associations':associations,'skipped':skipped,
        'interpretation':['Exact geometry only for correct full prefixes with distinct variables.',
            'Capacity is a real length boundary for worst-position attention mass, not an accuracy guarantee.',
            'No claim of a new theorem; parameter sharing makes this an exact softmax calculation.',
            'All errors in generated prefixes and imperfect FFN computation lie outside the mass-only calculation.',
            'Initial/projection and final training effects are not separated by this final-checkpoint-only diagnostic.'],
        'source':str(args.runs.resolve()),'checkpoints':len(summaries)}
    (args.output/'capacity.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'formula_validation':validation,'checkpoints':len(summaries),'skipped':len(skipped),'output':str(args.output)},indent=2))


if __name__=='__main__':main()
