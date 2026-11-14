"""Scientific controls: exact continuation, data pairing, and selection boundary."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from replication_experiment import train_run, epoch_data_seed, select_alpha


def tiny_spec(policy):
    return {'task': 'C6', 'seed': 3, 'nvars': 16, 'token_dim': 4,
            'length': 2, 'epochs': 3, 'samples': 9, 'batch': 4, 'lr': 1e-4,
            'data_policy': policy, 'data_seed': 10003, 'probe_samples': 3,
            'validation_seed': 901, 'validation_lengths': [2, 4],
            'eval_every': 1, 'save_every': 1, 'milestones': [1, 3]}


@pytest.mark.parametrize('policy', ['fixed', 'fresh'])
def test_resume_preserves_optimizer_rng_and_data(tmp_path, policy):
    spec = tiny_spec(policy)
    uninterrupted = train_run(tmp_path / 'full', spec, device='cpu')
    assert train_run(tmp_path / 'resumed', spec, device='cpu', stop_after=1) is None
    resumed = train_run(tmp_path / 'resumed', spec, device='cpu')
    a = torch.load(uninterrupted, weights_only=False)
    b = torch.load(resumed, weights_only=False)
    assert a['data_hashes'] == b['data_hashes']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
    hashes = [x['sha256'] for x in a['data_hashes']]
    assert len(set(hashes)) == (1 if policy == 'fixed' else 3)
    with pytest.raises(RuntimeError, match='Configuration changed'):
        train_run(tmp_path / 'resumed', {**spec, 'lr': 0.1}, device='cpu')


def test_first_epoch_pairing_and_fresh_seeds():
    assert epoch_data_seed(2, 1, 'fixed') == epoch_data_seed(2, 1, 'fresh') == 10002
    assert len({epoch_data_seed(2, e, 'fresh') for e in range(1, 901)}) == 900
    assert epoch_data_seed(2, 2, 'fixed') == 10002


def test_validation_selection_aggregates_seeds_and_prefers_default_on_tie():
    def panel(values):
        return {'evaluations': [{'inverse_temperature': a, 'length': 5,
                'rollout': {'metrics': {'value_chain_accuracy': v}}} for a, v in values.items()]}
    assert select_alpha([panel({0.5: 1., 1.: 1., 2.: 1.})])['alpha'] == 1.
    assert select_alpha([panel({1.: 1., 2.: 0.8}), panel({1.: 0.4, 2.: 0.8})])['alpha'] == 2.
