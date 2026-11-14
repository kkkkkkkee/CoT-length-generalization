"""A relabeling must preserve identity relations and complete LEGO semantics."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import Task
from routing_augmentation import rename_variables


def test_per_example_bijection_preserves_equality_and_nonvariables():
    nvars = 11
    tokens = torch.tensor([0, *range(1, nvars + 1), 4, 4, 9,
                           nvars + 1, nvars + 12, -1], dtype=torch.long)
    inputs = tokens.repeat(7, 1)
    original = inputs.clone()
    renamed = rename_variables(inputs, nvars, torch.Generator().manual_seed(92))
    assert renamed.shape == inputs.shape and renamed.dtype == inputs.dtype
    assert renamed.device == inputs.device and renamed.data_ptr() != inputs.data_ptr()
    assert torch.equal(inputs, original)
    mask = (inputs >= 1) & (inputs <= nvars)
    assert torch.equal(renamed[~mask], inputs[~mask])
    assert torch.equal(renamed[:, 1:nvars + 1].sort(-1).values,
                       torch.arange(1, nvars + 1).expand(len(inputs), -1))
    assert torch.equal(renamed[:, :, None] == renamed[:, None, :],
                       inputs[:, :, None] == inputs[:, None, :])
    assert torch.unique(renamed[:, 1:nvars + 1], dim=0).shape[0] == len(inputs)


def test_full_trace_keeps_actions_values_and_logical_links():
    task = Task('S5', nvars=17)
    sample = task.generate(13, 7, torch.Generator().manual_seed(87))
    clauses = sample['clauses']
    before = clauses.clone()
    renamed = rename_variables(clauses, task.nvars, torch.Generator().manual_seed(72))
    length = sample['actions'].shape[1]
    # Predicate target/source still name consecutive answers in the same trace.
    assert torch.equal(renamed[:, :length, 0], renamed[:, length + 1:, 3])
    assert torch.equal(renamed[:, :length, 2], renamed[:, length:-1, 3])
    assert torch.equal(renamed[:, :, 1], clauses[:, :, 1])
    assert torch.equal(renamed[:, :, 4], clauses[:, :, 4])
    values = renamed[:, length:, 4] - task.value_offset
    actions = renamed[:, :length, 1] - task.action_offset
    assert torch.equal(values[:, 1:], task.action_table[actions, values[:, :-1]])
    assert torch.equal(renamed[:, length:, 3].sort(-1).values.diff(dim=-1) > 0,
                       torch.ones(13, length, dtype=torch.bool))
    assert torch.equal(clauses, before)


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_generator_reproducibility_and_advancement_on_input_device(device):
    inputs = torch.arange(18, device=device).repeat(4, 1)
    first = rename_variables(inputs, 13, torch.Generator(device=device).manual_seed(123))
    second = rename_variables(inputs, 13, torch.Generator(device=device).manual_seed(123))
    other = rename_variables(inputs, 13, torch.Generator(device=device).manual_seed(124))
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    generator = torch.Generator(device=device).manual_seed(123)
    assert torch.equal(first, rename_variables(inputs, 13, generator))
    assert not torch.equal(first, rename_variables(inputs, 13, generator))


def test_empty_and_noncontiguous_inputs_are_not_mutated():
    empty = torch.empty(0, 3, 5, dtype=torch.long)
    assert rename_variables(empty, 8, torch.Generator()).shape == empty.shape
    inputs = torch.arange(40, dtype=torch.int32).reshape(4, 10).t()
    before = inputs.clone()
    renamed = rename_variables(inputs, 8, torch.Generator().manual_seed(8))
    assert torch.equal(inputs, before)
    assert renamed.shape == inputs.shape and renamed.dtype == inputs.dtype
    assert torch.equal(renamed[inputs > 8], inputs[inputs > 8])


def test_invalid_shape_dtype_and_vocabulary_are_rejected():
    with pytest.raises(ValueError, match='leading example'):
        rename_variables(torch.tensor(1), 3, torch.Generator())
    with pytest.raises(TypeError, match='integer token'):
        rename_variables(torch.ones(2, 3), 3, torch.Generator())
    with pytest.raises(ValueError, match='positive integer'):
        rename_variables(torch.ones(2, 3, dtype=torch.long), 0, torch.Generator())
