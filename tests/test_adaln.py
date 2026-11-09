from types import SimpleNamespace
import json

import pytest
import torch
from safetensors.torch import save_file

from fasth3_pipeline.adaln import AdaLNTable, TableProjection, checkpoint_identity
from fasth3_pipeline.storage import INDEX
from fasth3_pipeline.export_adaln import schedule


def test_adaln_table_selects_ordered_timestep_tuple_and_clears_scope(tmp_path):
    for name in (INDEX, 'mxfp8_manifest.json', 'config.json'):
        (tmp_path / name).write_text(name)
    path = tmp_path / 'table.safetensors'
    rows = torch.arange(54).view(9, 6).bfloat16()
    save_file({'block-0': rows}, path,
              metadata={'schema': 'fasth3-adaln-v1', 'checkpoint': checkpoint_identity(tmp_path),
                        'timesteps': json.dumps([[0.25], [0.25, 0.5]])})
    table = AdaLNTable(path, tmp_path, [SimpleNamespace(adaln_proj=SimpleNamespace(hidden_size=1))])
    projection = TableProjection(table, 0)
    with table.use(torch.tensor([0.25, 0.5])):
        assert torch.equal(torch.cat(projection(torch.empty(0)), dim=-1), rows[3:])
    with pytest.raises(RuntimeError, match='scope'):
        projection(torch.empty(0))
    with pytest.raises(ValueError, match='does not cover'):
        with table.use(torch.tensor([0.5])):
            pass


def test_adaln_schedule_covers_reference_modalities():
    times = schedule()
    assert len(times) == len(set(times))
    assert {len(row) for row in times} == {1, 2, 3, 4}
