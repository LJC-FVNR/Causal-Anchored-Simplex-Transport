import torch

from tasks.tensor_dir import TensorDirectorySequenceTask


def _write_split(path):
    seq = torch.tensor(
        [
            [
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
                [0.5, 0.4, 0.1],
                [0.4, 0.5, 0.1],
            ]
        ],
        dtype=torch.float32,
    )
    torch.save({"inputs": seq, "lengths": torch.tensor([4])}, path)


def test_tensor_dir_loads_split_payload(tmp_path):
    _write_split(tmp_path / "train.pt")
    _write_split(tmp_path / "val.pt")
    _write_split(tmp_path / "test.pt")

    task = TensorDirectorySequenceTask(str(tmp_path))
    task.prepare()

    assert task.prob_dim == 3
    assert task.available_splits() == ("train", "val", "test")
    payload = task.get_split_payload("train")
    assert tuple(payload.inputs.shape) == (1, 4, 3)
    assert task.max_block_size() == 3
