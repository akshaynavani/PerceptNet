import json

import pytest

from perceptnet.config import Config


@pytest.mark.slow
def test_one_epoch_on_synthetic(tmp_path):
    from perceptnet.eval.evaluate import run_evaluation
    from perceptnet.train.trainer import Trainer

    cfg = Config()
    cfg.data.input_size = (64, 64)
    cfg.data.batch_size = 4
    cfg.data.synthetic_train_size = 8
    cfg.data.synthetic_val_size = 4
    cfg.data.num_workers = 1
    cfg.train.epochs = 1
    cfg.train.warmup_steps = 1
    cfg.train.eval_every_epochs = 1
    cfg.train.output_dir = str(tmp_path)
    result = Trainer(cfg).train()
    assert "map" in result and result["val_loss"] > 0
    assert (tmp_path / "best_model.msgpack").exists()
    assert json.loads((tmp_path / "history.json").read_text())

    report = run_evaluation(cfg, tmp_path / "final_model.msgpack")
    assert 0.0 <= report["map"] <= 1.0
