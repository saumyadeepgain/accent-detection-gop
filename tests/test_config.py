"""Config loading, overrides, CLI flags and validation."""

import argparse
import os

import pytest
import yaml

from src.config import (
    Config,
    add_config_arguments,
    config_from_args,
    config_keys,
    load_config,
    short_aliases,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_YAML = os.path.join(REPO, "config.yaml")


def leaf_keys(data: dict) -> set[str]:
    keys = set()
    for k, v in data.items():
        if isinstance(v, dict) and k in (
            "audio",
            "data",
            "augment",
            "model",
            "train",
            "eval",
            "gop",
        ):
            keys |= {f"{k}.{sub}" for sub in v}
        else:
            keys.add(k)
    return keys


def test_config_yaml_and_dataclasses_list_the_same_keys():
    with open(CONFIG_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert leaf_keys(data) == set(config_keys())


def test_config_yaml_loads_validates_and_round_trips(tmp_path):
    cfg = load_config(CONFIG_YAML)
    assert cfg.audio.sample_rate == 16000 and cfg.model.layers == [6, 9]
    assert cfg.gop.rivals["ɡ"] == ["k"]  # IPA keys survive YAML
    path = tmp_path / "saved.yaml"
    cfg.save(str(path))
    assert load_config(str(path)).to_dict() == cfg.to_dict()


@pytest.mark.parametrize(
    "override, key, expected",
    [
        ("train.lr=1e-4", ("train", "lr"), 1e-4),  # YAML would read 1e-4 as a string
        ("train.epochs=5", ("train", "epochs"), 5),
        ("train.amp=false", ("train", "amp"), False),
        ("model.layers=[4,8]", ("model", "layers"), [4, 8]),
        ("model.layers=3,5", ("model", "layers"), [3, 5]),
        ("train.resume=null", ("train", "resume"), None),
        ("data.test_sources=[Svarah]", ("data", "test_sources"), ["Svarah"]),
        ("gop.rivals={v: [w]}", ("gop", "rivals"), {"v": ["w"]}),
        ("train.out_dir=runs/007", ("train", "out_dir"), "runs/007"),  # str stays str
        (
            "model.hf_overrides={hidden_dropout: 0.1}",
            ("model", "hf_overrides"),
            {"hidden_dropout": 0.1},
        ),
    ],
)
def test_overrides_are_typed(override, key, expected):
    cfg = load_config(None, [override])
    assert getattr(getattr(cfg, key[0]), key[1]) == expected


@pytest.mark.parametrize(
    "override, error",
    [
        ("train.lrr=1", KeyError),
        ("train.epochs=abc", ValueError),
        ("model.layers=[a]", ValueError),
        ("model.pooling=max", ValueError),
        ("data.val_frac=0.6", ValueError),  # + test_frac 0.15 is fine, but 0.6 + 0.6 is not
    ],
)
def test_bad_values_fail_loudly(override, error):
    overrides = [override] + (["data.test_frac=0.6"] if "val_frac" in override else [])
    with pytest.raises(error):
        load_config(None, overrides)


def test_unknown_key_suggests_the_right_one():
    with pytest.raises(KeyError, match="train.lr"):
        load_config(None, ["train.lrr=1"])


def test_every_key_is_a_flag_with_short_aliases():
    parser = add_config_arguments(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--lr",
            "3e-4",
            "--batch_size",
            "8",
            "--model.backbone",
            "mel_cnn",
            "--set",
            "eval.batch_size=4",
            "--set",
            "train.epochs=2",
        ]
    )
    cfg = config_from_args(args)
    assert (cfg.train.lr, cfg.train.batch_size, cfg.model.backbone) == (3e-4, 8, "mel_cnn")
    assert (cfg.eval.batch_size, cfg.train.epochs) == (4, 2)
    aliases = short_aliases()
    assert aliases["train.lr"] == "lr" and aliases["train.batch_size"] == "batch_size"
    assert "eval.batch_size" not in aliases  # ambiguous name goes to the higher-priority section
    eval_first = short_aliases(priority=("eval", "train"))
    assert eval_first["eval.batch_size"] == "batch_size"


def test_validation_warns_about_no_op_gain_augmentation():
    cfg = Config()
    cfg.augment.gain_prob = 0.5
    assert any("gain_prob" in w for w in cfg.validate())
    cfg.audio.normalize = False  # hf backbone + raw amplitudes: gain matters, no warning
    assert not any("gain_prob" in w for w in cfg.validate())
