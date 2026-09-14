import torch
from torch.utils.data import DataLoader
from src.utils.config import DataConfig
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary, ReasoningDataset, collate_reasoning_batch


def test_generator_produces_correct_count():
    config = DataConfig(num_train=100, num_val=20, num_test=20)
    gen = ArithmeticGenerator(config=config, seed=42)
    dataset = gen.generate_dataset()

    assert len(dataset["train"]) == 100
    assert len(dataset["val"]) == 20
    assert len(dataset["test"]) == 20


def test_affordability_ground_truth():
    config = DataConfig(num_train=200, task_family="affordability", max_operations=3)
    gen = ArithmeticGenerator(config=config, seed=100)
    records = gen.generate_dataset()["train"]

    for r in records:
        facts = r["input_facts"]
        person = None
        money = 0
        price = 0

        for f in facts:
            if f["type"] == "entity" and person is None:
                person = f["name"]
            elif f["type"] == "attribute" and f["entity"] == person and f["key"] == "money":
                money = f["value"]
            elif f["type"] == "operation" and f["entity"] == person and f["key"] == "money":
                if f["op"] == "add":
                    money += f["value"]
                elif f["op"] == "sub":
                    money -= f["value"]
            elif f["type"] == "attribute" and f["key"] == "price":
                price = f["value"]

        expected_label = 1 if money >= price else 0
        assert r["ground_truth"] == expected_label, f"Mismatch in record {r['id']}: money={money}, price={price}, label={r['ground_truth']}"


def test_class_balance():
    config = DataConfig(num_train=1000, task_family="affordability")
    gen = ArithmeticGenerator(config=config, seed=42)
    records = gen.generate_dataset()["train"]

    pos = sum(1 for r in records if r["ground_truth"] == 1)
    neg = sum(1 for r in records if r["ground_truth"] == 0)

    assert pos == 500
    assert neg == 500


def test_deterministic_seed():
    config = DataConfig(num_train=50, task_family="affordability")
    gen1 = ArithmeticGenerator(config=config, seed=777)
    gen2 = ArithmeticGenerator(config=config, seed=777)

    d1 = gen1.generate_dataset()
    d2 = gen2.generate_dataset()

    assert d1 == d2


def test_dataset_tensor_shapes():
    config = DataConfig(num_train=32, task_family="affordability")
    gen = ArithmeticGenerator(config=config, seed=42)
    records = gen.generate_dataset()["train"]

    vocab = Vocabulary()
    vocab.build_from_records(records)

    dataset = ReasoningDataset(records, vocab)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_reasoning_batch)

    batch = next(iter(loader))
    assert batch["facts"].shape[0] == 8
    assert batch["facts"].shape[2] == 5
    assert batch["fact_mask"].shape[0] == 8
    assert batch["label"].shape[0] == 8
    assert batch["fact_mask"].sum() > 0


def test_vocabulary_round_trip():
    records = [
        {
            "id": "1",
            "input_facts": [
                {"type": "entity", "name": "John"},
                {"type": "attribute", "entity": "John", "key": "money", "value": 50},
                {"type": "operation", "op": "add", "entity": "John", "key": "money", "value": 20},
            ],
            "ground_truth": 1,
        }
    ]
    vocab = Vocabulary()
    vocab.build_from_records(records)

    for fact in records[0]["input_facts"]:
        encoded = vocab.encode_fact(fact)
        assert len(encoded) == 5
        decoded = vocab.decode_fact(encoded)
        assert decoded["type"] == fact["type"]
        if "name" in fact:
            assert decoded["name"] == fact["name"]
        if "value" in fact:
            assert decoded["value"] == fact["value"]


def test_save_and_load_jsonl(tmp_path):
    config = DataConfig(num_train=20)
    gen = ArithmeticGenerator(config=config, seed=42)
    records = gen.generate_dataset()["train"]

    target_file = tmp_path / "data" / "train.jsonl"
    ArithmeticGenerator.save_jsonl(records, target_file)
    assert target_file.exists()

    loaded = ArithmeticGenerator.load_jsonl(target_file)
    assert len(loaded) == len(records)
    assert loaded == records
