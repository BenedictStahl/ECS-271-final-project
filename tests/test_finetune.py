from pathlib import Path
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from finetune import (Detector, encoder_parts, optimizer_groups, balanced_loss,
                      schedule_factor, ImageViews, restore_delta, save_checkpoint, load_finetuned, backward_step)
from train_probe import CHECKPOINTS


class FakeEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.embeddings = nn.Linear(768, 768)
        layers = nn.ModuleList([nn.Sequential(nn.Linear(768, 8), nn.Linear(8, 768), nn.Dropout(.2)) for _ in range(12)])
        norm = nn.LayerNorm(768)
        if name == "deit3":
            self.blocks, self.norm, self.num_prefix_tokens = layers, norm, 1
        elif name == "dinov3":
            self.model = nn.Module()
            self.model.layer, self.norm = layers, norm
            self.config = SimpleNamespace(num_register_tokens=4)
        else:
            self.encoder = nn.Module()
            self.encoder.layers, self.post_layernorm = layers, norm

    def forward_features(self, pixels):
        value = self.embeddings(pixels)
        layers, norm = encoder_parts(self, self.name)
        for layer in layers:
            value = value + layer(value)
        value = norm(value)
        prefix = 1 if self.name == "deit3" else 5 if self.name == "dinov3" else 0
        return torch.cat([torch.zeros(value.shape[0], prefix, 768), value], dim=1)

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.forward_features(pixel_values))


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.head = self.root / "head.npz"
        np.savez(self.head, weight=np.linspace(-.1, .1, 768), bias=.2, pooling="mean_patch_v1")
        torch.set_num_threads(2)

    def test_exact_unfreezing_modes_and_optimizer_groups_for_all_encoders(self):
        for name in ("deit3", "siglip2", "dinov3"):
            model = Detector(FakeEncoder(name), name, 2, self.head)
            layers, norm = encoder_parts(model.encoder, name)
            allowed = {id(p) for module in [*list(layers[-2:]), norm, model.head] for p in module.parameters()}
            self.assertEqual({id(p) for p in model.parameters() if p.requires_grad}, allowed)
            model.training_mode()
            self.assertFalse(model.encoder.training)
            self.assertFalse(model.encoder.embeddings.training)
            self.assertTrue(all(not layer.training for layer in layers[:-2]))
            self.assertTrue(all(layer.training for layer in layers[-2:]))
            groups = optimizer_groups(model, dict(head_lr=1e-4, encoder_lr=1e-5, weight_decay=.01))
            params = [p for group in groups for p in group["params"]]
            self.assertEqual(len(params), len({id(p) for p in params}))
            self.assertEqual({id(p) for p in params}, allowed)
            for group in groups:
                for p in group["params"]:
                    self.assertEqual(group["weight_decay"], .01 if p.ndim >= 2 else 0.)
                    self.assertEqual(group["lr"], 1e-4 if any(p is h for h in model.head.parameters()) else 1e-5)

    def test_backward_and_delta_restore_preserve_frozen_prefix(self):
        model = Detector(FakeEncoder("deit3"), "deit3", 2, self.head)
        model.training_mode()
        logits = model(torch.randn(2, 196, 768))
        loss = balanced_loss(logits, torch.tensor([0., 1.]), torch.tensor([.5, .5]), torch.tensor([2.5, .625]))
        loss.backward()
        for parameter in model.parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            else:
                self.assertIsNone(parameter.grad)
        state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(1)
        restore_delta(model, {"trainable_state": state})
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, state[n] if n in state else frozen[n], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "names differ"):
            restore_delta(model, {"trainable_state": dict(state, extra=torch.tensor(0))})

    def test_parent_weighting_and_scheduler_boundaries(self):
        logits, y = torch.tensor([-.2, .7]), torch.tensor([0., 1.])
        classes = torch.tensor([2.5, .625])
        original = balanced_loss(logits, y, torch.ones(2), classes)
        repeated = balanced_loss(logits.repeat_interleave(2), y.repeat_interleave(2), torch.full((4,), .5), classes)
        torch.testing.assert_close(original, repeated)
        self.assertAlmostEqual(schedule_factor(0, 100, .05), .2)
        self.assertEqual(schedule_factor(4, 100, .05), 1.)
        self.assertEqual(schedule_factor(100, 100, .05), 0.)
        self.assertTrue(0 < schedule_factor(50, 100, .05) < 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA needed for actual GradScaler overflow recovery")
    def test_amp_overflow_skips_update_reduces_scale_and_same_batch_can_retry(self):
        model = nn.Linear(2, 1).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.)
        pixels = torch.ones(2, 2, device="cuda")
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        norm, scale = backward_step(model, optimizer, scaler, model(pixels).sum(), 1.)
        self.assertIsNone(norm)
        self.assertEqual(scale, 1024.)
        self.assertEqual(scaler.get_scale(), 512.)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n], atol=0, rtol=0)
        hook.remove()
        optimizer.zero_grad(set_to_none=True)
        norm, scale = backward_step(model, optimizer, scaler, model(pixels).sum(), 1.)
        self.assertIsNone(scale)
        self.assertTrue(torch.isfinite(norm))
        self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()))

    def test_checkpoint_roundtrip_reconstructs_base_and_rejects_wrong_revision(self):
        torch.manual_seed(42)
        model = Detector(FakeEncoder("deit3"), "deit3", 2, self.head).eval()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        metadata = dict(model="deit3", checkpoint=CHECKPOINTS["deit3"], config=dict(unfrozen_blocks=2))
        path = self.root / "checkpoint.pt"
        save_checkpoint(path, model, optimizer, scheduler, scaler, metadata)
        pixels = torch.randn(2, 196, 768)
        expected = model(pixels)
        with patch("finetune.load_encoder", side_effect=lambda *a, **k: FakeEncoder("deit3")):
            torch.manual_seed(42)
            restored = load_finetuned(path, torch.device("cpu"))
            torch.testing.assert_close(restored(pixels), expected, atol=0, rtol=0)
            changed = torch.load(path, weights_only=True)
            changed["metadata"]["checkpoint"] = [CHECKPOINTS["deit3"][0], "wrong-revision"]
            torch.save(changed, path)
            with self.assertRaisesRegex(ValueError, "base revision"):
                load_finetuned(path, torch.device("cpu"))

    def test_image_views_are_fixed_train_only_and_hash_checked(self):
        p = self.root / "train/real/00000.jpg"
        p.parent.mkdir(parents=True)
        Image.new("RGB", (256, 240), (50, 100, 150)).save(p)
        row = dict(id="replacement:train/real/00000.jpg", path="train/real/00000.jpg", split="train",
                   source="real", target=0, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        dataset = ImageViews([row], {"dataset_root": str(self.root)}, "deit3", augment=True)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["id"], row["id"])
        self.assertEqual(dataset[0]["weight"], .5)
        self.assertEqual(dataset[1]["pixel_sha256"], dataset[1]["pixel_sha256"])
        torch.testing.assert_close(dataset[1]["pixels"], dataset[1]["pixels"], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "selected train or validation"):
            ImageViews([dict(row, split="test")], {"dataset_root": str(self.root)}, "deit3")
        p.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
