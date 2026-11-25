"""Pooling/XAI checks, including non-image token exclusion and affine head export."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import torch
    from train_probe import patch_tokens, patch_scores
except ModuleNotFoundError as error:
    if error.name not in {"torch", "torchvision", "numpy", "pyarrow"}:
        raise
    torch = None


@unittest.skipIf(torch is None, "Training libraries required")
class PatchPoolingTests(unittest.TestCase):
    def test_excludes_special_tokens_and_preserves_spatial_order(self):
        patches = torch.arange(196.).reshape(1, 196, 1).expand(1, 196, 768)
        for name, prefix in (("deit3", 1), ("dinov3", 5), ("siglip2", 0)):
            tokens = torch.cat([torch.full((1, prefix, 768), 10000.), patches], dim=1)
            class Encoder:
                num_prefix_tokens = 1
                config = SimpleNamespace(num_register_tokens=4)
                def forward_features(self, images):
                    return tokens
                def __call__(self, **kwargs):
                    return SimpleNamespace(last_hidden_state=tokens)
            selected = patch_tokens(Encoder(), name, torch.zeros(1, 3, 224, 224))
            torch.testing.assert_close(selected, patches)
            scores, logits = patch_scores(selected, torch.ones(768), -2.)
            self.assertEqual(tuple(scores.shape), (1, 14, 14))
            self.assertEqual(scores[0, 0, 1].item(), 768.)
            self.assertEqual(scores[0, 1, 0].item(), 14 * 768.)
            torch.testing.assert_close(logits, torch.tensor([97.5 * 768 - 2]))

    def test_gradient_and_signed_scores(self):
        tokens = torch.randn(2, 196, 768, requires_grad=True)
        weight = torch.randn(768, requires_grad=True)
        scores, logits = patch_scores(tokens, weight, 0.7)
        torch.testing.assert_close(logits, scores.mean((1, 2)) + 0.7)
        logits.sum().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(torch.isfinite(weight.grad).all())

    def test_standardization_can_be_folded_into_one_linear_layer(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(7)
        x = rng.normal(size=(24, 768)) * 2 + 3
        head = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, np.arange(24) % 2)
        scaler, classifier = head.steps[0][1], head.steps[1][1]
        w = classifier.coef_[0] / scaler.scale_
        b = classifier.intercept_[0] - w @ scaler.mean_
        np.testing.assert_allclose(x @ w + b, head.decision_function(x), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
