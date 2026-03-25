import unittest
import torch
import math
import numpy as np
from train_gpt import BayesianBackoffCache

class TestBayesianBackoffCache(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.cache = BayesianBackoffCache(
            vocab_size=self.vocab_size,
            max_order=3,
            min_cache_count=0.1,
            recency_decay=1.0, # Deterministic for most tests
            entropy_threshold=0.0,
            confidence_threshold=0.0,
            mix_alpha=0.5
        )

    def test_01_basic_observation(self):
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        self.assertEqual(len(self.cache._token_history), 2)

    def test_02_order_2_match(self):
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        lp = self.cache.get_cache_log_probs([1])
        self.assertIsNotNone(lp)
        self.assertAlmostEqual(lp[2].exp().item(), 1.0)

    def test_03_order_3_match(self):
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        self.cache.observe(3, 2)
        lp = self.cache.get_cache_log_probs([1, 2])
        self.assertIsNotNone(lp)
        self.assertAlmostEqual(lp[3].exp().item(), 1.0)

    def test_04_backoff_priority(self):
        # (1, 2) -> 3 (order 3)
        # (2,)   -> 4 (order 2)
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        self.cache.observe(3, 2)
        self.cache.observe(2, 10) # Different pos
        self.cache.observe(4, 11)
        
        lp = self.cache.get_cache_log_probs([1, 2])
        # Higher order (3) should win
        self.assertEqual(lp.argmax().item(), 3)

    def test_05_backoff_to_lower_order(self):
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        self.cache.observe(3, 2)
        # Query [5, 2] -> should match (2,) -> 3
        lp = self.cache.get_cache_log_probs([5, 2])
        self.assertEqual(lp.argmax().item(), 3)

    def test_06_recency_decay(self):
        self.cache.recency_decay = 0.5
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        # Decay happens on NEXT observe
        self.cache.observe(3, 2) 
        # (1,) -> 2 count should be 0.5
        counts = self.cache.ngram_counts[2][(1,)][2]
        self.assertAlmostEqual(counts, 0.5)

    def test_07_entropy_gate(self):
        self.cache.entropy_threshold = 2.0
        # Low entropy model LP
        model_lp = torch.zeros(self.vocab_size)
        model_lp[0] = 0.0 # Prob 1.0, entropy 0
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        mixed = self.cache.mix_with_model(model_lp, [1])
        # Should stay as model_lp
        self.assertTrue(torch.allclose(mixed, model_lp))

    def test_08_confidence_gate(self):
        self.cache.confidence_threshold = 0.9
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        self.cache.observe(1, 2)
        self.cache.observe(3, 3)
        # (1,) -> {2: 1.0, 3: 1.0}. Max prob 0.5 < 0.9.
        model_lp = torch.full((self.vocab_size,), -10.0) 
        mixed = self.cache.mix_with_model(model_lp, [1])
        self.assertTrue(torch.allclose(mixed, model_lp))

    def test_09_agreement_gate(self):
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        # Cache predicts 2 with prob 1.0
        # Model predicts 5 with prob 1.0
        model_lp = torch.full((self.vocab_size,), -100.0)
        model_lp[5] = 0.0
        mixed = self.cache.mix_with_model(model_lp, [1])
        # Disagree -> model wins
        self.assertTrue(torch.allclose(mixed, model_lp))

    def test_10_deduplication(self):
        self.cache.observe(1, 5)
        self.cache.observe(1, 5) # Duplicate
        self.assertEqual(len(self.cache._token_history), 1)

    def test_11_reset(self):
        self.cache.observe(1, 0)
        self.cache.reset()
        self.assertEqual(len(self.cache._token_history), 0)
        self.assertEqual(len(self.cache.ngram_counts[2]), 0)

    def test_12_min_cache_count(self):
        self.cache.min_cache_count = 5.0
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        lp = self.cache.get_cache_log_probs([1])
        self.assertIsNone(lp)

    def test_13_max_order_limit(self):
        cache = BayesianBackoffCache(100, max_order=2)
        cache.observe(1, 0)
        cache.observe(2, 1)
        cache.observe(3, 2)
        # Sequence: [1, 2, 3]. 
        # Order 2 counts for (2,): {3: 1.0}.
        # Order 3 counts: Should NOT EXIST.
        self.assertEqual(len(cache.ngram_counts), 1) # Only order 2
        lp = cache.get_cache_log_probs([1, 2])
        self.assertIsNotNone(lp)
        self.assertEqual(lp.argmax().item(), 3)
        # Verify it didn't use order 3 by checking a context that only 
        # exists at order 3 (if we had it).

    def test_14_sliding_window_history(self):
        # Dictionary history handles sparse/global positions natively
        self.cache.observe(1, 100)
        self.cache.observe(2, 101)
        # Prediction at 102 should have context [1, 2]
        ctx = self.cache.get_context_at(102)
        self.assertEqual(ctx, [1, 2])
        # Prediction at 101 should have context [1]
        ctx_prev = self.cache.get_context_at(101)
        self.assertEqual(ctx_prev, [1])

    def test_15_vocab_size_boundary(self):
        self.cache.observe(99, 0)
        self.cache.observe(0, 1)
        lp = self.cache.get_cache_log_probs([99])
        self.assertEqual(lp[0].exp().item(), 1.0)

    def test_16_mixing_alpha_math(self):
        self.cache.mix_alpha = 0.2
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        model_lp = torch.full((self.vocab_size,), math.log(1/100))
        mixed = self.cache.mix_with_model(model_lp, [1])
        # Mixed prob for 2: 0.2 * 1.0 + 0.8 * (0.01) = 0.2 + 0.008 = 0.208
        self.assertAlmostEqual(mixed[2].exp().item(), 0.208, places=5)

    def test_17_return_types(self):
        self.cache.observe(1, 0)
        self.cache.observe(2, 1)
        lp = self.cache.get_cache_log_probs([1])
        self.assertIsInstance(lp, torch.Tensor)
        self.assertEqual(lp.dtype, torch.float32)

if __name__ == "__main__":
    unittest.main()
