import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import unittest
import torch
from model import ClauseTransformer, ModelConfig


class TestModel(unittest.TestCase):
    def test_causal_no_future_leak(self):
        torch.manual_seed(12)
        m = ClauseTransformer(ModelConfig(20,token_dim=8)).eval()
        x = torch.randint(0,20,(2,10,5))
        changed = x.clone()
        changed[:,6:] = torch.randint(0,20,(2,4,5))
        with torch.no_grad():
            torch.testing.assert_close(m(x)[:,:6],m(changed)[:,:6])

    def test_cached_greedy_matches_full_prefix(self):
        torch.manual_seed(13)
        for residual in [True,False]:
            m = ClauseTransformer(ModelConfig(20,token_dim=8,residual=residual)).eval()
            for temperature in [0.5,1.0,4.0]:
                full = torch.randint(0,20,(3,6,5))
                incoming = full
                cache = None
                for _ in range(5):
                    with torch.no_grad():
                        expected = m(full,temperature=temperature)[:,-1]
                        actual,cache = m.decode_last(incoming,cache,temperature)
                    torch.testing.assert_close(expected,actual,atol=2e-6,rtol=2e-5)
                    incoming = actual.argmax(-1).unsqueeze(1)
                    full = torch.cat((full,incoming),dim=1)


if __name__ == "__main__":
    unittest.main()
