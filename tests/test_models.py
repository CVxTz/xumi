import torch

from xumi.ml.models import Seq2Seq


def test_models():
    n_classes = 100

    source = torch.randint(low=0, high=n_classes, size=(20, 16))
    target = torch.randint(low=0, high=n_classes, size=(20, 32))

    s2s = Seq2Seq(out_vocab_size=n_classes, pad_idx=0)

    out = s2s((source, target))

    assert list(out.size()) == [20, 32, 100]
