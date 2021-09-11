import random
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from xumi.perturbations.apply_perturbations import apply_perturbation_to_text
from xumi.ml.models import Seq2Seq
from xumi.text import Text


MAX_LEN = 256


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, hf_tokenizer):
        self.samples = samples
        self.n_samples = len(self.samples)
        self.hf_tokenizer = hf_tokenizer

    def __len__(self):
        return self.n_samples // 100  # Smaller epochs

    def __getitem__(self, _):
        idx = random.randint(0, self.n_samples - 1)

        text_str = self.samples[idx]
        text = Text(original=text_str)

        apply_perturbation_to_text(text)

        x = self.hf_tokenizer.encode(text.transformed).ids
        y = self.hf_tokenizer.encode(text.original).ids

        x = x[:MAX_LEN]
        y = y[:MAX_LEN]

        # print("input x :", self.hf_tokenizer.decode(x))
        # print("output y :", self.hf_tokenizer.decode(y))

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


def generate_batch(data_batch, pad_idx):
    src, trg = [], []
    for (src_item, trg_item) in data_batch:
        src.append(src_item)
        trg.append(trg_item)
    src = pad_sequence(src, padding_value=pad_idx, batch_first=True)
    trg = pad_sequence(trg, padding_value=pad_idx, batch_first=True)
    return src, trg


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--epochs", default=2000)
    parser.add_argument(
        "--init_model_path",
        default=str(Path(__file__).absolute().parents[2] / "output" / "checker-v2.ckpt")
    )
    parser.add_argument(
        "--data_path", default="/media/jenazzad/Data/ML/nlp/wikisent2.txt"
    )
    parser.add_argument(
        "--tokenizer_path",
        default=str(
            Path(__file__).absolute().parents[2] / "resources" / "tokenizer.json"
        ),
    )
    parser.add_argument(
        "--base_path", default=str(Path(__file__).absolute().parents[2] / "output")
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    data_path = args.data_path
    init_model_path = args.init_model_path
    tokenizer_path = args.tokenizer_path
    base_path = Path(args.base_path)
    base_path.mkdir(exist_ok=True)

    tokenizer = Tokenizer.from_file(tokenizer_path)

    with open(data_path) as f:
        data = f.read().split("\n")

    train, val = train_test_split(data, test_size=0.05, random_state=1337)

    train_data = Dataset(samples=train, hf_tokenizer=tokenizer)
    val_data = Dataset(samples=val, hf_tokenizer=tokenizer)

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
        collate_fn=partial(generate_batch, pad_idx=tokenizer.token_to_id("[PAD]")),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
        collate_fn=partial(generate_batch, pad_idx=tokenizer.token_to_id("[PAD]")),
    )

    model = Seq2Seq(
        out_vocab_size=tokenizer.get_vocab_size(),
        pad_idx=tokenizer.token_to_id("[PAD]"),
        lr=1e-5,
        dropout=0.1,
    )

    if init_model_path:
        model.load_state_dict(torch.load(init_model_path)["state_dict"])

    logger = TensorBoardLogger(
        save_dir=str(base_path),
        name="logs",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_acc", mode="max", dirpath=base_path, filename="checker"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)
