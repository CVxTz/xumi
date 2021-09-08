from pathlib import Path

from tokenizers import Tokenizer
import torch

from xumi.ml.train import MAX_LEN
from xumi.ml.models import Seq2Seq


def predict(transformed_text, model, tokenizer, cls_index, sep_index, max_len=MAX_LEN):

    src = tokenizer.encode(transformed_text).ids

    src = torch.tensor(src, dtype=torch.long)

    src = src.unsqueeze(0)

    memory = model.encode_src(src)

    trg = torch.zeros((src.shape[0], max_len), dtype=torch.long)

    trg[:, 0] = cls_index

    for i in range(1, max_len):
        print([i] * 100)
        print(tokenizer.decode(trg.squeeze().numpy()))
        output = model.decode_trg(trg[:, :i], memory=memory)
        output = output.argmax(2)

        is_end = output == sep_index
        is_end, _ = is_end.max(1)
        if is_end.sum() == output.shape[0]:
            break

        next_vals = output[:, -1]
        trg[:, i] = next_vals

    trg = trg.squeeze().numpy()

    return tokenizer.decode(trg)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=str(Path(__file__).absolute().parents[2] / "output" / "checker.ckpt"),
    )
    parser.add_argument(
        "--tokenizer_path",
        default=str(
            Path(__file__).absolute().parents[2] / "resources" / "tokenizer.json"
        ),
    )

    args = parser.parse_args()

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path

    tokenizer = Tokenizer.from_file(tokenizer_path)

    model = Seq2Seq(
        out_vocab_size=tokenizer.get_vocab_size(),
        pad_idx=tokenizer.token_to_id("[PAD]"),
        lr=1e-6,
        dropout=0.1,
    )

    device = torch.device("cpu")

    model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])

    model.eval()

    s = "The KING vultre (Sarcoramphus papa) is, a larg bird fouund in Central and South Amerika"
    # The king vulture (Sarcoramphus papa) is a large bird found in Central and South America.

    corrected = predict(
        transformed_text=s,
        model=model,
        tokenizer=tokenizer,
        cls_index=tokenizer.token_to_id("[CLS]"),
        sep_index=tokenizer.token_to_id("[SEP]"),
    )

    print(corrected)
