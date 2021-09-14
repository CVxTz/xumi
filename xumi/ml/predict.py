from pathlib import Path

import torch
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer

from xumi.ml.models import Seq2Seq
from xumi.ml.train import MAX_LEN
from xumi.perturbations.apply_perturbations import apply_perturbation_to_text
from xumi.text import Text


def predict(transformed_text, model, tokenizer, cls_index, sep_index, max_len=MAX_LEN):
    src = tokenizer.encode(transformed_text).ids

    src = torch.tensor(src, dtype=torch.long)

    src = src.unsqueeze(0)

    memory = model.encode_src(src)

    trg = torch.zeros((src.shape[0], max_len), dtype=torch.long)

    trg[:, 0] = cls_index

    for i in range(1, max_len):
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
        default=str(
            Path(__file__).absolute().parents[2] / "output" / "checker-v4.ckpt"
        ),
    )
    parser.add_argument(
        "--tokenizer_path",
        default=str(
            Path(__file__).absolute().parents[2] / "resources" / "tokenizer.json"
        ),
    )
    parser.add_argument(
        "--data_path", default="/media/jenazzad/Data/ML/nlp/wikisent2.txt"
    )
    parser.add_argument(
        "--base_path", default=str(Path(__file__).absolute().parents[2] / "output")
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

    s = "The KING vultre were an larg bird fouund in Central and South Amerika"
    # The king vulture is a large bird found in Central and South America.

    corrected = predict(
        transformed_text=s,
        model=model,
        tokenizer=tokenizer,
        cls_index=tokenizer.token_to_id("[CLS]"),
        sep_index=tokenizer.token_to_id("[SEP]"),
    )
    print(corrected)

    if args.data_path and args.base_path and Path(args.data_path).is_file():

        base_path = Path(args.base_path)
        base_path.mkdir(exist_ok=True)
        data_path = args.data_path

        with open(data_path) as f:
            data = f.read().split("\n")

        train, val = train_test_split(data, test_size=0.05, random_state=1337)

        gt_val = []
        perturbed_val = []
        predicted_val = []

        for sentence in tqdm(val[:100]):
            gt_val.append(sentence)
            text = Text(original=sentence)
            apply_perturbation_to_text(text, freq=3)
            perturbed_val.append(text.transformed)
            predicted = predict(
                transformed_text=text.transformed,
                model=model,
                tokenizer=tokenizer,
                cls_index=tokenizer.token_to_id("[CLS]"),
                sep_index=tokenizer.token_to_id("[SEP]"),
            )

            predicted_val.append(predicted)

        df = pd.DataFrame(
            {
                "ground_truth": gt_val,
                "perturbed": perturbed_val,
                "corrected": predicted_val,
            }
        )

        df.to_csv(base_path / "sample_predictions.csv", index=False)
