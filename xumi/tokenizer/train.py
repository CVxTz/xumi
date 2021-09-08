from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace, Sequence
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "/media/jenazzad/Data/ML/nlp/reddit_full_text.txt",
            "/media/jenazzad/Data/ML/nlp/amazon_full_text.txt",
        ],
    )
    parser.add_argument("--output_path", default="../../resources/tokenizer.json")
    args = parser.parse_args()

    files = args.files
    output_path = args.output_path

    tokenizer = Tokenizer(BPE(unk_token="[UNK]", continuing_subword_prefix="##"))
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        continuing_subword_prefix="##",
        vocab_size=5_000,
        limit_alphabet=1000,
        min_frequency=500,
    )

    tokenizer.pre_tokenizer = Sequence([Whitespace()])
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    tokenizer.train(files, trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    output = tokenizer.encode("Helllllo, y'all! How are you 😁 ?")
    print(output)

    print(output.tokens)
    print(output.ids)
    print(output.offsets)
    print(output.overflowing)

    tokenizer.save(output_path, pretty=True)

    tokenizer = Tokenizer.from_file(output_path)

    output = tokenizer.encode("Hellllllo, y'all! How are you 😁 ?")

    print(output.tokens)
    print(output.ids)
    print(output.offsets)
    print(output.overflowing)

    print(tokenizer.decode(output.ids))
