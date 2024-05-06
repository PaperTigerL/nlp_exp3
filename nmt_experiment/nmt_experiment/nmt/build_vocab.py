import json
import argparse


def build_vocab(corpus_path: str, vocab_path: str):
    vocab = {"<|EOS|>": 0, "<|PAD|>": 1, "<|UNK|>": 2}

    with open(corpus_path, mode="r", encoding="utf-8") as fin:
        for line in fin:
            tokens = line.strip().split()
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

    with open(vocab_path, mode="w", encoding="utf-8") as fout:
        json.dump(vocab, fout, ensure_ascii=False)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument("--vocab-path", type=str, required=True)
    args = parser.parse_args()
    build_vocab(args.corpus_path, args.vocab_path)


if __name__ == "__main__":
    cli_main()
