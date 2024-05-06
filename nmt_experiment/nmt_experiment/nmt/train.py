import os
from collections import OrderedDict
import sys
import tqdm
import json
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn.functional as F

from nmt.config import NMTConfig
from nmt.transformer import (
    Encoder, 
    Decoder,
    Transformer
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

logger = logging.getLogger("nmt.train")


def load_vocab(path):
    with open(path, mode="r", encoding="utf-8") as fin:
        vocab = json.load(fin)
    return vocab


def load_data(corpus_path, vocab):
    data = []

    with open(corpus_path, mode="r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()

            tokens = line.strip().split()

            token_ids = []
            for token in tokens:
                # 2 is UNK
                token_id = vocab.get(token, 2)
                token_ids.append(token_id)
            
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            data.append(token_ids)
    
    return data


class MTDataset(torch.utils.data.Dataset):
    def __init__(self, src_corpus_path, tgt_corpus_path, src_vocab, tgt_vocab):
        self.src_data = load_data(src_corpus_path, src_vocab)
        self.tgt_data = load_data(tgt_corpus_path, tgt_vocab)

        assert len(self.src_data) == len(self.tgt_data)
        self.eos_token = torch.tensor([0])

    def __getitem__(self, index):
        src_input = self.src_data[index]

        src_input = torch.cat([src_input, self.eos_token])

        tgt_input = self.tgt_data[index]

        target = torch.cat([tgt_input, self.eos_token])

        tgt_input = torch.cat([self.eos_token, tgt_input])


        return {
            "src_input": src_input,
            "tgt_input": tgt_input,
            "target": target
        }


    def __len__(self):
        return len(self.src_data)


def padding_batch(values):
    size = max(v.shape[0] for v in values)

    batch_size = len(values)

    # 1 is padding
    res = values[0].new(batch_size, size).fill_(1)

    for i, v in enumerate(values):
        res[i, : len(v)].copy_(v)

    return res


def collate_fn(batch):
    src_input = padding_batch(
        [item["src_input"] for item in batch]
    )
    tgt_input = padding_batch(
        [item["tgt_input"] for item in batch]
    )
    target = padding_batch(
        [item["target"] for item in batch]
    )
    return {
        "src_input": src_input,
        "tgt_input": tgt_input,
        "target": target
    }


def label_smoothed_nll_loss(lprobs, target, epsilon, padding_idx=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if padding_idx is not None:
        pad_mask = target.eq(padding_idx)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    
    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(config_path):
    config = NMTConfig.load_config(config_path)

    setup_seed(config.seed)
    
    src_vocab = load_vocab(config.src_vocab_path)
    tgt_vocab = load_vocab(config.tgt_vocab_path)

    logger.info("Source language vocab size: {}".format(len(src_vocab)))
    logger.info("Target language vocab size: {}".format(len(tgt_vocab)))

    train_data = MTDataset(
        config.src_train_corpus_path, 
        config.tgt_train_corpus_path,
        src_vocab,
        tgt_vocab
    )

    valid_data = MTDataset(
        config.src_valid_corpus_path, 
        config.tgt_valid_corpus_path,
        src_vocab,
        tgt_vocab
    )

    encoder = Encoder(
        d_model=config.d_model,
        num_layers=config.num_encoder_layers, 
        num_heads=config.num_heads, 
        d_ff=config.d_ff,
        dropout=config.dropout,
        vocab_size=len(src_vocab) 
    )

    decoder = Decoder(
        d_model=config.d_model, 
        num_layers=config.num_decoder_layers, 
        num_heads=config.num_heads, 
        d_ff=config.d_ff,
        dropout=config.dropout,
        vocab_size=len(tgt_vocab) 
    )

    model = Transformer(encoder, decoder)
    model.train()

    device = torch.device("cuda") if config.cuda else torch.device("cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2)
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data, 
        config.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, 
        config.batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )

    for i in range(config.epoch):

        for step, sample in enumerate(tqdm.tqdm(train_dataloader)):
            
            src_input = sample["src_input"]
            tgt_input = sample["tgt_input"]
            target = sample["target"]

            src_input = src_input.to(device)
            tgt_input = tgt_input.to(device)
            target = target.to(device)

            output = model(
                src_input=src_input,
                tgt_input=tgt_input,
            )

            lprobs = F.log_softmax(output, dim=-1)

            lprobs = lprobs.view(-1, lprobs.shape[-1])
            target = target.view(-1)

            loss, nll_loss = label_smoothed_nll_loss(lprobs, target, config.label_smoothing, 1)

            num_tokens = target.ne(1).long().sum()
            
            loss = loss / num_tokens
            nll_loss = nll_loss / num_tokens

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if (step + 1) % config.log_interval == 0:
                logger.info(f"step: {step}, loss: {loss.item(): .6f}, nll_loss: {nll_loss.item(): .6f}")
        
        valid_loss = 0.0
        vali_nll_loss = 0.0
        tokens_count = 0
        for sample in tqdm.tqdm(valid_dataloader):
            
            src_input = sample["src_input"]
            tgt_input = sample["tgt_input"]
            target = sample["target"]

            src_input = src_input.to(device)
            tgt_input = tgt_input.to(device)
            target = target.to(device)

            with torch.no_grad():
                output = model(
                    src_input=src_input,
                    tgt_input=tgt_input,
                )

                lprobs = F.log_softmax(output, dim=-1)

                lprobs = lprobs.view(-1, lprobs.shape[-1])
                target = target.view(-1)

                loss, nll_loss = label_smoothed_nll_loss(lprobs, target, config.label_smoothing, 1)

                num_tokens = target.ne(1).long().sum()

                valid_loss = valid_loss + loss.item()
                vali_nll_loss = vali_nll_loss + nll_loss.item()
                tokens_count = tokens_count + num_tokens
        
        valid_loss = valid_loss / tokens_count
        vali_nll_loss = vali_nll_loss / tokens_count
        logger.info(f"valid_loss: {valid_loss: .6f}, vali_nll_loss: {vali_nll_loss: .6f}")

        torch.save(model.state_dict(), os.path.join(config.save_dir, f"checkpoint{i + 1}.pt"))


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    args = parser.parse_args()
    train(args.config_path)


if __name__ == "__main__":
    cli_main()
