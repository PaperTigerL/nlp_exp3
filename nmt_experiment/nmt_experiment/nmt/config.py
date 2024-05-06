import yaml


class NMTConfig:
    def __init__(
        self,
        src_vocab_path: str,
        tgt_vocab_path: str,
        src_train_corpus_path: str,
        tgt_train_corpus_path: str,
        src_valid_corpus_path: str,
        tgt_valid_corpus_path: str,
        save_dir: str,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        epoch: int = 30,
        batch_size: int = 32,
        dropout: float = 0.1,
        learning_rate: float = 0.0007,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.98,
        label_smoothing: float = 0.1,
        log_interval: int = 100,
        cuda: bool = False,
        seed: int = 1,
    ):
        self.src_vocab_path = src_vocab_path
        self.tgt_vocab_path = tgt_vocab_path
        self.src_train_corpus_path = src_train_corpus_path
        self.tgt_train_corpus_path = tgt_train_corpus_path
        self.src_valid_corpus_path = src_valid_corpus_path
        self.tgt_valid_corpus_path = tgt_valid_corpus_path
        self.save_dir = save_dir
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.epoch = epoch
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.label_smoothing = label_smoothing
        self.log_interval = log_interval
        self.cuda = cuda
        self.seed = seed


    @classmethod
    def load_config(cls, config_path: str):
        with open(config_path, mode="r", encoding="utf-8") as fin:
            config = yaml.safe_load(fin)
        return cls(**config)


if __name__ == "__main__":
    NMTConfig.load_config("/mnt/d/file/nlp/exp3/nmt_experiment/nmt_experiment/config.yaml")
