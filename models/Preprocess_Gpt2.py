import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, AutoTokenizer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = configs.gpu
        print(f"cuda:{self.device}")
        
        self.gpt2 = GPT2Model.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(configs.llm_ckp_dir)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.max_length = 256
        self.vocab_size = self.gpt2_tokenizer.vocab_size
        self.hidden_dim_of_gpt2 = self.gpt2.config.hidden_size

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        self.encoder = nn.Linear(self.hidden_dim_of_gpt2, self.hidden_dim_of_gpt2)
        self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.vocab_size)

    def tokenizer(self, x):
        tokens = self.gpt2_tokenizer(
            x,
            return_tensors="pt",
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )['input_ids']
        tokens = tokens.to(self.device)
        embeddings = self.gpt2.get_input_embeddings()(tokens)
        return embeddings

    def forecast(self, x_mark_enc):
        embeddings = torch.cat([self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))], dim=0)
        outputs = self.gpt2(inputs_embeds=embeddings).last_hidden_state
        last_output = outputs[:, -1, :]
        return last_output

    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)
