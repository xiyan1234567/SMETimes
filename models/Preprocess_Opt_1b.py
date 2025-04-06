import torch
import torch.nn as nn
from transformers import (
    OPTForCausalLM,
    AutoTokenizer
)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = f"cuda:{configs.gpu}"
        print(self.device)

        self.opt = OPTForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            torch_dtype=torch.float16
        ).to(self.device)

        self.opt_tokenizer = AutoTokenizer.from_pretrained(configs.llm_ckp_dir)
        if self.opt_tokenizer.pad_token is None:
            self.opt_tokenizer.pad_token = self.opt_tokenizer.eos_token

        self.vocab_size = self.opt_tokenizer.vocab_size
        self.hidden_dim_of_opt = 2048

        for name, param in self.opt.named_parameters():
            param.requires_grad = False

    def tokenizer(self, text_list):
        tokenized_output = self.opt_tokenizer(
            text_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        input_ids = tokenized_output["input_ids"].to(self.device)
        embeddings = self.opt.get_input_embeddings()(input_ids)
        
        return embeddings
    
    def forecast(self, text_list):
        inputs_embeds = self.tokenizer(text_list) 
        
        outputs = self.opt.model(inputs_embeds=inputs_embeds)
        text_outputs = outputs.last_hidden_state
        text_outputs = text_outputs[:, -1, :]
        
        return text_outputs
    
    def forward(self, text_list):
        return self.forecast(text_list)