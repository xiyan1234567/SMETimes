import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from layers.mlp import MLP

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        
        self.fusion_gate = nn.Parameter(torch.zeros([])) if configs.mix_embeds else None
        
        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
        )
        self.hidden_dim_of_llama = 2048
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
        
        for name, param in self.llama.named_parameters():
            param.requires_grad = False

        self.num_experts = configs.num_experts
        self.lambda_reg = configs.lambda_reg
        self.experts = nn.ModuleList([
            nn.Linear(self.hidden_dim_of_llama, self.hidden_dim_of_llama, bias=False)
            for _ in range(self.num_experts)
        ])
        self.gate = nn.Linear(self.hidden_dim_of_llama, self.num_experts)
            
        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_llama)
            self.decoder = nn.Linear(self.hidden_dim_of_llama, self.token_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_llama, 
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_llama, self.token_len,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation)

        self.register_buffer('_dummy', torch.zeros(1))
    
    
    def _apply_moe(self, x):
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        
        gates = self.gate(x)  # [B*S, K]
        gates = torch.softmax(gates, dim=-1)
        
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B*S, K, D]
        
        moe_out = (gates.unsqueeze(-1) * expert_outputs).sum(1)
        moe_out = moe_out.view(orig_shape)
        
        reg_loss = gates.norm(p=1) * self.lambda_reg
        moe_out = moe_out + 0. * reg_loss * self._dummy
        
        return moe_out
    
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)

        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]

        # Adaptive Fusion
        if self.fusion_gate is not None:
            alpha = torch.sigmoid(self.fusion_gate)
            times_embeds = alpha * self.encoder(fold_out) + (1-alpha) * x_mark_enc
        else:
            times_embeds = self.encoder(fold_out)

        outputs = self.llama.model(
            inputs_embeds=times_embeds)[0]
        
        # MoE
        outputs = self._apply_moe(outputs)

        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)
        
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        
        return dec_out
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

    
    
    
    
    
