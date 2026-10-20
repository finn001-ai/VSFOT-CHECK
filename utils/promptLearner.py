import torch
import torch.nn as nn


class PromptLearner(nn.Module):
    """可学习的提示上下文（CoOp / COSMO 式）：所有类共享 n_ctx 个上下文向量，后接各自的类名，经冻结的 CLIP 文本编码器得到类原型

    提示词与原方法相同（"a photo of a {类名}"），上下文用前 n_ctx 个词（"a photo of a"）的词嵌入初始化，
    所以初始时输出与 clip_model.encode_text(clip.tokenize(prompts)) 完全一致；只有 self.ctx 是可学习参数。
    """

    def __init__(self, clip_model, tokens, n_ctx=4):
        super().__init__()
        self.clip = [clip_model]                      # 不注册为子模块：CLIP 参数不进入本模块的 parameters()
        self.n_ctx = n_ctx
        self.register_buffer("tokens", tokens)        # [C, 77]
        with torch.no_grad():
            emb = clip_model.token_embedding(tokens).float()           # [C, 77, D]
        self.register_buffer("prefix", emb[:, :1].clone())             # 起始符
        self.register_buffer("suffix", emb[:, 1 + n_ctx:].clone())     # 类名、句末、结束符和填充
        self.ctx = nn.Parameter(emb[0, 1:1 + n_ctx].clone())           # [n_ctx, D]，"a photo of a"

    def forward(self):
        """返回 L2 归一化前的文本特征 [C, FV]（同 clip_model.encode_text）"""
        m = self.clip[0]
        C = self.tokens.shape[0]
        x = torch.cat([self.prefix, self.ctx.unsqueeze(0).expand(C, -1, -1), self.suffix], dim=1).type(m.dtype)
        x = x + m.positional_embedding.type(m.dtype)
        x = m.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)          # NLD -> LND -> NLD
        x = m.ln_final(x).type(m.dtype)
        return x[torch.arange(C, device=x.device), self.tokens.argmax(dim=-1)] @ m.text_projection
