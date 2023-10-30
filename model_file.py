import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
      super().__init__()

      self.no_of_patches = (image_size//patch_size)**2
      self.embed_dim = embed_dim

      self.convlayer = nn.Conv2d(in_channels=1, out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)


    def forward(self, x):

      out = self.convlayer(x)

      out = out.view(-1, self.embed_dim, self.no_of_patches)
      out = out.transpose(1, 2)

      return out

def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        # TODO
        super().__init__()

        # assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.QKV_projection = nn.Linear(embed_dim, 3 * embed_dim)

        self.attn_drop = nn.Dropout(0)
        
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        self.proj_drop = nn.Dropout(0)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.QKV_projection.weight)
        self.QKV_projection.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_projection.weight)
        self.output_projection.bias.data.fill_(0)



    def forward(self, x):
      # TODO

        batch_size, seq_length, embed_dim = x.size()
        QKV = self.QKV_projection(x)

        QKV = QKV.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)

        QKV = QKV.permute(0, 2, 1, 3)

        q, k, v = QKV.chunk(3, dim=-1)

        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        attention = self.attn_drop(attention)
        values = torch.matmul(attention, v)
    
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.output_projection(values)
        o = self.proj_drop(o)

        return o

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        # TODO
        super().__init__()

        # Attention layer
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)


        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.Dropout(dropout),
#             nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim,eps=1e-6)
        self.dropout = nn.Dropout(dropout)




    def forward(self, x):
        # TODO

        # Attention part
        attn_out = self.self_attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, num_classes, dropout=0.1):
        # TODO

        super().__init__()

        self.patch_embedding_layer =  PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        seq_length += 1

        self.transformerlayers=nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)])

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, embed_dim).normal_(std=0.02))  

        self.head_layer = nn.Linear(embed_dim, num_classes)



    def forward(self, x):
        # TODO

        x = self.patch_embedding_layer(x)

        batch_size = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(batch_size, -1, -1)    

        x = torch.cat((batch_class_token, x), dim=1)

        x = x+self.pos_embedding



        for layer in self.transformerlayers:
            x = layer(x)

        x = x[:, 0]

        x = self.head_layer(x)

        return x

# # # Example usage:
# image_size = 28
# patch_size = 16
# in_channels = 1
# embed_dim = 1024
# num_heads = 16
# mlp_dim = 4096
# num_layers = 24
# num_classes = 10

# # # dropout = 0.2
# dropout = 0

# # # batch_size = 256
# batch_size = 64

# model = VisionTransformer(image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, num_classes, dropout)
# input_tensor = torch.randn(1, in_channels, image_size, image_size)
# output = model(input_tensor)
# print(output.shape)

# import numpy as np
        
# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print("no of params:", params)