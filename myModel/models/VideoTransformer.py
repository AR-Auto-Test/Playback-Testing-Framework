import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoTransformer(nn.Module):
    def __init__(self, frame_embed_dim, meta_embed_dim, n_frames, n_metadata, n_classes, nhead, num_layers, dropout=0.1):
        super(VideoTransformer, self).__init__()
        self.frame_embed = nn.Linear(n_frames * frame_embed_dim, frame_embed_dim)
        self.meta_embed = nn.Linear(n_metadata, meta_embed_dim)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=frame_embed_dim + meta_embed_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(frame_embed_dim + meta_embed_dim, n_classes)
        
    def forward(self, frames, metadata):
        # 假设frames的形状是 [batch_size, n_frames, frame_embed_dim]
        # 假设metadata的形状是 [batch_size, n_metadata]
        frame_embeddings = self.frame_embed(frames.view(frames.size(0), -1))
        meta_embeddings = self.meta_embed(metadata)
        
        # 合并帧嵌入和元数据嵌入
        embeddings = torch.cat((frame_embeddings, meta_embeddings), dim=1).unsqueeze(1)  # 增加一个维度作为序列长度
        
        transformer_output = self.transformer_encoder(embeddings)
        output = self.classifier(transformer_output.squeeze(1))
        return output
