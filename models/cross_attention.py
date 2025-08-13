import torch
from torch import nn

# Implementation based on paper: 10.3389/fsurg.2022.1029991
class CrossAttention(nn.Module):
    def __init__(self, n_meta_data, n_feat_conv, meta_encoder_output_size = 512, linear_layer_size=512):
        super(CrossAttention, self).__init__()
        self.meta_encoder = nn.Sequential(
            nn.Linear(n_meta_data, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, meta_encoder_output_size), nn.BatchNorm1d(meta_encoder_output_size), nn.ReLU(),
        )
        self.meta_linear = nn.Linear(meta_encoder_output_size, linear_layer_size)

        self.meta_self_attention = nn.MultiheadAttention(embed_dim=meta_encoder_output_size, num_heads=meta_encoder_output_size, batch_first=True)
        self.bn_meta = nn.LayerNorm(meta_encoder_output_size)
        self.meta_cross_attention =  nn.MultiheadAttention(embed_dim=linear_layer_size, num_heads=2, batch_first=True)

        self.img_linear = nn.Linear(n_feat_conv, linear_layer_size)
        self.bn_img_features = nn.LayerNorm(n_feat_conv)
        self.img_self_attention =nn.MultiheadAttention(embed_dim=n_feat_conv, num_heads=n_feat_conv, batch_first=True)
        self.img_cross_attention = nn.MultiheadAttention(embed_dim=linear_layer_size, num_heads=2, batch_first=True)

    def forward(self, img_features, meta_data):
        meta_data = self.meta_encoder(meta_data)
        meta_data = self.bn_meta(meta_data)
        img_features = self.bn_img_features(img_features)

        # self attention
        att_meta_data, _ = self.meta_self_attention(meta_data, meta_data, meta_data)
        att_img, _ = self.img_self_attention(img_features, img_features, img_features)
        
        # projection to linear_layer_size with linear network
        att_meta_data_projected = self.meta_linear(att_meta_data)
        att_img_projected = self.img_linear(att_img)
        
        # cross-attention
        img_features_cross_attention, _ = self.img_cross_attention(att_img_projected, att_meta_data_projected, att_meta_data_projected)
        meta_data_cross_attention, _ = self.meta_cross_attention(att_meta_data_projected, att_img_projected, att_img_projected)

        # residual connection
        att_meta_data = att_meta_data_projected + meta_data_cross_attention
        img_features = att_img_projected + img_features_cross_attention

        # concatenation
        return torch.cat([img_features, att_meta_data], dim=1)
