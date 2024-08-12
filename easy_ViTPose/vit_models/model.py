import torch.nn as nn

from .backbone.vit import ViT
from .head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead


__all__ = ['ViTPose']

# nn.Module: Base class for all neural network modules.
class ViTPose(nn.Module):
    def __init__(self, cfg: dict) -> None:
        '''
        super().__init__()是 Python 3 中引入的簡化語法。
        在 Python 3 中，您可以直接使用 super() 不帶任何參數來調用父類的方法。這種寫法更加簡潔、直觀，是 Python 3 推薦的用法。

        super(ViTPose, self).__init__()
        這種寫法在 Python 2 中是必須的，因為在 Python 2 中，super() 需要明確地傳遞當前類和實例作為參數。
        即使在 Python 3 中，這種寫法仍然有效，但它更加繁瑣，沒有利用 Python 3 提供的簡化語法。
        '''
        super(ViTPose, self).__init__()
        
        '''
        略過type了，不過backbone的type是ViT
        {'img_size': (256, 192), 'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'ratio': 1, 'use_checkpoint': False, 'mlp_ratio': 4, 'qkv_bias': True, 'drop_path_rate': 0.3}
        '''
        backbone_cfg = {k: v for k, v in cfg['backbone'].items() if k != 'type'}
        '''
        略過type了，不過head的type是TopdownHeatmapSimpleHead
        {'in_channels': 768, 'num_deconv_layers': 2, 'num_deconv_filters': (256, 256), 'num_deconv_kernels': (4, 4), 'extra': {'final_conv_kernel': 1}, 'loss_keypoint': {'type': 'JointsMSELoss', 'use_target_weight': True}, 'out_channels': 133}
        '''
        head_cfg = {k: v for k, v in cfg['keypoint_head'].items() if k != 'type'}
        
        # 直接用了，難怪略過type
        # /home/charl0tte/easy_ViTPose/easy_ViTPose/vit_models/backbone/vit.py
        # ** 符號用在函數調用時，表示將一個字典解包為關鍵字參數（keyword arguments)
        # (Pdb) p backbone_cfg
        # {'img_size': (256, 192), 'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'ratio': 1, 'use_checkpoint': False, 'mlp_ratio': 4, 'qkv_bias': True, 'drop_path_rate': 0.3}
        self.backbone = ViT(**backbone_cfg)
        # (Pdb) p self.backbone
        # ViT(
        #   (patch_embed): PatchEmbed(
        #     (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), padding=(2, 2))
        #   )
        #   (blocks): ModuleList(
        #     (0): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): Identity()
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #     (1): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): DropPath(p=0.027272729203104973)
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #     (2): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): DropPath(p=0.054545458406209946)
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #     (3): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): DropPath(p=0.08181818574666977)
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #     (4): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): DropPath(p=0.10909091681241989)
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #     (5): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): DropPath(p=0.13636364042758942)
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #     (6): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): DropPath(p=0.16363637149333954)
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #     (7): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): DropPath(p=0.19090908765792847)
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #     (8): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): DropPath(p=0.2181818187236786)
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #     (9): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): DropPath(p=0.2454545497894287)
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #     (10): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): DropPath(p=0.27272728085517883)
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #     (11): Block(
        #       (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (attn): Attention(
        #         (qkv): Linear(in_features=768, out_features=2304, bias=True)
        #         (attn_drop): Dropout(p=0.0, inplace=False)
        #         (proj): Linear(in_features=768, out_features=768, bias=True)
        #         (proj_drop): Dropout(p=0.0, inplace=False)
        #       )
        #       (drop_path): DropPath(p=0.30000001192092896)
        #       (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #         (drop): Dropout(p=0.0, inplace=False)
        #       )
        #     )
        #   )
        #   (last_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        # )

        # # (Pdb) p head_cfg
        # {'in_channels': 768, 'num_deconv_layers': 2, 'num_deconv_filters': (256, 256), 'num_deconv_kernels': (4, 4), 'extra': {'final_conv_kernel': 1}, 'loss_keypoint': {'type': 'JointsMSELoss', 'use_target_weight': True}, 'out_channels': 133}
        # 自製的 from .head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
        # /home/charl0tte/easy_ViTPose/easy_ViTPose/vit_models/head/topdown_heatmap_simple_head.py 
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg)
        # (Pdb) p self.keypoint_head
        # TopdownHeatmapSimpleHead(
        #   (deconv_layers): Sequential(
        #     (0): ConvTranspose2d(768, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU(inplace=True)
        #     (3): ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (5): ReLU(inplace=True)
        #   )
        #   (final_layer): Conv2d(256, 133, kernel_size=(1, 1), stride=(1, 1))
        # )

        # 所以經過self.keypoint_head後，會生成133張2d圖，一張圖有一個keypoint

    def forward_features(self, x):
        return self.backbone(x)
    
    def forward(self, x):
        # > /home/charl0tte/easy_ViTPose/easy_ViTPose/vit_models/backbone/vit.py(775)forward()
        # > /home/charl0tte/easy_ViTPose/easy_ViTPose/vit_models/head/topdown_heatmap_simple_head.py(233)forward()
        return self.keypoint_head(self.backbone(x))

