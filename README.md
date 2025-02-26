# BasicScoreMatching

To train the VAE:

```python3 trainVAE.py -config <config_path>```

To train the score network

```python3 experiment.py -config <config_path>```

To scale up the model, change the unet_params section in the config file. For instance:
```
unet_params:
  image_size: *latent_size
  layers_per_block: 4
  block_out_channels: [128, 256, 512]
  down_block_types: ["DownBlock2D", "AttnDownBlock2D", "DownBlock2D"]
  up_block_types: ["UpBlock2D", "AttnUpBlock2D", "UpBlock2D"]
  unet_output_channels: 32
  head_depth: 2
  head_hidden_channels: 128
  class_emb_channels: 128
  down_channels: [ 128, 256, 256, 256]
  mid_channels: [ 256, 256]
  down_sample: [ False, False, False ]
  attn_down : [True, True, True]
  time_emb_dim: 256
  norm_channels : 32
  num_heads : 16
  conv_out_channels : 128
  num_down_layers: 2
  num_mid_layers: 2
  num_head_mid_layers: 4
  num_up_layers: 2
  condition_config:
    condition_types: ['class']
    class_condition_config :
      num_classes : 9
      cond_drop_prob : 0.1
  feature_out: True
```
  Results in about 100M parameters.
