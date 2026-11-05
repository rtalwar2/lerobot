# diffusion_audio (audio fork)

Diffusion Policy with the audio fork's additions, as a separate policy type next to upstream's `diffusion`
(which is airo-ugent/lerobot's v0.6.1 unchanged):

- an AST audio encoder (`audio_backbone`, AudioSet weights by default) on the spectrogram observation
  `observation.audio.spectogram_values`, shape `(channels, time, 128)` per frame; `audio_channel` picks the channel
- `fusion_mode`: `"concat"` (Diffusion Policy) or `"modality_attention"` (ManiWAV-style image/audio fusion)
- the fork's RGB encoder: `image_pooling` `"avg"` or `"spatial_softmax"` into a 512-d feature,
  `use_separate_rgb_encoder_per_camera` default `False` (upstream: spatial softmax, 64-d, `True`)
- `rgb_encoder_init_checkpoint` / `audio_encoder_init_checkpoint`: encoders initialised from locally pretrained
  checkpoints (ignored by `from_pretrained`, a saved policy carries its own weights)
- `audio_backbone=None` gives an image-only policy with the fork's RGB encoder

Install with the `diffusion_audio` extra (adds `transformers`).

Declare the audio feature explicitly, with identity normalisation: the encoder normalises the raw log-mel
filterbank itself with `audio_norm_mean` / `audio_norm_std` (`(x - mean) / (2 * std)`, as for AST). If the features
are inferred from the dataset instead, the spectrogram is typed STATE and normalised by the preprocessor as well.

```json
"input_features": {"observation.audio.spectogram_values": {"type": "AUDIO", "shape": [3, 298, 128]}, ...},
"normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX", "AUDIO": "IDENTITY"}
```

The AST architecture config is stored in the policy config (`audio_backbone_config`) the first time the policy is
built, so a trained policy loads without the Hugging Face Hub.

## Migrating from the audio fork on v0.5.1

- Configs and saved checkpoints (`config.json`) with `"type": "diffusion"` and audio fields: change the type to
  `"diffusion_audio"`.
- Imports: `lerobot.policies.diffusion.modeling_diffusion` -> `lerobot.policies.diffusion_audio.modeling_diffusion_audio`
  (`DiffusionAudioEncoder`, `DiffusionRgbEncoder`, `OBS_AUDIO`; `DiffusionPolicy` -> `DiffusionAudioPolicy`).
- `predict_action_chunk` / `select_action` return the actions only (no attention maps or images any more), and
  `DiffusionRgbEncoder` returns its features only (was `(features, None)`).
