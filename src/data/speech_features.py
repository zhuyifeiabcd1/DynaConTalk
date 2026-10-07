"""Speech features of the DynaConTalk models, shared by the data preprocessing and the WebUI.

  rhythm    amplitude envelope, short-time energy, binary onsets (3)
  mel       128-band mel power spectrogram, no log transform (128)
  semantic  HuBERT-Large last hidden state, resampled from 50 fps (1024)
  clip_text CLIP ViT-B/32 text embedding of the transcript (512)

All at 30 fps from 16 kHz mono audio. Extract with TF32 disabled, as the training data was.
"""
import librosa
import numpy as np
import torch
import torch.nn.functional as F

POSE_FPS = 30


def extract_rhythm(audio_16k, sr=16000, frame_length=1024, hop_length=512, target_fps=POSE_FPS):
    """3D rhythm: amplitude envelope + short-time energy + binary onsets."""
    from numpy.lib import stride_tricks

    # The envelope is the maximum over a frame_length window starting at each of the first
    # len // hop_length samples (the convention of the training features); only those
    # windows are evaluated.
    shape = (audio_16k.shape[-1] - frame_length + 1, frame_length)
    strides = (audio_16k.strides[-1], audio_16k.strides[-1])
    rolling_view = stride_tricks.as_strided(audio_16k, shape=shape, strides=strides)
    amplitude_envelope = np.max(np.abs(rolling_view[:len(audio_16k) // hop_length]), axis=1)

    energy = np.array([
        np.sum(np.abs(audio_16k[i:i + frame_length] ** 2))
        for i in range(0, len(audio_16k), hop_length)
    ])
    energy = energy[:len(amplitude_envelope)]

    audio_onset_f = librosa.onset.onset_detect(y=audio_16k, sr=sr, hop_length=hop_length, units="frames")
    onset_array = np.zeros(len(amplitude_envelope), dtype=float)
    valid_onsets = audio_onset_f[audio_onset_f < len(onset_array)]
    onset_array[valid_onsets] = 1.0

    features = np.stack([amplitude_envelope, energy, onset_array], axis=1)

    duration = len(audio_16k) / sr
    num_frames = max(int(duration * target_fps), 1)
    resampled = np.zeros((num_frames, 3), dtype=np.float32)
    for i in range(3):
        resampled[:, i] = np.interp(
            np.linspace(0, len(features) - 1, num_frames), np.arange(len(features)), features[:, i]
        )
    return resampled


def extract_mel(audio_16k, sr=16000, n_mels=128, target_fps=POSE_FPS):
    """128-band mel power spectrogram, no log transform, no normalization."""
    mel_hop = int(sr / target_fps)
    mel_spec = librosa.feature.melspectrogram(y=audio_16k, sr=sr, n_mels=n_mels, hop_length=mel_hop)
    mel_spec = mel_spec[..., :-1]
    return np.swapaxes(mel_spec, -1, -2).astype(np.float32)


def load_hubert(device):
    from transformers import HubertModel, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device).eval()
    return processor, model


def extract_hubert(processor, model, audio_16k, device, target_fps=POSE_FPS):
    """HuBERT-Large last hidden state, ~20 s chunks overlapping by one receptive field,
    linearly resampled from 50 fps to target_fps."""
    kernel, stride = 400, 320
    clip_length = stride * 1000

    input_values_all = processor(audio_16k, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride

    res_lst = []
    with torch.no_grad():
        for i in range(num_iter):
            if i == 0:
                start_idx, end_idx = 0, clip_length - stride + kernel
            else:
                start_idx = clip_length * i
                end_idx = start_idx + (clip_length - stride + kernel)
            res_lst.append(model(input_values_all[:, start_idx:end_idx]).last_hidden_state[0])
        input_values = input_values_all[:, clip_length * num_iter:] if num_iter > 0 else input_values_all
        if input_values.shape[1] >= kernel:
            res_lst.append(model(input_values).last_hidden_state[0])

    features = torch.cat(res_lst, dim=0).cpu()
    assert abs(features.shape[0] - expected_T) <= 1, f"HuBERT output {features.shape[0]} vs expected {expected_T}"
    if features.shape[0] < expected_T:
        features = F.pad(features, (0, 0, 0, expected_T - features.shape[0]))
    else:
        features = features[:expected_T]
    features = features.numpy()

    from scipy.interpolate import interp1d

    hubert_fps = 16000 / stride
    target_len = max(int(len(audio_16k) / 16000 * target_fps), 1)
    src_times = np.arange(features.shape[0]) / hubert_fps
    tgt_times = np.clip(np.arange(target_len) / target_fps, src_times[0], src_times[-1])
    interp_fn = interp1d(src_times, features, axis=0, kind="linear", fill_value="extrapolate")
    return interp_fn(tgt_times).astype(np.float32)


def load_clip(device):
    import clip

    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, clip


def encode_text(clip_model, clip_module, sentence, device):
    if not sentence:
        return np.zeros(512, dtype=np.float32)
    with torch.no_grad():
        tokens = clip_module.tokenize([sentence], truncate=True).to(device)
        return clip_model.encode_text(tokens).float().squeeze(0).cpu().numpy().astype(np.float32)
