import numpy as np
import librosa
import scipy
from hparam import hparams

def convert_audio(wav_path):
    wav = load_wav(wav_path)
    mel = melspectrogram(wav).astype(np.float32)
    return mel.transpose(), wav

def load_wav(filename) :
    x = librosa.load(filename, sr=hparams.sample_rate)[0]
    return x

def save_wav(y, filename) :
    scipy.io.wavfile.write(filename, hparams.sample_rate, y.astype(np.int16))

mel_basis = None

def linear_to_mel(spectrogram):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)

def build_mel_basis():
    return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels, fmin=hparams.fmin, fmax=hparams.fmax)

def normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

def denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
    return np.power(10.0, x * 0.05)

def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - hparams.ref_level_db
    return normalize(S)

def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)

def stft(y):
    return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=hparams.hop_length, win_length=hparams.win_length)
