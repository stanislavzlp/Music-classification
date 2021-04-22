"""
Подготовка данных для обучения модели:
 - каждая песня нарезается на непересекающиеся 10-секундные фрагменты,
 - для каждого из них строится спектрограмма,
 - спектрограмма нормализуется (приводится к диапазону 0-1),
 - спектрограмма сохраняется на диск в виде .npy - массива (либо в Train либо в Validation в соответствии с пропорцией).
"""

from glob import glob

import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


FRAGMENT_LEN = 10
TEST_SIZE = 0.25

tracks_dir = 'Data/songs'
music_paths = sorted(glob(tracks_dir + '/*.mp3') + glob(tracks_dir + '/*.flac'))

for track_id, track in tqdm(enumerate(music_paths)):
    loaded_track, sr = librosa.load(track)
    n_fragments = (len(loaded_track) // sr) // FRAGMENT_LEN
    for ind, i in tqdm(enumerate(range(n_fragments))):
        track_patch = loaded_track[i * sr * FRAGMENT_LEN: (i + 1) * sr * FRAGMENT_LEN]
        patch_spectrogram = librosa.feature.melspectrogram(y=track_patch, sr=sr)
        patch_spectrogram = librosa.power_to_db(patch_spectrogram, ref=np.max)
        patch_spectrogram = MinMaxScaler().fit_transform(patch_spectrogram)

        file_name = f'{track_id}_{ind}.npy'
        split = 'Train' if np.random.rand() >= TEST_SIZE else 'Validation'
        np.save(f'Data/Patches/{split}/{file_name}', patch_spectrogram)
