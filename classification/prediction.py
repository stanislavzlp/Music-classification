import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

FRAGMENT_LEN = 10
genres = pd.read_csv('Data/genres.csv')
genres = list(genres.columns[1:])


def predict(music_path: str,
            model_path: str,
            topk: int = 4,
            strategy: str = 'max'):
    """
    Основная функция для предсказания жанров музыки:
    Трек нарезается на 10-ти секундные фрагменты, на каждом из них модель делает предсказания
    Далее (в зависимости от strategy) эти предсказания либо усредняются, либо берётся максимум.
    """

    assert strategy in ['max', 'mean'], 'Only max and mean strategies are available!'

    FRAGMENT_LEN = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_path)
    model.to(DEVICE)
    model.eval()

    predictions = []

    loaded_track, sr = librosa.load(music_path)
    n_fragments = (len(loaded_track) // sr) // FRAGMENT_LEN
    for ind, i in tqdm(enumerate(range(n_fragments))):
        track_patch = loaded_track[i * sr * FRAGMENT_LEN: (i + 1) * sr * FRAGMENT_LEN]
        patch_spectrogram = librosa.feature.melspectrogram(y=track_patch, sr=sr)
        patch_spectrogram = librosa.power_to_db(patch_spectrogram, ref=np.max)
        patch_spectrogram = MinMaxScaler().fit_transform(patch_spectrogram)
        patch_spectrogram = torch.from_numpy(patch_spectrogram[np.newaxis, np.newaxis, ...]).type(torch.float32).to(
            DEVICE)

        with torch.no_grad():
            prediction = model(patch_spectrogram)
        predictions.append(prediction)

    torch_preds = torch.stack(predictions, axis=-1).squeeze()

    if strategy == 'max':
        prediction = torch.max(torch_preds, axis=1).values
    elif strategy == 'mean':
        prediction = torch.mean(torch_preds, axis=1).values

    topk_preds = torch.topk(prediction, topk)
    final_preds = [genres[i] for i in topk_preds.indices]

    return final_preds, topk_preds.values.cpu().detach().numpy()


if __name__ == '__main__':

    model_path = 'saved_model_0.8155053490990991'
    music_path = "Data/songs/Amen_(Feat_Aquilo)-Enigma.mp3"

    preds, probs = predict(music_path, model_path=model_path, topk=10)

    print(f'Predicted labels: {preds}')
    print(f'With probabilities: {probs}')
