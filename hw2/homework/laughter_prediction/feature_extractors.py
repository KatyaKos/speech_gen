import os
import tempfile
import librosa
import numpy as np
import pandas as pd


class FeatureExtractor:
    def extract_features(self, wav_path, frame_sec=0.01):
        y, sr = librosa.load(wav_path, dtype=float)
        size = int(sr * frame_sec)
        mfcc = []
        fbank = []
        for i in range(0, len(y) - size, int(size / 5)):
            val = librosa.feature.mfcc(y=y.astype(np.float)[i: i + int(size / 5)], sr=sr)
            mfcc.append(np.mean(val, axis=1))
            val = librosa.feature.melspectrogram(y=y.astype(np.float)[i: i + int(size / 5)], sr=sr)
            fbank.append(np.mean(np.log(val), axis=1))
        mfcc = np.vstack(mfcc)
        fbank = np.vstack(fbank)
        mfcc = pd.DataFrame(mfcc, columns=list(map(lambda j: "mfcc_{}".format(j), list(range(mfcc.shape[1])))))
        fbank = pd.DataFrame(fbank, columns=list(map(lambda j: "fbank_{}".format(j), list(range(fbank.shape[1])))))
        return mfcc, fbank


class PyAAExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""
    def __init__(self):
        self.extract_script = "./extract_pyAA_features.py"
        self.py_env_name = "ipykernel_py2"

    def extract_features(self, wav_path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            feature_save_path = tmp_file.name
            cmd = "python \"{}\" --wav_path=\"{}\" " \
                  "--feature_save_path=\"{}\"".format(self.extract_script, wav_path, feature_save_path)
            os.system("source activate {}; {}".format(self.py_env_name, cmd))

            feature_df = pd.read_csv(feature_save_path)
        return feature_df
