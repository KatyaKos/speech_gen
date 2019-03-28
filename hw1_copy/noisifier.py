import numpy as np
import random
import librosa
import os
import argparse

FORMATS = ['.wav', '.flac']
SR = 25000


def is_format(file):
    return file.endswith('.wav') or file.endswith('.flac')


def no_format(file):
    n = len(file)
    for format in FORMATS:
        if file[n - len(format):] == format:
            return file[:n - len(format)]
    return None


def load_audio_sec(file_path, sr=SR):
    data, _ = librosa.core.load(file_path, sr)
    if len(data) > sr:
        data = data[:sr]
    else:
        data = np.pad(data, pad_width=(0, max(0, sr - len(data))), mode="constant")
    return data


def get_noises(dir):
    files = []
    for file in os.listdir(dir):
        if is_format(file):
            files.append(file)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to directory with input files to be noised.',
                        default='input/')
    parser.add_argument('-o', '--output', help='Path to directory with resulting files with noise.',
                        default='output/')
    parser.add_argument('-n', '--noise', help='Path to directory with noises.',
                        default='noise/')
    args = parser.parse_args()
    noises = get_noises(args.noise)
    noises_num = len(noises)

    for input_file in os.listdir(args.input):
        if is_format(input_file):
            audio = load_audio_sec(os.path.join(args.input, input_file))
            noise_file = noises[np.random.choice(noises_num)]
            noise = load_audio_sec(os.path.join(args.noise, noise_file))

            audio += 0.005 * noise
            new_name = os.path.join(os.path.join(no_format(input_file), no_format(noise_file)), ".wav")
            librosa.output.write_wav(os.path.join(args.output,  new_name), audio, sr=SR)


if __name__ == '__main__':
    main()
