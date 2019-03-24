import numpy as np
import random
import librosa
import os
import argparse

FORMATS = ['.wav', '.flac']
SR = 25000


def is_format(file):
    flag = False
    n = len(file)
    for format in FORMATS:
        flag  = flag or file[n - len(format):] == format
    return flag


def no_format(file):
    n = len(file)
    for format in FORMATS:
        if file[n - len(format):] == format:
            return file[:n - len(format)]
    return "unknown"


def load_audio_sec(file_path, sr=SR):
    data, _ = librosa.core.load(file_path, sr)
    if len(data) > sr:
        data = data[:sr]
    else:
        data = np.pad(data, pad_width=(0, max(0, sr - len(data))), mode="constant")
    return data


def choose_noise(dir):
    n = 0
    for file in os.listdir(dir):
        if is_format(file):
            n += 1
    i = random.randrange(n)
    for file in os.listdir(dir):
        if is_format(file):
            if i == 0:
                return file
            i -= 1
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to directory with input files to be noised.',
                        default='input/')
    parser.add_argument('-o', '--output', help='Path to directory with resulting files with noise.',
                        default='output/')
    parser.add_argument('-n', '--noise', help='Path to directory with noises.',
                        default='noise/')
    args = parser.parse_args()

    for input_file in os.listdir(args.input):
        if is_format(input_file):
            audio = load_audio_sec(os.path.join(args.input, input_file))
            noise_file = choose_noise(args.noise)
            if noise_file is None:
                print("ERROR! No noise was chosen")
                exit(1)
            noise = load_audio_sec(os.path.join(args.noise, noise_file))

            audio += 0.005 * noise
            librosa.output.write_wav(os.path.join(args.output, no_format(input_file) + no_format(noise_file) + ".wav"),
                                     audio, sr=SR)


if __name__ == '__main__':
    main()
