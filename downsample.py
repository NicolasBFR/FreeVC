import io
import os
import argparse
from pathlib import Path

import librosa
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from scipy.io import wavfile
from tqdm import tqdm
import zipfile


def yield_wavs(zip_file: zipfile.ZipFile):
    zip_path = zipfile.Path(zip_file, at="VCTK-Corpus/wav48/")
    res = []
    for f in filter(
            lambda x: all(
                substring not in x.name for substring in ["s5", "p280", "p315"]
            ),
            zip_path.iterdir(),
    ):
        res += list(f.iterdir())
    return res


def process(wav_path: zipfile.Path):
    wav_name = wav_path.name
    # speaker 's5', 'p280', 'p315' are excluded,
    speaker = wav_name[:4]
    # wav_path = os.path.join(args.in_dir, speaker, wav_name)
    wav, sr = librosa.load(wav_path.open(mode='rb'))
    wav, _ = librosa.effects.trim(wav, top_db=20)
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = 0.98 * wav / peak
    wav = librosa.resample(wav, orig_sr=sr, target_sr=args.sr)
    save_name = wav_name.replace("_mic2.flac", ".wav")
    save_path = os.path.join(args.out_dir, speaker, save_name)
    wavfile.write(
        save_path,
        args.sr,
        (wav * np.iinfo(np.int16).max).astype(np.int16)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("--in_zip", type=Path, default="VCTK-Corpus.zip", help="path to source archive")
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument("--out_dir", type=Path, default="./dataset/vctk-16k", help="path to target dir")
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)

    with zipfile.ZipFile(args.in_zip) as z:
        for e in zipfile.Path(z, at="VCTK-Corpus/wav48/").iterdir():
            (args.out_dir / e.name).mkdir(exist_ok=True)

        files = yield_wavs(z)
        process(files[0])
        # with ThreadPoolExecutor(args.num_workers) as executor:
        #     executor.map(process, files)
