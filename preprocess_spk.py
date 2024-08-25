from speaker_encoder.voice_encoder import SpeakerEncoder
from speaker_encoder.audio import preprocess_wav
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import argparse


def init_worker(out_dir, weights_fpath):
    global out
    global encoder
    out = out_dir
    encoder = SpeakerEncoder(weights_fpath)


def compute_spk_embed(wav_path):
    wav = preprocess_wav(wav_path)

    embed = encoder.embed_utterance(wav)

    fname_save = out / wav_path.parts[-2] / (wav_path.parts[-1].rstrip("wav") + "npy")
    np.save(fname_save, embed, allow_pickle=False)

    return fname_save.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=Path, default='dataset/vctk-16k/')
    parser.add_argument('--num_workers', type=int, default=cpu_count() - 2)
    parser.add_argument('--out_dir_root', type=Path, default='dataset/spk')
    parser.add_argument('--spk_encoder_ckpt', type=str,
                        default='speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    args = parser.parse_args()

    args.out_dir_root.mkdir(exist_ok=True, parents=True)

    for d in args.in_dir.iterdir():
        (args.out_dir_root / d.name).mkdir(exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker,
                             initargs=(args.out_dir_root, args.spk_encoder_ckpt)) as executor:
        executor.map(compute_spk_embed, args.in_dir.rglob("*.wav"))
