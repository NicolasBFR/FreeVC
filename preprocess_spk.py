import os, sys
from speaker_encoder.voice_encoder import SpeakerEncoder
from speaker_encoder.audio import preprocess_wav
from pathlib import Path
import numpy as np
from os.path import join, basename, split
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor 
from functools import partial
import glob
import argparse


def build_from_path(in_dir, out_dir, weights_fpath, num_workers=1):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    wavfile_paths = glob.glob(os.path.join(in_dir, '*.wav'))
    wavfile_paths= sorted(wavfile_paths)
    for wav_path in wavfile_paths:
        futures.append(executor.submit(
            partial(_compute_spkEmbed, out_dir, wav_path, weights_fpath)))
    return [future.result() for future in tqdm(futures)]

def _compute_spkEmbed(out_dir, wav_path, weights_fpath):
    utt_id = os.path.basename(wav_path).rstrip(".wav")
    fpath = Path(wav_path)
    wav = preprocess_wav(fpath)

    encoder = SpeakerEncoder(weights_fpath)
    embed = encoder.embed_utterance(wav)
    fname_save = os.path.join(out_dir, f"{utt_id}.npy")
    np.save(fname_save, embed, allow_pickle=False)
    return os.path.basename(fname_save)

def preprocess(in_dir, out_dir_root, spk, weights_fpath, num_workers):
    out_dir = os.path.join(out_dir_root, spk)
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir, weights_fpath, num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, 
        default='dataset/vctk-16k/')
    parser.add_argument('--num_workers', type=int, default=cpu_count()-2)
    parser.add_argument('--out_dir_root', type=str, 
        default='dataset')
    parser.add_argument('--spk_encoder_ckpt', type=str, \
        default='speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    args = parser.parse_args()

    sub_folder_list = os.listdir(args.in_dir)

    spk_embed_out_dir = os.path.join(args.out_dir_root, "spk")
    os.makedirs(spk_embed_out_dir, exist_ok=True)

    for spk in sub_folder_list:
        print("Preprocessing {} ...".format(spk))
        in_dir = os.path.join(args.in_dir, spk)
        if not os.path.isdir(in_dir): 
            continue
        preprocess(in_dir, spk_embed_out_dir, spk, args.spk_encoder_ckpt, args.num_workers)


