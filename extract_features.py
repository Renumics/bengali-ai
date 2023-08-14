# The imports you will need
import torch
import multiprocessing

import pickle
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import librosa
from mutagen.mp3 import MP3

# from renumics import spotlight
# from renumics.spotlight import Audio, Embedding
# from sliceguard import SliceGuard
from sliceguard.embeddings import generate_audio_embeddings, generate_text_embeddings

def main():
    # Configure the path to your dataset here
    DATASET_DIR = "/home/daniel/data/bengaliai/bengaliai-speech"
    dataset_dir = Path(DATASET_DIR)

    # Load the data
    df = pd.read_csv(dataset_dir / "train.csv")
    # Generate the audio paths
    df["path"] = str(dataset_dir / "train_mp3s") + "/" + df["id"] + ".mp3"

    print(len(df))
          
    audio_embeddings = generate_audio_embeddings(df["path"].values)

    with open("audio_embeddings.pkl", "wb") as f:
        pickle.dump(audio_embeddings, f)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    # multiprocessing.set_start_method("spawn", force=True)
    main()