import os
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import argparse
from pathlib import Path
import pickle


def compute_column_distribution(data, bins=50):
    """Compute histogram distribution for each column of the DataFrame."""
    histograms = {}
    for column in data.columns:
        hist, bin_edges = np.histogram(data[column], bins=bins, density=True)
        histograms[column] = (hist, bin_edges)
    return histograms


def compare_distributions(hist1, hist2):
    """Compare two histogram distributions using the Wasserstein distance."""
    distances = []
    for col in hist1:
        dist = wasserstein_distance(hist1[col][0], hist2[col][0])
        distances.append(dist)
    return np.mean(distances)


def select_top_n_files(data_path, n_files, subset, bins):
    """Select top n files representing the distribution of all files."""
    all_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')])[:subset]
    overall_data = pd.concat([pd.read_csv(file) for file in tqdm(all_files, desc="Loading files")], ignore_index=True)
    overall_data = overall_data.drop(columns=['t', 'steerCommand'])
    overall_hist = compute_column_distribution(overall_data, bins=bins)
    
    file_scores = []
    for file in tqdm(all_files, desc="Scoring files"):
        data = pd.read_csv(file)
        data = data.drop(columns=['t', 'steerCommand'])
        file_hist = compute_column_distribution(data, bins=bins)
        score = compare_distributions(overall_hist, file_hist)
        file_scores.append((file, score))
    
    # Sort files by score (lower score means closer distribution) and select top n
    top_n_files = sorted(file_scores, key=lambda x: x[1])[:n_files]
    
    dfs = []
    for file, _ in tqdm(top_n_files, desc='Scoring selection'):
        data = pd.read_csv(file)
        data = data.drop(columns=['t', 'steerCommand'])
        dfs.append(data)
    final_df = pd.concat(dfs)
    final_hist = compute_column_distribution(final_df, bins=bins)
    final_score = compare_distributions(overall_hist, final_hist)
    
    print(f'The final distance of the selected files and the overall distribution is: {final_score}')
    
    return [file for file, score in top_n_files]


def save_filenames(top_n_files, file_out):
    with open(f'{file_out}.pkl', 'wb') as handle:
        pickle.dump(top_n_files, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(data_path, n_files, subset, bins, file_out):
    top_n_files = select_top_n_files(data_path, n_files, subset, bins=bins)
    save_filenames(top_n_files, file_out)
    print(f"Filenames saved to {file_out}.pkl")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_files", type=int, required=True)
    parser.add_argument("--subset", type=int, default=20_000)
    parser.add_argument("--bins", default='fd')
    parser.add_argument("--file_out", type=str, default='out')
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    assert data_path.is_dir(), "data_path should be a directory"
    
    main(data_path, args.n_files, args.subset, args.bins, args.file_out)
    