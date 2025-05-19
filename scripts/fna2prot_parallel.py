from pathlib import Path
import multiprocessing
import sys
from fna2prot import run_fna2prot

def process_file(fna_path_and_evalue):
    fna_path, evalue = fna_path_and_evalue
    try:
        return run_fna2prot(str(fna_path), evalue)
    except Exception as e:
        print(f"Error processing {fna_path}: {e}")
        return None

def run_all(fna_files, evalue, nproc=4):
    with multiprocessing.Pool(processes=nproc) as pool:
        results = pool.map(process_file, [(f, evalue) for f in fna_files])
    return results