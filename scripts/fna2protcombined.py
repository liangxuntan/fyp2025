from pathlib import Path
from typing import List
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pyrodigal
import re
import multiprocessing
import sys
import os
import logging

from datetime import datetime

# Generate a timestamp string for this run
_now = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_LOG_DIR = Path("/Users/liangxuntan/Code/fyp2025/data/logs")
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / f"protein_prediction_{_now}.log"

def setup_logging(logfile=DEFAULT_LOG_FILE, level=logging.INFO):
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfmt = "[%(asctime)s %(levelname)s] %(message)s"
    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=level, format=logfmt, handlers=handlers, force=True)

def sci_to_conf(evalue: str) -> float:
    """Convert scientific-notation E-value string to Pyrodigal confidence (0–100)."""
    val = float(evalue)
    if not 0 < val < 1:
        raise ValueError("E-value must be between 0 and 1 (exclusive)")
    return 100 * (1 - val)

def run_fna2prot(fnafile: str, evalue: str = "1e-3") -> Path:
    """
    Predict proteins from nucleotide FASTA file using Pyrodigal.
    Returns the path to the output protein FASTA file.
    """
    fasta_path = Path(fnafile)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {fasta_path}")

    match = re.search(r"/(\d+)_GCF_", str(fasta_path))
    if match:
        number = match.group(1)
    else:
        number = fasta_path.stem  # fallback to filename stem if no match

    outputdir = fasta_path.parent.parent / "prodigal_output"
    outputdir.mkdir(parents=True, exist_ok=True)
    outfile = outputdir / f"{number}_predictedproteins.prot"

    conf_threshold = sci_to_conf(evalue)
    gene_finder = pyrodigal.GeneFinder(meta=True)
    predicted: List[SeqRecord] = []

    for record in SeqIO.parse(str(fasta_path), "fasta"):
        for gene in gene_finder.find_genes(str(record.seq)):
            if gene.confidence() >= conf_threshold:
                protein_seq = gene.translate()
                prot_id = f"{record.id}_gene_{gene.begin}_{gene.end}"
                predicted.append(SeqRecord(protein_seq, id=prot_id, description=""))

    SeqIO.write(predicted, outfile, "fasta")
    logging.info(f"Written {len(predicted)} protein sequences to '{outfile}'")

    return outfile

def process_file(fna_path_and_evalue):
    setup_logging()
    fna_path, evalue = fna_path_and_evalue
    try:
        result = run_fna2prot(str(fna_path), evalue)
        logging.info(f"Finished: {fna_path} → {result}")
        return result
    except Exception as e:
        logging.error(f"Error processing {fna_path}: {e}")
        return None

def run_all(fna_files, evalue="1e-3", nproc=4):
    with multiprocessing.Pool(processes=nproc) as pool:
        results = pool.map(process_file, [(f, evalue) for f in fna_files])
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Predict protein-coding genes with Pyrodigal (supports batch processing)"
    )
    parser.add_argument("fasta_dir", help="Directory with .fna files")
    parser.add_argument(
        "-e",
        "--evalue",
        default="1e-3",
        help="E-value threshold (scientific notation)",
    )
    parser.add_argument(
        "-n",
        "--nproc",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of processes to use"
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=str(DEFAULT_LOG_FILE),
        help="Log file path"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.logfile, level=logging.INFO)

    fasta_dir = Path(args.fasta_dir).expanduser().resolve()
    fna_files = sorted(fasta_dir.glob("*.fna"))

    if not fna_files:
        logging.warning(f"No .fna files found in {fasta_dir}")
        sys.exit(1)

    logging.info(f"Starting prediction on {len(fna_files)} files with {args.nproc} processes.")
    results = run_all(fna_files, args.evalue, args.nproc)
    logging.info("Protein prediction output files:")
    for r in results:
        logging.info(r)

if __name__ == "__main__":
    main()