#!/Users/liangxuntan/anaconda3/envs/mne/bin/python
"""
Predict protein-coding genes from a nucleotide FASTA using Pyrodigal.
"""

from pathlib import Path
from typing import List

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pyrodigal
import re


def sci_to_conf(evalue: str) -> float:
    """Convert scientific-notation E-value string to Pyrodigal confidence (0–100)."""
    val = float(evalue)
    if not 0 < val < 1:
        raise ValueError("E-value must be between 0 and 1 (exclusive)")
    return 100 * (1 - val)


def run_fna2prot(fnafile: str, evalue: str = "1e-50") -> Path:
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
    print(f"Written {len(predicted)} protein sequences to '{outfile}'")

    return outfile


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict protein-coding genes with Pyrodigal"
    )
    parser.add_argument("fnafile", help="Input contig FASTA (*.fna / *.fa / *.fasta)")
    parser.add_argument(
        "-e",
        "--evalue",
        default="1e-50",
        help="E-value threshold (scientific notation)",
    )
    args = parser.parse_args()

    run_fna2prot(args.fnafile, args.evalue)


if __name__ == "__main__":
    main()