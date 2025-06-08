# Predicting Microbial Cultivation Media through Genomics-Guided Machine Learning

## Project Overview

This project uses machine learning to predict microbial cultivation media recipes directly from genome-derived protein family profiles, leveraging unique recipes and genome data from DSMZ Mediadive and NCBI. The open-source model achieves high accuracy for key media components and reveals protein families linked to nutrient requirements, accelerating efforts to culture novel microorganisms.

---

## Directory Structure

<pre>
├── jnotebooks
│   ├── datacollection.ipynb      # Jupyter notebook for data collection
│   ├── modelbuilding.ipynb       # Jupyter notebook for model development
├── scripts
│   ├── taxid2fna.sh              # Shell script to fetch FASTA from taxids
│   ├── prot2pfam.sh              # Shell script for Pfam annotation
│   ├── run_taxid2fna.sh          # Batch processing script
│   ├── mlfunctions3.py           # Core ML functions in Python
│   ├── fna2protcombined.py       # FASTA to protein conversion
│   ├── gmcfunctions3.py          # Genomic to media composition functions
│   ├── PfamDB.json               # Pfam database file
│   ├── HPC_hmmer_annotation_scripts
│   │   ├── run_prot2pfam_full.sh     # HPC batch script for Pfam annotation
│   │   ├── prot2pfam_cluster.sh      # Cluster-optimized Pfam annotation
</pre>

---

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/liangxuntan/fyp2025.git
    cd yourrepo
    ```

2. **Set up the environment:**  
   *(Example with conda, adapt as needed)*
   ```bash
   conda env create -f environment.yml
   conda activate microbe_media

3. **Download required data:**  
   datacollection.ipynb to download media recipe and NCBI genome data

4. **Build model:**  
   modelbuilding.ipynb to train model and run analysis

---

## Usage

- **Data Collection:**  
    - `jnotebooks/datacollection.ipynb` — Jupyter notebook for data download, cleaning, and preprocessing.

- **Model Building:**  
    - `jnotebooks/modelbuilding.ipynb` — Notebook for model training, validation, and evaluation.

- **Scripts:**  
    - `scripts/taxid2fna.sh` — Fetch genome FASTA files using NCBI Taxonomy IDs.
    - `scripts/prot2pfam.sh` — Annotate proteins with Pfam domains using HMMER.
    - `scripts/mlfunctions3.py` — Machine learning pipeline functions for model training and prediction.
    - `scripts/gmcfunctions3.py` — Functions for mapping genomics data to media composition, and download data.
    - `scripts/HPC_hmmer_annotation_scripts/` — Scripts for running large-scale Pfam annotation on HPC clusters.

---

## Contributing

Contributions, suggestions, and pull requests are welcome!  
Feel free to open an issue or submit PRs for improvements or bug fixes.

---


*This project aims to accelerate efforts in culturing novel microorganisms through data-driven, genome-guided media design. For questions or collaborations, please reach out!*
