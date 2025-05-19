import requests
import pandas as pd
from pandas import json_normalize
import bacdive
from collections.abc import Iterable
import subprocess
from pathlib import Path
from fna2prot_parallel import run_all
import collections
from collections import defaultdict

def fetch_medium_compositions(medianums):
    """
    Given an iterable of DSMZ medium numbers, query the DSMZ MediaDive API
    and return one merged DataFrame whose columns are the medium numbers
    and whose index is the compound name.

    Parameters
    ----------
    medianums : Iterable[int | str]
        Medium numbers accepted by the MediaDive REST endpoint.

    Returns
    -------
    pd.DataFrame
        A wide table with one row per compound (“name”) and one column per
        medium number, containing g/L values. Media not found (HTTP≠200)
        simply create NaNs in the result.
    """
    merged_df = pd.DataFrame()

    for num in medianums:
        url = f"https://mediadive.dsmz.de/rest/medium-composition/{num}"
        resp = requests.get(url)

        if resp.status_code == 200:
            data = resp.json().get("data", [])
            df = json_normalize(data)[["name", "g_l"]].rename(columns={"g_l": str(num)})

            merged_df = df if merged_df.empty else pd.merge(
                merged_df, df, on="name", how="outer"
            )
        else:
            print(f"Error {resp.status_code} for medium {num}")

    return merged_df

def fetch_medium_strains(medianums):
    """
    Query the MediaDive “medium-strains” endpoint for each medium number
    and return a single DataFrame with all results side-by-side.

    Parameters
    ----------
    medianums : Iterable[int | str]
        DSMZ medium numbers.

    Returns
    -------
    pd.DataFrame
        Columns come in pairs:
            ┌───────────────┬────────┐
            │ species_<num> │ id_<num>│
            └───────────────┴────────┘
        where <num> is the medium number.
    """
    dfs = []

    for num in medianums:
        url = f"https://mediadive.dsmz.de/rest/medium-strains/{num}"
        resp = requests.get(url)

        if resp.status_code == 200:
            data = resp.json().get("data", [])
            df = (
                json_normalize(data)[["species", "bacdive_id"]]
                .rename(
                    columns={
                        "species": f"species_{num}",
                        "bacdive_id": f"id_{num}",
                    }
                )
            )
            dfs.append(df)
        else:
            print(f"Error {resp.status_code} for medium {num}")

    # Concatenate along columns (axis=1); rows align automatically
    return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

def get_first_bacdive_ids(medianums):
    """
    For each medium number, fetch the first BacDive strain ID listed by
    the MediaDive “medium-strains” endpoint.

    Parameters
    ----------
    medianums : Iterable[int | str]

    Returns
    -------
    dict
        {medium_number: first_bacdive_id or None}
    """
    bacdive_id_dict = {}

    for num in medianums:
        url = f"https://mediadive.dsmz.de/rest/medium-strains/{num}"
        resp = requests.get(url)

        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                df = json_normalize(data)
                first_id = df.iloc[0].get("bacdive_id") if "bacdive_id" in df.columns else None
                if first_id is None:
                    print(f"No 'bacdive_id' in data for {num}")
                bacdive_id_dict[num] = first_id
            else:
                print(f"No data found for {num}")
                bacdive_id_dict[num] = None
        else:
            print(f"Error {resp.status_code} for {num}")
            bacdive_id_dict[num] = None

    return bacdive_id_dict



def map_species_taxids_to_media(
    medianums: Iterable[int | str],
    bacdive_id_dict: dict,
    email: str,
    password: str,
):
    """
    For each medium number → BacDive strain ID, fetch the strain record and
    pull the NCBI species–level tax-id.  Return a dict {tax_id: medium_num}.

    Parameters
    ----------
    medianums : iterable
        Medium numbers you want to look up.
    bacdive_id_dict : dict
        Mapping produced earlier: {medium_num: first_bacdive_id or None}.
    email, password : str
        BacDive credentials for BacdiveClient.

    Returns
    -------
    dict
        {NCBI_species_taxid: medium_number}
    """
    client = bacdive.BacdiveClient(email, password)
    taxid_to_medium = {}

    for num in medianums:
        bacdive_id = bacdive_id_dict.get(num)
        print(f"medium {num}: BacDive ID → {bacdive_id}")

        if bacdive_id is None:
            print(f"No valid BacDive ID for medium {num}")
            continue

        # search() queues the record; retrieve() actually yields it
        client.search(id=int(bacdive_id))

        for record in client.retrieve(["NCBI tax id"]):
            # BacDive returns a dict keyed by the strain number
            strain_key = next(iter(record))
            ncbi_entries = record[strain_key]  # always a list

            for entry in ncbi_entries:
                ncbi_info = entry.get("NCBI tax id")

                # ncbi_info can be dict or list-of-dicts
                info_iter = (
                    ncbi_info if isinstance(ncbi_info, list) else [ncbi_info]
                )

                for sub in info_iter:
                    if not isinstance(sub, dict):
                        continue
                    if sub.get("Matching level") == "species":
                        tax_id = sub.get("NCBI tax id")
                        if tax_id:
                            print(f"  » species-level taxid {tax_id}")
                            taxid_to_medium[tax_id] = num
        # loop continues for next medium number

    return taxid_to_medium

def run_taxid2fna(taxid_list, script_path="/Users/liangxuntan/Code/fyp2025/scripts/run_taxid2fna.sh"):
    """
    Run `run_taxid2fna.sh` passing each TaxID in `taxid_list` as a separate CLI argument.

    Parameters
    ----------
    taxid_list : list of int or str
        List of NCBI TaxIDs to pass to the script.
    script_path : str or Path, optional
        Path to the shell script (default is user-specific path).

    Raises
    ------
    FileNotFoundError
        If the script is not found at `script_path`.
    ValueError
        If taxid_list is empty or None.
    subprocess.CalledProcessError
        If the script exits with a non-zero status.
    """
    script_path = Path(script_path).expanduser().resolve()

    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    if not taxid_list:
        raise ValueError("No TaxIDs supplied.")

    taxid_args = [str(tid) for tid in taxid_list if tid is not None]

    if not taxid_args:
        raise ValueError("No valid TaxIDs after filtering None.")

    cmd = [str(script_path), *taxid_args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_protein_prediction(fasta_dir_path, evalue="1e-30", nproc=4):
    """
    Run protein prediction on all .fna files in the specified directory.

    Parameters
    ----------
    fasta_dir_path : str or Path
        Path to the directory containing .fna files.
    evalue : str, optional
        E-value threshold for run_all (default "1e-30").
    nproc : int, optional
        Number of processors to use (default 4).

    Returns
    -------
    list of Path
        List of output files generated by run_all.
    """
    fasta_dir = Path(fasta_dir_path).expanduser().resolve()
    fna_files = sorted(fasta_dir.glob("*.fna"))
    
    if not fna_files:
        print(f"No .fna files found in {fasta_dir}")
        return []
    
    results = run_all(fna_files, evalue, nproc=nproc)
    
    print("Protein prediction output files:")
    for r in results:
        print(r)
    
    return results

# Example usage:
# run_protein_prediction("/Users/liangxuntan/Code/fyp2025/data/labelled_genomes")


def run_prot2pfam(prodigal_dir_path, 
                  pfamscript_path="/Users/liangxuntan/Code/fyp2025/scripts/prot2pfam.sh",
                  jobs=8):
    """
    Run the prot2pfam.sh script on all .prot files in the specified directory.

    Parameters
    ----------
    prodigal_dir_path : str or Path
        Directory containing .prot files.
    pfamscript_path : str or Path, optional
        Path to the prot2pfam.sh script (default is the user's script path).
    jobs : int, optional
        Number of parallel jobs to run (default is 8).

    Returns
    -------
    CompletedProcess or None
        The result object from subprocess.run including stdout and stderr,
        or None if no .prot files are found.
    """
    prodigal_dir = Path(prodigal_dir_path).expanduser().resolve()
    pfamscript = Path(pfamscript_path).expanduser().resolve()

    prodigal_files = sorted(prodigal_dir.glob("*.prot"))
    if not prodigal_files:
        print(f"No .prot files found in {prodigal_dir}")
        return None

    file_args = [str(f) for f in prodigal_files]
    cmd = [str(pfamscript), "-j", str(jobs)] + file_args
    print("Running:", " ".join(cmd))

    result = subprocess.run(cmd)  # stdout and stderr go directly to the console
    if result.returncode != 0:
        print(f"Error running prot2pfam.sh, return code: {result.returncode}")
    return result

def build_pfam_matrix(
    hmmer_dir_path, 
    taxid2media_dict,
    map_taxid: bool = True,
    save_path: str = None
):
    """
    Build a Pfam count matrix from hmmer output files and optionally map taxid indices to media IDs.

    Parameters
    ----------
    hmmer_dir_path : str or Path
        Directory containing *.tbl files named <taxid>_pfam.tbl.
    taxid2media_dict : dict
        Mapping from integer taxid to media id(s). Values can be int or list of ints.
    map_taxid : bool, optional, default=True
        If True, remap taxid index to media IDs using taxid2media_dict.
    save_path : str, optional, default=None
        If provided, path to save the resulting DataFrame as a CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with taxid or media IDs as index, Pfam names as columns, and counts as values.
    """

    hmmer_dir = Path(hmmer_dir_path).expanduser().resolve()
    hmmer_files = sorted(hmmer_dir.glob("*.tbl"))

    count_dict = {}

    for f in hmmer_files:
        pfam_counts = collections.Counter()
        with f.open() as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.split()
                pfam_name = fields[0]  # 1st column = Pfam HMM name
                pfam_counts[pfam_name] += 1

        taxid = f.stem.split('_')[0]  # filename format: <taxid>_pfam.tbl
        count_dict[taxid] = pfam_counts

    pfam_df = pd.DataFrame.from_dict(count_dict, orient="index").fillna(0).astype(int)

    if map_taxid:
        media_map = {
            str(k): (v if isinstance(v, list) else [v])
            for k, v in taxid2media_dict.items()
        }

        used_counter = defaultdict(int)
        missing_taxids = []

        def pick_media_id(taxid):
            if taxid not in media_map:
                missing_taxids.append(taxid)
                return taxid
            i = used_counter[taxid]
            id_list = media_map[taxid]
            used_counter[taxid] += 1
            return id_list[i] if i < len(id_list) else id_list[-1]

        pfam_df.index = [pick_media_id(tid) for tid in pfam_df.index]

        try:
            pfam_df.index = pfam_df.index.astype(int)
        except ValueError:
            pass

        if missing_taxids:
            print("TaxIDs without media ID:", sorted(set(str(t) for t in missing_taxids)))

    if save_path:
        pfam_df.to_csv(save_path)

    return pfam_df

# Example usage: 
# df = build_pfam_matrix(
#     hmmer_dir_path="results/hmmer",
#     taxid2media_dict={123: [1, 2], 456: 3},
#     map_taxid=True,
#     save_path="pfam_matrix.csv"
# )

def get_media_ingredient_matrix(medianums, binary=True, save_path=None):
    """
    Fetch medium composition data for given medianums from BacDive API,
    merge them into a DataFrame (medium IDs x ingredient names).

    Parameters:
    -----------
    medianums : list of int
        List of medium IDs to query.
    binary : bool, optional (default=True)
        If True, convert concentrations to presence (1) / absence (0).
        If False, keep original g/L numeric values.
    save_path : str or None, optional (default=None)
        If provided, saves the resulting DataFrame to this path as a CSV file.

    Returns:
    --------
    pd.DataFrame
        DataFrame with medium IDs as rows and ingredient names as columns.
        Values are binary presence/absence if binary=True, else original g/L values.
    """
    merged_df = pd.DataFrame()

    with requests.Session() as s:
        for num in medianums:
            url = f'https://mediadive.dsmz.de/rest/medium-composition/{num}'
            resp = s.get(url)

            if resp.status_code == 200:
                data = resp.json().get('data')
                if data:
                    idf = json_normalize(data)[["name", "g_l"]]
                    idf = idf.rename(columns={"g_l": str(num)})
                    merged_df = (
                        idf if merged_df.empty
                        else pd.merge(merged_df, idf, on="name", how="outer")
                    )
            else:
                print(f"Error {resp.status_code} for medium ID {num}")

    if merged_df.empty:
        print("No valid data retrieved.")
        return pd.DataFrame()

    rownames = merged_df["name"].values
    medianum_strs = merged_df.columns[1:]  # skip 'name'

    media_df = merged_df.set_index("name").T
    media_df.index = medianum_strs.astype(int)
    media_df.columns = rownames

    if binary:
        media_df = (media_df > 0).astype(int)

    if save_path:
        media_df.to_csv(save_path)
        print(f"Data saved to {save_path}")

    return media_df

#example usage:
# matrix = get_media_ingredient_matrix([1, 2, 3], binary=False, save_path="media_matrix.csv")
