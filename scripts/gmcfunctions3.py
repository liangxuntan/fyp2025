import requests
import pandas as pd
from pandas import json_normalize
import bacdive
from collections.abc import Iterable
import subprocess
from pathlib import Path
from fna2protcombined import run_all
import collections
from collections import defaultdict
import math
from typing import Tuple, Union
import json
from collections import Counter
import logging
from pathlib import Path

def fetch_ingredient_id_to_name(api_url="https://mediadive.dsmz.de/rest/ingredients"):
    """
    Fetch ingredient ID-to-name mapping from MediaDive API.

    Parameters
    ----------
    api_url : str, optional
        URL for the ingredients endpoint (default is MediaDive REST ingredients).

    Returns
    -------
    dict
        Mapping from ingredient ID (int) to ingredient name (str).
        Example: {12: 'Agar', 34: 'Glucose', ...}
        Returns empty dict if request fails.
    """
    try:
        with requests.Session() as s:
            resp = s.get(api_url)
            if resp.status_code == 200:
                data = resp.json().get('data')
                if data:
                    ingredients_df = json_normalize(data)[["id", "name"]]
                    return {
                        row["id"]: row["name"]
                        for _, row in ingredients_df.iterrows()
                    }
    except Exception as e:
        print(f"Error fetching ingredient mapping: {e}")

    return {}  # Fallback if request or parsing fails


def get_media_ingredient_matrix(
    medianums,
    convert_ingID=True,
    idmappings=None,
    binary=True,
    save_path=None
):
    """
    Fetch medium composition data for given medianums from MediaDive API,
    merge them into a DataFrame (medium IDs x ingredient names).

    Parameters
    ----------
    medianums : list of int
        List of medium IDs to query.
    convert_ingID : bool, optional (default=True)
        If True, convert ingredient IDs to names using idmappings.
    idmappings : dict or None, optional
        Ingredient ID to name mapping. If None, will fetch from MediaDive.
    binary : bool, optional (default=True)
        If True, convert concentrations to presence (1) / absence (0).
    save_path : str or Path, optional (default=None)
        If provided, saves the resulting DataFrame to this path as a CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with medium IDs as rows and ingredient names as columns.
    """
    merged_df = pd.DataFrame()
    with requests.Session() as s:
        for num in medianums:
            url = f'https://mediadive.dsmz.de/rest/medium-composition/{num}'
            resp = s.get(url)
            if resp.status_code == 200:
                data = resp.json().get('data')
                if data:
                    idf = json_normalize(data)[["id", "g_l"]]
                    idf = idf.rename(columns={"g_l": str(num)})
                    merged_df = (
                        idf if merged_df.empty
                        else pd.merge(merged_df, idf, on="id", how="outer")
                    )
            else:
                print(f"Error {resp.status_code} for medium ID {num}")

    if merged_df.empty:
        print("No valid data retrieved.")
        return pd.DataFrame()

    # Fill missing values with zeros
    merged_df = merged_df.fillna(0)

    # Map ingredient IDs to names if needed
    if convert_ingID:
        if idmappings is None:
            idmappings = fetch_ingredient_id_to_name()
        merged_df["name"] = merged_df["id"].map(idmappings)
        if merged_df["name"].isnull().any():
            print("Warning: Some ingredient IDs could not be mapped to names.")
            merged_df = merged_df.dropna(subset=["name"])
        merged_df = merged_df.drop(columns=["id"])
        merged_df = merged_df.set_index("name")
    else:
        merged_df = merged_df.set_index("id")

    # Transpose: now rows=medianum, cols=ingredient name or ID
    media_df = merged_df.T
    # Set index as int (medium ID)
    media_df.index = media_df.index.astype(int)

    # Binarize if requested
    if binary:
        media_df = (media_df > 0).astype(int)

    # Save if needed
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        media_df.to_csv(save_path)
        print(f"Data saved to {save_path}")

    return media_df

def num2bacid_dict(
    medianums,
    remove_none=True,
    take_first=False,
    save_path=None
):
    """
    For each medium number, fetch all valid BacDive strain IDs listed by
    the MediaDive “medium-strains” endpoint.

    A valid ID is numerical (int or str of digits), not None, and not NaN.

    Parameters
    ----------
    medianums : Iterable[int | str]
    remove_none : bool, optional
        If True, remove entries with None as bacdive_id in the output.
    take_first : bool, optional (default=False)
        If True, return only the first valid BacDive ID per medium.
        If False, return all valid IDs as a list per medium.
    save_path : str or Path, optional
        If provided, save the dictionary as a JSON file to this path.

    Returns
    -------
    dict
        {medium_number: list_of_valid_bacdive_ids or None}
        (or {medium_number: first_valid_bacdive_id or None} if take_first is True)
        (or without None values if remove_none=True)
    """
    bacdive_id_dict = {}

    for num in medianums:
        url = f"https://mediadive.dsmz.de/rest/medium-strains/{num}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                valid_ids = []
                for entry in data:
                    bid = entry.get("bacdive_id")
                    if bid is not None and not (isinstance(bid, float) and math.isnan(bid)):
                        try:
                            valid_ids.append(int(bid))
                        except (ValueError, TypeError):
                            continue
                if not valid_ids:
                    print(f"No valid 'bacdive_id' found for {num}")
                bacdive_id_dict[num] = valid_ids  # Always a list, even if empty
            else:
                print(f"Error {resp.status_code} for {num}")
                bacdive_id_dict[num] = None
        except requests.RequestException as e:
            print(f"Request failed for {num}: {e}")
            bacdive_id_dict[num] = None

    if take_first:
        bacdive_id_dict = {k: v[0] if isinstance(v, list) and v else None for k, v in bacdive_id_dict.items()}

    if remove_none:
        bacdive_id_dict = {k: v for k, v in bacdive_id_dict.items() if v is not None}

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(bacdive_id_dict, f, indent=2)
        print(f"Saved BacDive ID dictionary to {save_path}")

    return bacdive_id_dict

def map_taxids_1to1_fast(
    num2bac_dic,
    save_path=None,  # NEW OPTION
    min_instance=None,
    min_instance_frac=0.3,
    email='tanliangxun2000@gmail.com',
    password='pass1234',
):
    """
    For each medium, assign the first taxid that appears min_instance times among its BacDive IDs,
    then skip to the next medium. If min_instance is None, use min_instance_frac.
    If save_path is given, saves output dict as JSON.
    Returns: {medium_num: taxid}
    """
    client = bacdive.BacdiveClient(email, password)
    num2taxid = {}
    for num, bacids in num2bac_dic.items():
        if not bacids:
            continue  # skip mediums with no BacDive IDs
        taxids_count = Counter()
        # Decide threshold: explicit integer or fractional
        if min_instance is not None:
            threshold = min(int(min_instance), len(bacids))
        else:
            threshold = max(1, math.ceil(min_instance_frac * len(bacids)))
        assigned = False
        for bacid in bacids:
            print(f"mediaID = {num} BacDive ID = {bacid}")
            client.search(id=int(bacid))
            for record in client.retrieve(["NCBI tax id"]):
                strain_key = next(iter(record))
                ncbi_entries = record[strain_key]
                for entry in ncbi_entries:
                    ncbi_info = entry.get("NCBI tax id")
                    info_iter = ncbi_info if isinstance(ncbi_info, list) else [ncbi_info]
                    for sub in info_iter:
                        if not isinstance(sub, dict):
                            continue
                        if sub.get("Matching level") == "species":
                            tax_id = sub.get("NCBI tax id")
                            if tax_id:
                                print(f"  » species-level taxid {tax_id}")
                                taxids_count[tax_id] += 1
                                if taxids_count[tax_id] >= threshold:
                                    num2taxid[num] = tax_id
                                    assigned = True
                                    break
                    if assigned:
                        break
                if assigned:
                    break
            if assigned:
                break

    # Save dictionary if requested
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(num2taxid, f, indent=2)
        print(f"Saved taxid mapping dictionary to {save_path}")

    return num2taxid

def map_taxlevel_1to1_fast(
    num2bac_dic,
    save_path=None,
    taxlevel='family',
    min_instance=None,
    min_instance_frac=0.3,
    email='tanliangxun2000@gmail.com',
    password='pass1234',
):
    """
    For each medium, assign the first taxlevel name (e.g. family, genus) that appears threshold times among its BacDive IDs,
    then skip to the next medium. If min_instance is None, use min_instance_frac.

    Parameters
    ----------
    num2bac_dic : dict
        {medium_number: [bacdive_id1, bacdive_id2, ...]}
    save_path : str or Path, optional
        If provided, saves the dictionary as a JSON file.
    taxlevel : str, optional
        Taxonomic level to map (e.g. 'family', 'genus', 'order', 'species').
    min_instance : int or None, optional
        Absolute count threshold for assigning a taxlevel.
    min_instance_frac : float, optional
        If min_instance is None, minimum fraction of bacdive IDs that must match a taxlevel.
    email : str
        Your BacDive email login.
    password : str
        Your BacDive password.

    Returns
    -------
    dict
        {medium_number: taxlevel_name} mapping for assigned taxlevels.

    Usage
    -----
    >>> result = map_taxlevel_1to1_fast(
    ...     num2bac_dic,
    ...     save_path="num2family.json",
    ...     taxlevel="family",
    ...     min_instance=None,
    ...     min_instance_frac=0.3,
    ...     email='your@email.com',
    ...     password='yourpassword'
    ... )
    >>> print(result)
    {1: "Bacillaceae", 2: "Pseudomonadaceae", ...}
    # Output is also saved to 'num2family.json'
    """
    client = bacdive.BacdiveClient(email, password)
    num2taxlevel = {}
    for num, bacids in num2bac_dic.items():
        if not bacids:
            continue  # skip mediums with no BacDive IDs
        taxl_count = Counter()
        # Decide threshold: explicit integer or fractional
        if min_instance is not None:
            threshold = min(int(min_instance), len(bacids))
        else:
            threshold = max(1, math.ceil(min_instance_frac * len(bacids)))
        assigned = False
        for bacid in bacids:
            print(f"mediaID = {num} BacDive ID = {bacid}")
            client.search(id=int(bacid))
            for record in client.retrieve([taxlevel]):
                strain_key = next(iter(record))
                ncbi_entries = record[strain_key]
                for entry in ncbi_entries:
                    tname = entry.get(taxlevel)
                    # Only increment for non-NaN, non-None, non-empty values
                    if tname and not (isinstance(tname, float) and math.isnan(tname)):
                        print(f"Found {taxlevel}: {tname}")
                        taxl_count[tname] += 1
                        if taxl_count[tname] >= threshold:
                            num2taxlevel[num] = tname
                            assigned = True
                            break
                    if assigned:
                        break
                if assigned:
                    break
            if assigned:
                break

    # Save dictionary if requested
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(num2taxlevel, f, indent=2)
        print(f"Saved taxlevel mapping dictionary to {save_path}")

    return num2taxlevel

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



def run_protein_prediction(fasta_dir_path,
                           evalue="1e-3", 
                           nproc=4,
                           taxids=None):
    """
    Run protein prediction on all .fna files in the specified directory.
    If taxids is provided, only process files whose basename starts with any of those taxids (before the first '_').

    Parameters
    ----------
    fasta_dir_path : str or Path
        Path to the directory containing .fna files.
    evalue : str, optional
        E-value threshold for run_all (default "1e-3").
    nproc : int, optional
        Number of processors to use (default 4).
    taxids : set, list, or None
        Taxids to filter for in filenames (default None = process all files)

    Returns
    -------
    list of Path
        List of output files generated by run_all.
    """
    fasta_dir = Path(fasta_dir_path).expanduser().resolve()
    fna_files = sorted(fasta_dir.glob("*.fna"))

    if not fna_files:
        logging.warning(f"No .fna files found in {fasta_dir}")
        return []

    # If filtering by taxids, select only those files where the filename starts with taxid
    if taxids is not None:
        taxids = set(str(tid) for tid in taxids)
        filtered_files = []
        for f in fna_files:
            filename = f.name
            # Extract the leading number (taxid) before the first underscore
            leading = filename.split('_', 1)[0]
            if leading in taxids:
                filtered_files.append(f)
        fna_files = filtered_files
        if not fna_files:
            logging.warning(f"No .fna files match provided taxids in {fasta_dir}")
            return []

    logging.info(f"Starting prediction on {len(fna_files)} files with {nproc} processes.")
    results = run_all(fna_files, evalue, nproc=nproc)

    logging.info("Protein prediction output files:")
    for r in results:
        logging.info(r)

    return results


def run_prot2pfam(
    prodigal_dir_path, 
    jobs=8,
    evalue="1e-20",
    taxids=None
):
    """
    Run the prot2pfam.sh script on all .prot files in the specified directory.

    Parameters
    ----------
    prodigal_dir_path : str or Path
        Directory containing .prot files.
    jobs : int, optional
        Number of parallel jobs to run (default is 8).
    evalue : str, optional
        E-value threshold to use with hmmscan (default is '1e-20').
    taxids : set, list, or None
        Taxids to filter for in filenames (default None = process all files)

    Returns
    -------
    subprocess.CompletedProcess or None
        The result object from subprocess.run including stdout and stderr,
        or None if no .prot files are found.
    """
    # Hardcoded script path
    pfamscript = Path("/Users/liangxuntan/Code/fyp2025/scripts/prot2pfam.sh").expanduser().resolve()
    prodigal_dir = Path(prodigal_dir_path).expanduser().resolve()

    prodigal_files = sorted(prodigal_dir.glob("*.prot"))
    if not prodigal_files:
        print(f"No .prot files found in {prodigal_dir}")
        return None

    # Taxid filtering
    if taxids is not None:
        taxids = set(str(tid) for tid in taxids)
        filtered_files = []
        for f in prodigal_files:
            filename = f.name
            leading = filename.split('_', 1)[0]
            if leading in taxids:
                filtered_files.append(f)
        prodigal_files = filtered_files
        if not prodigal_files:
            print(f"No .prot files match provided taxids in {prodigal_dir}")
            return None

    file_args = [str(f) for f in prodigal_files]
    cmd = [str(pfamscript), "-j", str(jobs), "-e", str(evalue)] + file_args
    print("Running:", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running prot2pfam.sh (code {e.returncode})")
        return e


def pfam_acc_to_name(json_path="/Users/liangxuntan/Code/fyp2025/scripts/PfamDB.json"):
    """
    Load a PfamDB-style JSON file and return a mapping from accession to name.

    Parameters
    ----------
    json_path : str, optional
        Path to the PfamDB JSON file (default: './PfamDB.json').

    Returns
    -------
    dict
        Dictionary mapping Pfam accession (str) to Pfam name (str).
    """
    df = pd.read_json(json_path)
    df_expanded = pd.json_normalize(df["metadata"])
    return df_expanded.set_index("accession")["name"].to_dict()

def build_pfam_matrix2(
    hmmeroutputdir,  # directory containing all input .tbl files
    num2taxid,       # dictionary mapping medianum to taxid
    ethreshold='1e-20',   # E-value threshold for including hits
    save_path: str = None,  # optional path to save output DataFrame as CSV
    binary: bool = False,   # if True, output will be binary matrix (presence/absence)
    acc_to_name_dict=None,  # optional: dict mapping pfam accession ID to domain name
):
    # If no Pfam accession-to-name mapping is given, load default
    if acc_to_name_dict is None:
        acc_to_name_dict = pfam_acc_to_name()

    # Prepare path and get sorted list of all .tbl files in directory
    hmmer_dir = Path(hmmeroutputdir).expanduser().resolve()
    hmmer_files = sorted(hmmer_dir.glob("*.tbl"))

    count_dict = {}  # stores counts for each sample

    # Iterate over all .tbl files and count Pfam hits passing threshold
    for file in hmmer_files:
        pfam_counts = collections.Counter()
        with file.open() as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue  # skip comments and blank lines
                pfam_acc = line.split()[1].split(".")[0]  # get Pfam accession (remove version)
                pfam_eval = line.split()[4]  # get E-value
                if float(pfam_eval) <= float(ethreshold):  # filter by E-value threshold
                    pfam_counts[pfam_acc] += 1  # count occurrence

        taxid = file.stem.split('_')[0]  # extract taxid from filename (assumes format: <taxid>_pfam.tbl)
        count_dict[taxid] = pfam_counts  # store counts for this sample

    # Build DataFrame: rows=samples (taxids), columns=Pfam accessions, values=counts
    pfam_df = pd.DataFrame.from_dict(count_dict, orient="index").fillna(0).astype(int)

    # Rename columns from Pfam accession to Pfam domain name (if mapping provided)
    pfam_df = pfam_df.rename(columns=acc_to_name_dict)

    # Create a mapping from taxid to medianum for index renaming
    taxid2num = {str(v): str(k) for k, v in num2taxid.items()}
    pfam_df.index = pfam_df.index.map(str)  # ensure all indices are strings

    # Warn if there are taxids in pfam_df not found in taxid2num mapping
    missing = set(pfam_df.index) - set(taxid2num.keys())
    if missing:
        print(f"Warning: Some taxids are missing in taxid2num: {missing}")

    # Keep only rows with taxids that are in taxid2num and rename them to medianum
    pfam_df = pfam_df[pfam_df.index.isin(taxid2num.keys())].rename(index=taxid2num)

    # Drop duplicate indices after renaming (keeps first by default)
    def drop_duplicate_index(df, keep='first'):
        if df.index.duplicated().any():
            print("Warning: Duplicate index values detected after renaming!")
            print("Duplicate index values:", df.index[df.index.duplicated(keep=False)].unique().tolist())
        return df[~df.index.duplicated(keep=keep)]

    pfam_df = drop_duplicate_index(pfam_df, keep='first')

    # Attempt to convert index to integer type for consistency
    try:
        pfam_df.index = pfam_df.index.astype(int)
    except ValueError:
        pass  # If cannot convert (some non-integer index), ignore

    # Convert to binary presence/absence matrix if requested
    if binary:
        pfam_df = (pfam_df > 0).astype(int)

    # Save DataFrame to CSV file if save_path is given
    if save_path:
        save_path = Path(save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pfam_df.to_csv(save_path)

    # Return the resulting DataFrame
    return pfam_df


            
