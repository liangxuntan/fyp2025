#!/usr/bin/env bash
# prot2pfam_cluster.sh – run hmmscan on many FASTA files in parallel (cluster-optimized, all arguments required)
set -euo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 -c threads -j jobs -e evalue prot1.faa [prot2.faa …]
  -c  Threads given to each hmmscan run      (REQUIRED, integer >0)
  -j  How many hmmscan runs in parallel      (REQUIRED, integer >0)
  -e  E-value threshold for hmmscan          (REQUIRED, e.g. 1e-20)
  All additional arguments are input .prot files (absolute or relative path)
EOF
  exit 1
}

# ---------- parse options ----------
threads=0
jobs=0
evalue=""

while getopts ":c:j:e:h" opt; do
  case $opt in
    c) threads="$OPTARG" ;;
    j) jobs="$OPTARG" ;;
    e) evalue="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done
shift $((OPTIND-1))

[[ "$threads" =~ ^[0-9]+$ && "$threads" -gt 0 ]] || { echo "Missing or invalid -c value"; usage; }
[[ "$jobs" =~ ^[0-9]+$ && "$jobs" -gt 0 ]]       || { echo "Missing or invalid -j value"; usage; }
[[ -n "$evalue" ]]                               || { echo "Missing -e value"; usage; }
[[ "$#" -gt 0 ]]                                 || { echo "No input .prot files given!"; usage; }

# ---------- dirs ----------
runlog_dir="/home/liangxun/logs"
dest_dir="/home/liangxun/hmm_output"
model_dir="/home/liangxun/hmm_database"

mkdir -p "$runlog_dir" "$dest_dir"
timestamp=$(date +%Y%m%d_%H%M%S)
logfile="${runlog_dir}/prot2pfam_${timestamp}.log"

inputfiles=( "$@" )

tmpdir="/tmp/liangxun_${RANDOM}_$$"
hmmfiles_tmp="${tmpdir}/model"
output_tmp="${tmpdir}/output"
input_tmp="${tmpdir}/input"
mkdir -p "$hmmfiles_tmp" "$output_tmp" "$input_tmp"

cp "$model_dir"/Pfam-A.hmm* "$hmmfiles_tmp"
for f in "${inputfiles[@]}"; do cp "$f" "$input_tmp"; done

for suf in .h3f .h3i .h3m .h3p; do
    [[ -f "$hmmfiles_tmp/Pfam-A.hmm$suf" ]] || { echo "Error: Missing $hmmfiles_tmp/Pfam-A.hmm$suf" | tee -a "$logfile"; exit 1; }
done

cd "$hmmfiles_tmp"

command -v hmmscan >/dev/null 2>&1 || { echo "hmmscan not found in PATH!"; exit 1; }

echo "▶ Starting parallel hmmscan ( $(date) ) on ${#inputfiles[@]} file(s)" | tee -a "$logfile"

export output_tmp logfile threads evalue hmmfiles_tmp

par_run() {
  protfile="$1"
  [[ -f "$protfile" ]] || { echo "✖ File not found: $protfile" | tee -a "$logfile"; exit 0; }
  basename=$(basename "$protfile")
  idnum="${basename%%_*}"
  tblout="${output_tmp}/${idnum}_pfam.tbl"

  echo "→ Processing $basename ➜ $tblout" | tee -a "$logfile"
  start=$(date +%s)

  hmmscan --cpu "$threads" -E "$evalue" \
        --acc --tblout "$tblout" "$hmmfiles_tmp/Pfam-A.hmm" "$protfile" \
        > /dev/null 2>&1

  secs=$(( $(date +%s) - start ))
  echo "✓ Finished $basename in ${secs}s" | tee -a "$logfile"
}
export -f par_run

echo "Running $jobs job(s) × $threads thread(s) each" | tee -a "$logfile"

parallel --jobs "$jobs" par_run ::: "$input_tmp"/*

cd "$dest_dir"

cp "$output_tmp"/*.tbl "$dest_dir"
rm -rf "$tmpdir"

echo "✔ All done ( $(date) ). Full log: $logfile"
ls -lh "$dest_dir"/*.tbl | tee -a "$logfile"
echo "Done at $(date)" | tee -a "$logfile"

