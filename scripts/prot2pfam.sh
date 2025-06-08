#!/usr/bin/env bash
# prot2pfam_parallel.sh – run hmmscan on many FASTA files in parallel
set -euo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 [-c threads] [-j jobs] [-e evalue] prot1.faa [prot2.faa …]

  -c  Threads given to each hmmscan run      (default: 1)
  -j  How many hmmscan runs in parallel      (default: auto = floor(total_cores / -c))
  -e  E-value threshold for hmmscan          (default: 1e-20)
EOF
  exit 1
}

# ---------- parse options ----------
threads=1        # --cpu for hmmscan
jobs=0           # 0 → decide later
evalue="1e-20"   # default e-value

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

[[ "$#" -gt 0 ]] || usage
[[ "$threads" =~ ^[0-9]+$ && "$threads" -gt 0 ]] || { echo "Invalid -c value"; exit 1; }
if [[ "$jobs" -ne 0 && ! "$jobs" =~ ^[0-9]+$ ]]; then
  echo "Invalid -j value"; exit 1
fi
[[ "$evalue" =~ ^[0-9.eE-]+$ ]] || { echo "Invalid -e value"; exit 1; }

# ---------- dirs ----------
scripts_dir="/Users/liangxuntan/Code/fyp2025/scripts"
runlog_dir="/Users/liangxuntan/Code/fyp2025/data/logs"
dest_dir="/Users/liangxuntan/Code/fyp2025/data/hmmer_output"
model_dir="/Users/liangxuntan/Code/fyp2025/data/hmm_scoringmodel"

mkdir -p "$runlog_dir" "$dest_dir"
timestamp=$(date +%Y%m%d_%H%M%S)
logfile="${runlog_dir}/prot2pfam_${timestamp}.log"

# ---------- env ----------
command -v hmmscan >/dev/null 2>&1 || source ~/anaconda3/bin/activate mne

cd "$model_dir"
echo "▶ Starting parallel hmmscan ( $(date) ) on $# file(s)" | tee -a "$logfile"

export dest_dir logfile threads evalue                # to subshells

par_run() {
  protfile="$1"
  [[ -f "$protfile" ]] || { echo "✖ File not found: $protfile" | tee -a "$logfile"; exit 0; }

  basename=$(basename "$protfile")
  idnum="${basename%%_*}"
  tblout="${dest_dir}/${idnum}_pfam.tbl"

  echo "→ Processing $basename ➜ $tblout" | tee -a "$logfile"
  start=$(date +%s)

  hmmscan --cpu "$threads" -E "$evalue" --domE "$evalue" \
        --acc --tblout "$tblout" Pfam-A.hmm "$protfile" \
        > /dev/null 2>&1

  secs=$(( $(date +%s) - start ))
  echo "✓ Finished $basename in ${secs}s" | tee -a "$logfile"
}
export -f par_run

# ---------- decide job count ----------
if [[ "$jobs" -eq 0 ]]; then
  total_cores=$(getconf _NPROCESSORS_ONLN)
  jobs=$(( total_cores / threads - 1 ))           # leave one core free
  (( jobs < 1 )) && jobs=1
fi
echo "Running $jobs job(s) × $threads thread(s) each" | tee -a "$logfile"

parallel --jobs "$jobs" par_run ::: "$@"

echo "✔ All done ( $(date) ). Full log: $logfile"