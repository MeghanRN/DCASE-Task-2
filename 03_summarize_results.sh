#!/usr/bin/env bash
# --------------------------------------------------------------------------
# 03_summarize_results.sh
#
# Summarize per-section metrics (AUC, pAUC, precision, recall, F1) for
# DCASE-Task-2 baseline runs.  Results are gathered from
#   results/<dev|eval>_data/baseline_MSE/<Machine>/*.csv
#   results/<dev|eval>_data/baseline_MAHALA/<Machine>/*.csv
# and merged into a single CSV in
#   results/<dev|eval>_data/baseline/summarize/<DATASET>/summary.csv
#
# Usage:
#   ./03_summarize_results.sh  <DATASET>  <-d|--dev|-e|--eval>
# --------------------------------------------------------------------------
set -euo pipefail

### ------------- Parse Args -------------
dataset=${1:-}
mode=${2:-}

if [[ -z "$dataset" || -z "$mode" ]]; then
  echo "Usage: $0 <DATASET> <-d|--dev|-e|--eval>"
  exit 1
fi

case "$mode" in
  -d|--dev)  split="dev_data"  ;;
  -e|--eval) split="eval_data" ;;
  *) echo "Mode must be -d/--dev or -e/--eval"; exit 1 ;;
esac

### ------------- Config -------------
systems=(
  results/${split}/baseline_MSE
  results/${split}/baseline_MAHALA   # remove if you only ran MSE mode
)

export_dir=results/${split}/baseline/summarize/${dataset}
mkdir -p "$export_dir"
summary_csv=${export_dir}/summary.csv

echo "Dataset   : $dataset"
echo "Split     : $split"
echo "Systems   : ${systems[*]}"
echo "Export CSV: $summary_csv"
echo

### ------------- Helper: extract metrics from a csv row -------------
# Expected per-section result.csv row format (no header):
# section,auc_src,auc_tgt,pauc,precision_src,precision_tgt,recall_src,recall_tgt,f1_src,f1_tgt
get_val () {  # $1=id  $2=col-idx
  awk -F',' -v id="$1" -v col="$2" '$1==id {print $col}' "$3"
}

### ------------- Build summary -------------
echo "machine,auc_src,auc_tgt,pauc,precision_src,precision_tgt,recall_src,recall_tgt,f1_src,f1_tgt" > "$summary_csv"

for sys_dir in "${systems[@]}"; do
  for mach_dir in "$sys_dir"/*/; do
    machine=$(basename "$mach_dir")
    res_csv="${mach_dir}/result.csv"        # produced by your test script
    [[ -f "$res_csv" ]] || continue

    auc_src=$(get_val 00 2 "$res_csv")
    auc_tgt=$(get_val 00 3 "$res_csv")
    pauc=$(get_val 00 4 "$res_csv")
    prec_src=$(get_val 00 6 "$res_csv")
    prec_tgt=$(get_val 00 7 "$res_csv")
    rec_src=$(get_val 00 8 "$res_csv")
    rec_tgt=$(get_val 00 9 "$res_csv")
    f1_src=$(get_val 00 10 "$res_csv")
    f1_tgt=$(get_val 00 11 "$res_csv")

    echo "${machine},${auc_src},${auc_tgt},${pauc},${prec_src},${prec_tgt},${rec_src},${rec_tgt},${f1_src},${f1_tgt}" \
         >> "$summary_csv"
  done
done

echo "Summary written to $summary_csv"