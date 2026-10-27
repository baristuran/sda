#!/usr/bin/env bash
# Compute physical statistics for a set of generated Rayleigh-Benard fields.
#
# Runs the two-step pipeline:
#   1. npz_to_h5.py  -- convert the generated-fields npz array into the HDF5
#                       ('theta') format that stats.py streams. Skipped when the
#                       input is already such an .h5 file (e.g. prior_fields.h5
#                       from sample_prior.py, or a previous npz_to_h5.py output).
#   2. stats.py      -- spectra / PDFs / Nusselt etc. of the generated fields
#                       vs. the ground-truth test set, written to <output_dir>.
#
# Usage:
#   ./compute_stats.sh <generated.npz|generated.h5> <output_dir> [options]
#
# Required:
#   <generated>       npz holding the generated fields (e.g. results/daps_enkf.npz),
#                     or an .h5 with a physical-units 'theta' dataset (used as is)
#   <output_dir>      directory for the .h5 and all stats outputs (created if needed)
#
# Options:
#   --key KEY         npz array to analyze (default: posterior; also 'prior'/'truth');
#                     ignored for .h5 input
#   --rayleigh RA     Rayleigh number (default: 1e7, matching this dataset)
#   --prandtl PR      Prandtl number  (default: 1.0)
#   --batch N         trajectories per streaming batch (default: 8)
#   --standardized    the npz arrays are in standardised units (destandardise them);
#                     not supported for .h5 input
#   --python PATH     python interpreter (default: the sda conda env)
#
# Example:
#   ./compute_stats.sh results/daps_enkf.npz results/stats_enkf --key posterior
#   ./compute_stats.sh results/prior_fields.h5 results/stats_prior

set -euo pipefail

# -- defaults ---------------------------------------------------------------
KEY="posterior"
RAYLEIGH="1e7"
PRANDTL="1.0"
BATCH="8"
STD_FLAG=""
PYTHON="/home/baris-turan/miniconda3/envs/sda/bin/python"

# -- parse args -------------------------------------------------------------
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --key)          KEY="$2";      shift 2 ;;
        --rayleigh)     RAYLEIGH="$2"; shift 2 ;;
        --prandtl)      PRANDTL="$2";  shift 2 ;;
        --batch)        BATCH="$2";    shift 2 ;;
        --standardized) STD_FLAG="--standardized-input"; shift ;;
        --python)       PYTHON="$2";   shift 2 ;;
        -h|--help)      sed -n '2,32p' "$0"; exit 0 ;;
        -*)             echo "unknown option: $1" >&2; exit 1 ;;
        *)              POSITIONAL+=("$1"); shift ;;
    esac
done

if [[ ${#POSITIONAL[@]} -ne 2 ]]; then
    echo "usage: $0 <generated.npz|generated.h5> <output_dir> [options]  (see --help)" >&2
    exit 1
fi

INPUT="${POSITIONAL[0]}"
OUTDIR="${POSITIONAL[1]}"

# The python scripts import the local `utils` module, so run them from the
# directory this script lives in (the rayleigh-benard experiment dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve paths to absolute BEFORE changing directory (they may be relative to
# the caller's cwd). stats.py joins --out-path onto its own PATH, so an absolute
# output dir keeps the outputs where the user asked.
if [[ ! -f "$INPUT" ]]; then
    echo "error: input file not found: $INPUT" >&2
    exit 1
fi
INPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

cd "$SCRIPT_DIR"

case "$INPUT" in
    *.h5|*.hdf5)
        # Already in stats.py's format: no conversion. stats.py reads it as
        # physical units, so it cannot be destandardised here.
        if [[ -n "$STD_FLAG" ]]; then
            echo "error: --standardized is not supported for .h5 input" >&2
            exit 1
        fi
        H5="$INPUT"
        echo "==> [1/2] npz_to_h5.py: skipped, input is already HDF5 ($H5)"
        ;;
    *.npz)
        H5="$OUTDIR/$(basename "${INPUT%.npz}").h5"
        echo "==> [1/2] npz_to_h5.py: '$KEY' from $INPUT -> $H5"
        "$PYTHON" npz_to_h5.py "$INPUT" --key "$KEY" --out "$H5" --batch "$BATCH" $STD_FLAG
        ;;
    *)
        echo "error: input must be .npz or .h5: $INPUT" >&2
        exit 1
        ;;
esac

echo "==> [2/2] stats.py: $H5 -> $OUTDIR  (Ra=$RAYLEIGH, Pr=$PRANDTL)"
"$PYTHON" stats.py --prior-file "$H5" --out-path "$OUTDIR" \
    --rayleigh "$RAYLEIGH" --prandtl "$PRANDTL" --batch "$BATCH"

echo "==> done. statistics written to $OUTDIR"
