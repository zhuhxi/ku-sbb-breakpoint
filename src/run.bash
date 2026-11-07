# usage: bash run.bash [path/to/ku-applied-ai-challenge-2026-ku-sbb.zip]
set -e
cd "$(dirname "$0")"
if [ -n "$1" ]; then mkdir -p data && unzip -o -q "$1" -d data; fi
python train.py --data data --out outputs
