#!/usr/bin/env bash
# Fetch the WASSA 2024 Track 3 (EMP) / Track 4 (PER) data from the Google Drive links
# on https://codalab.lisn.upsaclay.fr/competitions/18810 (Datasets tab) into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw && cd data/raw

gdown -q 1A5Fy1u60n1aVyAD38UxH2bu5mMMShg0S -O track3_train.csv
gdown -q 1dXjYrT9vRSQN6HKSAJTE0T4Eq2gdJVCu -O track3_dev.csv
gdown -q 1cvPbT4BCD55ec8jz8YkXkX1PmcTeqi0e -O track4_train.csv
gdown -q 17JtV0fitOW0NU0k6Qy7Kjn-4CZ3F-Rkp -O track4_dev.csv
# test inputs, test gold labels (test_data_with_labels/) and the official evaluation.py
gdown -q --folder 1cUODSmUauENB1IX69wUa7IjYHcbgBT78 -O test
# news articles: ec_article_info_train.csv / ec_article_info_dev.csv
gdown -q --folder 1FGmmmECAMvZDrdSOzMdlW8FRxSh8n7Bv -O articles

find . -type f | sort
