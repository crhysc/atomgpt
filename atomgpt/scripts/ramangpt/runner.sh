#!/bin/bash

python make_raman_alpaca.py \
  --raman-json ramandb.json \
  --test-ratio 0.1 \
  --train-out alpaca_prop_train.json \
  --test-out alpaca_prop_test.json \
  --freq-decimals 9 \
  --activity-decimals 6

