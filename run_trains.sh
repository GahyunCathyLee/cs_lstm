#!/usr/bin/env bash
set +e

# ── legacy experiments ──────────────────────────────────────────────────────
python3 train.py --config configs/baseline.yaml
python3 train.py --config configs/importance.yaml

# ── v4_lit_g2_slot3  |  c0: ego+nb(6D)  |  5 trials ────────────────────────
python3 train.py --config configs/v4lit_c0_t1.yaml
python3 train.py --config configs/v4lit_c0_t2.yaml
python3 train.py --config configs/v4lit_c0_t3.yaml
python3 train.py --config configs/v4lit_c0_t4.yaml
python3 train.py --config configs/v4lit_c0_t5.yaml

# ── v4_lit_g2_slot3  |  c2: ego+nb(6D)+I_y  |  5 trials ───────────────────
python3 train.py --config configs/v4lit_c2_t1.yaml
python3 train.py --config configs/v4lit_c2_t2.yaml
python3 train.py --config configs/v4lit_c2_t3.yaml
python3 train.py --config configs/v4lit_c2_t4.yaml
python3 train.py --config configs/v4lit_c2_t5.yaml
