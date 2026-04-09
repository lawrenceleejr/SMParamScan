# SMParamScan

Scan SM quark Yukawa couplings and compute the entanglement entropy of Higgs boson decays, following [arXiv:2511.17321](https://arxiv.org/abs/2511.17321).

Uses [HDECAY](https://www.fuw.edu.pl/~kalino/fortran/hdecay.f) (Djouadi, Kalinowski, Spira) for Higgs branching ratio calculations, run inside a Docker container that auto-rebuilds when configuration changes.

## Setup

```bash
pip install -r requirements.txt
```

Docker is required for the default mode. If Docker is unavailable, the tool automatically falls back to compiling HDECAY locally (requires `gfortran` and `wget`).

## Usage

Run a full scan over all 6 quark Yukawa couplings:

```bash
python run_scan.py
```

Options:

```
--quarks b,c,t        Select quarks to scan (default: u,d,s,c,b,t)
--npoints 100         Number of scan points per quark (default: 100)
--kappa-min 0.01      Minimum coupling modifier (default: 0.01)
--kappa-max 100       Maximum coupling modifier (default: 100.0)
--output-dir plots    Output directory for plots (default: plots/)
```

Example — scan only bottom and top with 50 points:

```bash
python run_scan.py --quarks b,t --npoints 50
```

## Physics

The entanglement entropy is the linear entropy (Tsallis-2) of the reduced density matrix for Higgs decay products:

```
EE = 1 - sum_i (P_i / N_c^i) * BR_i^2
```

where `P_i` is the spin factor and `N_c^i` the color multiplicity for each decay channel. Quark Yukawa couplings are varied by scaling the corresponding quark mass input to HDECAY (`kappa_q = m_q / m_q^SM`).

## Output

Individual PDF plots for each quark (`ee_vs_kappa_b.pdf`, etc.) plus a combined 6-panel summary (`ee_vs_kappa_all_quarks.pdf`).
