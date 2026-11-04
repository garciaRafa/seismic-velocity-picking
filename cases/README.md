# Test case bank

Each file is one synthetic CDP gather. Two generators:

- `scripts/generate_cases.py` (`configs/test_cases.json`): 1D model,
  hyperbolic moveout with the RMS velocity, 30 Hz Ricker wavelet,
  white Gaussian noise relative to each trace maximum;
- `scripts/model_fd.py` (`configs/synthetic_fd_model.json`, `fd_*`
  cases): acoustic finite-difference modeling of a 2D model with
  Devito, with or without free surface (surface multiples).

Load a case with `src.seismic.cases.load_case(path)`.

| Case | Description | Traces | Offsets (m) | dt (ms) | Record (ms) | Noise | Water bottom (ms) | V_rms (m/s) |
|---|---|---|---|---|---|---|---|---|
| fd_layered_cdp1800_fs | FD acoustic, fd_layered model at x = 1.8 km, free surface (with multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2412 |
| fd_layered_cdp1800_nofs | FD acoustic, fd_layered model at x = 1.8 km, absorbing top (no surface multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2412 |
| fd_layered_cdp2200_fs | FD acoustic, fd_layered model at x = 2.2 km, free surface (with multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2403 |
| fd_layered_cdp2200_nofs | FD acoustic, fd_layered model at x = 2.2 km, absorbing top (no surface multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2403 |
| fd_layered_cdp2500_fs | FD acoustic, fd_layered model at x = 2.5 km, free surface (with multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2396 |
| fd_layered_cdp2500_nofs | FD acoustic, fd_layered model at x = 2.5 km, absorbing top (no surface multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2396 |
| fd_layered_cdp2900_fs | FD acoustic, fd_layered model at x = 2.9 km, free surface (with multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2385 |
| fd_layered_cdp2900_nofs | FD acoustic, fd_layered model at x = 2.9 km, absorbing top (no surface multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2385 |
| fd_layered_cdp3200_fs | FD acoustic, fd_layered model at x = 3.2 km, free surface (with multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2378 |
| fd_layered_cdp3200_nofs | FD acoustic, fd_layered model at x = 3.2 km, absorbing top (no surface multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2378 |
| fd_layered_cdp3400_fs | FD acoustic, fd_layered model at x = 3.4 km, free surface (with multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2344 |
| fd_layered_cdp3400_nofs | FD acoustic, fd_layered model at x = 3.4 km, absorbing top (no surface multiples) | 50 | 100–3000 | 2 | 2500 | 0% | 533 | 1500–2344 |
| layers3 | Three flat layers (1500, 2000, 2600 m/s): simple sanity case | 50 | 100–3000 | 2 | 3000 | 5% | 800 | 1500–2217 |
| marmousi2_cdp02000 | Marmousi2 column at x = 2.5 km | 50 | 100–3000 | 2 | 3000 | 5% | 602 | 1500–2311 |
| marmousi2_cdp04800 | Marmousi2 column at x = 6.0 km | 50 | 100–3000 | 2 | 3000 | 5% | 602 | 1500–2272 |
| marmousi2_cdp06800 | Marmousi2 column at x = 8.5 km (case used in the experiments) | 50 | 100–3000 | 2 | 3000 | 5% | 602 | 1500–2511 |
| marmousi2_cdp06800_noise20 | Same column as marmousi2_cdp06800 with 4x more noise (20%) | 50 | 100–3000 | 2 | 3000 | 20% | 602 | 1500–2511 |
| marmousi2_cdp09600 | Marmousi2 column at x = 12.0 km | 50 | 100–3000 | 2 | 3000 | 5% | 602 | 1500–2598 |
| marmousi2_cdp12000 | Marmousi2 column at x = 15.0 km | 50 | 100–3000 | 2 | 3000 | 5% | 602 | 1500–2628 |
