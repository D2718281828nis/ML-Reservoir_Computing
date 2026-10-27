# Reservoir Computing: a hands-on course

Feed-forward neural networks against **reservoir computing** (echo state networks, liquid state machines, next-generation RC) for time-series forecasting:
an interactive explainer that trains every model in your browser, plus notebooks that build each model step by step.

**Interactive page:** https://d2718281828nis.github.io/ML-Reservoir_Computing/ · на русском: https://d2718281828nis.github.io/ML-Reservoir_Computing/ru/

## The idea in one paragraph

To forecast a time series with a neural network you can give a **feed-forward network** a window of the last values and train every weight with
backpropagation. Or you can let the series drive a large **random recurrent network that is never trained** (the *reservoir*) and fit only a linear
*readout* of its state, in a single ridge-regression solve. "FFN vs RC" is really two separate choices:

|                                 | every weight trained (backprop) | only the readout trained (one solve) |
|---------------------------------|---------------------------------|--------------------------------------|
| **memory = a window**           | FFN / MLP                       | ELM (random features), NG-RC (polynomial features) |
| **memory = a recurrent state**  | RNN, LSTM, GRU                  | ESN, LSM (reservoir computing)       |

The explainer races the corners against each other to test each choice on its own.

## What the explainer measures

All numbers are produced by the page itself (and again, in numpy, by notebook 00):

| Experiment | Result |
|---|---|
| Mackey–Glass, 20 steps ahead, window 20 | ESN ≈ 0.0004 and ELM ≈ 0.0005 NRMSE after one ridge solve (~0.1–0.2 s); FFN ≈ 0.009 after 150 epochs (~2 s) |
| Same, window 3 | every window model collapses (FFN ≈ 0.18, ELM ≈ 0.13); the ESN, which has no window, is unaffected |
| Echo state property | "ρ < 1" is a rule of thumb: ρ = 1.2 still forgets its initial state; ρ = 2 does too if the input is strong |
| Memory capacity (white noise, N = 100) | about 30 delays; it fades instead of stopping like a window |
| Daily electricity demand, Spain 2018 | all learned models beat "same day last week" (NRMSE 0.34–0.38 vs 0.70 for tomorrow); FFN slightly ahead |
| ECG, 4–100 ms ahead | ESN best at 4 ms (0.08 vs 0.20); further out ESN and FFN are close |
| Closed loop on Mackey–Glass | ESN stays valid ≈ 540–860 steps, FFN ≈ 65–150 |
| Lorenz-63, 400 training points | NG-RC ≈ 4 Lyapunov times with 84 weights and 1 warm-up point; ESN ≈ 3.5 (it catches up with ≥ 1,000 points) |

## Learning path

| # | Notebook | Topic | Colab |
|---|---|---|---|
| 00 | [FFN vs reservoir computing](notebooks/00_ffn_vs_reservoir_computing.ipynb) | **start here** — every experiment of the explainer in numpy: FFN, ESN, echo state property, memory, ELM, forecasting with baselines, closed loop, NG-RC | [open](https://colab.research.google.com/github/D2718281828nis/ML-Reservoir_Computing/blob/master/notebooks/00_ffn_vs_reservoir_computing.ipynb) |
| 01 | [ESN, LSM and RNN on a sine generator](notebooks/01_esn_lsm_rnn_sine_generator.ipynb) | a first reservoir; how to read an MSE against the target's variance | [open](https://colab.research.google.com/github/D2718281828nis/ML-Reservoir_Computing/blob/master/notebooks/01_esn_lsm_rnn_sine_generator.ipynb) |
| 02 | [ESN vs LSTM](notebooks/02_esn_vs_lstm.ipynb) | accuracy against training time | [open](https://colab.research.google.com/github/D2718281828nis/ML-Reservoir_Computing/blob/master/notebooks/02_esn_vs_lstm.ipynb) |
| 03 | [NG-RC for a forced oscillator](notebooks/03_ngrc_forced_oscillator.ipynb) | delay taps, polynomial features, fixed points | [open](https://colab.research.google.com/github/D2718281828nis/ML-Reservoir_Computing/blob/master/notebooks/03_ngrc_forced_oscillator.ipynb) |
| 04 | [NG-RC on Lorenz and the double scroll](notebooks/04_ngrc_lorenz_double_scroll.ipynb) | Gauthier et al. (2021) reproduced; inference task | [open](https://colab.research.google.com/github/D2718281828nis/ML-Reservoir_Computing/blob/master/notebooks/04_ngrc_lorenz_double_scroll.ipynb) |
| 05 | [ESN on Lorenz with rescompy](notebooks/05_esn_lorenz_rescompy.ipynb) | a reservoir library on a chaotic system | [open](https://colab.research.google.com/github/D2718281828nis/ML-Reservoir_Computing/blob/master/notebooks/05_esn_lorenz_rescompy.ipynb) |
| 06 | [Liquid state machine](notebooks/06_lsm_spiking_network.ipynb) | a spiking reservoir in PyTorch | [open](https://colab.research.google.com/github/D2718281828nis/ML-Reservoir_Computing/blob/master/notebooks/06_lsm_spiking_network.ipynb) |
| 07 | [ECG forecast with a reservoir library](notebooks/07_ecg_forecast_rc_library.ipynb) | `reservoir_computing`'s forecaster on real data | [open](https://colab.research.google.com/github/D2718281828nis/ML-Reservoir_Computing/blob/master/notebooks/07_ecg_forecast_rc_library.ipynb) |
| 08 | [ECG forecast with a warmed-up ESN](notebooks/08_ecg_forecast_esn_warmup.ipynb) | window-to-window prediction; why the washout matters | [open](https://colab.research.google.com/github/D2718281828nis/ML-Reservoir_Computing/blob/master/notebooks/08_ecg_forecast_esn_warmup.ipynb) |
| 09 | [Basins of attraction](notebooks/09_basins_of_attraction_duffing.ipynb) | dynamical-systems background: fixed points, stability, basins | [open](https://colab.research.google.com/github/D2718281828nis/ML-Reservoir_Computing/blob/master/notebooks/09_basins_of_attraction_duffing.ipynb) |

Each notebook starts with a short note on what it shows, what it needs, and how to read its results.
[`notebooks/archive/`](notebooks/archive/) keeps a duplicate of 03 and the Bayesian-readout notebook, which needs packages that are no longer in this repository.

## Repository layout

```
explainer/            the interactive page (GitHub Pages): index.html (EN), ru/index.html (RU), js/, css/, assets/, tools/
notebooks/            the learning path above; archive/ for notebooks kept for history
dataset/              ECG (1 kHz), Spanish energy market (daily), ACEA electricity load (10-minute, MAT), data_loaders.py
.github/workflows/    pages.yml publishes explainer/ to GitHub Pages
```

## Run it

```bash
pip install -r requirements.txt                 # numpy, scipy, pandas, scikit-learn, matplotlib, jupyter
jupyter lab notebooks/00_ffn_vs_reservoir_computing.ipynb

cd explainer && python3 -m http.server 8000     # the page, at http://localhost:8000
node explainer/tools/check.mjs                  # checks the page's math against numpy (Node ≥ 20)
```

Some notebooks need extra packages, named in their first cell (torch, tensorflow, plotly, rescompy, reservoir_computing).

## Key terms

- **Reservoir** — a fixed, random, high-dimensional dynamical system driven by the input; its state is a nonlinear, fading summary of the input history.
- **Readout** — the only trained part: a linear map from the reservoir state to the output, fitted by ridge regression.
- **Echo state property** — the reservoir forgets its initial state, so its state depends only on the input history. Shaped by the spectral radius ρ, the leak rate α and the input gain.
- **NG-RC / NVAR** — next-generation RC replaces the reservoir with delay taps `[x(t), x(t−τ), …]` and their polynomial products; mathematically, a nonlinear vector autoregression.
- **Lyapunov time** — the time over which nearby trajectories of a chaotic system separate by a factor e; the natural unit for "how far ahead can we predict".

## References

- H. Jaeger, *The "echo state" approach to analysing and training recurrent neural networks*, GMD Report 148 (2001).
- W. Maass, T. Natschläger, H. Markram, *Real-time computing without stable states*, Neural Computation 14 (2002).
- H. Jaeger, H. Haas, *Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless communication*, Science 304 (2004).
- M. Lukoševičius, *A practical guide to applying echo state networks* (2012).
- D. J. Gauthier, E. Bollt, A. Griffith, W. A. S. Barbosa, *Next generation reservoir computing*, Nature Communications 12 (2021).
- One of the best reservoir-computing repositories: [eloMG/Probabilistic-load-forecasting-with-Reservoir-Computing](https://github.com/eloMG/Probabilistic-load-forecasting-with-Reservoir-Computing).

## Citation

The Bayesian-readout notebook and `dataset/data_loaders.py` come from the work below. If you use them in your research, please cite:

```
@ARTICLE{10360823,
  author={Guerra, Michele and Scardapane, Simone and Bianchi, Filippo Maria},
  journal={IEEE Access},
  title={Probabilistic Load Forecasting With Reservoir Computing},
  year={2023},
  volume={11},
  number={},
  pages={145989-146002},
  doi={10.1109/ACCESS.2023.3343467}}
```

## License

MIT, see [LICENSE](LICENSE).
