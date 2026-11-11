# Floor Checker

Does your world model beat copying the last frame?

A latent world model predicts the embedding of the next frame. Its score, the share of the
target's variation it explains, is often above 0.9, but copying the last frame's embedding often
scores above 0.9 too. This page measures how far above copying a model sits:

- **floor**: the score of copying the last frame the model was given; it equals 2ρ − 1, where ρ is
  the correlation between consecutive frames' embeddings;
- **lift** = (score − floor) / (1 − floor): 0 is no better than copying, 1 is perfect.

Open the page, then:

1. **Guess, then check** — released world models (DINO-WM, its retrain, JEPA-WM, LeWorldModel),
   trained and untrained.
2. **Why copying scores so high** — a slider for ρ.
3. **Bring your own** — drop three `.npy` arrays (predictions, targets, last frame seen); everything
   is computed in your browser and nothing is uploaded.

It runs offline: download this repository and open `index.html`.

## Files

| File | What |
|---|---|
| `index.html` | the page |
| `lift.js` | the arithmetic: `.npy` reading, variance explained, lift |
| `data.js` | the measured results, generated from the paper's results store |

## Paper

*Your world model may be worse than copying the last frame.* Michael Surazhsky, Yuval Elovici
(Cyber@BGU) and Ronen Brafman (Department of Computer Science), Ben-Gurion University of the Negev.
NeurIPS 2026 Workshop on Robot Learning with World Models.
