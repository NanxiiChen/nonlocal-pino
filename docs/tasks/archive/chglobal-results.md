| Global control | Common phase pooled / terminal | Native mu pooled / positive-time | Phi range | Trainer seconds |
|---|---:|---:|---:|---:|
| Differential/spatial inverse | 51.01% / 61.25% | 71.7% / 122.4% | -0.99..0.99 | 255.5 |
| Differential/space-time inverse | 97.97% / 99.85% | 62.3% / 106.5% | -0.98..0.98 | 218.7 |
| Integral, joint MLP | 84.26% / 73.40% | 159.9% / 273.2% | -1.29..1.73 | 286.7 |
| Neumann inverse, joint MLP | 55.84% / 64.96% | 91.5% / 156.3% | -1.09..1.08 | 276.2 |
| Integral, global basis | 35.61% / 63.40% | 678.3% / 1158.8% | -3.94..3.49 | 200.5 |
| Global basis + energy | 89.13% / 90.09% | 69.2% / 118.2% | -1.05..1.03 | 282.6 |
| Global basis, R/t | 21.54% / 43.18% | 258.1% / 440.9% | -1.76..1.67 | 197.8 |
| Global basis, differential inverse | 57.84% / 65.55% | 90.3% / 154.3% | -1.02..1.01 | 251.9 |
| Global basis, causal differential inverse | 56.77% / 64.46% | 105.6% / 180.4% | -1.08..1.01 | 252.0 |
| Global basis, differential inverse, width128 | 65.63% / 73.93% | 59.4% / 101.5% | -1.07..1.03 | 308.0 |

Trainer times exclude initial preprocessing. Newton iterations require Hessian products and are not equivalent to L-BFGS iterations. All outcomes retained; none use reference training labels.
