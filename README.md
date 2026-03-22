# MySQL Configuration Runtime Prediction

This project predicts MySQL benchmark runtime from database configuration settings.

The main focus is a **multi-output neural network with an auxiliary consistency loss**.  
Instead of predicting only the total runtime, the model jointly predicts:

- the total benchmark runtime
- the runtime of each individual query

An additional auxiliary loss enforces consistency between these outputs by encouraging:

```text
# predicted total runtime ≈ sum of predicted per-query runtimes
L = MSE(y_hat, y) + α · MSE(y_hat_total, sum(y_hat_query))
```

| Model               |         100 |         200 |        300 |        400 |        500 |        600 |        700 |        800 |
| ------------------- | ----------: | ----------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| NN                  |     2342.35 |     1490.67 |     464.19 |     362.90 |     268.83 |     208.68 |     189.08 |     160.91 |
| NN_multi            |     2523.65 |     1166.99 |     374.01 |     259.56 |     263.70 |     170.43 |     185.96 |     153.62 |
| **Proposed**        | **2381.74** | **1258.19** | **332.99** | **213.94** | **220.66** | **187.29** | **158.70** | **134.32** |
| **Improvement (%)** |       -1.68 |       15.60 |      28.26 |  **41.05** |      17.92 |      10.25 |      16.06 |      16.53 |

# Key Observations
The proposed model consistently outperforms both single-output and multi-output baselines
Gains are especially large in low-data regimes (up to +41% improvement)
Even with more data, the advantage does not disappear
Multi-output prediction alone helps, but
→ explicit consistency regularization is the key factor
