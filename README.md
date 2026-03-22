# MySQL Configuration Runtime Prediction

This project predicts MySQL benchmark runtime from database configuration settings.

The main focus is a **multi-output neural network with an auxiliary consistency loss**.  
Instead of predicting only the total runtime, the model jointly predicts:

- the total benchmark runtime
- the runtime of each individual query

An additional auxiliary loss enforces consistency between these outputs by encouraging:

```text
predicted total runtime ≈ sum of predicted per-query runtimes
L = MSE(y_hat, y) + α · MSE(y_hat_total, sum(y_hat_query))
```
