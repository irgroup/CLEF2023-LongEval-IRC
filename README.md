# LongEval
## Dataset `WT` Stats
|            | len stream |  len token |   len stop | len stop unique |   len stem | len stem unique |
|-----------:|-----------:|-----------:|----------------:|-----------:|----------------:|------------|
|      count | 1570734.00 | 1570734.00 |      1570734.00 | 1570734.00 |      1570734.00 | 1570734.00 |
|       mean |    4846.15 |     794.11 |          559.47 |     370.46 |          559.47 |     344.47 |
|        std |    3228.08 |     532.73 |          386.06 |     239.44 |          386.06 |     222.99 |
|        min |       3.00 |       0.00 |            0.00 |       0.00 |            0.00 |       0.00 |
|        25% |    1998.00 |     327.00 |          230.00 |     171.00 |          230.00 |     162.00 |
|        50% |    4224.00 |     693.00 |          480.00 |     328.00 |          480.00 |     305.00 |
|        75% |    8169.00 |    1311.00 |          878.00 |     556.00 |          878.00 |     509.00 |
|        max |   26645.00 |    7065.00 |         6981.00 |    1480.00 |         6981.00 |    1472.00 |

## Index 
```python -m src.create_index --index WT```


## Create Passages from a dataset:
```python -m src.create_passages --index WT```

# Systems

| System                                | Run name                 | Command                                          | Description                  |
|---------------------------------------|--------------------------|--------------------------------------------------|------------------------------|
| BM25                                  | `IRCologne-BM25.WT`      | `python -m systems.BM25 --index WT`              | Baseline                     |
| DPH                                   | `IRCologne-DPH.WT`       | `python -m systems.BM25 --index WT`              |                              |
| PL2                                   | `IRCologne-PL2.WT`       | `python -m systems.BM25 --index WT`              |                              |
| TF_IDF                                | `IRCologne-TF_IDF.WT`    | `python -m systems.BM25 --index WT`              |                              |
| XSqrA_M                               | `IRCologne-XSqrA_M.WT`   | `python -m systems.BM25 --index WT`              |                              |
|                                       |                          |                                                  |                              |
| BM25 > axio                           | `IRCologne-BM25_axio.WT` | `python -m systems.BM25 --index WT`              |                              |
| BM25 > Bo1                            | `IRCologne-BM25_Bo1.WT`  | `python -m systems.BM25 --index WT`              |                              |
| BM25 > RM3                            | `IRCologne-BM25_RM3.WT`  | `python -m systems.BM25 --index WT`              |                              |
|                                       |                          |                                                  |                              |
| RRF(BM25, BM25>Bo2, XSqrA_M, PL2)     | `IRCologne-RRF(BBXP).WT` | `python -m systems.BM25 --index WT`              |                              |
| RRF(BM25>RM3, BM25>Bo2, XSqrA_M, PL2) | `IRCologne-RRF(BRXP).WT` | `python -m systems.BM25 --index WT`              |                              |
| RRF(BM25, XSqrA_M, PL2)               | `IRCologne-RRF(BXP).WT`  | `python -m systems.BM25 --index WT`              |                              |
|                                       |                          |                                                  |                              |
| BM25 > monoT5                         | `IRCologne_monoT5.WT`    | `python -m systems.monoT5 --index WT`            | Baseline                     |
| BM25 > monoT5_WT                      |                          | `python -m systems.monoT5_WT --index WT`         | pre trained on WT            |
| BM25 > monoT5_WT                      |                          | `python -m systems.monoT5_WT --index WT --train` | pre trained on WT test slice |
|                                       |                          |                                                  |                              |
||||
|BM25 > LambdaMART || Trained on LETOR features |
|BM25 > LambdaMART || Trained on LETOR features. Test slice only! |
|BM25 > LambdaMART_WT || Trained on LETOR and USE Features |
|BM25 > LambdaMART_WT || Trained on LETOR and USE Features. Test slice only!|