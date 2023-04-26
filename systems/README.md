# BM25
- BM25: BM25 baseline
- BM25+RM3: RM3 reranker

## Naming:
| WT (t)                | ST (t')               | LT (t'')              |
| --------------------- | --------------------- | --------------------- |
| IRCologne_BM25.WT     | IRCologne_BM25.ST     | IRCologne_BM25.LT     |
| IRCologne_BM25_RM3.WT | IRCologne_BM25+RM3.ST | IRCologne_BM25+RM3.LT |


## Results:
| name                  | map                 | P_20                | ndcg               | ndcg_cut_20         |
|-----------------------|---------------------|---------------------|--------------------|---------------------|
| IRCologne_BM25.WT     | 0.14523270287562143 | 0.07357723577235771 | 0.281723358942147  | 0.21251216027152317 |
| IRCologne_BM25_RM3.WT | 0.1409423488476195  | 0.06910569105691057 | 0.2762760864249693 | 0.19790412330025706 |
 <br><br>
 


# LambdaMART
- IRCologne\_LambdaMART: LambdaMART witrh LETOR features
- IRCologne\_LambdaMART\_BERT: LabdaMART with BERT features

| WT (t)                | ST (t')               | LT (t'')              |
| --------------------- | --------------------- | --------------------- |
| IRCologne_LambdaMART.WT     | IRCologne_LambdaMART.ST     | IRCologne_LambdaMART.LT     |
| IRCologne_LambdaMART_BERT.WT | IRCologne_LambdaMART_BERT.ST | IRCologne_LambdaMART_BERT.LT |

 <br><br>


# ColBERT
- IRCologne_ColBERT: ColBERT from PyTerrier Ceckpoint
- IRCologne_ColBERT_LE: BERT model Fine-Tuned as ColBER with the LongEval passages dataset

## Naming:
| WT (t)                | ST (t')               | LT (t'')              |
| --------------------- | --------------------- | --------------------- |
| IRCologne_ColBERT.WT     | IRCologne_ColBERT.ST     | IRCologne_ColBERT.LT     |
| IRCologne_ColBERT_LE.WT | IRCologne_ColBERT_LE.ST | IRCologne_ColBERT_LE.LT |

## Results:
| name                    | map                  | P_20                | ndcg                | ndcg_cut_20          |
|-------------------------|----------------------|---------------------|---------------------|----------------------|
| IRCologne_ColBERT.WT    | 0.19260795117019522  | 0.07967479674796751 | 0.3227943930754179  | 0.25575666214554404  |
| IRCologne_ColBERT_LE.WT | 0.014550832430080929 | 0.00853658536585366 | 0.13280381598566277 | 0.018442705469450395 |
