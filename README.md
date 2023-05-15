
# CLEF2023-LongEval-IRC

## Setup
- `conda create env -f conda_env && conda activate LongEval`


## Experiments

### BM25 and Fuse Runs
Fusion of three runs: BM25 with Bo1 query expansion, XSqrA_M and PL2. All parameters are PyTerrier default, and test terms from three docs were used for expansion.

- `python -m src.create_index --index WT & python -m src.create_index --index ST & python -m src.create_index --index LT`
- `python -m systems.BM25 --index WT && python -m systems.BM25 --index ST && python -m systems.BM25 --index LT`


### Doc2Query
Default PyTerrier BM25 run on an expanded index with Doc2Query. Each document was expanded with ten queries from the [macavaney/doc2query-t5-base-msmarco](https://huggingface.co/macavaney/doc2query-t5-base-msmarco) model. The indexing with document expansion took around ten hours per sub-collection on a single NVIDIA GeForce RTX 3070 GPU. The model was not further trained.

- `python -m src.create_index --index WT --d2q && python -m src.create_index --index ST --d2q && python -m src.create_index --index LT --d2q`
- `python -m systems.d2q+BM25 --index WT_d2q && python -m systems.d2q+BM25 --index LT_d2q && python -m systems.d2q+BM25 --index ST_d2q`


### E5
Dense retrieval system from [intfloat/e5-base](https://huggingface.co/intfloat/e5-base) embeddings and a Faiss IndexFlatL2 index. The creation of the embeddings and indexing took around 6 hours on a single NVIDIA GeForce RTX 3070 GPU per sub-collection. The model was not further trained.

- `python -m src.create_index_e5 --index WT_e5_base --batch_size 150 --save 1000 && \ 
python -m src.create_index_e5 --index ST_e5_base --batch_size 150 --save 1000 && \ 
python -m src.create_index_e5 --index LT_e5_base --batch_size 150 --save 1000`
- `python -m system.e5 --index WT_e5_base && python -m system.e5 --index ST_e5_base && python -m system.e5 --index LT_e5_base`

### BM25+ColBERT
BM25 ranking that was reranked by colBERT. The full documents were used (and truncated after 512 subword tokens). Completing one run took under 30 minutes on a single NVIDIA GeForce RTX 2080Ti GPU.

- `python -m systems.BM25+colBERT --index WT && python -m systems.BM25+colBERT --index ST && python -m systems.BM25+colBERT --index LT`


### BM25+monoT5
BM25 ranking that was reranked by [castorini/monot5-base-msmarco](https://huggingface.co/castorini/monot5-base-msmarco). The full documents were used (and truncated after 512 subword tokens). Completing one run took around 5 hours on a single NVIDIA GeForce RTX 2080Ti GPU.

- `python -m systems.BM25+monoT5 --index WT && python -m systems.BM25+monoT5 --index ST && python -m systems.BM25+monoT5 --index LT`

## Results on the train topics of the WT sub-collection
 |System | map | bpref | recip_rank | P_20 | ndcg | ndcg_cut_20 |
 |:--- | :---: | :---: | :---: | :---: | :---: | :---: |
 | BM25 | 0.1452 | 0.3245 | 0.2604 | 0.0654 | 0.2884 | 0.2087 | 
 | RRF(BM25+Bo1-XSqrA_M-PL2) | 0.1511 | 0.3466 | 0.2686 | 0.0673 | 0.3040 | 0.2155 | 
 | d2q(10)>BM25 | 0.1799 | 0.3361 | 0.2918 | 0.0781 | 0.3117 | 0.2494 | 
 | BM25+monoT5 | 0.1809 | 0.3494 | 0.3216 | 0.0768 | 0.3208 | 0.2490 | 
 | BM25+colBERT | 0.1682 | 0.3447 | 0.3046 | 0.0692 | 0.3082 | 0.2310 | 
 | e5_base | 0.1545 | 0.3483 | 0.2826 | 0.0634 | 0.2910 | 0.2128 | 


 ## References
- Bassani, E., & Romelli, L. (2022). ranx.fuse: A python library for metasearch. In M. A. Hasan & L. Xiong (Hrsg.), Proceedings of the 31st ACM international conference on information & knowledge management, atlanta, GA, USA, october 17-21, 2022 (S. 4808–4812). ACM. https://doi.org/10.1145/3511808.3557207
- Cheriton, D. R. (2019). From doc2query to docTTTTTquery.
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In J. Allan, J. A. Aslam, M. Sanderson, C. Zhai, & J. Zobel (Hrsg.), Proceedings of the 32nd annual international ACM SIGIR conference on research and development in information retrieval, SIGIR 2009, boston, MA, USA, july 19-23, 2009 (S. 758–759). ACM. https://doi.org/10.1145/1571941.1572114
- Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In J. X. Huang, Y. Chang, X. Cheng, J. Kamps, V. Murdock, J.-R. Wen, & Y. Liu (Hrsg.), Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020 (S. 39–48). ACM. https://doi.org/10.1145/3397271.3401075
- Macdonald, C., & Tonellotto, N. (2020). Declarative experimentation in information retrieval using PyTerrier. In K. Balog, V. Setty, C. Lioma, Y. Liu, M. Zhang, & K. Berberich (Hrsg.), ICTIR ’20: The 2020 ACM SIGIR international conference on the theory of information retrieval, virtual event, norway, september 14-17, 2020 (S. 161–168). ACM. https://doi.org/10.1145/3409256.3409829
- Nogueira, R. F., Yang, W., Lin, J., & Cho, K. (2019). Document expansion by query prediction. CoRR, abs/1904.08375. http://arxiv.org/abs/1904.08375
- Nogueira, R., Jiang, Z., Pradeep, R., & Lin, J. (2020). Document Ranking with a Pretrained Sequence-to-Sequence Model. 708–718. https://doi.org/10.18653/v1/2020.findings-emnlp.63
- Pradeep, R., Nogueira, R., & Lin, J. (2021). The Expando-Mono-Duo Design Pattern for Text Ranking with Pretrained Sequence-to-Sequence Models (arXiv:2101.05667). http://arxiv.org/abs/2101.05667
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21, 140:1-140:67.
- Robertson, S., Walker, S., Jones, S., Hancock-Beaulieu, M., & Gatford, M. (1994). Okapi at TREC-3.
- Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2022). Text embeddings by weakly-supervised contrastive pre-training. CoRR, abs/2212.03533. https://doi.org/10.48550/arXiv.2212.03533
