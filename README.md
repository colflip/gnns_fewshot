# gnns_fewshot [[address](https://github.com/colflip/gnns_fewshot)]

Code implementation of GNNs in few-shot learning: GCN, GAT, GraphSAGE to the node classification task of some datasets.

## dependencies

* Python ≥ 3.10
* PyTorch ≥ 11.3
* pyg ≥ 1.12.0

## results

### novel class is num_class*2/5

|      | model/shot/dataset | Cora          | CiteSeer      | Photo         | cs            | Computers     | CoraFull       |
|------|--------------------|---------------|---------------|---------------|---------------|---------------|----------------|
| GCN  | 1                  | 72.43±1.97[2] | 64.9±1.75[2]  | 78.69±2.49[3] | 83.51±0.88[6] | 74.35±1.99[4] | 34.56±0.22[28] |
|   | 3                  | 85.6±1.13[2]  | 75.67±1.66[2] | 91.12±0.54[3] | 91.92±0.2[6]  | 87.88±0.74[4] | 54.92±0.17[28] |
|   | 5                  | 89.28±0.8[2]  | 79.29±1.39[2] | 93.32±0.32[3] | 93.75±0.14[6] | 90.64±0.46[4] | 62.46±0.15[28] |
| GAT  | 1                  | 69.37±2.34[2] | 61.08±1.59[2] | 40.52±3.98[3] | 73.39±1.53[6] | 30.95±3.1[4]  | 34.64±0.26[28] |
|   | 3                  | 84.07±1.31[2] | 72.12±1.87[2] | 61.33±8.15[3] | 89.95±0.28[6] | 56.33±7.8[4]  | 54.55±0.17[28] |
|   | 5                  | 88.79±0.9[2]  | 77.12±1.57[2] | 74.29±7.08[3] | 92.17±0.17[6] | 69.63±7.27[4] | 62.05±0.15[28] |
| GraphSAGE | 1                  | 71.74±1.75[2] | 64.41±1.78[2] | 48.61±1.77[3] | 72.7±2.07[6]  | 36.4±0.91[4]  | 23.67±0.26[28] |
|  | 3                  | 83.32±1.15[2] | 72.98±1.59[2] | 69.95±2.03[3] | 87.24±0.64[6] | 62.1±1.7[4]   | 47.93±0.18[28] |
|  | 5                  | 87.48±0.92[2] | 78.41±1.25[2] | 82.17±1.13[3] | 90.6±0.31[6]  | 75.69±1.31[4] | 57.68±0.16[28] |


### novel class is 2

|  | **model/shot/dataset** | **Cora**   | **CiteSeer** | **Photo**  | **cs**     | **Computers** | **CoraFull** | **PubMed** |
|---|------------------------|------------|--------------|------------|------------|---------------|--------------|------------|
| GCN | 1                      | 72.12±1.96 | 64.82±1.76   | 83.82±2.76 | 94.6±0.86  | 83.53±3.21    |
|   | 3                      | 85.62±1.09 | 75.43±1.73   | 93.87±0.58 | 97.86±0.17 | 93.9±0.94     |
|   | 5                      | 89.18±0.77 | 79.28±1.41   | 95.38±0.31 | 98.08±0.15 | 95.56±0.43    |
| GAT | 1                      | 68.66±2.31 | 61.27±1.58   | 56.78±4.56 | 86.42±2.27 | 51.03±3.78    |
|   | 3                      | 84.22±1.3  | 71.94±1.84   | 68.62±7.07 | 95.57±0.47 | 65.88±7.03    |
|   | 5                      | 88.92±0.92 | 77.54±1.53   | 76.12±6.48 | 96.82±0.27 | 75.34±7.65    |
| GraphSAGE | 1                      | 71.38±1.78 | 64.55±1.64   | 60.6±1.82  | 89.86±1.79 | 57.57±1.32    |
|   | 3                      | 83.63±1.11 | 73.38±1.57   | 76.25±2.6  | 96.03±0.5  | 69.92±2.49    |
|   | 5                      | 87.7±0.84  | 77.54±1.27   | 84.35±1.58 | 96.3±0.33  | 80.01±2.3     |

## license

It is under the MIT license. See the [LICENSE](LICENSE) file for details.

***

“一切恩爱会、无常难得久、生世多畏惧、命危于晨露，**由爱故生忧，由爱故生怖，若离于爱者，无忧亦无怖。**”

![](https://pic1.zhimg.com/80/v2-5fa69cb8df03fc653aac644d611392ce_720w.webp?source=1940ef5c)
