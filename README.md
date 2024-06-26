# Redefining Contributions: Shapley-Driven Federated Learning [IJCAI 2024]

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://tnurbek.github.io/shapfed/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2406.00569)

> [**Redefining Contributions: Shapley-Driven Federated Learning [IJCAI 2024]**](https://arxiv.org/abs/2406.00569)<br>
> [Nurbek Tastan](https://tnurbek.github.io/), [Samar Fares](https://www.linkedin.com/in/samarfares/), [Toluwani Aremu](https://www.toluwaniaremu.com/), [Samuel Horvath](https://samuelhorvath.github.io/), [Karthik Nandakumar](https://www.sprintai.org/nkarthik) 

Official implementation of the paper: "Redefining Contributions: Shapley-Driven Federated Learning" [IJCAI 2024]. 

## Overview
![main figure](https://tnurbek.github.io/assets/img/shapfed.png)

<b>Overview of our proposed ShapFed algorithm:</b> Each participant $i$ transmits their locally computed iterates $w_i$ to the server. The server then, (i) computes class-specific Shapley values (CSSVs) using the last layer parameters (gradients) $\hat{w}$ (as illustrated in Figure 2), (ii) aggregates the weights by employing normalized contribution assessment values $\tilde{\gamma}_i$ for each participant $i$, and (iii) broadcasts the personalized weights $\bar{w}_i$ to each participant, using their individual, not-normalized contribution values $\gamma_i$. 

> **<p align="justify"> Abstract:** Federated learning (FL) has emerged as a pivotal approach in machine learning, enabling multiple participants to collaboratively train a global model without sharing raw data. While FL finds applications in various domains such as healthcare and finance, it is challenging to ensure global model convergence when participants do not contribute equally and/or honestly. To overcome this challenge, principled mechanisms are required to evaluate the contributions made by individual participants in the FL setting. Existing solutions for contribution assessment rely on general accuracy evaluation, often failing to capture nuanced dynamics and class-specific influences. This paper proposes a novel contribution assessment method called ShapFed for fine-grained evaluation of participant contributions in FL. Our approach uses Shapley values from cooperative game theory to provide a granular understanding of class-specific influences. Based on ShapFed, we introduce a weighted aggregation method called ShapFed-WA, which outperforms conventional federated averaging, especially in class-imbalanced scenarios. Personalizing participant updates based on their contributions further enhances collaborative fairness by delivering differentiated models commensurate with the participant contributions. Experiments on CIFAR-10, Chest X-Ray, and Fed-ISIC2019 datasets demonstrate the effectiveness of our approach in improving utility, efficiency, and fairness in FL systems.


## Dependencies
```
pip install -r requirements.txt
```

## Run ShapFed algorithm
Default dataset: synthetic dataset. 
```
python3 main_synthetic.py --model_num 4 --aggregation 2 --split heterogeneous --num_rounds 50 --num_lepochs 1 
```

## Citation 
```bibtex
@InProceedings{tastan2024redefining,
    author    = {Tastan, Nurbek and Fares, Samar and Aremu, Toluwani and Horvath, Samuel and Nandakumar, Karthik},
    title     = {Redefining Contributions: Shapley-Driven Federated Learning}, 
    booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
    year      = {2024},
}
```