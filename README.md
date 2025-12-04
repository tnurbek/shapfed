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
If you like the work, please consider citing us: 

```bibtex
@inproceedings{tastan2024redefining,
  title     = {{Redefining Contributions: Shapley-Driven Federated Learning}},
  author    = {Tastan, Nurbek and Fares, Samar and Aremu, Toluwani and Horváth, Samuel and Nandakumar, Karthik},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson}, 
  pages     = {5009--5017},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
}

@article{tastan2025cycle,
    title={{{CYC}le: Choosing Your Collaborators Wisely to Enhance Collaborative Fairness in Decentralized Learning}},
    author={Nurbek Tastan and Samuel Horv{\'a}th and Karthik Nandakumar},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=ygqNiLQqfH},
    note={}
}

@InProceedings{tastan2025aequa,
  title = 	 {{Aequa: Fair Model Rewards in Collaborative Learning via Slimmable Networks}},
  author =       {Tastan, Nurbek and Horv\'{a}th, Samuel and Nandakumar, Karthik},
  booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  pages = 	 {59210--59236},
  year = 	 {2025},
  editor = 	 {Singh, Aarti and Fazel, Maryam and Hsu, Daniel and Lacoste-Julien, Simon and Berkenkamp, Felix and Maharaj, Tegan and Wagstaff, Kiri and Zhu, Jerry},
  volume = 	 {267},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--19 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v267/main/assets/tastan25a/tastan25a.pdf},
  url = 	 {https://proceedings.mlr.press/v267/tastan25a.html}
}

```