## A Single-Step, Sharpness-Aware Minimization is All You Need to Achieve Efficient and Accurate Sparse Training

Jie Ji, Gen Li, Jingjing Fu, Fatemeh Afghah, Linke Guo, Xiaoyong Yuan, Xiaolong Ma

Clemson University jji@g.clemson.edu

## Abstract

Sparse training stands as a landmark approach in addressing the considerable training resource demands imposed by the continuously expanding size of Deep Neural Networks (DNNs). However, the training of a sparse DNN encounters great challenges in achieving optimal generalization ability despite the efforts from the state-of-the-art sparse training methodologies. To unravel the mysterious reason behind the difficulty of sparse training, we connect network sparsity with the structure of neural loss functions and identify that the cause of such difficulty lies in a chaotic loss surface. In light of such revelation, we propose S 2 -SAM, characterized by a Single-step Sharpness-Aware Minimization that is tailored for Sparse training. For the first time, S 2 -SAM innovates the traditional SAM-style optimization by approximating sharpness perturbation through prior gradient information, incurring zero extra cost . Therefore, S 2 -SAM not only exhibits the capacity to improve generalization but also aligns with the efficiency goal of sparse training. Additionally, we study the generalization result of S 2 -SAM and provide theoretical proof for convergence. Through extensive experiments, S 2 -SAM demonstrates its universally applicable plug-and-play functionality, enhancing accuracy across various sparse training methods. Code available at https://github.com/jjsrf/SSAM-NEURIPS2024 .

## 1 Introduction

The arrival of the Artificial General Intelligence (AGI) [1] era has urged an ever-expanding realm of artificial intelligence, bringing significant growth in deep neural networks (DNNs) depth and intricacy. The efficient training of DNN has thus emerged as an uttermost imperative, demanding immediate and concerted efforts.

To train an overparameterized, large-scale DNN efficiently while preserving its high accuracy, stateof-the-art literature has introduced sparse training [2-6] as a straightforward solution that reduces both parameter footprint and computation cost. With the presence of a large portion of zeros in the network, both forward and backward computation for training can be saved by skipping those zeros, as well as reducing the memory consumption since the zeros are not necessary to be stored. However, training a sparse neural network is difficult since the optimization under sparse regime readily converges to stationary points with a sub-optimal generalization accuracy (i.e., saddle points) [7]. Due to the difficulty of directly (i.e., without the time-consuming linear interpolation) assessing if the sparse solution is in the saddle point, finding workaround approaches for training a better sparse neural network is critical, especially for the practical usage in efficient learning [8-17].

To address the above difficulty, many efforts have been made. For instance, the Static Sparse Training (SST) such as the Lottery Ticket Hypothesis (LTH) [20], SNIP [21], GraSP [18] and SynFlow [22]

Figure 1: The loss surface visualization for training a sparse neural network using ResNet-32 on CIFAR-10. We select two representative sparse training methods [18, 3] and incorporate different levels of sparsity. We also quantify the loss surface behavior using coefficient Ra [19] to evaluate sharpness. With increased sparsity, Ra becomes larger, indicating sharper and steeper surface.

<!-- image -->

determine a static sparse pattern at training initialization. On the other hand, Dynamic Sparse Training (DST) such as SET [2], RigL [23], and MEST [3] iteratively updates the sparse topology during training to find a better sparse model. However, those methods either suffer from suboptimal generalization ability due to heuristic nature of their methodology or metric settings, or experience difficulties in parameter setting and dynamic sparsity scheduling. With the increase of the sparsity, these phenomenon become more severe.

We identify that the cause of the sparse network learning difficulties lies in the variable learning dynamics, which is closely related to the network topology. When incorporating sparsity into a neural network, the effective structure of the original network becomes narrower. According to the study of neural loss function structure [24], a wide network resulted in flat minima and wide regions of apparent convexity, which helps prevent the chaotic behavior that occurs during training. Evidently, higher sparsity indicates a narrower structure, which suggests more chaotic behavior is expected during training, thus degrading the accuracy. We perform different types of sparse training with different levels of sparsity and plot the loss surface in Figure 1 to demonstrate the convexity of the training behavior. According to the sharpness of the loss basin, we observe a transition from a smooth and close-to-convex surface to a steep one as sparsity is introduced and increased. Since higher sparsity levels lead to more chaotic behavior, in landscapes that are narrow and sharp with a large Ra value, we might expect to encounter more chaotic training behavior that degrades generalization ability.

It seems that achieving coexistence of sparsity and good generalization ability during training is challenging. Therefore, we raise the following question: is there a simple, effective method that can improve generalization ability of training a sparse neural network, without sacrificing efficiency (sparsity) and incurring zero extra cost? We believe the answer to the above question lies in the sharpness of the loss surface. Inspired by the Sharpness-Aware Minimization (SAM) [25] technique that finds flatter minima that have uniformly low loss in nearby regions, we argue that such technique is especially suitable for sparse training since the steep loss surface induced by sparsity can be directly mitigated. However, to leverage sharpness for better generalization, SAM uses an additional full training step (i.e., forward and backward propagation) to quantify and evaluate loss on the summation of current weights and a constrained perturbation, which roughly doubles the computation of training. Such extra costs contradict the goal of sparse training. Although some literature [26, 27] have been proposed to reduce the computation cost for SAM, the computation is still significant, posing a roadblock to the wide application of SAM to the realm of sparse training.

In this paper, we propose a novel approach to achieve sharpness-aware minimization with zero extra cost , tailored for the sparse training regime to maintain efficiency while improving generalization ability. It is the first time that a S ingle-step S harpnessA ware M inimization is proposed for S parse training (S 2 -SAM). Different from the traditional two-step computation regime of SAM, S 2 -SAM uses a unique single-step approach that leverages sharpness and trains weights with one training step. Specifically, S 2 -SAM uses the weight gradients from the prior step to approximate the perturbation to the weights, thus solving the sharpness evaluation without performing an extra full training step. Therefore, S 2 -SAM incurs zero extra cost to achieve sharpness-aware training, which aligns with the

efficiency goal of sparse training. We also study the generalization result of S 2 -SAM and provide theoretical proof for convergence. We demonstrate that S 2 -SAM provides a straightforward plug-andplay functionality on variety of sparse training methods, significantly boosting their accuracy. Our contributions are summarized as follows:

- We identify that the difficulty of training a sparse neural network lies in the increasingly chaotic and steep loss surface when sparsity is introduced and increased.
- Wedevelop a novel Single-step Sharpness-Aware Minimization technique tailored for Sparse training (S 2 -SAM), and it is the first time that a SAM-style optimization with zero extra computation cost has been proposed.
- We study the generalization result of S 2 -SAM and provide theoretical proof to demonstrate that the S 2 -SAM is guaranteed for convergence.
- Through systematic evaluations, we show that S 2 -SAM provides a plug-and-play functionality applied to a variety of sparse training methods, and consistently improves accuracy on different networks and datasets.

## 2 Proposed Method

## 2.1 Preliminary of Sharpness-Aware Minimization

SAM is an optimization technique designed to enhance neural network generalization and mitigate overfitting. It minimizes the maximum loss in a neighborhood around the current parameters, as opposed to solely focusing on the loss at the current point. This approach identifies flatter minima that have uniformly low loss in nearby regions, ultimately contributing to improved generalization performance.

Specifically, consider a family of models parameterized by w ∈ W ⊆ R d ; L is the loss function, and S denotes the training dataset. SAM aims to minimize the following upper bound of the PAC-Bayesian generalization error: for any ρ &gt; 0 ,

Figure 2: Illustration of the optimization mechanism of S 2 -SAM. The perturbation on the current weights is approximated by the weight gradients from prior step. Please see Section 2.2 for detailed discussion.

<!-- image -->

<!-- formula-not-decoded -->

To solve the above minimax problem, at each iteration t , SAM updates the following steps:

<!-- formula-not-decoded -->

where 1 /p + 1 /q = 1 , ρ &gt; 0 is a hyperparameter, λ &gt; 0 is the parameter for weight decay, and η t &gt; 0 is the learning rate. By setting p = q = 2 and introducing an intermediate variable u t , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 2.2 The Proposed S 2 -SAM Method

As shown in Equation (2), SAM needs to compute the gradient twice at each iteration, involving additional computation costs. Further, the two-step gradient computation is not parallelizable, which presents challenges for deployment in large-scale training scenarios. Thus, we propose a new algorithm that only needs to compute the gradient once in each iteration. Different from prior efficient SAMworks [26-28] aiming to reduce the computation by introducing periodically SAM steps or data

selection, our proposed framework S 2 -SAM achieves zero extra computation cost while maintaining the improved generalization ability of sparse network training.

In Figure 2, we demonstrate that the perturbation on the current weights is approximated by the weight gradients g t -1 from the prior step. The rational behind such design is that the gradient direction (i.e., w t -2 to w t -1 ) from prior step is optimizing the prior loss to a considerable extent, thus it can be used to represent a sharp direction among all perturbation directions at t -1 . Since the loss surface of a sparse network is too chaotic, a perturbation with relatively high degree of sharpness (i.e., the prior gradient) can still find a neighborhood with near-maximum loss. The following equations specifying g t -1 information are used to substitute the ∇ L S ( w t -1 ) step in Equation (2). Thus, we only need to compute the gradient once in each iteration:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark 1 . The parameter ρ can vary in terms of iteration t : ρ t = √ c/t , where c &gt; 0 is a constant.

## 2.3 Generalization Analysis

In this section, we study the generalization result of S 2 -SAM. First, we give some notations, where most of them are followed by [29, 30]. Let w S = A ( S ) be a solution that generated by a random algorithm A based on dataset S . Recall that problem (8)

<!-- formula-not-decoded -->

is called empirical risk minimization in the literature, and the true risk minimization is given by

<!-- formula-not-decoded -->

We define its optimal solution: w ∗ ∈ arg min w ∈W F ( w ) . Then the excess risk bound (ERB) is defined as

<!-- formula-not-decoded -->

It has been shown that the ERB can be upper bounded by optimization error and generalization error [29, 30]. We notice that there are several works [29, 30] studying the generalization result of SGD for non-convex setting under different conditions, such as bounded stochastic gradient ∥∇ w ℓ ( y , f ( w ; x )) ∥ ≤ G and decaying learning rate η t ≤ c t with a constant c &gt; 0 , where t is the optimization iteration. In this paper, we are not interested in establishing a fast rate in ERB under different conditions, but we want to explore the generalization ability of S 2 -SAM with the fewest possible modifications when building a bridge between theory and practice. For example, weight decay is a widely used trick when training deep neural networks. With the use of weight decay, the empirical risk minimization in practice becomes min w ∈W { ̂ F S ( w ) := F S ( w ) + λ 2 ∥ w ∥ 2 } . Then, we define some notations as follows. Specifically, let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Followed by [30], we use the following decomposition of testing error:

<!-- formula-not-decoded -->

where the upper bound is the optimization error plus the generalization error.

Next, we present some notations and assumptions that will be used in the convergence analysis. Throughout this paper, we also make the following assumptions for solving the problem (8).

Assumption 1. Assume the following conditions hold: (i) The stochastic gradient of F S ( w ) is unbiased, i.e., E ( x , y ) [ ∇ ℓ ( y , f ( w ; x ))] = ∇ F S ( w ) , and the variance of stochastic gradient is bounded, i.e., there exists a constant σ 2 &gt; 0 , such that

<!-- formula-not-decoded -->

(ii) F S ( w ) is smooth with an L-Lipchitz continuous gradient, i.e., it is differentiable and there exists a constant L &gt; 0 such that ∥∇ F S ( w ) -∇ F S ( u ) ∥ ≤ L ∥ w -u ∥ , ∀ w , u ∈ W .

Assumption 2. There exists a constant µ &gt; 0 such that 2 µ ( F S ( w ) -F S ( w ∗ S )) ≤ ∥∇ F S ( w ) ∥ 2 , ∀ w ∈ W , where w ∗ S ∈ min w ∈W F S ( w ) is a optimal solution (PL condition [31]).

Now consider that A = S 2 -SAM. We define the gradient update rule G ̂ ℓ,η as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we have the following lemma, which is similar to Lemma 2.5 and Lemma 4.2 in [29] that use recursive definition through variable G ′ and w ′ t .

Lemma 1. Assume that ℓ ( y , f ( w , x )) is L-smooth and B-Lipschitz. Let w t +1 = G ( w t ) and another sequence w ′ t +1 = G ′ ( w ′ t ) , then

<!-- formula-not-decoded -->

Please see the proof of Lemma 1 in Appendix B.

Theorem 1. Under Assumption 1, assume that ℓ ( y , f ( w , x )) is L-smooth and B-Lipschitz, suppose ̂ F S ( w ) satisfies Assumption 2 and min w ∈W ̂ F S ( w ) ≤ F S ( w ∗ S ) + λ 2 ∥ w t ∥ 2 with λ = 2 L , where w t is the intermediate solution of A , then

<!-- formula-not-decoded -->

where A is SGD .

The w ∗ S ∈ min w ∈W F S ( w ) is an optimal solution, and we show that the generalization error is bounded. Based on Lemma 1, the proof of Theorem 1 is derived in Appendix C.

## 3 Experimental Results

In this section, we carry out comprehensive experiments to demonstrate how S 2 -SAM improves sparse training performance. We test S 2 -SAM on CIFAR-10/100 [34] with ResNet-32 [32] and VGG-19 [35], and we also perform experiments on ImageNet-1K [36] and ImageNet-C [37] based on ResNet-50 [38].

Following the recent developments in sparse training techniques, we apply S 2 -SAM on static sparse training such as LTH [20], SNIP [21], GraSP [18], as well as dynamic sparse training methods such as SET [2], DSR [33], RigL [23], MEST [3], CHEX [10] and Chase [39]. We apply S 2 -SAM to the official codes or published implementations to show performance gains. All of our experiments are performed on NVIDIA 4 × A6000 GPUs. We repeat training experiments for 3 times and report the mean and standard deviation of the accuracy. For training throughput evaluation, we adopt the original settings of each baseline method and record the throughput on 4 × A6000 GPUs.

Table 1: Test accuracy (%) of pruned ResNet-32 on CIFAR-10/100.

| Datasets          | CIFAR-10          | CIFAR-10          | CIFAR-10          | CIFAR-100          | CIFAR-100          | CIFAR-100          |
|----------|----------|----------|----------|----------|----------|----------|
| Pruning ratio          | 90%          | 95%          | 98%          | 90%          | 95%          | 98%          |
| ResNet-32          | 94.58 (Dense)          |          |          | 74.89 (Dense)          |          |          |
| LT [20] LT+ S 2 -SAM (ours)          | 92.31 92.58 ± 0.07 (0.27 ↑ )          | 91.06 91.47 ± 0.10 (0.41 ↑ )          | 88.78 89.35 ± 0.11 (0.57 ↑ )          | 68.99 69.34 ± 0.09 (0.35 ↑ )          | 65.02 65.45 ± 0.11 (0.43 ↑ )          | 57.37 57.76 ± 0.13 (0.39 ↑ )          |
| SNIP [21] SNIP+ S 2 -SAM (ours)          | 92.59 ± 0.10 93.17 ± 0.16 (0.58 ↑ )          | 91.01 ± 0.21 91.59 ± 0.22 (0.58 ↑ )          | 87.51 ± 0.31 88.08 ± 0.29 (0.57 ↑ )          | 68.89 ± 0.45 69.33 ± 0.28 (0.44 ↑ )          | 65.02 ± 0.69 65.66 ± 0.49 (0.64 ↑ )          | 57.37 ± 1.43 58.25 ± 0.77 (0.88 ↑ )          |
| GraSP [32] GraSP+ S 2 -SAM (ours)          | 92.38 ± 0.21 92.87 ± 0.14 (0.49 ↑ )          | 91.39 ± 0.25 91.98 ± 0.22 (0.59 ↑ )          | 88.81 ± 0.14 89.66 ± 0.29 (0.85 ↑ )          | 69.24 ± 0.24 69.98 ± 0.22 (0.74 ↑ )          | 66.50 ± 0.11 67.12 ± 0.18 (0.62 ↑ )          | 58.43 ± 0.43 59.45 ± 0.19 (1.02 ↑ )          |
| SET [2] SET+ S 2 -SAM (ours)          | 92.30 92.92 ± 0.23 (0.62 ↑ )          | 90.76 91.50 ± 0.19 (0.74 ↑ )          | 88.29 88.78 ± 0.20 (0.49 ↑ )          | 69.66 70.23 ± 0.20 (0.57 ↑ )          | 67.41 68.28 ± 0.15 (0.87 ↑ )          | 62.25 63.56 ± 0.19 (1.31 ↑ )          |
| DSR [33] DSR+ S 2 -SAM (ours)          | 92.97 93.49 ± 0.21 (0.52 ↑ )          | 91.61 92.08 ± 0.22 (0.47 ↑ )          | 88.46 89.11 ± 0.17 (0.65 ↑ )          | 69.63 70.11 ± 0.16 (0.48 ↑ )          | 68.20 68.87 ± 0.16 (0.67 ↑ )          | 61.24 62.00 ± 0.17 (0.76 ↑ )          |
| RigL [23] RigL+ S 2 -SAM (ours) RigL (ERK) [23] RigL (ERK)+ S 2 -SAM (ours)          | 93.07 93.55 ± 0.14 (0.48 ↑ ) 93.55 93.75 ± 0.19 (0.20 ↑ )          | 91.83 92.11 ± 0.21 (0.28 ↑ ) 92.39 92.81 ± 0.08 (0.42 ↑ )          | 89.00 90.40 ± 0.17 (1.40 ↑ ) 90.22 91.16 ± 0.11 (0.94 ↑ )          | 70.34 72.38 ± 0.11 (2.04 ↑ ) 70.62 72.56 ± 0.07 (1.94 ↑ )          | 68.22 70.29 ± 0.14 (2.07 ↑ ) 68.47 70.33 ± 0.10 (1.86 ↑ )          | 64.07 64.98 ± 0.06 (0.91 ↑ ) 64.14 65.15 ± 0.12 (1.01 ↑ )          |
| MEST (EM) [3] MEST (EM) + S 2 -SAM (ours) MEST (EM&S) [3] MEST (EM&S) + S 2 -SAM (ours) | 92.56 ± 0.07 93.43 ± 0.12 (0.87 ↑ ) 93.27 ± 0.14 93.39 ± 0.17 (0.12 ↑ ) | 91.15 ± 0.29 91.58 ± 0.07 (0.43 ↑ ) 92.44 ± 0.13 92.97 ± 0.17 (0.53 ↑ ) | 89.22 ± 0.11 91.22 ± 0.14 (2.00 ↑ ) 90.51 ± 0.11 91.32 ± 0.18 (0.81 ↑ ) | 70.44 ± 0.26 71.95 ± 0.13 (1.51x ↑ 71.30 ± 0.31 72.74 ± 0.08 (1.44 ↑ ) | 68.43 ± 0.32 70.04 ± 0.10 (1.61 ↑ ) 70.36 ± 0.05 71.85 ± 0.09 (1.49 ↑ ) | 64.59 ± 0.27 65.69 ± 0.34 (1.10 ↑ ) 67.16 ± 0.25 69.13 ± 0.20 (1.97 ↑ ) |

## 3.1 Accuracy Enhancement for State-of-the-art Sparse Training Methods

CIFAR-10 and CIFAR-100. We show S 2 -SAM is universally applicable on various sparse training methods. The results on ResNet-32 is shown in Table 1. Due to limited space, we demonstrate results of VGG-19 on CIFAR-10 in Table A.1 in the appendix. For each baseline method, we perform training with S 2 -SAM at 90%, 95% and 98% sparsity, and compare the accuracy with the original accuracy. We set hyper-parameter ρ = 0 . 05 for all experiments. From the comparison results, we can notice that all sparse training methods trained with S 2 -SAM obtain consistent and significant improvements, not only for static sparse training but also for dynamic sparse training. We stress that for all experiments, we only use one setting for S 2 -SAM, which indicates an easy implementation that has the potential for wide applicability. We also find that S 2 -SAM works better on high sparsity. For all sparse training with a 98% sparsity ratio, S 2 -SAM improves the original baseline accuracy by an average of 0.77% on CIFAR-10 and 1.12% on CIFAR-100, which shows S 2 -SAM is especially suitable for challenging tasks.

ImageNet-1K. We evaluate S 2 -SAM on ImageNet-1K dataset. The results are shown in Table 2. We report the top-1 accuracy of the sparse training results, as well the overall training cost with FLOPs. We set hyper-parameter ρ = 1 . 0 for all experiments. Compared to the baseline methods, our result shows higher test accuracy at the same computational cost. Similarly, S 2 -SAM also shows better gain on a higher sparsity ratio for most of the sparse training baselines, which indicates the consistency of using our methods on different scales of datasets. To make a fair comparison with different training recipes, we slightly scale up our training epochs to have the same or less overall training FLOPs to compare with longer training baselines, and we find out that S 2 -SAM shows stable improvement on those methods.

## 3.2 Sparse Training with Structured Sparsity

In this section, we evaluate how S 2 -SAM performs on structured sparse training methods. Note that all previous baselines use unstructured sparsity where the zeros are scattered in the network. Such unstructured sparse training shows the potential of training a sparse model, but cannot fully represent training acceleration ability. Therefore, we apply S 2 -SAM to two representative structured sparse trainings, CHEX [10] and Chase [39] which dynamically train sparse networks with channel-level sparsity. The results are shown in Table 3. From the results, we can see that the average accuracy improvement is around 0.5%. Considering structured (channel-wise) sparsity is one challenging sparsity type to achieve good accuracy, these results indicate S 2 -SAM works very well on structured sparse training methods.

## 3.3 Improvement on the Roughness of the Loss Surface

In this part, we plot the loss surface of the sparse training with and without S 2 -SAM. The comparisons are demonstrated in Figure 3, where we perform our experiments with three representative sparse training methods SNIP [21], GraSP [18] and MEST [3] at 90%, 95% and 98% sparsity ratios on

Table 2: Results of ResNet-50 on ImageNet-1K.

| Method          | Sparsity Distribution     | Top-1 Accuracy (%)        | Training FLOPs   | Inference FLOPs   | Top-1 Accuary (%)          | Training FLOPs   | Inference FLOPs   |
|----------|----------|----------|----------|----------|----------|----------|----------|
| ResNet-50          | dense          | 76.9          | ( × e18)         | ( × e9)          | 76.9          | ( × e18)         | ( × e9)          |
| Sparsity          | 80%          |          |          |          |          | 90%          |          |
| LT [20] LT + S 2 -SAM (ours)          | n/a ↑ ) n/a          | 72.6 73.11 ± 0.08 (0.51   |          | 2.7 2.7          | 70.1 70.78 ± 0.05 (0.68 ↑ )      | n/a n/a          | 1.7 1.7          |
| SNIP [21] SNIP + S 2 -SAM (ours)          | non-uniform 1.67 70.55    | 69.7 ± 0.05 (0.85 ↑ )     | 1.67          | 2.8 2.8          | 62.0 62.62 ± 0.07 (0.62 ↑ )      | 0.91 0.91        | 1.9 1.9          |
| GraSP [32] GraSP + S 2 -SAM (ours)          | 1.67 72.66 (0.56 ↑ ) 1.67 | 72.1 ± 0.10          |          | 2.8 2.8          | 68.1 68.78 ± 0.12 (0.68 ↑ )      | 0.91 0.91        | 1.9 1.9          |
| SET [2] SET + S 2 -SAM (ours)          | 0.74          | 72.9 )          | 0.74          | 1.7 1.7          | 69.6 70.41 ± 0.08 (0.81 ↑ )      | 0.10 0.10        | 0.1          |
| DSR [33] DSR + S 2 -SAM (ours)          | non-uniform          | 73.66 ± 0.11 (0.76 ↑      |          | 3.3          |          |          | 0.1          |
|          | 74.08          | 73.3 ± 0.17          | 1.28          | 3.3 1.7          | 71.6 72.32 ± 0.13 (0.72 ↑ )      | 0.96 0.96        | 2.5 2.5          |
| RigL [23] RigL + S 2 -SAM (ours)          | (0.78 ↑ ) 1.28          | 74.6 75.39 ± 0.12 (0.79 ↑ | 0.74 0.74        | 1.7 1.7          | 72.0 72.44 ± 0.06 (0.44 ↑ ) 73.6 | 0.39 0.39        | 0.9 0.9          |
| MEST (EM) [3] MEST (EM) + S 2 -SAM          | uniform )          | 75.7 76.35 ± 0.02 (0.65 ↑ | 1.10 1.10 1.27   | 1.7 1.7          | 74.58 ± 0.03 (0.98 ↑ ) 75.0      | 0.48 0.48        | 0.9 0.9          |
| (ours)          | )          | 75.7 76.44 ± 0.06 (0.74   | 1.27          | 1.7          | 75.36 ± 0.04 (0.36 ↑ )          | 0.65 0.65        | 0.9          |
| MEST (EM&S) [3] MEST (EM&S) + S 2 -SAM (ours)   | ↑ )          | -          | -          | -          | 73.0 ± ↑          | 0.63          | 0.9          |
| Top-KAST [40] 2          |          |          | -          | -          | 73.82 0.17 (0.82 )          | 0.63          | 0.9          |
| Top-KAST + S -SAM (ours)          |          | -          |          |          | 75.9 )          | 0.80          | 0.9          |
|          |          | 76.7          | 1.84          | 1.7          | ↑          | 0.80          |          |
|          |          |          |          |          |          |          | 0.9          |
| MEST 1 . 7 × [3] MEST 1 . 7 × + S 2 -SAM (ours) | uniform          | 77.73 ± 0.11 (1.03 ↑ )    | 1.84          | 1.7          | 76.82 ± 0.12 (0.92          | 1.95          | 0.9          |
|          |          | 77.72 ± 0.05 (1.12 ↑      |          | 1.7          | 76.88 ± 0.13 (1.18          | 1.95          |          |
| RigL 5 × + S 2 -SAM (ours)          | )          |          |          | 1.7          |          |          |          |
|          |          |          |          |          | ↑          |          |          |
|          |          |          | 3.71          |          |          |          |          |
|          |          |          |          |          |          |          | 0.9          |
| 5 ×          |          |          |          |          |          |          |          |
| RigL [23]          |          |          |          |          |          |          |          |
|          |          |          |          |          | 75.7          |          |          |
|          |          | 76.6          |          |          |          |          |          |
|          |          |          |          |          |          |          | 0.9          |
|          | 3.71          |          |          |          |          |          |          |
|          |          |          |          |          | )          |          |          |

Figure 3: Loss surface sharpness comparison of different sparse training methods with original training and with S 2 -SAM. We also quantitatively evaluate the coefficient Ra . Using S 2 -SAM compared to the original method results in a smaller Ra , indicating a wider and smoother loss surface, which suggests improved generalization ability.

<!-- image -->

CIFAR-10 dataset with ResNet-32. We observe that as sparsity increases, suggesting more chaotic training behavior, the coefficient Ra also increases, indicating a steeper loss surface and a narrower basin. While with S 2 -SAM, we can see that under the same sparsity, the basin of the loss surface widens and enlarges as Ra values decrease, suggesting a smoother loss trajectory during training of a sparse network.

## 3.4 Training Speed Comparison on GPU

S 2 -SAM is a highly efficient approach to optimizing the sharpness of the loss surface, which incurs zero extra cost for achieving its functionality. Compared to the traditional SAM, S 2 -SAM uses fewer computations and has better training performance on GPUs. In Table 4, we obtain the computation cost as well as the training speed in terms of throughput (i.e., imgs/sec) on 4 × NVIDIA A6000 GPUs

using ResNet-50 on the ImageNet-1K dataset. We record the throughput for different sparse training methods with their original training, with SAM [25], and with S 2 -SAM. From the results, we can see that S 2 -SAM achieves the same training throughput (with only negligible &lt; 20 imgs/s decrease due to processing) as the baseline methods, where no sharpness optimization is involved. Compared to original SAM, we can see that training throughput with SAM is less than half of the ones that train with S 2 -SAM, which aligns with the observation that SAM uses twice the computation cost of S 2 -SAM or original training. Although SAM yields slightly better accuracy, the associated training cost outweighs the benefits, rendering it impractical.

Table 3: Accuracy of S 2 -SAM on structured sparse training CHEX [10] and Chase [39].

<!-- image -->

| Methods   | Networks   | Training   | FLOPs   | Accuracy (%)    |
|----------|----------|----------|---------|----------|
| CHEX      | ResNet-34  | Original   | 2.0G    | 73.50          |
| CHEX      | ResNet-34  | S 2 -SAM   | 2.0G    | 73.94 (0.44 ↑ ) |
| CHEX      | ResNet-50  | Original   | 1.0G    | 76.00          |
| CHEX      | ResNet-50  | S 2 -SAM   | 1.0G    | 76.51 (0.51 ↑ ) |
| Chase     | ResNet-34  | Original ∗ | 2.1G    | 72.34          |
| Chase     | ResNet-34  | S 2 -SAM   | 2.1G    | 72.77 (0.43 ↑ ) |
| Chase     | ResNet-50  | Original   | 1.3G    | 75.62          |
| Chase     | ResNet-50  | S 2 -SAM   | 1.3G    | 76.17 (0.55 ↑ ) |

* ResNet-34 original result of Chase is our own implementation.

## 3.5 Robustness Improvement by S 2 -SAM

Since sparse training approaches have potentially high usage in practical scenarios, the system's robustness against perturbation (e.g., inclement weather conditions for image/video tasks) is usually critical. We perform experiments to evaluate the sparse model (80% sparsity) robustness against perturbations in Table 5. Our intuition is that the model trained with its sharpness optimized has a wider loss basin, which indicates higher endurance on perturbations since the loss won't change much when it is located in a wide and flat region. We adopt ImageNet-C [37] which contains a test set with the same images as ImageNet-1K but with nineteen types of corruptions applied with five different levels of severity. We train the sparse networks using different methods on ImageNet-1K, and then test their accuracy on ImageNet-C. We report test accuracy for both datasets. We can see that when the sparse model is trained using the original method, the accuracy of ImageNet-C is significantly lower than the accuracy of the original ImageNet-1K test set (around 35% lower). With S 2 -SAM, the test accuracy on ImageNet-C shows promising improvement. The robust accuracy improves by an average of 3.23%.

Table 5: Testing accuracy on ImageNet-C test set. We compare the results with and without S 2 -SAM using 80% sparsity.

Methods

SNIP

SNIP + S

2

GraSP

GraSP + S

2

-SAM

MEST (EM)

2

MEST (EM) + S

RigL

RigL + S

Table 6: Testing accuracy on dense model training. We compare original training with S 2 -SAM in same settings.

| Networks          | Params. Count       | Original Accuracy (%)   | S 2 -SAM Accuracy (%)          |
|----------|----------|----------|----------|
| CIFAR-10          | CIFAR-10          | CIFAR-10          | CIFAR-10          |
| ResNet-32 MobileNet-V2 VGG-19       | 1.86M 2.30M 20.03M  | 94.58 94.13 94.21       | 94.99 (0.41 ↑ ) 94.55 (0.42 ↑ ) 94.48 (0.27 ↑ ) |
| ImageNet-1K          | ImageNet-1K         | ImageNet-1K          | ImageNet-1K          |
| EfficientNet-B0 ResNet-34 ResNet-50 | 5.30M 21.80M 25.50M | 76.54 74.09 76.90       | 77.10 (0.56 ↑ ) 74.58 (0.49 ↑ ) 77.32 (0.42 ↑ ) |

2

-SAM

ImageNet-1K

Accuracy (%)

69.70

70.55

(0.85

72.10

72.66

(0.56

75.70

76.35

(0.65

74.60

75.39

(0.79

↑

↑

↑

↑

)

)

)

)

ImageNet-C

Accuracy (%)

31.12

34.87

(3.75

32.24

35.17

(2.93

33.87

36.98

(3.11

33.68

36.80

(3.12

↑

↑

↑

↑

)

)

)

)

## 3.6 Applying S 2 -SAM to Dense Model Training

We also apply S 2 -SAM to the dense model training. The results are shown in Table 6. We test on two datasets CIFAR-10 and ImageNet-1K with two additional networks MobileNet-V2 [41] and EfficientNet-B0 [42]. For ResNet-32 and VGG-19 on CIFAR-10, we follow the settings in [3] and train for 160 epochs, and for MobileNet-V2, we use the 1.0 width version and train for 350 epochs

Table 4: Training speed of SAM [25] and S 2 -SAM for different sparse training at 90% sparsity.

| Methods   | Training          | Accuracy (%)      | Throughput ( ↑ )          |
|----------|----------|----------|----------|
| GraSP     | Original SAM S 2 -SAM | 68.10 68.95 68.78 | 2148 imgs/s 1021 imgs/s 2132 imgs/s |
| RigL      | Original SAM S 2 -SAM | 72.00 72.75 72.44 | 3133 imgs/s 1508 imgs/s 3098 imgs/s |
| MEST (EM) | Original SAM S 2 -SAM | 73.60 74.88 74.58 | 2981 imgs/s 1398 imgs/s 2977 imgs/s |

-SAM

-SAM

for better convergence [43]. For ResNet family on ImageNet-1K, we follow the setting in [3] and train for 150 epochs, and we train EfficientNet-B0 for 350 epochs for better convergence [44]. From the results, we can see that S 2 -SAM still improves the accuracy of the dense networks. We find out that the less the parameter count of the network, the better effectiveness of S 2 -SAM it achieves. This phenomenon proves our finding that a narrower network is harder to train, and S 2 -SAM is an effective solution for better generalization. We must also stress that S 2 -SAM is focusing on sparse neural network training, and we will leave the study of S 2 -SAM on dense model training for our future research.

## 4 Related Works

Static Sparse Training Static sparse training determines the structure of the sparse network through the application of a pruning algorithm in the early stages of training. The lottery ticket hypothesis (LTH) [20, 45, 46] uses iterative magnitude-based pruning (IMP) to find a subnetwork that can be trained from scratch without losing accuracy. SNIP [21], GraSP [18], SynFlow [22] determine a static sparse pattern at training initialization by obtaining gradient information with a few iterations of dense training. FISH [47] acquires fixed subnetwork by pre-computing a sparse mask using Fisher information.

Dynamic Sparse Training Dynamic sparse training starts with a randomly selected sparse network structure and adapts it throughout the training process in an effort to find a better sparse structure. Sparse Evolutionary Training (SET) [2] prunes small magnitude weights and grows back randomly at the end of each training epoch. Deep R [48] uses a combination of stochastic parameter updates and dynamic sparse parameterization for training. Dynamic Sparse Reparameterization (DSR) [33] proposes to redistribute parameters between layers during training. Sparse Networks from Scratch (SNFS) [49] creates the sparse momentum algorithm, which finds the layers and weights that effectively reduce the error by using exponentially smoothed gradients (momentum). In RigL [23], the gradients of all the weights are computed when the model needs to be updated to grow new connections. Top-KAST [40] proposes a scalable and performant sparse-to-spars DST framework for maximum efficacy. Powerpropagation [50] suggests a novel neural network weight parameterization that largely preserves low-magnitude parameters from learning. ITOP [51] investigates the underlying DST mechanism and finds that the advantages of DST result from a time-based search for all potential factors. MEST [3] designs a memory-economic sparse training framework targeting accurate and fast execution on edge devices. By co-training dense and sparse models, AD/AC [52] suggests a technique that, at the conclusion of training, produces precise sparse-dense model pairings. Chase [39] dynamically translates the unstructured sparsity into channel-level sparsity to achieve direct speedup on GPU.

Sharpness-Aware Minimization Sharpness-Aware Minimization was first introduced in [25]. This optimization technique is designed to identify flatter minima characterized by consistently low loss in neighboring regions, aiming to enhance the model generalization capability during training. Nevertheless, the computational cost of Sharpness-Aware Minimization is doubled due to its two-step gradient computation regime, presenting challenges for deployment in large-scale training scenarios. To reconcile such, ESAM [26] introduces two novel and efficient training strategies: stochastic weight perturbation and sharpness-sensitive data selection, enhancing the efficiency of the SAM process without compromising its generalization performance. LookSAM [27] only periodically calculates the inner gradient ascent to significantly reduce the additional training cost of SAM. SAF [28] introduces a novel trajectory loss based on KL-divergence to measure the rate of change in training loss along the model update trajectory and replace the SAM sharpness measure. CrAM [53] optimizes over the compression projection applied to the intermediate model at every training step, which results in a compressible models that can be pruned in an on-shot manner after training. However, all those methods need extra computation cost, and not target on sparse training method.

## 5 Conclusion

In this paper, we propose a novel Single-step Sharpness-Aware Minimization that is tailored for Sparse training (S 2 -SAM), which revolutionizes the originally computation-intensive sharpness-aware optimization into a highly efficient tool with zero extra cost. In light of the improved optimization trajectory in the loss surface, S 2 -SAM successfully enhances the accuracy of the sparse network

training, as well as the robustness of the sparse model in practical scenarios. S 2 -SAM offers seamless plug-and-play functionality, showcasing its potential for widespread applicability in the evolving landscape of efficient training. The research is inherently scientific, and we anticipate no adverse societal impact stemming from its findings.

## Acknowledgments

This work is partly supported by the National Science Foundation CCF-2427875, CCF-2312616, CNS-2232048, CNS-2008049, CNS-2431440, National Aeronautics and Space Administration (NASA) 80NSSC23K1393, and the Army Research Office W911NF-24-1-0044. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of NSF, NASA, the Army Research Office or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

## References

- [1] Ben Goertzel and Cassio Pennachin. Artificial general intelligence , volume 2. Springer, 2007.
- [2] Decebal Constantin Mocanu, Elena Mocanu, Peter Stone, Phuong H Nguyen, Madeleine Gibescu, and Antonio Liotta. Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science. Nature Communications , 9(1):1-12, 2018.
- [3] Geng Yuan, Xiaolong Ma, Wei Niu, Zhengang Li, Zhenglun Kong, Ning Liu, Yifan Gong, Zheng Zhan, Chaoyang He, Qing Jin, Siyue Wang, Minghai Qin, Bin Ren, Yanzhi Wang, Sijia Liu, and Xue Lin. Mest: Accurate and fast memory-economic sparse training framework on the edge. In Advances in Neural Information Processing Systems (NeurIPS) , 2021.
- [4] Lu Yin, Gen Li, Meng Fang, Li Shen, Tianjin Huang, Zhangyang Wang, Vlado Menkovski, Xiaolong Ma, Mykola Pechenizkiy, Shiwei Liu, et al. Dynamic sparsity is channel-level sparsity learner. Advances in Neural Information Processing Systems , 36, 2024.
- [5] Gen Li, Lu Yin, Jie Ji, Wei Niu, Minghai Qin, Bin Ren, Linke Guo, Shiwei Liu, and Xiaolong Ma. Neurrev: Train better sparse neural network practically via neuron revitalization. In The Twelfth International Conference on Learning Representations , 2024.
- [6] Jie Ji, Gen Li, Lu Yin, Minghai Qin, Geng Yuan, Linke Guo, Shiwei Liu, and Xiaolong Ma. Advancing dynamic sparse training by exploring optimization opportunities. In Forty-first International Conference on Machine Learning , 2024.
- [7] Utku Evci, Fabian Pedregosa, Aidan Gomez, and Erich Elsen. The difficulty of training sparse neural networks. ICML Workshop Deep Phenomena , 2019.
- [8] Alex Bronstein, Pablo Sprechmann, and Guillermo Sapiro. Learning efficient structured sparse models. In International Conference on Machine Learning , 2012.
- [9] Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. Journal of Machine Learning Research , 22(241):1-124, 2021.
- [10] Zejiang Hou, Minghai Qin, Fei Sun, Xiaolong Ma, Kun Yuan, Yi Xu, Yen-Kuang Chen, Rong Jin, Yuan Xie, and Sun-Yuan Kung. Chex: Channel exploration for cnn model compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 12287-12298, 2022.
- [11] Xiaolong Ma, Minghai Qin, Fei Sun, Zejiang Hou, Kun Yuan, Yi Xu, Yanzhi Wang, Yen-Kuang Chen, Rong Jin, and Yuan Xie. Effective model sparsification by scheduled grow-and-prune methods. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2022.
- [12] Bo Hui, Da Yan, Xiaolong Ma, and Wei-Shinn Ku. Rethinking graph lottery tickets: Graph sparsity matters. In The Twelfth International Conference on Learning Representations , 2023.

- [13] Haoyu Ma, Chengming Zhang, Xiaolong Ma, Geng Yuan, Wenkai Zhang, Shiwei Liu, Tianlong Chen, Dingwen Tao, Yanzhi Wang, Zhangyang Wang, et al. Hrbp: Hardware-friendly regrouping towards block-based pruning for sparse cnn training. In Conference on Parsimony and Learning , pages 282-301. PMLR, 2024.
- [14] Jiancheng Liu, Parikshit Ram, Yuguang Yao, Gaowen Liu, Yang Liu, PRANAY SHARMA, Sijia Liu, et al. Model sparsity can simplify machine unlearning. Advances in Neural Information Processing Systems , 36, 2024.
- [15] Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh, Yaqing Wang, Yiling Jia, Gen Li, Ajay Kumar Jaiswal, Mykola Pechenizkiy, Yi Liang, Michael Bendersky, Zhangyang Wang, and Shiwei Liu. Outlier weighed layerwise sparsity (OWL): A missing secret sauce for pruning LLMs to high sparsity. In Proceedings of the 41st International Conference on Machine Learning , volume 235, pages 57101-57115. PMLR, 2024.
- [16] Shuyao Wang, Yongduo Sui, Jiancan Wu, Zhi Zheng, and Hui Xiong. Dynamic sparse learning: A novel paradigm for efficient recommendation. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining , pages 740-749, 2024.
- [17] Haizhong Zheng, Xiaoyan Bai, Beidi Chen, Fan Lai, and Atul Prakash. Learn to be efficient: Build structured sparsity in large language models. arXiv preprint arXiv:2402.06126 , 2024.
- [18] Chaoqi Wang, Guodong Zhang, and Roger Grosse. Picking winning tickets before training by preserving gradient flow. In International Conference on Learning Representations (ICLR) , 2020.
- [19] NO Myers. Characterization of surface roughness. Wear , 5(3):182-189, 1962.
- [20] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. ICLR , 2018.
- [21] Namhoon Lee, Thalaiyasingam Ajanthan, and Philip Torr. Snip: Single-shot network pruning based on connection sensitivity. In International Conference on Learning Representations (ICLR) , 2019.
- [22] Hidenori Tanaka, Daniel Kunin, Daniel L Yamins, and Surya Ganguli. Pruning neural networks without any data by iteratively conserving synaptic flow. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS) , volume 33, 2020.
- [23] Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro, and Erich Elsen. Rigging the lottery: Making all tickets winners. In International Conference on Machine Learning (ICML) , pages 2943-2952. PMLR, 2020.
- [24] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets. Advances in neural information processing systems , 31, 2018.
- [25] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. In International Conference on Learning Representations , 2021.
- [26] Jiawei Du, Hanshu Yan, Jiashi Feng, Joey Tianyi Zhou, Liangli Zhen, Rick Siow Mong Goh, and Vincent YF Tan. Efficient sharpness-aware minimization for improved training of neural networks. arXiv preprint arXiv:2110.03141 , 2021.
- [27] Yong Liu, Siqi Mai, Xiangning Chen, Cho-Jui Hsieh, and Yang You. Towards efficient and scalable sharpness-aware minimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 12360-12370, 2022.
- [28] Jiawei Du, Daquan Zhou, Jiashi Feng, Vincent Tan, and Joey Tianyi Zhou. Sharpness-aware training for free. Advances in Neural Information Processing Systems , 35:23439-23451, 2022.
- [29] Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In International conference on machine learning , pages 1225-1234. PMLR, 2016.

- [30] Zhuoning Yuan, Yan Yan, Rong Jin, and Tianbao Yang. Stagewise training accelerates convergence of testing error over sgd. Advances in Neural Information Processing Systems , 32, 2019.
- [31] Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximalgradient methods under the polyak-łojasiewicz condition. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2016, Riva del Garda, Italy, September 19-23, 2016, Proceedings, Part I 16 , pages 795-811. Springer, 2016.
- [32] Chaoqi Wang, Guodong Zhang, and Roger Grosse. Picking winning tickets before training by preserving gradient flow. In International Conference on Learning Representations , 2020.
- [33] Hesham Mostafa and Xin Wang. Parameter efficient training of deep convolutional neural networks by dynamic sparse reparameterization. In International Conference on Machine Learning (ICML) , pages 4646-4655. PMLR, 2019.
- [34] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.
- [35] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 , 2014.
- [36] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision , 115:211-252, 2015.
- [37] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261 , 2019.
- [38] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR , 2016.
- [39] Lu Yin, Gen Li, Meng Fang, Li Shen, Tianjin Huang, Zhangyang Wang, Vlado Menkovski, Xiaolong Ma, Mykola Pechenizkiy, and Shiwei Liu. Dynamic sparsity is channel-level sparsity learner. In Advances in Neural Information Processing Systems (NeurIPS) , 2023.
- [40] Siddhant Jayakumar, Razvan Pascanu, Jack Rae, Simon Osindero, and Erich Elsen. Top-kast: Top-k always sparse training. Advances in Neural Information Processing Systems , 33:2074420754, 2020.
- [41] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 4510-4520, 2018.
- [42] Mingxing Tan and Quoc Le. EfficientNet: Rethinking model scaling for convolutional neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97, pages 6105-6114. PMLR, 2019.
- [43] Zhuangwei Zhuang, Mingkui Tan, Bohan Zhuang, Jing Liu, Yong Guo, Qingyao Wu, Junzhou Huang, and Jinhui Zhu. Discrimination-aware channel pruning for deep neural networks. Advances in neural information processing systems , 31, 2018.
- [44] Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, and Jian Ren. Efficientformer: Vision transformers at mobilenet speed. Advances in Neural Information Processing Systems , 35:12934-12949, 2022.
- [45] Alex Renda, Jonathan Frankle, and Michael Carbin. Comparing rewinding and fine-tuning in neural network pruning. In International Conference on Learning Representations , 2020. arXiv:2003.02389.
- [46] Xiaohan Chen, Yu Cheng, Shuohang Wang, Zhe Gan, Zhangyang Wang, and Jingjing Liu. Earlybert: Efficient bert training via early-bird lottery tickets. ACL-IJCNLP , 2021.
- [47] Yi-Lin Sung, Varun Nair, and Colin A Raffel. Training neural networks with fixed sparse masks. Advances in Neural Information Processing Systems , 34:24193-24205, 2021.

- [48] Guillaume Bellec, David Kappel, Wolfgang Maass, and Robert Legenstein. Deep rewiring: Training very sparse deep net-works. In International Conference on Learning Representations (ICLR) , 2018.
- [49] Tim Dettmers and Luke Zettlemoyer. Sparse networks from scratch: Faster training without losing performance. arXiv preprint arXiv:1907.04840 , 2019.
- [50] Jonathan Schwarz, Siddhant Jayakumar, Razvan Pascanu, Peter E Latham, and Yee Teh. Powerpropagation: A sparsity inducing weight reparameterisation. Advances in neural information processing systems , 34:28889-28903, 2021.
- [51] Shiwei Liu, Lu Yin, Decebal Constantin Mocanu, and Mykola Pechenizkiy. Do we actually need dense over-parameterization? in-time over-parameterization in sparse training. In International Conference on Machine Learning , pages 6989-7000. PMLR, 2021.
- [52] Alexandra Peste, Eugenia Iofinova, Adrian Vladu, and Dan Alistarh. Ac/dc: Alternating compressed/decompressed training of deep neural networks. Advances in neural information processing systems , 34:8557-8570, 2021.
- [53] Alexandra Peste, Adrian Vladu, Eldar Kurtic, Christoph H Lampert, and Dan Alistarh. Cram: A compression-aware minimizer. arXiv preprint arXiv:2207.14200 , 2022.

## Appendix

## A Results of VGG-19 on CIFAR-10/100

We test S 2 -SAM on VGG-19 using CIFAR-10 and CIFAR-100. The results are shown in Table A.1. We demonstrate results on both uniform and ERK distributions, and achieve SOTA results on CIFAR10/100.

Table A.1: Test accuracy (%) of pruned VGG-19 on CIFAR-10/100.

| Datasets          | CIFAR-10          | CIFAR-10          | CIFAR-10          | CIFAR-100          | CIFAR-100          | CIFAR-100          |
|----------|----------|----------|----------|----------|----------|----------|
| Pruning ratio          | 90%          | 95%          | 98%          | 90%          | 95%          | 98%          |
| VGG-19          | 94.20          |          |          | 74.17          |          |          |
| LT [20] LT+ S 2 -SAM (ours)          | 93.51 93.82 ± 0.12 (0.31 ↑ )          | 92.92 93.41 ± 0.11 (0.49 ↑ )          | 92.34 92.92 ± 0.14 (0.58 ↑ )          | 72.78 73.08 ± 0.08 (0.30 ↑ )          | 71.44 71.69 ± 0.10 (0.25 ↑ )          | 68.95 69.78 ± 0.17 (0.83 ↑ )          |
| SNIP [21] SNIP+ S 2 -SAM (ours)          | 93.63 ± 0.06 93.91 ± 0.11 (0.28 ↑ )          | 93.43 ± 0.20 93.91 ± 0.21 (0.48 ↑ )          | 92.05 ± 0.28 92.86 ± 0.28 (0.81 ↑ )          | 72.84 ± 0.22 73.33 ± 0.21 (0.49 ↑ )          | 71.83 ± 0.23 72.41 ± 0.35 (0.58 ↑ )          | 58.46 ± 1.10 60.12 ± 0.42 (1.66 ↑ )          |
| GraSP [32] GraSP+ S 2 -SAM (ours)          | 93.30 ± 0.14 93.88 ± 0.11 (0.58 ↑ )          | 93.43 ± 0.18 93.75 ± 0.21 (0.32 ↑ )          | 92.19 ± 0.12 92.95 ± 0.23 (0.76 ↑ )          | 71.95 ± 0.18 72.44 ± 0.27 (0.49 ↑ )          | 71.23 ± 0.12 72.91 ± 0.25 (1.68 ↑ )          | 68.90 ± 0.41 69.98 ± 0.31 (1.08 ↑ )          |
| SET [2] SET+ S 2 -SAM (ours)          | 92.46 92.97 ± 0.22 (0.51 ↑ )          | 91.73 92.55 ± 0.20 (0.82 ↑ )          | 89.18 90.18 ± 0.19 (1.00 ↑ )          | 72.36 72.65 ± 0.14 (0.29 ↑ )          | 69.81 70.11 ± 0.14 (0.30 ↑ )          | 65.94 66.76 ± 0.17 (0.82 ↑ )          |
| DSR [33] DSR+ S 2 -SAM (ours)          | 93.75 94.24 ± 0.13 (0.49 ↑ )          | 93.86 94.27 ± 0.15 (0.41 ↑ )          | 93.13 93.81 ± 0.27 (0.68 ↑ )          | 72.31 72.78 ± 0.20 (0.47 ↑ )          | 71.98 72.55 ± 0.22 (0.57 ↑ )          | 70.70 71.69 ± 0.24 (0.99 ↑ )          |
| RigL [23] RigL+ S 2 -SAM (ours) RigL (ERK) [23] RigL (ERK)+ S 2 -SAM (ours)          | 93.12 93.61 ± 0.16 (0.49 ↑ ) 93.77 94.07 ± 0.21 (0.30 ↑ )          | 92.43 92.87 ± 0.21 (0.44 ↑ ) 92.75 93.33 ± 0.12 (0.58 ↑ )          | 90.65 91.88 ± 0.21 (1.23 ↑ ) 90.87 91.79 ± 0.12 (0.92 ↑ )          | 71.14 71.98 ± 0.18 (0.84 ↑ ) 71.34 72.12 ± 0.20 (0.78 ↑ )          | 69.02 70.16 ± 0.18 (1.14 ↑ ) 69.21 69.95 ± 0.18 (0.74 ↑ )          | 64.87 66.01 ± 0.25 (1.14 ↑ ) 65.02 66.13 ± 0.19 (1.11 ↑ )          |
| MEST (EM) [3] MEST (EM) + S 2 -SAM (ours) MEST (EM&S) [3] MEST (EM&S) + S 2 -SAM (ours) | 93.07 ± 0.36 93.76 ± 0.13 (0.69 ↑ ) 93.61 ± 0.36 94.51 ± 0.11 (0.90 ↑ ) | 92.59 ± 0.41 93.18 ± 0.12 (0.59 ↑ ) 93.46 ± 0.41 93.98 ± 0.11 (0.52 ↑ ) | 90.55 ± 0.44 91.86 ± 0.23 (1.31 ↑ ) 92.30 ± 0.44 91.55 ± 0.14 (0.75 ↑ ) | 71.23 ± 0.37 71.93 ± 0.11 (0.70 ↑ ) 72.52 ± 0.37 72.90 ± 0.17 (0.38 ↑ ) | 69.08 ± 0.41 69.98 ± 0.10 (0.90 ↑ ) 71.21 ± 0.41 72.42 ± 0.09 (1.21 ↑ ) | 64.92 ± 0.34 66.08 ± 0.16 (1.16 ↑ ) 69.02 ± 0.34 71.01 ± 0.13 (1.99 ↑ ) |

## B Proof of Lemma 1

Proof. We consider two cases.

- (1) When G = G ′ and ℓ ( · , f ( w ; · )) is L

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C Proof of Theorem 1

Proof. By Theorem 3.2 of [29], we have

<!-- formula-not-decoded -->

where (a) uses Lemma 1; (b) uses λ = 2 L ; (c) uses ∆ t 0 = 0 ; (d) uses ηL &lt; 1 . Then, by Lemma 3.11 of [29] and ̂ ℓ is ( L + λ ) -smooth, we have

<!-- formula-not-decoded -->

where the last inequality holds by selecting t 0 = 1 and λ = 2 L . Then, by Theorem 2.2 of [29]

<!-- formula-not-decoded -->

which is the generalization error. Next, we will bound the optimization error. Since ̂ F S is ( L + λ ) -smooth and satisfies µ -PL condition, then follow the standard analysis, we have

Since ̂ F S is ( L + λ ) -smooth, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is due to ℓ ( y , f ( w , x )) is B -Lipschitz. Therefore, we get

<!-- formula-not-decoded -->

Since ̂ F S satisfies µ -PL condition, then

<!-- formula-not-decoded -->

Then by (12), (17) and (21), we get

<!-- formula-not-decoded -->

By using the conditions that min w ∈W ̂ F S ( w ) ≤ F S ( w ∗ S ) + λ 2 ∥ w t ∥ 2 , definitions (11), we get

<!-- formula-not-decoded -->

Therefore, we have the following inequality by (22) and (23):

<!-- formula-not-decoded -->

By Lemma 5.1 of [29], (24) implies the ERB is

<!-- formula-not-decoded -->

By setting η = O (1 / √ n ) and T = O ( n ) then E R, A , S [ F ( w R )] -F ( w ∗ ) ≤ O (1 / √ n ) , where A is S 2 -SAM.

##