## Sharpness-diversity tradeoff: improving flat ensembles with SharpBalance

```
Haiquan Lu 1 ∗ , Xiaotian Liu 2 ∗ , Yefan Zhou 2 ∗ , Qunli Li 3 ∗ , Kurt Keutzer 4 , Michael W. Mahoney 4 , 5 , 6 , Yujun Yan 2 , Huanrui Yang 4 , Yaoqing Yang 2 1 Nankai University 2 Dartmouth College 3 University of California San Diego 4 University of California at Berkeley 5 International Computer Science Institute 6 Lawrence Berkeley National Laboratory
```

## Abstract

Recent studies on deep ensembles have identified the sharpness of the local minima of individual learners and the diversity of the ensemble members as key factors in improving test-time performance. Building on this, our study investigates the interplay between sharpness and diversity within deep ensembles, illustrating their crucial role in robust generalization to both in-distribution (ID) and out-of-distribution (OOD) data. We discover a trade-off between sharpness and diversity: minimizing the sharpness in the loss landscape tends to diminish the diversity of individual members within the ensemble, adversely affecting the ensemble's improvement. The trade-off is justified through our theoretical analysis and verified empirically through extensive experiments. To address the issue of reduced diversity, we introduce SharpBalance , a novel training approach that balances sharpness and diversity within ensembles. Theoretically, we show that our training strategy achieves a better sharpness-diversity trade-off. Empirically, we conducted comprehensive evaluations in various data sets (CIFAR-10, CIFAR-100, TinyImageNet) and showed that SharpBalance not only effectively improves the sharpness-diversity trade-off, but also significantly improves ensemble performance in ID and OOD scenarios. Our code has been made open-source. †

## 1 Introduction

There has been interest in understanding the properties of neural networks (NNs) and their implications for robust generalization to both in-distribution (ID) and out-of-distribution (OOD) data [Hendrycks and Dietterich, 2019a]. Two properties of particular importance, sharpness (or flatness) [Granziol, 2020, Andriushchenko et al., 2023, Yang et al., 2021, Dinh et al., 2017, Yao et al., 2020] and diversity [Laviolette et al., 2017, Fort et al., 2019, Yao et al., 2020, Dietterich, 2000, Ortega et al., 2022, Theisen et al., 2023], have been shown to have a significant influence on performance. In the context of deep ensembles [Ovadia et al., 2019, Lakshminarayanan et al., 2017, Fort et al., 2019, Mehrtash et al., 2020, Ganaie et al., 2022], diversity (which measures the variance in output between independently-trained models) is shown to be critical in enhancing ensemble accuracy. Sharpness, on the other hand, quantifies the curvature of local minima and is believed to be empirically correlated with an individual model's generalization ability.

* First four authors contributed equally.

† https://github.com/haiquanlu/SharpBalance

Figure 1: (Sharpness-diversity trade-off and SharpBalance ). (a) Caricature illustrating the sharpness-diversity trade-off that emerges in an ensemble's loss landscape induced by the Sharpnessaware Minimization ( SAM ) optimizer. We propose SharpBalance to address this trade-off. Each black circle represents an individual NN in a three-member ensemble. The distance between circles represents the diversity between NNs and the ruggedness of the basin represents the sharpness of each NN. (b) Theoretically proving the existence of the sharpness-diversity trade-off and improvement from SharpBalance , plotting the analytic representation of sharpness and diversity from Theorem 1 and Theorem 2 by changing the perturbation radius ρ of SAM . SharpBalance achieves a larger diversity for the same level of sharpness. (c) Empirical results of verifying sharpness-diversity trade-off improvement from SharpBalance . Each marker represents a three-member ResNet18 ensemble trained on CIFAR-10. Diversity is measured by the variance of individual models' predictions, and sharpness is measured by the adaptive worst-case sharpness, both defined in Section 2.

<!-- image -->

Recent research on loss landscapes [Yang et al., 2021] highlights that a single structural property of the loss landscape is insufficient to fully capture a model's generalizability, and it underscores the importance of a joint analysis of sharpness and diversity. Despite significant efforts in studying sharpness and diversity individually, a gap persists in understanding their relationship, particularly in the context of ensemble learning. Our work seeks to bridge this gap by investigating ensemble learning through the lens of loss landscapes, with a specific focus on the interplay between sharpness and diversity.

Sharpness-diversity trade-off. Our examination of loss landscape structure for ensembling revealed a 'trade-off' between the diversity of individual NNs and the sharpness of the local minima to which they converge. This trade-off introduces a potential limitation to the achievable performance of the deep ensemble: the test accuracy of individual NN can be improved as the sharpness is reduced, but it simultaneously reduces diversity, thereby compromising the ensembling improvement (evidence in Section 4.2 and 4.4). This trade-off is visually summarized in the lower transition branch in Figure 1a. We also developed theories (in Section 3) to verify the trade-off. The theoretical results characterizing this phenomenon are visualized in Figure 1b, and the experimental observation is presented in Figure 1c. In Section 4.2, we also verified the existence of the trade-off by varying the experimental setting to include different datasets and different levels of overparameterization (e.g., changing model width).

SharpBalance mitigates the trade-off and improves ensembling performance. To address the challenge presented by the sharpness-diversity tradeoff, we propose a novel ensemble training method called SharpBalance . This method aims to simultaneously reduce the sharpness of individual NNs and prevent diversity reduction among them, as demonstrated in the upper transition branch of Figure 1a. This method is designed based on our theoretical results, which suggest that training different ensemble members using a loss function that aims to reduce sharpness on different subsets of the training data can improve the trade-off between sharpness and diversity. Our theoretical results are summarized in Figure 1b. Aligned with theoretical insights, our SharpBalance method lets each ensemble member minimize the sharpness objective exclusively on a subset of training data, termed the sharpness-aware set . The sharpness-aware set of each ensemble member is diversified by an adaptive strategy based on data-dependent sharpness measures. As shown in Figure 1c, we verify

that SharpBalance improves the sharpness-diversity tradeoff in training the ResNet18 ensemble on CIFAR10. We conducted experiments on three classification datasets to show that SharpBalance boosts ensembling performance in ID and OOD data.

Our contributions are summarized as follows:

- Comprehensive identification of the sharpness-diversity trade-off: This work provides a thorough examination of the phenomenon sharpness-diversity trade-off where reducing the sharpness of individual models can decrease diversity between models within an ensemble. We demonstrate this effect through extensive experiments across various settings,using different sharpness and diversity measures, as well as different model capacities. Our findings show that this trade-off can negatively affect the ensemble improvements.
- Novel theory: We prove the existence of the trade-off under a novel theoretical framework based on rigorous analysis of sharpness-aware training objectives [Foret et al., 2021, Behdin and Mazumder, 2023]. Our analysis borrows tools from analyzing Wishart moments [Bishop et al., 2018], and characterizes the exact dynamics of training, bias-variance tradeoff, and the upper and lower bounds of sharpness. Notably, our novel theoretical analysis generalizes existing analysis to ensemble members trained with different data, which is the key to analyzing our own training method SharpBalance .
- Effective approach: To mitigate the sharpness-diversity trade-off, we introduce SharpBalance , an ensemble training approach. Our theoretical framework demonstrates that SharpBalance provably achieves improvements on the sharpness-diversity trade-off by reducing sharpness while mitigating the decrease in diversity. Empirically, we confirm this improvement and demonstrate that SharpBalance enhances overall ensemble performance, outperforming baseline methods in CIFAR-10, CIFAR-100 [Krizhevsky, 2009], TinyImageNet [Le and Yang, 2015], and their corrupted versions to assess OOD performance.

We provide a more detailed discussion on related work in Appendix B.

## 2 Background

Preliminaries. We use a NN denoted as f θ : R d in → R d out , where θ ∈ R p denotes the trainable parameters. The training dataset comprises n data-label pairs D = { ( x 1 , y 1 ) , . . . , ( x n , y n ) } . The training loss of NN f θ over a dataset D can be defined as L D ( θ ) = 1 n ∑ n i =1 ℓ ( f θ ( x i ) , y i ) . Here ℓ ( · ) is a loss function, which, for instance, can be the cross entropy loss or ℓ 2 loss. We construct a deep ensemble consisting of m distinct NNs f θ 1 , . . . , f θ m . For classification tasks, the ensemble's output is derived by averaging the predicted logits of these individual networks. We use flat ensemble to mean the deep ensemble in which each ensemble member is trained using a sharpness-aware optimization method [Foret et al., 2021], differentiating it from other ensemble approaches.

Diversity metrics. Distinct measures of diversity have been proposed in the literature [Laviolette et al., 2017, Fort et al., 2019, Dietterich, 2000, Baek et al., 2022, Ortega et al., 2022, Theisen et al., 2023], and they are primarily calculated using the predictions made by individual models. Ortega et al. [2022] define diversity D ( θ ) to be the variance of model outputs averaged over the data-generating distribution, which we adopt in the theoretical analysis:

<!-- formula-not-decoded -->

In our experiments, diversity is measured using variance defined above, as well as two other widely used metrics in ensemble learning, namely Disagreement Error Ratio (DER) [Theisen et al., 2023] defined in equation (2), and KL divergence [Kullback and Leibler, 1951] defined in equation (11) in the appendices. We show in Section 4.2 that our main claim generalizes to these three metrics in characterizing the diversity between members within an ensemble. Specifically, denote P as the distribution of model weights θ after training. Then, the DER is defined as

<!-- formula-not-decoded -->

where Dis( f θ , f θ ′ ) is the prediction disagreement [Masegosa, 2020, Mukhoti et al., 2021, Jiang et al., 2022] between two classifier f θ , f θ ′ , and E ( f θ ) is the prediction error.

Sharpness Metric. In accordance with the definition proposed by Foret et al. [2021], we characterize the first-order sharpness of a model as the worst-case perturbation within a radius of ρ 0 . Mathematically, the sharpness κ of a model θ is expressed as follows:

<!-- formula-not-decoded -->

Empirically, we measure the sharpness of the NN via the adaptive worst-case sharpness [Kwon et al., 2021, Andriushchenko et al., 2023]. The adaptive worst-case sharpness captures how much the loss can increase within the perturbation radius ρ 0 of θ :

<!-- formula-not-decoded -->

where θ = [ θ 1 , . . . , θ l ] , and T θ = diag ( | θ 1 | , . . . , | θ l | ) . T -1 θ is a normalization operator to make sharpness 'scale-free', that is, such that scaling operations on θ that do not alter NN predictions will not impact the sharpness measure.

Ensembling. We characterize the effectiveness of ensembling by the metric called ensemble improvement rate (EIR) [Theisen et al., 2023], which is defined as the ensembling improvement over the average performance of single models. Let E ens denote the test error of an ensemble; the EIR is then defined as follows:

<!-- formula-not-decoded -->

Sharpness Aware Minimization ( SAM ). SAM [Foret et al., 2021] has been shown to be an effective method for improving the generalization of NNs by reducing the sharpness of local minima. It essentially functions by penalizing the maximum loss within a specified radius ρ of the current parameter θ . The training objective of SAM is to minimize the following loss function:

<!-- formula-not-decoded -->

where λ is the hyperparameter of a standard ℓ 2 regularization term.

## 3 Theoretical Analysis of Sharpness-diversity Trade-off

This section theoretically analyzes the sharpness-diversity trade-off. The diversity among individual models is quantified using equation (1). The first theorem establishes the existence of a trade-off between sharpness and diversity. The second theorem demonstrates that training models with only a subset of data samples leads to a more favorable trade-off between these two metrics.

Sharpness and Diversity of SAM . Assume the training data matrix A ∈ R n tr × d in and test data matrix T ∈ R n te × d in are random with entries drawn from Gaussian N (0 , 1 d in I ) . Suppose the model weight at the 0-th time step, θ 0 , is initialized randomly such that E [ θ 0 ] = 0 and E [ θ 0 θ T 0 ] = σ 2 I and updated with a quadratic optimization objective through SAM . The learned weight matrix after k time steps is denoted as θ k . Let θ ∗ be the teacher model (i.e., ground-truth model) such that A θ ∗ = y ( A ) and T θ ∗ = y ( T ) , where y ( D ) is the label vector for data matrix D . Given a perturbation radius ρ 0 , the sharpness of a model after k iteration under the random matrix assumption is defined as

<!-- formula-not-decoded -->

which is the expected fluctuation of the model output after perturbation over the data distribution. For simplicity, we denote κ ( θ SAM k ; ρ 0 ) = κ SAM k for the rest of the paper. We derive an explicit formulation of diversity and upper and lower bounds of sharpness for models optimized with SAM in Theorem 1. Detailed proof can be found in Appendix C.1.

Theorem 1 (Diversity and Sharpness of SAM ) . Let θ 0 be initialized randomly such that E [ θ 0 ] = 0 and E [ θ 0 θ T 0 ] = σ 2 I . Suppose θ SAM k is the model weight after k iterations of training with SAM on A ∈ R n tr × d in and evaluated on T ∈ R n te × d in . Let η be the step size, ρ be the perturbation radius in SAM and ρ 0 be the radius for measuring sharpness κ SAM k . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Diversity

•-- Lower bound of sharpness (x-axis)

Upper bound of sharpness (x-axis)

1000°

10-

0-

Simulated p-0.5

p=2.5e-1

Lower Bound on Sharpness where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To provide a clearer understanding of the relationship between sharpness and diversity, Figure 2 presents a trade-off curve between these two metrics. The estimated sharpness and diversity are displayed on the x and y axes, respectively. Each point in the plot corresponds to a model trained using SAM with a different ρ value, showcasing the outcome of varying perturbation radius. In these experiments, we evaluated the sharpness and diversity of the models empirically and compared them to the estimates obtained using Theorem 1. The soundness of Theorem 1 and tightness of the derived bounds are further supported by empirical evidence, as depicted in Figure 2. Further verification results supporting our theoretical analysis are provided in Appendix C.3

Training with Data Subsets. Assume A is partitioned into S horizontal submatrices, such that A = [ A T 1 , A T 2 , . . . , A T S ] T . We show in Theorem 2 a similar analysis of the sharpness and diversity of ensembles for which each model is trained with a submatrix. Under this setting, we first selected a subset of data A s uniformly at random and then train the model with the selected subset with SAM .

Theorem 2 (Diversity and Sharpness when Models are Trained on Subsets) . Suppose the training data matrix A is partitioned into S horizontal submatrices. Let θ 0 be initialized randomly such that E [ θ 0 ] = 0 and E [ θ 0 θ T 0 ] = σ 2 I . Let θ SharpBal k be the model weight trained with SAM for k iterations on the submatrix A s ∈ R n tr S × d in , selected uniformly at random, and evaluated on test data T ∈ R n te × d in . Let η be the step size, ρ be the perturbation radius in SAM , ρ 0 be the radius for measuring sharpness κ SAM k , and r = n tr Sd in . Then

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of Theorem 2 is provided in Appendix C.2. Similar experimental validations are conducted to verify Theorem 2, with results also presented in Appendix C. The main insight from Theorem 2 is

Figure 2: (Theoretical vs. Simulated sharpness-diversity trade-off) . This figure illustrates the relationship between sharpness (upper and lower bounds) and diversity as predicted by Thereom 1 and as observed in simulations. Note that the upper and lower bounds correspond to the sharpness values plotted along the x-axis, with the upper bound positioned to the right and the lower bound to the left. Also, note that the bounds provided are for the expected sharpness, which means that random fluctuations can cause the simulation results to move beyond these bounds.

<!-- image -->

ó p-0.1

1

that training models on a randomly selected data subset offers a better trade-off between sharpness and diversity compared to training on the complete dataset. This idea is further illustrated in Figure 1b, where we compare the sharpness upper bound and diversity of models trained on the full dataset (labeled as SAM ) and those trained on subsets (labeled as SharpBalance ). The results demonstrate that SharpBalance achieves a more favorable trade-off. For a given level of sharpness, deep ensembles with models trained on subsets of the data exhibit higher diversity compared to those trained on the entire dataset. This indicates that minimizing sharpness on randomly sampled data subsets for each model within the ensemble promotes the diversity among the models, thereby enhancing the sharpness-diversity trade-off.

## 4 Experiments

In this section, we describe our experiments. In particular, following Section 4.1 where we describe our experimental setup, in Section 4.2, we provide an empirical evaluation across various datasets to explore the trade-off between sharpness and diversity. We also examine how this trade-off changes with different levels of overparameterization. Then, in Section 4.3 and 4.4, we elaborate the SharpBalance algorithm and compare its performance with baseline methods.

## 4.1 Experimental setup

Here, we describe the experiment setup for Section 4.2. Each ensemble member is trained individually using SAM with a consistent perturbation radius ρ , as defined in equation (5). We adjust ρ across different ensembles to achieve varying levels of minimized sharpness. Sharpness for each NN was measured using the adaptive worst-case sharpness metric, defined in equation (3). The sharpness measurement was done on the training set, using 100 batches of size 5. The diversity between NNs is measured using DER defined in equation (2). The diversity between ensemble members is tested on OOD data. We evaluated this trade-off using a variety of image classification datasets, including CIFAR-10, CIFAR-100 [Krizhevsky, 2009], TinyImageNet [Le and Yang, 2015], and their corrupted versions [Hendrycks and Dietterich, 2019b]. For the setup of Section 4.4, we used the same datasets and architecture. The hyperparameters of the baseline methods has been carefully tuned. The hyperparameters for conducting the experiments are detailed in Appendix D.

<!-- image -->

- (a) Measuring diversity via Variance

(b) Measuring diversity via DER

- (c) Measuring diversity via KL

Figure 3: (Varying diversity measure in empirical study). Three different metrics are employed to measure the diversity of individual models within an ensemble, i.e., Variance in equation (1), DER in equation (2), and KL divergence in equation (11). The results of the three metrics show consistent trends, demonstrating the sharpness-diversity trade-off: lower sharpness is correlated with lower diversity. The experiment is conducted by training a three-member ResNet18 ensemble on CIFAR10.

## 4.2 Empirical validation of Sharpness-diversity trade-off

We provide empirical observation to validate and explore the sharpness-diversity trade-off. Figure 3 presents the validation of observing the trade-off phenomenon on training ResNet18 ensembles on CIFAR10 applying three different metrics to measure the diversity. The results demonstrate that this trade-off phenomenon generalizes to the three diversity metrics defined in Section 2. Figure 4 presents the validation on three different datasets. In the following empirical study, DER will be the primary metric for measuring diversity of models.

Diver

Ensemble Improvement Rate

0.03

0.04

0.05

Ensemble improvement diversity decreases

0.4-

0.2

decrease while

0.4

0.6

Sharpness

Individual OOD Acc

74.75

75.00

75.25

Individual performance sharpness decreases

increase while

0.4

Sharpness

0.6

Figure 4: (Empirical observations of sharpness-diversity trade-off). The identified trade-off shows that while reducing sharpness enhances individual model performance, it concurrently lowers diversity and thus diminishes the ensemble improvement rate. First row : the color encoding represents the ensemble improvement rate (EIR) defined in equation (4), from red to blue means ensembling improvement decreases. Second row : the color encoding represents the individual ensemble member's OOD accuracy, from blue to red means individual performance becomes better. Each marker represents a three-member ResNet18 ensemble trained with SAM with a different perturbation radius.

<!-- image -->

Experimental results obtained with the other two metrics are available in Appendix E. The three sets of results first verify that minimizing individual member's sharpness indeed reduces diversity. This is confirmed by the consistent trends of markers moving from upper right to lower left. Second, the first row of Figure 4 shows that an ensemble with decreased diversity (lower in y -axis) shows a lower ensemble improvement rate (from red to blue), highlighting the negative impact of this trade-off. Lastly, the second row shows when the sharpness of the individual model is reduced (lower in x -axis), the individual model's OOD accuracy is improved (from blue to red), demonstrating the benefits of minimizing sharpness. We verify the robustness of the phenomenon by measuring the sharpness and diversity using different metrics in Appendix E.

Figure 5 illustrates the trade-off curves as the overparameterization level of the model is adjusted by changing width or sparsity (introduced using model pruning). This visualization confirms that the trade-off is a consistent phenomenon across models of different sizes, and the ensemble provides less improvement (blue color) at the lower left end of each trade-off curve. It also highlights that models with smaller or sparser configurations show a more significant trade-off effect, as evidenced by the steeper slopes and higher coefficient values of the linear fitting curves. As sparse ensembles are now being used to demonstrate the benefits of ensembling for efficient models [Liu et al., 2022, Diffenderfer et al., 2021, Whitaker and Whitley, 2022, Kobayashi et al., 2022, Zimmer et al., 2024], addressing the conflict between sharpness and diversity becomes particularly crucial.

## 4.3 Our SharpBalance method

Here, we describe the design and implementation of our main method, SharpBalance . Figure 6 provides an overview. Our approach is motivated by the theoretical analysis in Section 3, which suggests that having each ensemble member minimize sharpness on diverse subsets of the data can lead to a better trade-off between sharpness and diversity. SharpBalance aims to achieve the optimal balance by applying SAM to a carefully selected subset of the data, while performing standard optimization on the remaining samples. More specifically, for each ensemble member NN f θ i , our method divides the entire training dataset D into two distinct subsets: sharpness-aware set D i SAM and normal set D i Normal . The model is trained to optimize the sharpness reduction objective on D i SAM ,

0.2

Ensemble Improvement Rate

0.8

0.04

0.05

3 0.6- decrease while

Ensemble Improvement Rate

0.9 0.035

0.040

0.045

0.8-

0.7-°

decrease while

Figure 5: (Sharpness-diversity trade-off in models varying overparameterization levels). Different types of markers represent models with varying degrees of overparameterization, determined by changing the model width (a) or sparsity (b). Each marker represents a three-member ensemble trained with SAM with a different perturbation radius. The β reflects the rate of decline in the trade-off curve, calculated via applying linear fitting over the ensembles at each level of overparameterization. A higher β points to a steeper decline in the trade-off. Ensembles with narrower widths or increased sparsity display more pronounced trade-off effects. The model used in ResNet18 and the dataset is CIFAR-10.

<!-- image -->

while it optimizes the normal training objective on D i Normal . These training objectives are denoted as L SAM D i SAM ( θ i ) and L D i Normal ( θ i ) , respectively. The D i SAM is selected by an adaptive strategy from the whole dataset D : it is composed of the union of samples that are deemed 'sharp' by all other members of the ensemble except the i -th. Specifically, for each model, we pick the subset of data samples with the topk %highest 'per-data-sample sharpness.' Then, we take the union of all such subsets expect the i -th for creating the subset D i SAM . This partition of data samples can be efficiently computed in parallel as there is no sequential dependency on the training of the ensemble members. However, SharpBalance can be easily adapted for sequential training if memory constraints permit training only one model at a time.

Figure 6: (System diagram of SharpBalance ). Each ensemble member f θ i optimizes the sharpness reduction objective on subset D i SAM and the normal training objective on D i Normal . D i SAM is formed by selecting data samples from D that significantly affect the loss landscape sharpness of other ensemble members.

<!-- image -->

Per-data-sample sharpness. This metric is designed to efficiently assess the sharpness of a model for individual data samples. For each data point ( x j , y j ) , sharpness is quantified using the Fisher Information Matrix (FIM), which is expressed as ∇ θ ℓ ( f θ ( x j ) , y j ) ∇ θ ℓ ( f θ ( x j ) , y j ) T . Following a well-established approach [Bottou et al., 2018], we approximate the trace of the FIM by computing the squared ℓ 2 norm of the gradient: ∥∇ θ ℓ ( f θ ( x j ) , y j ) ∥ 2 2 . Other common sharpness metrics, such as worst-case sharpness, trace of the Hessian, or Hessian eigenvalues, are computationally slightly more expensive to approximate [Yao et al., 2020, 2021], but are expected to lead to similar results.

## 4.4 Empirical evaluation of SharpBalance

Figure 7: (Main results: SharpBalance improves the overall ensembling performance and mitigates the reduced ensembling improvement caused by sharpness-diversity trade-off). The three-member ResNet18 ensemble is trained with different methods on three datasets. The first row reports the OOD accuracy and the second row reports the ID accuracy. The lower part of each bar with the diagonal lines represents the individual model performance. The upper part of each bar represents the ensembling improvement. The results are reported by averaging three ensembles, and each ensemble is comprised of three models.

<!-- image -->

We evaluate SharpBalance by benchmarking it against both a standard Deep Ensemble, trained using SGD, and a Deep Ensemble enhanced with SAM . The results are presented in Figure 7 for CIFAR-10, CIFAR-100, and TinyImageNet. The comparison between the middle and left bars shows that SAM enhances individual model performance by reducing sharpness. However, this reduction in sharpness also diminishes the overall ensemble effectiveness by lowering diversity, exemplifying the sharpness-diversity trade-off discussed in Section 4.2. Further comparison between the right and middle bars shows that SharpBalance maintains or improves individual performance while improving ensemble effectiveness.

We also evaluate SharpBalance on different ensemble sizes. As shown in Figure 8, SharpBalance demonstrates more pronounced empirical improvements as the number of ensemble models increases. The accuracy difference between SharpBalance and the baseline methods becomes more significant, especially on corrupted data. Specifically, SharpBalance outperforms the baselines by up to 1.30% when ensembling 5 models on CIFAR100-C dataset.

To further evaluate SharpBalance , we provide corroborating results in Appendix F, which includes:

- We evaluate SharpBalance on different severity of the corruption on CIFAR10-C, CIFAR100-C and Tiny-ImageNet-C. SharpBalance increasingly outperforms the baselines as the severity of the corruption increases. We also evaluate the proposed method using uncertainty metrics such as negative log-likelihood and expected calibration error.
- We further evaluate SharpBalance on other model architectures and tasks, such as WideResNet, ViT, and ALBERT [Lan et al., 2020] on language tasks.
- We compare our method of measuring sharpness with another method of measuring the curvature of the loss around a data point [Garg and Roy, 2023] and show the strong correlation between these two methods.
- We further compare SharpBalance with ensemble baseline EoA [Arpit et al., 2022], an improved version of SAM (for which individual models in an ensemble are trained with different ρ values) and GSAM [Zhuang et al., 2022]. Results show that SharpBalance can significantly outperform the baselines.

96.50

296.25-

·96.00-

· 0006-

0.004-

2

CIFAR10

I

I

2

3

4

5

Number of models

CIFAR10

3

4

5

Number of models

Deep ensemble + SAM

CIFAR 100

78

77

76-

1

I

SharpBalance

CIFAR10-C

I

1

52

I

<!-- image -->

Accuray

Accuray

Figure 8: SharpBalance achieves more pronounced improvement when increasing the number of ensembling models. "EIR" represents the ensemble inprovement rate, which is defined in Section 2, the larger the better. x -axis represents the number of individual models in one ensemble.

- We demonstrate that, compared to training a deep ensemble with SAM , our method adds only minimal computational cost. The extra time complexity is dominated by the computation of Fisher trace for evaluating per-sample sharpness, which empirically increases the training time by 1%.

## 5 Conclusion

Our theoretical and empirical analyses demonstrate the existence of a sharpness-diversity trade-off when sharpness-minimization training methods are applied to deep ensembles. This leads to two main insights that are relevant for improving model performance. First, reducing the sharpness in individual models proves to be beneficial in enhancing the performance of the ensemble as a whole. Second, the accompanying reduction in diversity suggests that popular ensembling methods have limitations, and also highlights the potential for more sophisticated designs that promote diversity among models with lower sharpness. These results are particularly timely, given recent theoretical work on characterizing ensemble improvement [Theisen et al., 2023]. In response to these findings, we have proposed SharpBalance , which 'diagnoses' the training data by evaluating the sharpness of each sample and then fine-tunes the training of individual models to focus on a diverse subset of the sharpest training data samples. This targeted approach helps maintain diversity among models while also reducing their individual sharpness. Extensive evaluations indicate that SharpBalance not only improves the sharpness-diversity trade-off but also delivers superior OOD performance for both dense and sparse models across various datasets and architectures when compared to other ensembling approaches.

Limitations. One limitation of the study is that our theoretical analysis in Section 3 relies on the assumption that the data matrices A , T follow a Gaussian distribution and assumed the optimization objective to be quadratic, which may not always hold in practice. Despite the potentially strong assumptions, our empirical findings in Section 4 show that the conclusions remain robust in real-world datasets with various model architectures. This suggests the insights discovered in our study are applicable to a wider range of real-world scenarios, beyond just those strictly adhering to the Gaussian assumption. Nevertheless, future research could explore how such assumptions can be relaxed and extend the theoretical analysis to a weaker condition.

Acknowledgements. Michael W. Mahoney would like to acknowledge the UC Berkeley CLTC, ARO, IARPA (contract W911NF20C0035), NSF, and ONR for providing partial support of this work. Kurt Keutzer would like to acknowledge support from Berkeley Deep Drive. Yaoqing Yang would like to acknowledge support from DOE under Award Number DE-SC0025584, DARPA under Agreement

CIFAR 100-C

I

1

number HR00112490441, and Dartmouth College. Our conclusions do not necessarily reflect the position or the policy of our sponsors, and no official endorsement should be inferred.

## References

- Maksym Andriushchenko, Francesco Croce, Maximilian Mueller, Matthias Hein, and Nicolas Flammarion. A modern look at the relationship between sharpness and generalization. In International Conference on Machine Learning , 2023.
- Devansh Arpit, Huan Wang, Yingbo Zhou, and Caiming Xiong. Ensemble of averages: Improving model selection and boosting performance in domain generalization. Advances in Neural Information Processing Systems , 2022.
- Christina Baek, Yiding Jiang, Aditi Raghunathan, and J Zico Kolter. Agreement-on-the-line: Predicting the performance of neural networks under distribution shift. Advances in Neural Information Processing Systems , 35:19274-19289, 2022.
- Kayhan Behdin and Rahul Mazumder. On statistical properties of sharpness-aware minimization: Provable guarantees. arXiv preprint arXiv:2302.11836 , 2023.
- Adrian N Bishop, Pierre Del Moral, Angèle Niclas, et al. An introduction to wishart matrix moments. Foundations and Trends® in Machine Learning , 11(2):97-218, 2018.
- Léon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. SIAM review , 60(2):223-311, 2018.
- Leo Breiman. Random forests. Machine learning , 45(1):5-32, 2001.
- Anh Bui, Vy Vo, Tung Pham, Dinh Phung, and Trung Le. Diversity-aware agnostic ensemble of sharpness minimizers. arXiv preprint arXiv:2403.13204 , 2024.
- Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park. Swad: Domain generalization by seeking flat minima. Advances in Neural Information Processing Systems , 34:22405-22418, 2021.
- Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining , pages 785-794, 2016.
- Thomas G Dietterich. Ensemble methods in machine learning. In International workshop on multiple classifier systems , pages 1-15. Springer, 2000.
- James Diffenderfer, Brian Bartoldson, Shreya Chaganti, Jize Zhang, and Bhavya Kailkhura. A winning hand: Compressing deep networks can improve out-of-distribution robustness. Advances in neural information processing systems , 34:664-676, 2021.
- Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep nets. In International Conference on Machine Learning , pages 1019-1028. PMLR, 2017.
- Jiawei Du, Hanshu Yan, Jiashi Feng, Joey Tianyi Zhou, Liangli Zhen, Rick Siow Mong Goh, and Vincent Tan. Efficient sharpness-aware minimization for improved training of neural networks. In International Conference on Learning Representations , 2024.
- Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. In International Conference on Learning Representations , 2021.
- Stanislav Fort, Huiyi Hu, and Balaji Lakshminarayanan. Deep ensembles: A loss landscape perspective. arXiv preprint arXiv:1912.02757 , 2019.
- Yoav Freund. Boosting a weak learning algorithm by majority. Information and computation , 121(2): 256-285, 1995.
- Yoav Freund and Robert E Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences , 55(1):119-139, 1997.
- Mudasir A Ganaie, Minghui Hu, AK Malik, M Tanveer, and PN Suganthan. Ensemble deep learning: A review. Engineering Applications of Artificial Intelligence , 115:105151, 2022.

- Xiang Gao, Meera Sitharam, and Adrian E. Roitberg. Bounds on the jensen gap, and implications for mean-concentrated distributions. The Australian Journal of Mathematical Analysis and Applications , 2019.
- Isha Garg and Kaushik Roy. Samples with low loss curvature improve data efficiency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 20290-20300, 2023.
- Diego Granziol. Flatness is a false friend. arXiv preprint arXiv:2006.09091 , 2020.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations , 2019a.
- Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations , 2019b.
- Geoffrey E. Hinton and Drew van Camp. Keeping the neural networks simple by minimizing the description length of the weights. In Proceedings of the Sixth Annual Conference on Computational Learning Theory , 1993.
- Sepp Hochreiter and Jürgen Schmidhuber. Flat Minima. Neural Computation , 9(1):1-42, 1997.
- Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. In 34th Conference on Uncertainty in Artificial Intelligence 2018, UAI 2018 , pages 876-885. Association For Uncertainty in Artificial Intelligence (AUAI), 2018.
- Weisen Jiang, Hansi Yang, Yu Zhang, and James Kwok. An adaptive policy to employ sharpnessaware minimization. In International Conference on Learning Representations , 2023.
- Yiding Jiang, Vaishnavh Nagarajan, Christina Baek, and J Zico Kolter. Assessing generalization of SGD via disagreement. In International Conference on Learning Representations , 2022.
- Jean Kaddour, Linqing Liu, Ricardo Silva, and Matt J Kusner. When do flat minima optimizers work? Advances in Neural Information Processing Systems , 35:16577-16595, 2022.
- Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. In International Conference on Learning Representations , 2016.
- Sosuke Kobayashi, Shun Kiyono, Jun Suzuki, and Kentaro Inui. Diverse lottery tickets boost ensemble from a single pretrained model. In Proceedings of BigScience Episode #5 - Workshop on Challenges &amp; Perspectives in Creating Large Language Models , 2022.
- Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009.
- S. Kullback and R. A. Leibler. On Information and Sufficiency. The Annals of Mathematical Statistics , 22(1):79 - 86, 1951. doi: 10.1214/aoms/1177729694.
- Jungmin Kwon, Jeongseop Kim, Hyunseo Park, and In Kwon Choi. Asam: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks. In International Conference on Machine Learning , pages 5905-5914. PMLR, 2021.
- Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems , 30, 2017.
- Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. In International Conference on Learning Representations , 2020.

- François Laviolette, Emilie Morvant, Liva Ralaivola, and Jean-Francis Roy. Risk upper bounds for general ensemble methods with an application to multiclass classification. Neurocomputing , 219: 15-25, 2017.
- Ya Le and Xuan S. Yang. Tiny imagenet visual recognition challenge. 2015.
- Yoonho Lee, Huaxiu Yao, and Chelsea Finn. Diversify and disambiguate: Learning from underspecified data. In ICML 2022: Workshop on Spurious Correlations, Invariance and Stability , 2022.
- Shiwei Liu, Tianlong Chen, Zahra Atashgahi, Xiaohan Chen, Ghada Sokar, Elena Mocanu, Mykola Pechenizkiy, Zhangyang Wang, and Decebal Constantin Mocanu. Deep ensembling with no overhead for either training or testing: The all-round blessings of dynamic sparsity. In International Conference on Learning Representations , 2022.
- Andres Masegosa. Learning under model misspecification: Applications to variational and ensemble methods. Advances in Neural Information Processing Systems , 33:5479-5491, 2020.
- Alireza Mehrtash, Purang Abolmaesumi, Polina Golland, Tina Kapur, Demian Wassermann, and William Wells. Pep: Parameter ensembling by perturbation. Advances in neural information processing systems , 33:8895-8906, 2020.
- Jishnu Mukhoti, Andreas Kirsch, Joost van Amersfoort, Philip HS Torr, and Yarin Gal. Deterministic neural networks with appropriate inductive biases capture epistemic and aleatoric uncertainty. arXiv preprint arXiv:2102.11582 , 2021.
- Behnam Neyshabur, Srinadh Bhojanapalli, and Nathan Srebro. A PAC-bayesian approach to spectrally-normalized margin bounds for neural networks. In International Conference on Learning Representations , 2018.
- Luis A Ortega, Rafael Cabañas, and Andres Masegosa. Diversity and generalization in neural network ensembles. In International Conference on Artificial Intelligence and Statistics , pages 11720-11743. PMLR, 2022.
- Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. Advances in neural information processing systems , 32, 2019.
- Tianyu Pang, Kun Xu, Chao Du, Ning Chen, and Jun Zhu. Improving adversarial robustness via promoting ensemble diversity. In International Conference on Machine Learning , pages 4970-4979. PMLR, 2019.
- Jack Parker-Holder, Luke Metz, Cinjon Resnick, Hengyuan Hu, Adam Lerer, Alistair Letcher, Alexander Peysakhovich, Aldo Pacchiano, and Jakob Foerster. Ridge rider: Finding diverse solutions by following eigenvectors of the hessian. Advances in Neural Information Processing Systems , 33:753-765, 2020.
- Alexandre Rame, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, Patrick Gallinari, and Matthieu Cord. Diverse weight averaging for out-of-distribution generalization. Advances in Neural Information Processing Systems , 35:10821-10836, 2022.
- Ryan Theisen, Hyunsuk Kim, Yaoqing Yang, Liam Hodgkinson, and Michael W Mahoney. When are ensembles really effective? Advances in neural information processing systems , 2023.
- Giorgio Valentini and Thomas G Dietterich. Low bias bagged support vector machines. In Proceedings of the 20th International Conference on Machine Learning (ICML-03) , pages 752-759, 2003.
- Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018.
- Tim Whitaker and Darrell Whitley. Prune and tune ensembles: low-cost ensemble learning with sparse independent subnetworks. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 8638-8646, 2022.

- Yaoqing Yang, Liam Hodgkinson, Ryan Theisen, Joe Zou, Joseph E Gonzalez, Kannan Ramchandran, and Michael W Mahoney. Taxonomizing local versus global structure in neural network loss landscapes. Advances in Neural Information Processing Systems , 34:18722-18733, 2021.
- Zhewei Yao, Amir Gholami, Qi Lei, Kurt Keutzer, and Michael W Mahoney. Hessian-based analysis of large batch training and robustness to adversaries. Advances in Neural Information Processing Systems , 31, 2018.
- Zhewei Yao, Amir Gholami, Kurt Keutzer, and Michael W Mahoney. Pyhessian: Neural networks through the lens of the hessian. In 2020 IEEE international conference on big data (Big data) , pages 581-590. IEEE, 2020.
- Zhewei Yao, Amir Gholami, Sheng Shen, Mustafa Mustafa, Kurt Keutzer, and Michael Mahoney. Adahessian: An adaptive second order optimizer for machine learning. In proceedings of the AAAI conference on artificial intelligence , volume 35, pages 10665-10673, 2021.
- Juntang Zhuang, Boqing Gong, Liangzhe Yuan, Yin Cui, Hartwig Adam, Nicha C Dvornek, James s Duncan, Ting Liu, et al. Surrogate gap minimization improves sharpness-aware training. In International Conference on Learning Representations , 2022.
- Max Zimmer, Christoph Spiegel, and Sebastian Pokutta. Sparse model soups: A recipe for improved pruning via model averaging. In International Conference on Learning Representations , 2024.

## Appendix

## A Impact Statement

This paper uncovers a trade-off between sharpness and diversity in deep ensembles and introduces a novel training strategy to achieve an optimal balance between these two crucial metrics. While the proposed method could potentially be misused for malicious purposes, we believe that the study itself does not pose any direct negative societal impact. More importantly, this research advances the field of ensemble learning and contribute to the development of more reliable deep ensemble models. These advancements consequently result in enhanced robustness when dealing with OOD data and enable the quantification of uncertainty, thereby strengthening the reliability and applicability of deep learning systems in real-world scenarios.

## B Related work

Ensembling . Diversity is one of the major factors that contribute to the success of the ensembling method. Popular ensemble techniques have been developed for tree-type individual learners, which are known to have a high variance. This is evident such as in [Breiman, 2001, Chen and Guestrin, 2016, Freund, 1995, Freund and Schapire, 1997]. In contrast, more stable algorithms, such as support vector machines (SVM) type learners, are less commonly used for ensembles, unless they are tuned to a low-bias, high-variance regime, as explored in [Valentini and Dietterich, 2003]. When it comes to diversity and ensembling, NNs are known to exhibit properties different than traditional models, e.g., as described in recent theoretical and empirical work on loss landscapes and emsemble improvement [Theisen et al., 2023, Yang et al., 2021]. Therefore, ensembling techniques that work well for traditional models (e.g., tree-type models) often underperform the simple yet efficient deep ensembles method [Fort et al., 2019, Ortega et al., 2022] that uses the independent initialization and optimization. Previous literature has explored various new methods to learn diverse NNs [Lee et al., 2022, Rame et al., 2022, Pang et al., 2019, Parker-Holder et al., 2020]. Our work is different from previous work in that we study flat ensembles obtained from sharpness-aware training methods, especially focusing on diversifying flat ensembles by reducing the overlap between sharpness-aware data subsets. While our work demonstrates significant improvements in OOD generalization, it is known that (in some cases, see also Theisen et al. [2023]) deep ensembling is a simple, yet effective method to improve OOD performance [Diffenderfer et al., 2021]. Therefore, we compare the OOD performance of SharpBalance to deep ensembles.

Sharpness and generalization . A large body of work has studied the relationship between the sharpness (or flatness) of minima and the generalizability of models [Hochreiter and Schmidhuber, 1997, Hinton and van Camp, 1993, Keskar et al., 2016, Neyshabur et al., 2018, Yang et al., 2021, Kaddour et al., 2022, Yao et al., 2021, 2020]. Works such as those by [Hochreiter and Schmidhuber, 1997] and [Hinton and van Camp, 1993] use Bayesian learning and minimum description length to explain why we should train models to flat minima. [Keskar et al., 2016] introduces a sharpness-based metric, demonstrating how large-batch training can skew NNs towards sharp local minima, adversely affecting generalization. In addition, [Neyshabur et al., 2018] uses a PAC-Bayesian framework to prove bounds on generalization, which can be interpreted as the relationship between sharpness and test accuracy. Furthermore, [Cha et al., 2021] presents a theoretical exploration of the link between the sharpness of minima and OOD generalization.

Motivated by the good generalization property of flat minima, variants of sharpness-guided optimization techniques have been proposed [Yao et al., 2018, 2021, Du et al., 2024, Jiang et al., 2023], including sharpness-aware minimization [Foret et al., 2021]. The DiWA method [Rame et al., 2022] observed that SAM can decrease the diversity of models in the context of weight averaging (WA) [Izmailov et al., 2018]. However, WA imposes constraints on different models, requiring them to share the same initialization and stay close to each other in the parameter space. In contrast, our work focuses on deep ensembles that do not pose additional constraints on the training trajectories of individual ensemble members. Previous work by [Behdin and Mazumder, 2023] provided a theoretical characterization of important statistical properties for kernel regression models and single-layer ReLU networks, optimized using SAM on noisy datasets. Our theoretical analysis borrows ideas from [Behdin and Mazumder, 2023] and extends the analysis using random matrix theory. DASH was proposed in [Bui et al., 2024] to minimize the generalization loss by adding KL divergence constraint

on the output logits of ensemble members. The authors believe that the decrease in diversity is a result of models being initialized closely and updated with the same direction. In contrast, SharpBalance observed that the sharpness-diversity trade-off is ubiquitous across various settings and provides a rigorous theoretical quantification that characterizes the interplay of the two metrics. Compare to DASH, SharpBalance provably achieves improved performance and is simple, effective, and computationally cheap to implement.

## C Proof of Theorems in Section 3

Recall that SAM updates the model weights, ignoring the normalization constant and regularization, through the following recursive rule

<!-- formula-not-decoded -->

We first show an unrolling of the iterative optimization on a quadratic objective.

Theorem 3 (Unrolling SAM ) . Let θ ∗ be the teacher model. Let θ 0 be randomly initialized and updated with SAM to solve a quadratic objective L A ( θ ) = 1 2 ( θ -θ ∗ ) T A T A ( θ -θ ∗ ) . Then,

<!-- formula-not-decoded -->

where B = I -η A T A -ηρ ( A T A ) 2 .

Proof. The gradient of the objective f is given by ∇ f ( θ ) = A T A ( θ -θ ∗ ) . Therefore,

<!-- formula-not-decoded -->

With SAM update,

<!-- formula-not-decoded -->

where the last equation is obtained by recursively unrolling the weight by previous updates.

Theorem 3 offers a valuable tool to analyze the statistical behavior of the models optimized by SAM . However, one more ingredient is required to arrive at the interesting conclusions claimed in Section 3, the random matrix theory. Recall that the data matrix A ∈ R n tr × d in is random with entries drawn from Gaussian N (0 , I /d in ) . As a result, entries in A T A follows the Wishart distribution and according to Corollary 3.3 in Bishop et al. [2018], for k ≥ 1 ,

<!-- formula-not-decoded -->

where N k,i = 1 i ( k -1 i -1 )( k i -1 ) is the Narayana number. With the help of this Corollary, we now prove a proposition on the expectation of B k .

Proposition 1 (Expectation of Wishart Moments) . Let i, j be non-negative integers, then

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Proof. By Multinomial Theorem,

<!-- formula-not-decoded -->

Let m = k 2 +2 k 3 + j and taking the expectation with equation (6) gives

<!-- formula-not-decoded -->

If j = 0 , then there is a case when k 2 = k 3 = 0 , and the expectation of ( A T A ) 0 simply becomes I . Therefore,

<!-- formula-not-decoded -->

## C.1 Proof of Theorem 1

In this subsection, we show a proof for Theorem 1.

Proof. Apply Singular Value Decomposition (SVD) to obtain A = V Σ U T and A T A = U Σ 2 U T . Let D = Σ 2 . By Theorem 3,

<!-- formula-not-decoded -->

As a result, E θ 0 [ θ SAM k ] = θ ∗ -( I -η A T A -ηρ ( A T A ) 2 ) k θ ∗ = θ ∗ -B k θ ∗ . By definition,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since E T [ T T T ] = n te d in I and E [ θ 0 θ T 0 ] = σ 2 I . Hence,

<!-- formula-not-decoded -->

Recall that given a perturbation radius ρ 0 , the sharpness is defined as

<!-- formula-not-decoded -->

We first compute

<!-- formula-not-decoded -->

Similarly,

<!-- formula-not-decoded -->

Let λ min be the least eigenvalue of A T A . By subtracting equation (7) with equation (8), we have

<!-- formula-not-decoded -->

The smallest singular value λ min of a random n × d in matrix A can be bounded by the following inequality on the smallest singular value σ min ( A ) by Vershynin [2018], assuming n tr ≥ d in, then almost surely

<!-- formula-not-decoded -->

Therefore, E A [ λ min ] ≥ E A [ σ min ( A )] 2 ≥ (√ n tr d in -1 ) 2 . Now we show a lower bound on E A [ ρ 0 ∥ B k A T A θ ∗ ∥ 2 ] . By Gao et al. [2019], the Jensen gap ( E [ Z ]) 1 / 2 -E [( Z ) 1 / 2 ] is upper bounded by Var ( Z ) 2 when Z is non-negative and E [ Z ] = 1 . Notice that

<!-- formula-not-decoded -->

By normalizing Z and applying the Jensen gap upperbound, we have

<!-- formula-not-decoded -->

As a result,

<!-- formula-not-decoded -->

The derivation of the upper bound follows from a similar proof, ignoring the Jensen gap.

## C.2 Proof of Theorem 2

Below we show a proof of Theorem 2.

Proof. We apply SVD to A s to obtain A s = V s Σ s U T s and A T s A = U s Σ 2 s U T s . Let D s = Σ 2 s and B s = I -η A T s A s -ηρ ( A T s A s ) 2 . By Theorem 3 and a similar derivation in the proof of Theorem 1,

<!-- formula-not-decoded -->

Applying Proposition 1, we have where

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last equality is the result of applying E A [ B i s ] = ϕ ′ ( i, 0) with different combinations of B s , B s ′ , counting multiplicity. Similarly,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

When the model is trained on the submatrix, the sharpness of model θ SharpBal k is defined as

<!-- formula-not-decoded -->

From a similar analysis of the proof for Theorem 1,

<!-- formula-not-decoded -->

and with r = n tr Sd in ,

<!-- formula-not-decoded -->

The last equality is the result of applying E A [ B i s ( A T s A s ) j ] = ϕ ′ ( i, j ) with different combinations of B s , B s ′ , A T j A j , and A T l A l , counting multiplicity and the fact that E A [( A T s A s ) 2 ] = r (1 + r ) I . In conclusion, where

<!-- formula-not-decoded -->

The claims in Theorem 2 is further supported by the experimental validations with results presented in Figure 9.

Diversity

100-

1-

Upper bound of sharpness (x-axis)

Simulated

8p 1.0

Upper Bound on Sharpness

1.0

Sharpness

0.100

• 0.06

<!-- image -->

10.0

Figure 9: (Theoretical vs. Simulated sharpness-diversity trade-off in SharpBalance ) This figure illustrates the relationship between sharpness(upper bound) and diversity as predicted by Theorem 2 and as observed in simulations under different configurations. (a) validates our theoretical results by varying the perturbation radius ρ from 1 . 0 to 0 . 4 . (b) validates the derivation by varying number of iterations k from 1 to 15 . These results demonstrate the soundness of our derivation across a range of parameters.

## C.3 Empirical Verification of Theorem 1 and 2

To demonstrate the robustness and tightness of the bounds presented in Theorem 1, we provide verification results across a range of parameter configurations. Interestingly, the observed model behaviors closely align with the upper bound derived in Theorem 1, highlighting the effectiveness of our theoretical analysis in capturing the underlying dynamics of the ensemble. Figure 10 illustrates these results, with each sub-figure corresponding to a specific combination of k and η with ρ from range 0 . 5 to 0 . 3 . In these experiment, we generated 50 random data matrices A of size 3000 × 150 and test data T of size 1000 × 150 . For each random dataset, we initialized 50 random model weights θ 0 and collected the expected statistics of interest after training. To measure the sharpness κ SAM k , we employed projected gradient ascent to find the optimal perturbation, using a step size of 0 . 01 and a maximum of 50 steps. Similar experiments are performed to verify the derivations in Theorem 2 with results presented in Figure 11, with the number of partitions S = 10 .

## D Hyperparamter setting

## D.1 Datasets

We first evaluate on image classification datasets CIFAR-10 and CIFAR-100. The corresponding OOD robustness is evaluated on CIFAR-10C and CIFAR-100C [Hendrycks and Dietterich, 2019b]. The experiments are carried out on ResNet18 [He et al., 2016]. We use a batch size of 128, a momentum of 0.9, and a weight decay of 0.0005 for model training. TinyImageNet is an image classification dataset consisting of 100K images for training and 10K images for in-distribution testing. We evaluate ensemble's OOD robustness on TinyImageNetC [Hendrycks and Dietterich, 2019b].

## D.2 Hyperparamter setting for empirical sharpness-diversity trade-off

Here, we provide the hyperparameter for the experiments in Section 4.2. When using adaptive worst-case sharpness for sharpness measurement, the size of neighborhood γ defined in equation (3) needed to be specified, we use a γ of 0.5 for all the results in Figure 1 and Figure 5. Additionally, when training NNs in the ensemble, we change the perturbation radius ρ of SAM so that we can study the trade-off. The range of ρ for the results in Figure 1 is { 0 . 01 , 0 . 02 , 0 . 03 , 0 . 04 , 0 . 05 , 0 . 1 , 0 . 2 , 0 . 3 } , the range of ρ for the results in Figure 5 is { 0 . 01 , 0 . 015 , 0 . 02 , 0 . 025 , 0 . 03 , 0 . 05 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 } .

ers p=0.4

Upper bound of sharpness (x-axis)

Simulated k=1

6×10"

4 × 10*

23x10*

2 × 10*

10*

10

Driesly

Diversi

Simulated

Lower bound of sharpness (x-axis)

Upper bound of sharpness (x-axis)

3 × 10°

4 × 10°

6 × 10°

Sharpness

(a)k = 2, n = 0.1

Simulated

Lower bound of sharpness (x-axis)

Upper bound of sharpness (x-axis)

Sharpness

(d)k = 4, n = 0.1

Simulated

10"T

Lower bound of sharpness (x-axis)

Upper bound of sharpness (x-axis)

: 1022

Div

1021

10'°

10"

Sharpness

10'2

3 × 10°

2 × 103

Diversity a-o-o-o

10'-

Diversity

-0-0-0

09-000-00

Figure 10: (Theoretical vs. Simulated sharpness-diversity trade-off in SAM ). This figure compares the sharpness and diversity as predicted by Theorem 1 and as observed in simulations under various parameter configurations. Results demonstrates the robustness of our theoretical analysis and tightness of the derived sharpness upper bound.

<!-- image -->

## D.3 Hyperparamter setting for SharpBalance

Hyperparameter setting on CIFAR10/100 . For experiments on CIFAR10/100, we train an NN from scratch with basic data augmentations, including random cropping, padding by four pixels, and random horizontal flipping. We use a batch size of 128, a momentum of 0.9, and a weight decay of 0.0005. For deep ensemble, we train each model for 200 epochs.

In addition, we use 10% of the training set as the validation set for selecting ρ and k based on the ensemble's performance. We make a grid search for ρ over { 0 . 01 , 0 . 02 , 0 . 05 , 0 . 1 , 0 . 2 , 0 . 5 } . For SharpBalance , we use the same ρ as SAM and search k over { 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 , 0 . 6 } . T d is another hyperparameter introduced by SharpBalance , we use a T d of 10 for all experiments on CIFAR10, a T d of 100 and 150 respectively when training dense and sparse models on CIFAR100. See Table 1 for the optimal ρ and k after grid search.

Hyperparameter setting on TinyImageNet . For experiments on TinyImageNet, we adopt basic data augmentations, including random cropping, padding by four pixels, and random horizontal flipping. We train each model for 200 epochs. We use a batch size of 128, a momentum of 0.9, a weight decay of 5e-4, a T d of 100, an initial learning rate of 0.1, and decay it with a factor of 10 at Epoch 100 and 150. We search ρ and k in the same range as what we do on CIFAR10/100. See Table 1 for the optimal ρ and k after grid search.

0-00-00

10°-

0-0-0

Diversi

104

10'6-

≤ 1012

10'°

1025

1021

10"7.

Simulated

Upper bound of sharpness (x-axis)

102

10°

Sharpness

(a) k = 4, n = 0.5

Simulated

Upper bound of sharpness (x-axis)

10°

10°

10'

10°

Sharpness

(d) k = 8, n = 0.5

Simulated

Upper bound of sharpness (x-axis)

10°

10'°

Sharpness

10'2

10'T

10°-

Figure 11: (Theoretical vs. Simulated sharpness-diversity trade-off in SharpBalance ). This figure compares the sharpness and diversity as predicted by Theorem 2 and as observed in simulations under various parameter configurations. The observed model behaviors align closely with our derived upper bounds.

<!-- image -->

Table 1: Hyperparamter setting for results in Section 4.4, we report the optimal ρ and k after grid search. Each result in Figure 7 is averaged over three ensembles, which corresponds to 9 random seeds, the random seeds we use are { 13 , 17 , 27 , 113 , 117 , 127 , 43 , 59 , 223 } .

| Dataset      | Model          | Method          | ρ     | k     | T d   |
|----------|----------|----------|-------|-------|-------|
| CIFAR10      | ResNet18 ResNet18 | Deep Ensemble Deep Ensemble+ SAM | - 0.2 | - -   | - -   |
|          | ResNet18 ResNet18 | Deep Ensemble Deep Ensemble+     | -     | -     | - -   |
| CIFAR100     | ResNet18          | SAM          | 0.2   | - 0.5 |       |
| TinyImageNet | ResNet18          | SharpBalance          | 0.2   |       |       |
|          | ResNet18          | Deep Ensemble+ SAM          | 0.2   | -     | 100   |
|          |          | Deep Ensemble          | -     | -     | -     |
|          |          |          |       |       | -     |
|          | ResNet18          | SharpBalance          | 0.2   |       |       |
|          |          |          |       | 0.3   | 100   |

## E Ablation studies on loss landscape metrics

In this section, we show that the sharpness-diversity trade-off generalizes to different measurements of sharpness and diversity. The results are presented in Figure 12.

Diversity

0.0.000

Figure 12: (Ablation study of varying sharpness and diversity metrics to corroborate existence of sharpness-diversity trade-off). (a)(d) Varying sharpness metric by using the adaptive ℓ ∞ worstcase sharpness. (b)(e) Varying sharpness metric by using the adaptive ℓ 2 average case sharpness. (c)(f) Varying diversity metric by using the KL divergence. The sharpness-diversity trade-off is still observed in all the settings. The x -axis and y -axis are in log scale. The notation β stands for the slope of the linear regression function fitted on all the ensembles trained by SAM .

<!-- image -->

Sharpness metric. In the main paper, we use adaptive worst-case sharpness defined in equation (3), the parameter neighborhood is bounded by ℓ 2 norm. In this section, we consider two more sharpness metrics [Kwon et al., 2021, Andriushchenko et al., 2023]: adaptive worst-case sharpness with the parameter neighborhood bounded by ℓ ∞ norm (referred to as adaptive ℓ ∞ worst-case sharpness); and adaptive average case sharpness bounded by ℓ 2 norm (termed average case sharpness). The adaptive ℓ ∞ worst-case sharpness is defined as:

<!-- formula-not-decoded -->

The average case sharpness is defined as:

<!-- formula-not-decoded -->

where ρ 0 is the neighborhood size of current parameter θ . T θ is a normalization operator that ensures the sharpness measure is invariant with respect to the re-scaling operation of the parameter. The results, illustrated in Figures 12, corroborate our observation of a trade-off between sharpness and diversity.

Diversity metric . We consider Kullback-Leibler (KL) Divergence [Kullback and Leibler, 1951] as an alternative diversity metric, which is also widely used in previous literature to gauge the diversity of two ensemble members [Fort et al., 2019, Liu et al., 2022]. Specifically, the KL-divergence between the outputs of two ensemble members given a data sample ( x , y ) is defined as:

<!-- formula-not-decoded -->

We measure the KL divergence on each data sample in the test data and then average the measured KL divergence. The results for KL-divergence are shown in Figure 12, which demonstrate the trade-off remains consistent for different diversity metrics.

## F More results

## F.1 Evaluation on different corruption severity

SharpBalance 's main advantage lies in OOD scenarios. As shown in Table 2-4, SharpBalance consistently outperforms the baselines on different levels of corruption.

Table 2: Results of different severity levels on CIFAR10-C.

| Corruption Severity   | 1          | 2          | 3          | 4          | 5          |
|----------|----------|----------|----------|----------|----------|
| Deep ensemble         | 88.90         | 83.67         | 77.56         | 70.37         | 58.63         |
| Deep ensemble+ SAM    | 89.44         | 84.24         | 78.16         | 71.04         | 58.77         |
| SharpBalance          | 89.75 (+0.31) | 84.80 (+0.56) | 78.98 (+0.82) | 72.25 (+1.21) | 60.78 (+2.01) |

Table 3: Results of different severity levels on CIFAR100-C.

| Corruption Severity   | 1          | 2          | 3          | 4          | 5          |
|----------|----------|----------|----------|----------|----------|
| Deep ensemble         | 65.78         | 57.77         | 51.30         | 44.33         | 34.16         |
| Deep ensemble+ SAM    | 66.39         | 58.47         | 51.89         | 44.90         | 34.81         |
| SharpBalance          | 67.23 (+0.84) | 59.53 (+1.06) | 53.14 (+1.25) | 46.19 (+1.29) | 36.20 (+1.39) |

Table 4: Results of different severity levels on Tiny-ImageNet-C.

| Corruption Severity   | 1          | 2          | 3          | 4          | 5          |
|----------|----------|----------|----------|----------|----------|
| Deep ensemble         | 43.62         | 36.65         | 28.96         | 22.08         | 16.86         |
| Deep ensemble+ SAM    | 45.20         | 38.04         | 30.19         | 22.98         | 17.71         |
| SharpBalance          | 46.48 (+1.28) | 39.53 (+1.49) | 31.70 (+1.51) | 24.27 (+1.29) | 18.69 (+0.98) |

## F.2 Evaluation on different model architectures

We extend the evaluations on more architectures such as WideResNet (WRN), ViT, and ALBERT. Here we describe the experimental setup. For vision tasks with WRN, we trained the ensemble members from scratch on CIFAR-10 and CIFAR-100. For vision tasks with transformers, we constructed the three-member ensemble by fine-tuning the pre-trained ViT-T/16 model on the CIFAR100 dataset, evaluated on in-distribution and CIFAR100-C test sets. For language tasks, we constructed the three-member ensemble by fine-tuning the pre-trained ALBERT-Base model on Microsoft Research Paraphrase Corpus (MRPC) dataset and evaluated the performance on its validation set. The hyperparameter search and setup are the same as in Appendix D.3.

These results in Figure 13 and Table 5 confirm that SharpBalance consistently boosts both ID and OOD performance across the models and datasets studied.

Table 5: (Additional experiments on Transformer-architecture) . The ensemble test accuracy is reported and each ensemble comprises three members. The observation is consistent with the residual network results in the main paper: SAM improves the Deep Ensemble, and SharpBalance outperforms both two baselines.

| Method          | ViT-T/16   | ViT-T/16   | ALBERT-B MRPC   |
|----------|----------|----------|----------|
|          | CIFAR100   | CIFAR100-C |          |
| Deep ensemble       | 88.34      | 66.64      | 89.50          |
| Deep ensemble + SAM | 88.48      | 66.89      | 89.89          |
| SharpBalance        | 88.68      | 67.21      | 90.11          |

Figure 13: The three-member WRN-40-2 ensemble is trained with different methods on two datasets. The first row reports the OOD accuracy and the second row reports the ID accuracy. The lower part of each bar with the diagonal lines represents the individual model performance. The upper part of each bar represents the ensembling improvement. The results are reported by averaging three ensembles, and each ensemble is comprised of three models.

<!-- image -->

## F.3 Sharpness-aware set: hard vs easy examples

SharpBalance aims to achieve the optimal balance by applying SAM to a carefully selected subset of the data while performing standard optimization on the remaining samples. In our work, sharpness is determined by the curvature of the loss around the model's weights, whereas [Garg and Roy, 2023] determines it based on the curvature of the loss around a data point. In Figure 14, we rank 1000 samples using both metrics and found a strong correlation between these two.

Figure 14: Rank correlation between fisher trace and loss curvature around input data

<!-- image -->

## F.4 Comparison with more baselines

We compare SharpBalance with stronger ensemble method EoA [Arpit et al., 2022] and stronger SAMmethods. We carefully tuned the hyperparameters for EoA. EoA fine-tuned a pre-trained model; and in our paper, all models are trained from scratch. We compare SharpBalance with another SAM baseline: SAM + , where three individual models are trained with different ρ values, e.g., 0.05, 0.1, and 0.2, respectively. From Table 6, SharpBalance outperforms these two baselines both in-distribution and OOD generalization.

In Table 7, we combine GSAM [Zhuang et al., 2022] with Deep Ensemble as a new baseline method 'Deep Ensemble + GSAM', and incorporate the GSAM into our method SharpBalance. The results show that the new baseline with GSAM outperforms the original baseline in ID and OOD performance but still underperforms SharpBalance (w/ SAM). Furthermore, we enhance SharpBalance by replacing the SAM with GSAM, which leads to better ID performance.

| Dataset   | Method       | ACC          | cACC          |
|----------|----------|----------|----------|
| CIFAR10   | SAM + EoA    | 96.03         | 76.29         |
|          |          | 95.55         | 75.57         |
|          | SharpBalance | 96.18 (+0.15) | 77.32 (+1.03) |
| CIFAR100  | SAM +        | 79.67         | 51.28         |
|          | EoA          | 79.53         | 51.45         |
|          | SharpBalance | 79.84 (+0.17) | 52.46 (+1.01) |

Table 6: SharpBalance outperforms EoA and SAM + both in-distribution and OOD generalization on CIFAR10 and CIFAR100.

Table 7: (Comparing our method SharpBalance with stronger SAM baseline). The ensemble test accuracy is reported and each ensemble comprises three members. GSAM improves the original baseline method with SAM and SharpBalance . The model is ResNet18.

| Method          |   CIFAR100 Acc |   CIFAR100-C Acc |
|----------|----------|----------|
| Deep Ensemble        |          79.28 |          50.67 |
| Deep Ensemble + SAM  |          79.5  |          51.28 |
| SharpBalance (SAM)   |          79.84 |          52.46 |
| Deep Ensemble + GSAM |          79.74 |          51.37 |
| SharpBalance (GSAM)  |          80.01 |          51.92 |

Figure 15: (Uncertainty metrics on CIFAR100-C). 'ECE' represents expected calibration error, and 'NLL' represents negative log-likelihood. Both metrics are lower the better. The model architecture is ResNet-18. The uncertainty metrics demonstrate the superior performance of SharpBalance . x -axis represents the number of individual models in one ensemble.

<!-- image -->

## F.5 Evaluation on uncertainty metrics

In Figure 15, we present the results of uncertainty metrics, i.e., negative log-likelihood and expected calibration error. These uncertainty metrics exhibit trends similar to the accuracy metrics: "Deep Ensemble + SAM" outperforms "Deep Ensemble", and our method outperforms both baselines. The experiments were conducted using ResNet-18 on CIFAR100, with metrics reported on corrupted datasets. Additionally, we observe that both metrics improve as the number of ensemble members increases for all three methods.

## G Experiments Compute Resources

All codes are implemented in PyTorch, and the experiments are conducted on 3 Nvidia Quadro RTX 6000 GPUs for training an ensemble of 3 models. Compared to SAM , our method adds a minimal computational cost. The extra time comes from using Fisher trace to compute the per-sample sharpness. Therefore, computing the per-sample sharpness requires one single forward pass and one backward pass. We report the additional training cost in Table 8. SharpBalance only increases the training time by 1%: 0 . 83 (84 . 48 -0 . 83) × 100% ≈ 1% .

Table 8: Additional training cost introduced by SharpBalance . We train a ResNet18 on CIFAR10 for 200 epochs.

| Additional training cost   | Total training cost   |
|----------|----------|
| 0.83 min          | 84.48 min          |

##