## Addressing Spectral Bias of Deep Neural Networks by Multi-Grade Deep Learning

## Ronglong Fang, Yuesheng Xu ∗

Department of Mathematics and Statistics, Old Dominion University {rfang002, y1xu}@odu.edu

## Abstract

Deep neural networks (DNNs) have showcased their remarkable precision in approximating smooth functions. However, they suffer from the spectral bias , wherein DNNs typically exhibit a tendency to prioritize the learning of lower-frequency components of a function, struggling to effectively capture its high-frequency features. This paper is to address this issue. Notice that a function having only low frequency components may be well-represented by a shallow neural network (SNN), a network having only a few layers. By observing that composition of low frequency functions can effectively approximate a high-frequency function, we propose to learn a function containing high-frequency components by composing several SNNs, each of which learns certain low-frequency information from the given data. We implement the proposed idea by exploiting the multi-grade deep learning (MGDL) model, a recently introduced model that trains a DNN incrementally, grade by grade, a current grade learning from the residue of the previous grade only an SNN (with trainable parameters) composed with the SNNs (with fixed parameters) trained in the preceding grades as features. We apply MGDL to synthetic, manifold, colored images, and MNIST datasets, all characterized by presence of high-frequency features. Our study reveals that MGDL excels at representing functions containing high-frequency information. Specifically, the neural networks learned in each grade adeptly capture some low-frequency information, allowing their compositions with SNNs learned in the previous grades effectively representing the high-frequency features. Our experimental results underscore the efficacy of MGDL in addressing the spectral bias inherent in DNNs. By leveraging MGDL, we offer insights into overcoming spectral bias limitation of DNNs, thereby enhancing the performance and applicability of deep learning models in tasks requiring the representation of high-frequency information. This study confirms that the proposed method offers a promising solution to address the spectral bias of DNNs. The code is available on GitHub: Addressing Spectral Bias via MGDL .

## 1 Introduction

Deep neural networks (DNNs) have achieved tremendous success in various applications, including computer vision [15], natural language processing [38], speech recognition [26], and finance [27]. From a mathematical perspective, the success is mainly due to their high expressiveness, as evidenced by theoretical demonstrations showing their capability to approximate smooth functions with arbitrary precision [3, 10, 14, 17, 48]. Various mathematical aspects of DNNs as an approximation tool were recently investigated in [11, 16, 22, 25, 35, 45-47, 49]. However, it was noted in [30, 42] that the standard deep learning model, which is called single grade deep learning (SGDL) in this paper,

∗ Corresponding author: y1xu@odu.edu

trains a DNN end-to-end leading to learning bias towards low-frequency functions. While this bias may explain the phenomenon where DNNs with a large number of parameters can achieve low generalization error [7, 40, 41], DNNs trained by SGDL struggle to capture high-frequency components within a function even though they can well-represent its low-frequency components. This bias may potentially limit the applicability of DNNs to problems involving high-frequency features, such as image reconstruction [12, 24, 37], seismic wavefield modeling [39], high-frequency wave equations in homogenization periodic media [9], and high energy physics [28]. Especially, in medical image reconstruction such as PET/SPECT, high-frequency components play a crucial role in determining image resolution, which is critical in clinical practice, as higher resolution leads to earlier and more accurate disease diagnosis.

There have been some efforts to address this issue. A phrase shift DNN was proposed in [5], where the original dataset was first decomposed into subsets with specific frequency components, the highfrequency component was shifted downward to a low-frequency spectrum for learning, and finally, the learned function was converted back to the original high frequency. An adaptive activation function was proposed in [18] to replace the traditional activation by scaling it with a trainable parameter. A multiscale DNN was introduced in [6, 23], in which the input variable was first scaled with different scales and then the multiscale variables were combined to learn a DNN. It was proposed in [36] first to map the input variable to Fourier features with different frequencies and then to train the mapped data by DNNs. All these approaches can mitigate the spectral bias issue of DNNs to some extent.

Despite of encouraging progresses made in mitigating the spectral bias of DNNs, practical learning with DNNs remains a persistent challenge due to the bias, especially for learning from higherdimensional data. This issue deserves further investigation. We propose to address this issue by understanding how a high-frequency function can be more accurately represented by neural networks. On one hand, it has been observed [7, 30, 31] that a function having only low frequency components can be well represented by a shallow neural network (SNN), a network having only a few layers. On the other hand, the classical Jacobi-Anger identity expresses a complex exponential of a trigonometric function as a linear combination of its harmonics that can contain significant high-frequency components. Even though the complex exponential function and the trigonometric function both are of low frequency, their composition could contain high frequency components. This motivates us to decompose a function containing high-frequencies as a sum-composition form of low-frequency functions. That is, we decompose it into a sum of different frequency components, each of which is further broken down to a composition of low-frequency functions . In implementing this idea, we find that the multi-grade deep learning (MGDL) model recently introduced in [43, 44] matches seamlessly for constructing the sum-composition form for a function of high-frequency. It is the purpose of this study to introduce the general methodology in addressing the spectral bias issue of DNNs and implement it by employing MGDL as a technical tool. We demonstrate the efficacy of the proposed approach in four experiments with one-dimensional synthetic data, twodimensional manifold data, two-dimensional colored images, and very high-dimensional modified National Institute of Standards and Technology (MNIST) data. Our numerical results endorse that the proposed approach can effectively address the spectral bias issue, leading to substantial improvement in approximation accuracy in comparison with the traditional SGDL training approach.

Contributions of this paper include: (a) We propose a novel approach to address the spectral bias issue by decomposing a function containing high-frequencies as a sum of different frequency components, which are represented as compositions of low-frequency functions. (b) We investigate the efficacy of MGDL in decomposing a function of high-frequency into its 'sum-composition' form of SNNs. (c) We successfully apply the proposed approach to synthetic data in 1 and 2 dimensions and real data in 2 and 784 dimensions, showing that it can effectively address the spectral bias issue.

## 2 Proposed Approach and Multi-Grade Learning Model

We introduce a novel approach to tackle the spectral bias issue and review the MGDL model.

We begin with a quick review of the definition of DNNs. A DNN is a successive composition of an activation function composed with a linear transformation. Let R denote the set of all real numbers, and d, s be two positive integers. A DNN with depth D consists of an input layer, D -1 hidden layers, and an output layer. Let N D := { 1 , 2 , . . . , D } . For j ∈ { 0 }∪ N D , let d j denote the number of neurons in the j -th hidden layer with d 0 := d and d D := s . We use W j ∈ R d j × d j -1 and b j ∈ R d j

0.10

0.08-

0.06

0.04 -

0.02 -

0.00

Amplitude

25

- f

50

75 100 125 150 175 200

Frequency [Hz]

0.10-

0.08

0.06

o.

.04

0.02 -

Figure 1: Spectrum comparison of f := f 1 + f 2 ◦ f 1 + f 3 ◦ f 2 ◦ f 1 + f 4 ◦ f 3 ◦ f 2 ◦ f 1 and f j : Amplitude versus one-side frequency plots for function f (Left) and f j for j ∈ N 4 (Right). The function f is of highfrequency and functions f j all are of low-frequency.

<!-- image -->

to represent the weight matrix and bias vector, respectively, for the j -th layer. By σ : R → R we denote an activation function. When σ is applied to a vector, it means that σ is applied to the vector componentwise. For an input vector x := [ x 1 , x 2 , . . . , x d ] ⊤ ∈ R d , the output of the first layer is defined by H 1 ( x ) := σ ( W 1 x + b 1 ) . For a DNN with depth D ≥ 3 , the output of the ( j +1) -th hidden layer can be identified as a recursive function of the output of the j -th hidden layer, defined as H j +1 ( x ) := σ ( W j +1 H j ( x ) + b j +1 ) , for j ∈ N D -2 . Finally, the output of the DNN with depth D is an s -dimensional vector-valued function defined by

<!-- formula-not-decoded -->

Suppose that data samples D := { x ℓ , y ℓ } N ℓ =1 are chosen. The loss on D is defined as

<!-- formula-not-decoded -->

The traditional SGDL model is to minimize the loss function L defined by (2) with respect to Θ := { W j , b j } D j =1 , which yields the optimal parameters Θ ∗ := { W ∗ j , b ∗ j } D j =1 and the corresponding DNN N D (Θ ∗ ; · ) . When D is relatively small, for example, D &lt; 5 , we call N D an SNN. It is well-recognized that training an SNN is notably easier than training a DNN.

We motivate the proposed idea by a simple example. We consider the function f ( x ) , x ∈ [0 , 1] , whose Fourier transform is shown in Figure 1 (Left), where the Fourier transform is defined by ˆ f ( t ) := ∫ ∞ -∞ f ( x ) e -i 2 π tx d x . To compute the Fourier transform of f defined on [0 , 1] , we extend f to the entire real line by assigning its value to be zero for x / ∈ [0 , 1] . Observing from Figure 1 (Left), the function f has significant high-frequency components, with frequencies varying from 0 to 200 . The function f can be represented as

<!-- formula-not-decoded -->

where ◦ denotes the composition of two functions. Note that the Fourier transforms ˆ f j , j = 1 , 2 , 3 , 4 , are displayed in Figure 1 (Right). Clearly, the functions f j , j = 1 , 2 , 3 , 4 , are of low-frequency, with frequencies mainly concentrating on the interval [0 , 50] . This example surely demonstrates that a function of high-frequency can be expressed as a sum of compositions of lower-frequency functions. This observation leads to the proposed approach of addressing the spectral bias of DNNs to be studied in this paper. The legitimacy of the proposed idea may be reinforced by the Jacobi-Anger identity [2], which expresses a complex exponential of a trigonometric function as a linear combination of its harmonics. Even though both the complex exponential function and the trigonometric function are of low-frequency, their composition contains many high-frequency components. We now review the Jacobi-Anger identity, the identity named after the 19th-century mathematicians Carl Jacobi and Carl Theodor Anger. It has the form

<!-- formula-not-decoded -->

where i denotes the imaginary unit and J n ( a ) denotes the n -th Bessel function of the first kind, see details in [2]. Taking the real part of the both sides of the Jacobi-Anger identity (4), we obtain that

<!-- formula-not-decoded -->

The left-hand side of (5) is a composition of two low-frequency functions cos( a x ) and sin( b x ) , having frequencies a/ (2 π ) and b/ (2 π ) , respectively, while the right-hand side is a linear combination of cos( nb x ) with n taking all integers. The high-frequency of the composition can be estimated by a rule of thumb. Specifically, the left-hand side of (5) is a frequency-modulated sinusoidal signal

Amplitude f2

f3

[32, 33], with its frequencies spreading on an interval centered at zero. It follows from the well-known Carson bandwidth rule [8, 29, 32], regarded as a rule of thumb, that more than 98% frequencies are located within the interval [ -( ab + b ) / (2 π ) , ( ab + b ) / (2 π )] . Therefore, the highest frequency of cos( a sin( b x )) can be well-estimated by ab + b 2 π , which is greater than the frequencies of cos( a x ) and sin( b x ) when a &gt; 0 and b &gt; 1 . These suggest that a composition of two low-frequency functions may lead to a high-frequency function.

The example presented earlier, together with the Jacobi-Anger identity, inspires us to decompose a given function into a sum of different frequency components, each of which is a composition of lowerfrequency functions, a decomposition similar to equation (3) for the function f represented in Figure 1 (Left). In other words, for a function g of high-frequency, we decompose it in a 'sum-composition' form as

<!-- formula-not-decoded -->

where ⊙ k j =1 g j := g k ◦ · · · ◦ g 2 ◦ g 1 , and g j , j ∈ N k , are all of low-frequency. The function f represented in (3) is a special example of (6). In the context of approximation by neural networks, we prefer expressing g j by SNNs, as a function having only low-frequency components can be wellrepresented by an SNN. The MGDL model originated in [43, 44] furnishes exactly the decomposition (6), with each g j being an SNN. We propose to employ MGDL to learn the decomposition (6), where the low-frequency function g j is represented by an SNN.

It is worth explaining the motivation behind the MGDL model. MGDL was inspired by the human education system which is arranged in grades. In such a system, students learn a complex subject in grades, by decomposing it into sequential, simpler topics. Foundational knowledge learned in previous grades remains relatively stable and serves as a basis for learning in a present and future grades. This learning process can be modeled mathematically by representing a function that contains higher-frequency components by a 'sum-composition' form of low-frequency functions. MGDL draws upon this concept by decomposing the learning process into multiple grades, where each grade captures different levels of complexity.

We now review the MGDL model that learns given data D := { x ℓ , y ℓ } N ℓ =1 . Following [43], we split a DNN with depth D into L grades, with L &lt; D , each of which learns an SNN N D l , defined as (1), with depth D l , from the residue { e l ℓ } N ℓ =1 of the previous grade, where 1 &lt; D l &lt; D and ∑ L l =1 D l = D + L -1 . Let Θ l := { W l j , b l j } D l j =1 denote the parameters to be learned in grade l . We define recursively g 1 (Θ 1 ; x ) := N D 1 (Θ 1 ; x ) , g l +1 (Θ l +1 ; x ) := N D l +1 (Θ l +1 ; · ) ◦ H D l -1 (Θ ∗ l ; · ) ◦ . . . ◦ H D 1 -1 (Θ ∗ 1 ; · )( x ) , for l ∈ N L -1 , and the loss function of grade l by

<!-- formula-not-decoded -->

where Θ ∗ l := { W l ∗ j , b l ∗ j } D l j =1 are the optimal parameters learned by minimizing the loss function L l with respect to Θ l . The residues are defined by e 1 ℓ := y ℓ and e l +1 ℓ := e l ℓ -g l (Θ ∗ l ; x ℓ ) , for l ∈ N L -1 , ℓ ∈ N N . When minimizing the loss function L l (Θ l ; D ) of grade l , parameters Θ ∗ j , j ∈ N l -1 , learned from the previous l -1 grades are all fixed and H D l -1 -1 (Θ ∗ l -1 ; · ) ◦ . . . ◦ H D 1 -1 (Θ ∗ 1 ; · ) serves as a feature or 'basis". After L grades are learned, the function ¯ g L learned from MGDL is the summation of the function learned in each grade, that is,

<!-- formula-not-decoded -->

where g l (Θ ∗ l ; x ) := N D l (Θ ∗ l ; · ) ◦ H D l -1 -1 (Θ ∗ l -1 ; · ) ◦ . . . ◦ H D 1 -1 (Θ ∗ 1 ; · )( x ) , and H D k -1 -1 for 1 ≤ k ≤ L and N D L are SNNs learned in different grades. Thus, MGDL enables us to construct the desired 'sum-composition" form (6). When L = 1 , MGDL reduces to the traditional SGDL model.

In MGDL, we use the mean squared error (MSE) loss function. It was established in [43] that when the loss function is defined by MSE, MGDL either learns the zero function or results in a strictly decreasing residual error sequence (see, Theorem 1 in Appendix A). Since the regression problems conducted in this paper naturally align with MSE losses, it is a suitable choice. In practice, MGDL can also be applied with other loss functions, such as cross-entropy loss, when solving classification problems. In MGDL, the computation cost remains relatively consistent across all grades. For x l ℓ := H D l -1 -1 (Θ ∗ l -1 ; · ) ◦ H D l -2 -1 (Θ ∗ l -2 ; · ) ◦ . . . ◦ H D 1 -1 (Θ ∗ 1 ; · )( x ℓ ) , we recursively

let x 1 ℓ := x ℓ , x k ℓ := H D k -1 -1 (Θ ∗ k -1 ; · ) ◦ x k -1 ℓ , k = 2 , 3 , . . . , n. When training grade l , we use the output of grade l -1 , denoted as x l ℓ along the residual e l ℓ , which are already obtained. The training dataset in grade l consists of { ( x l ℓ , e l ℓ ) } N ℓ =1 . This dataset is used to train a new shallow network, which is independent of the previous l -1 grades. Moreover, x l ℓ can be computed recursively, ensuring that the computation cost for each grade remains relatively consistent.

MGDL avoids training a DNN from end to end. Instead, it trains several SNNs sequentially, with the current grade making use the SNNs learned from the previous grades as a feature and composing it with a new SNN to learn the residue of the previous grade. This allows MGDL to decompose a function that contains higher-frequency in a form of (6), with g j being a SNN learned from grade j . In this way, higher-frequency components in the data can be effectively learned in a grade-by-grade manner. Note that the training time of MGDL increases linearly with the number of grades. This makes MGDL an effective and scalable solution for tackling complex tasks. MGDL is an adaptive approach by nature. When the outcome of the present grade is not satisfactory, we can always add a new grade without changing the previous grades.

It is worth noting that while ResNet [15] also has a sum-composition form, MGDL differs from it significantly. The 'Composition' for ResNet refers to composition of layers, while that for MGDL emphasizes the composition of the SNNs sequentially learned in the previous grades. Moreover, ResNet learns all parameters of the entire sum of DNNs at once, training it from end to end, whereas MGDL learns the sum incrementally, grade by grade, in each grade training an SNN composed with the feature (the composition of the SNNs learned in the previous grades). MGDL also differs from the relay backpropagation approach proposed in [34], where a DNN is divided into multiple segments, each with its own loss function. The gradients from these losses are then propagated to lower layers of their respective segments and all segments are optimized all together by minimizing the sum of the losses. While MGDL trains SNNs in a multi-grade manner, each of which learns from the residue of the previous grade, freezing the previously learned SNNs (serving as features or 'bases').

MGDL is a principle applicable to various models, including standard DNNs, convolutional neural networks, and ResNet. In this paper, we demonstrate its feasibility by applying it to standard DNNs.

## 3 Numerical Experiments

In this section, we study MGDL empirically in addressing the spectral bias issue of SGDL. We consider four examples: Subsections 3.1, 3.2, and 3.4 investigate regression on synthetic, manifold, and MNIST data, respectively, for which the spectral bias phenomena of SGDL are identified in [30]. Section 3.3 deals with regression on colored images, which were studied in [36] by using the Fourier features to mitigate the spectral bias. Our goal is to compare the learning performance of MGDL with that of SGDL on these datasets and understand to what extent MGDL can overcome the spectral bias exhibited in SGDL.

The loss functions defined in (2) for SGDL and (7) for MGDL are used to compute the training and validation loss when D is chosen to be the training and validation data, respectively. We use the relative squared error (RSE) to measure the accuracy of predictions obtained from both SGDL and MGDL. Assume that N is a trained neural network. For a prediction value ˆ y ℓ := N ( x ℓ ) at x ℓ , we define RSE := ∑ N ℓ =1 ∥ ˆ y ℓ -y ℓ ∥ 2 2 / ∑ N ℓ =1 ∥ y ℓ ∥ 2 2 . When D represents the training, validation, and testing data, RSE is specialized as TrRSE, VaRSE, and TeRSE, respectively.

Details of the numerical experiments conducted in this section, including computational resources, the network structure of SGDL and MGDL for each example, the choice of activation function, the optimizer, parameters used in the optimization process, and supporting figures are provided in Appendix B.

3.1 Regression on the synthetic data. In this experiment, we compare the efficacy of SGDL and MGDL in learning functions of four different types of high-frequencies.

The experiment setup is as follows. Given frequencies κ := ( κ 1 , κ 2 , . . . , κ M ) with corresponding amplitudes α := ( α 1 , α 2 , . . . , α M ) , and phases φ := ( φ 1 , φ 2 , . . . , φ M ) , we consider approximating the function λ : [0 , 1] → R defined by

<!-- formula-not-decoded -->

by neural networks learned with SGDL and MGDL. We consider four settings, in all of which we choose M := 20 , κ j := 10 j and φ j ∼ U (0 , 2 π ) for j ∈ N 20 , where U denotes the uniform

102

0.5

10°

0.4

10-1

0.

.3

un 10-2

910-3

Amplitude

10-+

0.1

10-5

0.0

10-0

G 1

training, 1

G 2

validation, l training, 2

G 3

-=-- validation, 2

G 4

training, 3

validation, 3

training, 4

---- validation, 4

25

50 75 100 125 150 175 200

Frequency [HZ]

10000

20000

Epochs (SGDL)

30000

102

0.5

10°

10-2

0.4

0.

2

10-+

0.1

10-3.

10-0

0.35

0.30-

0.25

0.30

itude

· 030

0.20

0.20

0.15

0.15-

0.10-

Ampli

0.10

0.05

0.05

0.00

0.00

G 1

G 2

G 3

G 1

G 2

G 3

G 4

G 4

50 75 100 125 150 175 200

- training. 1

validation, 1

training. 2

---- validation, 2

training. 3

validation, 3

training, 4

--- validation, 4

75 100 125 150 175 200

G 130000G 230000G 330000G 430000G 530000

25

25

G 1

G 2

G 3

G 4

G 1

G 2

G 3

G 4

G 5

50 75 100 125 150 175 200

Amplitude

0.25

0.35

0.30

0.20

0.25

0.15

0.10

0.05

0.05

0.00

0.00

<!-- image -->

50 100 150 200 250 300 350 400

50 100 150 200 250 300 350 400

Figure 2: Amplitude versus one-side frequency plot for the learned functions learned across four grades of MGDL for settings 1-4.

Table 1: Comparison of SGDL and MGDL: Accuracy

|   setting | model          | t max - 4         | t min         | batch size   | time(s)          | TrRSE          | VaRSE          | TeRSE          |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|
|         1 | SGDL 10 MGDL 10    | 10 - 4 - 4 10 - 4 | 256 256       |          | 32 , 401 27 , 817 | 1 . 5 × 10 - 1 5 . 9 × 10 - 6 | 1 . 3 × 10 - 1 1 . 9 × 10 - 5 | 1 . 2 × 10 - 1 1 . 7 × 10 - 5 |
|         2 | SGDL 5 × 10 MGDL - | - 4 10 4          | 10 - 4 10 - 4 | 256 256      | 32 , 053 27 , 628 | 5 . 8 × 10 - 3 1 . 5 × 10 - 6 | 5 . 9 × 10 - 3 4 . 0 × 10 - 6 | 5 . 7 × 10 - 3 6 . 5 × 10 - 6 |
|         3 | SGDL MGDL          | 10 - 3 10 - 4     | 10 - 4 10 - 4 | 256 256      | 31 , 808 24 , 876 | 9 . 6 × 10 - 2 5 . 2 × 10 - 6 | 8 . 3 × 10 - 2 3 . 2 × 10 - 5 | 1 . 1 × 10 - 1 2 . 1 × 10 - 5 |
|         4 | SGDL MGDL          | 5 × 10 - 5 10 - 4 | 10 - 5 10 - 4 | 256 Full     | 41 , 063 9 , 875  | 7 . 9 × 10 - 1 1 . 1 × 10 - 3 | 7 . 5 × 10 - 1 2 . 2 × 10 - 3 | 7 . 7 × 10 - 1 1 . 3 × 10 - 3 |

distribution and the random seed is set to be 0 . In settings 1 , 2 , 3 , and 4 , we choose respectively the amplitudes α j := 1 , α j := 1 . 05 -0 . 05 j , α j ( x ) := e -x cos( j x ) , and α j := 0 . 05 j , for j ∈ N 20 . Note that in setting 3 , we explore the case where the amplitude varies as a function of x for each component. The amplitudes versus one-side frequencies for λ across the four settings are depicted in Figure 7 in Appendix B.1. For all the four settings, the training data consist of pairs { x ℓ , λ ( x ℓ ) } ℓ ∈ N 6000 , where x ℓ 's are equally spaced between 0 and 1 . The validation and testing data consist of pairs { x ℓ , λ ( x ℓ ) } ℓ ∈ N 2000 , where x ℓ 's are generated from a random uniform distribution on [0 , 1] , with the random seed set to be 0 and 1, respectively.

Numerical results for this example are reported in Figures 2 and 3, as well as 8 and 9 (in Appendix B.1), and Table 1. Figure 2 displays the amplitude versus the one-side frequency of the functions learned across four grades of MGDL in settings 1-3, and five grades in setting 4. In all the four settings, MGDL exhibits a pattern of learning low-frequency components in grade 1, middle-frequency components in grade 2, and high-frequency components in grades 3, 4 (and 5 for setting 4). We let N ∗ j := N D j (Θ ∗ j ; · ) and H ∗ j := H D j -1 (Θ ∗ j ; · ) . The SNN N ∗ 1 learned in grade 1 represents a low-frequency component of λ , the SNNs N ∗ 2 and N ∗ 3 learned in grades 2 and 3, composed with H ∗ 1 and H ∗ 2 ◦ H ∗ 1 , respectively, represents higher-frequency components of λ . Likewise, the SNN N ∗ 4 , learned in grade 4, composed with H ∗ 3 ◦ H ∗ 2 ◦ H ∗ 1 represents the highest-frequency component of λ . This fact is particularly evident in setting 4 (see, the fourth subfigure in Figure 2), where the amplitude within the data increases with the frequency. For settings 1, 2, and 3, grade 4 does not learn much. This is because for the functions of these three settings, the amplitudes of higher-frequencies are proportionally small. However, grade 4 is still important for these settings. As shown in Figure 3 (right), grade 4 reduces the loss from 10 -2 to 10 -5 for setting 1 and from 10 -4 to 10 -6 for settings 2 and 3. This indicates that if we want a high precision, we need to include grade 4. For setting 4, we need grade 5 to learn its highest frequency component. These findings suggest that MGDL is particularly well-suited for learning the high-frequency components of the function.

Figure 3: Comparison of SGDL (left) and MGDL (right): training and validation loss across settings 1-4.

<!-- image -->

Figure 4: Amplitude vs. one-side frequency plot for the learned function across four grades of MGDL: settings 1 (left) and 2 (right) with q = 0 .

Amplitude

G 1

G 2

G 3

G 4

Figure 3 compares the progress of the training and validation losses against the number of training epochs for SGDL and MGDL across the four settings. We observe that when learning a task involving high-frequency components by SGDL, the training and validation losses decrease slowly due to the spectral bias of DNN. While the same task is learned by MGDL, the learning process progresses through distinct grades. In grade 1, MGDL primarily learns low-frequency, resulting in a slow decrease in loss. In grade 2, the training loss and validation loss both decrease more rapidly due to the use of the composition of SNN N ∗ 2 with the feature H ∗ 1 , facilitating in learning high-frequency features. This accelerated learning aspect of MGDL is further evidenced in grades 3 and 4 (as well as grade 5 for setting 4). Table 1 compares the accuracy achieved by SGDL and MGDL. Within a comparable or even less training time, MGDL increases accuracy, measured by TeRSE from 10 -1 to 10 -5 , 10 -3 to 10 -6 , 10 -1 to 10 -5 , and 10 -1 to 10 -3 in settings 1, 2, 3 and 4, respectively. Across the four settings, TeRSE values are reduced by a factor of 592 ∼ 7 , 058 . These comparisons highlight MGDL's advantage in effectively learning high-frequency oscillatory functions.

Figure 8 in Appendix B.1 depicts the functions, in the Fourier domain, learned by SGDL (row 1) and MGDL (row 2) across the four settings, demonstrating that MGDL has a substantial reduction in the 'spectral bias' exhibited in SGDL. This is because high-frequency components are learned in a higher grade, where they are represented as the composition of a low-frequency component with the low-frequency components learned in the previous grades, and each grade focuses solely on learning a low-frequency component by an SNN. We also include Figure 9 in Appendix B.1 to compare the spectrum evolution between SGDL (1st row) and MGDL (2nd row) across settings 1-4. Notably, although in iterations SGDL and MGDL both learn low-frequency components first and then followed by middle and high-frequency components, MGDL learns in grade by grade, exhibiting significant outperformance.

3.2 Regression on the manifold data. The second experiment compares regression by SGDL and MGDL on two-dimensional manifold data, studied in [30] but with twice higher frequencies.

The goal of this experiment is to explore scenarios where data lies on a lower-dimensional manifold embedded within a higher-dimensional space. Such data is commonly referred to as manifold data [4]. Let γ be an injective mapping from [0 , 1] m → R d with m ≤ d and M := γ ([0 , 1] m ) denote the manifold data. A target function τ : M→ R defined on the manifold can be identified with function λ := τ ◦ γ defined on [0 , 1] m . Regressing the target function τ is therefore equivalent to finding f : R d → R such that f ◦ γ matches λ . Following [30], we set m := 1 , d := 2 , and choose the mapping γ as

<!-- formula-not-decoded -->

for a nonnegative integer q . Clearly, γ q : [0 , 1] → R 2 , and M := γ q ([0 , 1]) defines the manifold corresponding to a flower-shaped curve with q petals when q &gt; 0 , and a unit circle when q = 0 . Suppose that λ : [0 , 1] → R is the function defined by (9). Our task is to learn a DNN f : R 2 → R such that f ◦ γ q matches λ . We consider two settings for λ . In settings 1 and 2, we choose α j := 0 . 025 j and α j ( x ) := e -x cos( j x ) for j ∈ N 40 , respectively. For both the settings, we choose κ j := 10 j and φ j ∼ U (0 , 2 π ) for j ∈ N 40 with the random seed set to be 0 , and consider the cases where q := 4 and q := 0 . Note that the smaller q is, the more difficult the learning task is. The training data consists of pairs { γ q ( x ℓ ) , λ ( x ℓ ) } ℓ ∈ N 12000 , where x ℓ 's are equally spaced between 0 and 1 . The validation and testing data consist of pairs { γ q ( x ℓ ) , λ ( x ℓ ) } ℓ ∈ N 4000 , where x ℓ 's are generated from a random uniform distribution on [0 , 1] , with random seed set to be 0 and 1, respectively.

Numerical results for this example are reported in Figures 4-5, and 10 (in Appendix B.2), and Table 2. Figure 4 illustrates the frequency of functions learned across four grades of MGDL for settings 1 and 2, where q := 0 . In both of the settings, MGDL exhibits a pattern of learning low-frequency components in grade 1 , middle-frequency components in grade 2 , and high-frequency components in grades 3 and 4 . Therefore, the high-frequency components within the function mainly learned in higher grades, in which the learned function is a composition of the SNNs learned from several grades. That is, MGDL decomposes a high-frequency component as the composition of several lower-frequency components, facilitating effectively learning high frequency features within the data.

Table 2 compares the approximation accuracy achieved by SGDL and MGDL for settings 1 and 2. For SGDL, reducing the value of q makes the learning task for both settings more challenging, due to the spectral bias of DNNs. When q := 4 , 0 in setting 1 and q := 0 in setting 2, learning becomes especially challenging for SGDL. In such cases, MGDL significantly outperforms SGDL by achieving higher accuracy in approximately half to one-third of the training time for both settings. Figure 5 displays the training and validation loss for SGDL and MGDL. Figure 10 illustrates the

10°

10"

10-3

10-31

10- training, 4

---- validation, 4

training. O

--- validation, 0

10000

20000

Epochs (SGDL)

30000

....

10-

10-

• training, 4

---- validation, 4

training. 0

--- validation, 0

01

30000

30000

30000

training, 4

validation, 4 M

training, 0

... validation, O

10000

20000

30000

training, 4

validation, 4

training, 0

• validation, 0

G 1 30000 G 2 30000 G 3 30000 G 4 30000

Table 2: Accuracy comparison: SGDL versus MGDL. Epochs (SGDL) Epochs (MGDL)

Epochs (MGOL)

30000 04

Figure 5: Comparison of the training and validation loss for SGDL (1st and 3rd subfigures) and MGDL (2nd and 4th subfigures) in settings 1 (1st and 2nd subfigures) and 2 (3rd and 4th subfigures).

| setting   |   q | method   | t max   | t min   | batch size   | time (s)   | TrRSE          | VaRSE          | TeRSE          |
|----------|-----|----------|---------|---------|----------|----------|----------|----------|----------|
| 1         |   4 | SGDL     | 10 - 4  | 10 - 6  | 1024         | 28 , 832   | 2 . 7 × 10 - 1 | 2 . 4 × 10 - 1 | 2 . 8 × 10 - 1 |
| 1         |   4 | MGDL     | 10 - 3  | 10 - 4  | Full         | 15 , 519   | 4 . 9 × 10 - 5 | 1 . 8 × 10 - 4 | 1 . 1 × 10 - 4 |
|          |   0 | SGDL     | 10 - 4  | 10 - 5  | 1024         | 29 , 051   | 5 . 5 × 10 - 1 | 5 . 4 × 10 - 1 | 5 . 4 × 10 - 1 |
|          |   0 | MGDL     | 10 - 3  | 10 - 4  | Full         | 15 , 969   | 1 . 9 × 10 - 4 | 2 . 7 × 10 - 4 | 2 . 1 × 10 - 4 |
| 2         |   4 | SGDL     | 10 - 3  | 10 - 6  | 512          | 44 , 083   | 2 . 0 × 10 - 4 | 1 . 8 × 10 - 3 | 2 . 3 × 10 - 4 |
| 2         |   4 | MGDL     | 10 - 3  | 10 - 4  | Full         | 11 , 067   | 8 . 5 × 10 - 6 | 1 . 4 × 10 - 3 | 4 . 2 × 10 - 5 |
|          |   0 | SGDL     | 10 - 4  | 10 - 4  | 512          | 41 , 941   | 1 . 1 × 10 - 2 | 1 . 8 × 10 - 2 | 1 . 0 × 10 - 2 |
|          |   0 | MGDL     | 10 - 3  | 10 - 4  | Full         | 11 , 027   | 3 . 7 × 10 - 5 | 2 . 2 × 10 - 3 | 8 . 7 × 10 - 5 |

<!-- image -->

spectrum evolution throughout the learning process for settings 1 and 2. We observe that MGDL outperforms SGDL in accuracy during the learning process. Table 2 and Figure 10 consistently demonstrate that MGDL exhibits a substantial advancement in addressing the spectral bias.

3.3 Regression on two-dimensional colored Images. In the third experiment, we compare performance of MGDL and SGDL for regression of two-dimensional color images.

We test the models with the 'cat' image from website Cat, and the 'sea' and 'building' images from the Div2K dataset [1]. The input to the models is the pixel coordinates, and the output is the corresponding RGB values. The training dataset consists of a regularly spaced grid containing 1 / 4 of the image pixels, while the test dataset contains the full image pixels. We use peak signal-tonoise ratio (PSNR) defined as in (12) to evaluate the accuracy of images obtained from MGDL and SGDL. PSNR values computed over the train and test dataset are denoted by TrPSNR and TePSNR, respectively.

Numerical results for this experiment are presented in Table 3, Figure 6, and Figures 11-13 (in Appendix B.3). Table 3 compares the PSNR values of images obtained from MGDL and SGDL. For MGDL, it is evident that adding more grades consistently improves the image quality, as both training and testing PSNR values increase substantially with the addition of each grade across all images Cat, Sea, and Building. MGDL outperforms SGDL by 2 . 35 ∼ 3 . 93 dB for the testing PSNR values. Specifically, for images Cat, Sea and Building, MGDL surpasses SGDL by 2.35, 3.93 and 2.88 dB, respectively. This demonstrates the superiority of MGDL in representing images compared to SGDL. Figure 6 illustrates the training and testing PSNR values during the training process for the three images. It is evident that MGDL facilitates a smoother learning process as more grades are added, in comparison to SGDL. This observation aligns with the results presented in Table 3.

Predictions of each grade of MGDL and of SGDL are illustrated in Figures 11, 12, and 13 for images Cat, Sea, and Building, respectively. In all cases, grade 1 captures only the rough outlines of the images, representing the lower-frequency components. As we progress to grades 2, 3, and 4, more details corresponding to the higher-frequency components are added. This demonstrates the effectiveness of MGDL in progressively learning high-frequency features within the images. Moreover, the image quality achieved with MGDL is notably superior to that with SGDL.

3.4 Regression on MNIST data. In our fourth experiment, we apply MGDL to regress on highdimension data, the MNIST data. We compare the training and validation loss, and TeRSE for SGDL and MGDL, when learning high-frequency features from the data.

We set up this experiment following [30], with a focus on comparing performance of SGDL and MGDL in learning high-frequency features. We choose the handwriting digits from MNIST dataset [21], composed of 60,000 training samples and 10,000 testing samples of the digits '0' through

10%

10°

10-2

10-3

10-4-

10-3

10%

10°

10-1

1010-1

10-3

10-4

10-3

28-

26-

241

20-

18

16

28

26-

24-

YÑSA

18

16

SGDL

Training PSNR

Testing PSNR

2000

4000 6000

8000 10000

Epochs

(a) Cat: SGDL

MGDL

- Training PSNR

Testing PSNR

10000

20000

Epochs

30000

40000

28-

261

24-

,22-

18-

· 20-0

16-

14-

12

28

26

24

. 22

Z 20-

18

16

14

12

24

22

16

Figure 6: Comparison of PSNR values for SGDL and MGDL on images Cat, Sea, and Building: SGDL (a)-(c), MGDL (d)-(f).

<!-- image -->

Table 3: PSNR comparison: SGDL versus MGDL.

| image    | method   | grade   | learning rate   |   time (s) | TrPSNR   | TePSNR   |
|----------|----------|---------|----------|----------|----------|----------|
| cat      | MGDL     | 1       | 10 - 3          |         38 | 20 . 45  | 20 . 41  |
|          |          | 2       | 10 - 3          |         40 | 22 . 97  | 22 . 67  |
|          |          | 3       | 5 × 10 - 4      |         40 | 25 . 14  | 23 . 71  |
|          |          | 4       | 5 × 10 - 4      |         28 | 28 . 59  | 24 . 18  |
|          | SGDL     |         | 5 × 10 - 3      |         77 | 21 . 93  | 21 . 83  |
| sea      | MGDL     | 1       | 10 - 2          |        160 | 18 . 62  | 18 . 62  |
| sea      |          | 2       | 10 - 3          |        173 | 21 . 57  | 21 . 50  |
| sea      |          | 3       | 10 - 3          |        174 | 24 . 17  | 23 . 42  |
| sea      |          | 4       | 10 - 3          |        182 | 27 . 31  | 24 . 32  |
| sea      | SGDL     |         | 10 - 3          |        685 | 20 . 43  | 20 . 39  |
| building | MGDL     | 1       | 5 × 10 - 3      |        171 | 17 . 30  | 17 . 29  |
| building |          | 2       | 5 × 10 - 3      |        181 | 18 . 97  | 18 . 95  |
| building |          | 3       | 10 - 3          |        182 | 20 . 86  | 20 . 67  |
| building |          | 4       | 10 - 3          |        182 | 23 . 36  | 21 . 97  |
| building | SGDL     |         | 10 - 3          |        742 | 19 . 12  | 19 . 09  |

'9'. We represent each digit j = 0 , 1 , . . . , 9 by a one-hot vector e j +1 ∈ R 10 , whose ( j + 1) -th component is one and all others are zero, and denote by τ 0 : R 784 → R 10 the classifier, which is a piecewise constant function defined by τ 0 ( x ) := e j +1 if x represents the digit j . Wesplit the available training samples to form the training and validation data, denoted as D 0 := { x ℓ , τ 0 ( x ℓ ) } ℓ ∈ N n train and D ′ 0 := { x ′ ℓ , τ 0 ( x ′ ℓ ) } ℓ ∈ N n val respectively, with n train := 45 , 000 and n val := 15 , 000 , and use the testing samples as the testing data, denoted as D ′′ 0 := { x ′′ ℓ , τ 0 ( x ′′ ℓ ) } ℓ ∈ N ntest with n test := 10 , 000 . Clearly, D 0 , D 0 ′ and D ′′ are subsets of { [0 , 1] 784 , { e j +1 } 9 j =0 } . Letting ψ κ ( x ) := sin(2 πκ ∥ x ∥ 2 ) , corresponding to a radial wave defined on the input space R 784 , we define the target function by τ β,κ ( x ) := τ 0 ( x ) (1 + βψ κ ( x )) , where κ is the frequency of the wave and β is the amplitude. Note that τ 0 and βψ κ contribute respectively the lower-frequency and high-frequency components (regarded as noise) of the target function τ β,κ , as discussed in [30]. The modified training and validation data denoted by D β,κ := { x ℓ , τ β,κ ( x ℓ ) } ℓ ∈ N n train and D ′ β,κ := { x ′ ℓ , τ β,κ ( x ′ ℓ ) } ℓ ∈ N n val , respectively, are used to train DNNs. Our goal is to use SGDL and MDGL to regress the modified data D β,κ through minimizing their respective training loss, to compare their robustness to noise. The training loss is evaluated on D β,κ and validation loss is on D ′ β,κ . TrRSE, VaRSE, and TeRSE are evaluated on D β,κ D ′ β,κ , and D ′′ 0 , respectively, noting that D ′′ 0 are test data without noise.

SGDL

Training PSNR

Testing PSNR

SGDL

Training PSNR

Testing PSNR

Table 4: Accuracy comparison: SGDL versus MGDL with β = 1 .

|   κ | method          | time (s)          | TrRSE          | VaRSE          | TeRSE          |
|-----|----------|----------|----------|----------|----------|
|   1 | SGDL 3 , 298 MGDL 3 , | 3 . 1 × 10 - 1 109 3 . 5 × 10 - 1     | 4 . 1 × 10 - 1 3 . 9 × - 1    | 1 . 1 × 10 - 10 8 . 0 × 10 - | 1 2          |
|   5 | SGDL 3 , MGDL 3 ,     | 333 3 . 0 × 10 - 1 461 3 . 5 × 10 - 1 | 4 . 1 × 10 3          | - 1 . 9 × 10 - 1          | 1 . 1 × 10 - 1 7 . 8 × 10 - 2 |
|  10 | SGDL 3 , MGDL 3 ,     | 199 3 . 1 × 448 3 . 5 ×          | 10 - 1 4 . 10 - 1 3 .         | 1 × 10 - 1 9 × 10 - 1        | 1 . 0 × 10 - 1 7 . 8 × 10 - 2 |
|  50 | SGDL 3 , MGDL 3 ,     | 168 3 . 0 × 484 3 . 5 ×          | 10 - 1 4 . 1 × 10 - 1 3 . 9 × | 10 - 1 10 - 1          | 1 . 1 × 10 - 1 7 . 9 × 10 - 2 |

We choose the amplitude β from { 0 . 5 , 1 , 3 , 5 } . For each β , we vary the frequency κ from { 1 , 3 , 5 , 7 , 10 , 50 } . These choices of β and κ result in functions τ β,κ that have higher frequencies than those studied in [30]. Figures 14 and 15 (in Appendix B.4) compare the training and validation loss versus training time of SGDL (structure (13)) and MGDL (structures (14) and (15) respectively) for different values of frequency κ and amplitude β (depicted in the figure). We observe that for small β , for example, β = 0 . 5 , the results of the two models are comparable. When β = 1 , the training loss for SGDL keeps decreasing as training time increases, while the validation loss initially decreases and starts to increase after training of 1,600 seconds, and unlike SGDL, both the training loss and validation loss for MGDL keep decreasing. This trend indicates that over-fitting phenomenon occurs for SGDL and suggests that MGDL has a superior generalization capability. Further increasing the value of β , for example, β = 3 and β = 5 , the over-fitting phenomenon occurs in both of the models. An increase in β corresponds to a higher proportion of high-frequency components within τ β,κ . This increased proportion of the high-frequency component significantly impacts the validation loss of SGDL, a trend also observed in MGDL. However, the effect is comparatively less with MGDL than with SGDL. We present in Table 4 TrRSE, VaRSE, and TeRSE of SGDL (structure (13)) and MGDL (structure (14)) for β := 1 , where the results are obtained by choosing t min := 10 -5 and t max := 10 -4 , and the batch size to be 'Full', for all cases of κ and for both SGDL and MGDL. We observe that while TrRSE of SGDL is smaller than that of MGDL, both VaRSE, and TeRSE of SGDL are larger than the corresponding values of MGDL, suggesting occurrence of overfitting with SGDL. MGDL's improvement in accuracy, measured by TeRSE, is about 27 ∼ 29% . This experiment reveals that MGDL is a promising model for learning higher-dimensional data with a higher proportion of high-frequency features.

## 4 Conclusion

By observing that a high-frequency function may be well-represented by composition of several lower-frequency functions, we have proposed a novel approach to learn such a function. The proposed approach decomposes the function into a sum of multiple frequency components, each of which is compositions of lower-frequency functions. By leveraging the MGDL model, we can express high-frequency components by compositions of multiple SNNs of low-frequency. We have conducted numerical studies of the proposed approach by applying it to one-dimensional synthetic data, two-dimensional manifold data, colored images, and higher-dimensional MNIST data. Our studies have concluded that MGDL can effectively learn high-frequency components within data. Moreover, the proposed approach is easy to implement and not limited by dimension of the input variable. Consequently, it offers a promising approach to learn high-frequency features within (high-dimensional) dataset.

Limitation : Mathematical understanding of the spectral bias of DNNs is absent. Theoretical foundation for MGDL to address the spectral bias issue needs to be established. Numerical studies are preliminary, limited to four examples. More extensive numerical experiments will be conducted in our future work.

Acknowledgments: Y. Xu is indebted to Professor Wei Cai of Southern Methodist University for insightful discussion of the spectral bias of DNNs. Y. Xu is supported in part by the US National Science Foundation under grant DMS-2208386, and by the US National Institutes of Health under grant R21CA263876.

## References

- [1] Agustsson, E., &amp; Timofte, R. (2017) Ntire 2017 challenge on single image super-resolution: Dataset and study. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops , pp. 126-135.
- [2] Arfken, G. B., Weber, H. J., &amp; Harris, F. E. (2011) Mathematical Methods for Physicists: A Comprehensive Guide. Orlando, FL: Academic press.
- [3] Arpit, D., Jastrz˛ ebski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M. S., ... &amp; Lacoste-Julien, S. (2017) A closer look at memorization in deep networks. In International Conference on Machine Learning , pp. 233-242.
- [4] Bengio, Y., Courville, A.C., &amp; Vincent, P. (2012) Representation learning: A review and new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence , 35 (8): 1798-1828.
- [5] Cai, W., Li, X., &amp; Liu, L. (2020) A phase shift deep neural network for high frequency approximation and wave problems. SIAM Journal on Scientific Computing , 42 (5): A3285-A3312.
- [6] Cai, W., &amp; Xu, Z. Q. J. (2019) Multi-scale deep neural networks for solving high dimensional pdes. arXiv preprint arXiv:1910.11710.
- [7] Cao, Y., Fang, Z., Wu, Y., Zhou, D., &amp; Gu, Q. (2019) Towards understanding the spectral bias of deep learning. International Joint Conference on Artificial Intelligence , pp. 2205-2211.
- [8] Carson, J. R. (1922) Notes on the theory of modulation. Proceedings of the Institute of Radio Engineers 10 (1): 57-64.
- [9] Craster, R. V., Kaplunov, J., &amp; Pichugin, A. V. (2010) High-frequency homogenization for periodic media. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences , 466 (2120): 23412362.
- [10] Cybenko, G. (1989) Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems , 2 (4): 303-314.
- [11] Daubechies, I., DeVore, R., Foucart, S., Hanin, B., &amp; Petrova, G. (2022) Nonlinear approximation and (deep) ReLU networks. Constructive Approximation , 55 (1): 127-172.
- [12] Fang, R., Xu, Y., &amp; Yan, M. (2024) Inexact fixed-point proximity algorithm for the ℓ 0 sparse regularization problem. Journal of Scientific Computing , 100 (2): 58.
- [13] Glorot, X., &amp; Bengio, Y. (2010, March) Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics , pp. 249-256.
- [14] Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016) Deep Learning . Cambridge, MA: MIT Press.
- [15] He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016) Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pp. 770-778.
- [16] Huang, J., Jiao, Y., Li, Z., Liu, S., Wang, Y ., &amp; Yang, Y . (2022) An error analysis of generative adversarial networks for learning distributions. Journal of Machine Learning Research , 23 (116): 1-43.
- [17] Hornik, K., Stinchcombe, M., &amp; White, H. (1989) Multilayer feedforward networks are universal approximators. Neural Networks , 2 (5): 359-366.
- [18] Jagtap, A. D., Kawaguchi, K., &amp; Karniadakis, G. E. (2020) Adaptive activation functions accelerate convergence in deep and physics-informed neural networks. Journal of Computational Physics , 404 : 109136.
- [19] Jiang, J., &amp; Xu, Y. (2024) Deep neural network solutions for oscillatory Fredholm integral equations. Journal of Integral Equations and Applications 36 (1): 23-55.
- [20] Kingma, D.P., &amp; Ba, J. (2015) Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations (ICLR) .
- [21] LeCun, Y., Bottou, L., Bengio, Y., &amp; Haffner, P. (1998) Gradient-based learning applied to document recognition. Proceedings of the IEEE 86 (11): 2278-2324.

- [22] Li, Y., Lu, S., Mathé, P., &amp; Pereverzev, S. V. (2024). Two-layer networks with the ReLU k activation function: Barron spaces and derivative approximation. Numerische Mathematik , 156 (1): 319-344.
- [23] Liu, Z., Cai, W., &amp; Xu, Z. Q. J. (2020) Multi-scale deep neural network (MscaleDNN) for solving Poisson-Boltzmann equation in complex domains. Communications in Computational Physics , 28 (5): 19702001.
- [24] Lu, Y., Shen, L., &amp; Xu, Y. (2010) Integral equation models for image restoration: high accuracy methods and fast algorithms. Inverse Problems , 26 (4): 045006.
- [25] Mhaskar, H. N., &amp; Poggio, T. (2016) Deep vs. shallow networks: An approximation theory perspective. Analysis and Applications , 14 (06): 829-848.
- [26] Nassif, A. B., Shahin, I., Attili, I., Azzeh, M., &amp; Shaalan, K. (2019) Speech recognition using deep neural networks: A systematic review. IEEE Access , 7 : 19143-19165.
- [27] Ozbayoglu, A. M., Gudelek, M. U., &amp; Sezer, O. B. (2020) Deep learning for financial applications: A survey. Applied Soft Computing , 93 : 106384.
- [28] Perkins, D. H. (2000) Introduction to High Energy Physics . Cambridge: Cambridge University Press.
- [29] Pieper, R. J. (2001) Laboratory and computer tests for Carson's FM bandwidth rule. In Proceedings of the 33rd Southeastern Symposium on System Theory (IEEE) , pp., 145-149.
- [30] Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y ., &amp; Courville, A. (2019) On the spectral bias of neural networks. In International Conference on Machine Learning , pp. 5301-5310.
- [31] Ronen, B., Jacobs, D., Kasten, Y., &amp; Kritchman, S. (2019) The convergence rate of neural networks for learned functions of different frequencies. Advances in Neural Information Processing Systems , 32.
- [32] Schulze, J.P., Kraus, M., Lang, U., &amp; Ertl, T. (2003) Integrating pre-integration into the shear-warp algorithm. In IEEE VGTC / Eurographics International Symposium on Volume Graphics , pp. 109-118.
- [33] Shanmugam, K. S. (1979) Digital and Analog Communication Systems . New York: Wiley.
- [34] Shen, L., Lin, Z., &amp; Huang, Q. (2016) Relay backpropagation for effective learning of deep convolutional neural networks. In European Conference on Computer Vision , pp. 467-482.
- [35] Shen, Z., Yang, H., &amp; Zhang, S. (2022) Optimal approximation rate of ReLU networks in terms of width and depth. Journal de Mathématiques Pures et Appliquées , 157: 101-135.
- [36] Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., ... &amp; Ng, R. (2020) Fourier features let networks learn high frequency functions in low dimensional domains. Advances in Neural Information Processing Systems , pp. 7537-7547.
- [37] Tang, X., Schmidtlein, C. R., Li, S., &amp; Xu, Y. (2021) An integral equation model for PET imaging. International Journal of Numerical Analysis and Modeling , 18 (6): 834-864.
- [38] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... &amp; Polosukhin, I. (2017) Attention is all you need. Advances in Neural Information Processing Systems , pp. 5998-6008.
- [39] Wu, T., &amp; Xu, Y. (2022) Inverting incomplete Fourier transforms by a sparse regularization model and applications in seismic wavefield modeling. Journal of Scientific Computing , 92 (2): 48.
- [40] Xu, Z. J. (2018) Understanding training and generalization in deep learning by fourier analysis. arXiv preprint arXiv:1808.04295 .
- [41] Xu, Z. Q. J., Zhang, Y., &amp; Luo, T. (2022) Overview frequency principle/spectral bias in deep learning. arXiv preprint arXiv:2201.07395 .
- [42] Xu, Z. Q. J., Zhang, Y., &amp; Xiao, Y. (2019) Training behavior of deep neural network in frequency domain. In International Conference on Neural Information Processing , pp. 264-274.
- [43] Xu, Y. (forthcoming) Multi-grade deep learning. Communications on Applied Mathematics and Computation . arXiv preprint arXiv:2302.00150 published on February 1, 2023.
- [44] Xu, Y. (2023) Successive affine learning for deep neural networks. ArXiv, abs/2305.07996 , July 11, 2023.
- [45] Xu, Y. &amp; Zhang, H. (2022) Convergence of deep convolutional neural networks. Neural Networks , 153 : 553-563.

- [46] Xu, Y. &amp; Zhang, H. (2024) Convergence of deep ReLU networks. Neurocomputing 571 : 127174.
- [47] Xu, Y., &amp; Zhang, H. (2024) Uniform convergence of deep neural networks with Lipschitz continuous activation functions and variable widths. IEEE Transactions on Information Theory , 70 : No. 10, October 2024.
- [48] Zhang, C., Bengio, S., Hardt, M., Recht, B., &amp; Vinyals, O. (2021) Understanding deep learning (still) requires rethinking generalization. Communications of the ACM 64 (3): 107-115.
- [49] Zhou, D. X. (2020) Universality of deep convolutional neural networks. Applied and Computational Harmonic Analysis , 48 (2): 787-794.

## A Analysis of the Multi-Grade Deep Learning Model

It was established in [43] that when the loss function is defined in terms of the mean square error, a grade of MGDL either learns the zero function or results in a strictly decreasing residual error sequence.

Theorem 1. Let D be a compact subset of R s and L 2 ( D , R t ) denote the space of t -dimensional vector-valued square integral functions on D . If f ∈ L 2 ( D , R t ) , then for all i = 1 , 2 , . . . ,

<!-- formula-not-decoded -->

where N ∗ l is the SNN learned in grade l of MGDL, and for i = 1 , 2 , . . . , either f i +1 = 0 or

<!-- formula-not-decoded -->

All numerical examples presented in this paper validate Theorem 1.

## B Supporting material for Section 3

We provide in this appendix details for the numerical experiments in Section 3, including computational resources, network structures of SGDL and MGDL, the choice of activation function, as well as the optimizer and parameters used in the optimization process, and some supporting figures.

The experiments conducted in Sections 3.1, 3.2, and 3.4 of Section 3 were performed on X86\_64 server equipped with an Intel(R) Xeon(R) Gold 6148 CPU @ 2.4GHz (40 slots) or Intel(R) Xeon(R) CPU E5-2698 v3 @ 2.30GHz (32 slots). In contrast, the experiments described in Section 3.3 of Section 3 were performed on X86\_64 server equipped with AMD 7543 @ 2.8GHz (64 slots) and AVX512, 2 x Nvidia Ampere A100 GPU.

Wechoose ReLU as the activation function as in [30] for all the four experiments. For each experiment, for SGDL we test several network structures and choose the one produces the best performance. We then design MGDL, having the same total number of layers and the same number of neurons for each layer as the chosen SGDL structure, but their parameters are trained in multiple grades as described in Section 2. Details of the network structure will be described for each experiment.

The optimization problems for both SGDL and MGDL across the four experiments are solved by the Adam method [20] with 'Xavier' initialization [13]. In Sections 3.1, 3.2, and 3.4, the learning rate t k for the k -th epoch decays exponentially with each epoch [19], calculated as t k := t max e -γk , where γ := (1 /K ) ln( t max /t min ) represents the decay rate, with K being the total number of training epochs, t max and t min denoting the predefined maximum and minimum learning rates, respectively. In Section 3.3, we employed a fixed learning rate for both SGDL and MGDL, as numerical results indicate that the exponential decay learning rate performs poorly. We observe that when a network structure of SGDL is split into multiple grades of MGDL, the combined computing time required to train all grades in MGDL, each for K epochs, is comparable to the computing time needed to train the SGDL with K epochs, due to only SNNs involved in MGDL. Therefore, in all the four experiments, we train SGDL and all grades of MGDL for the same K epochs. We test SGDL and MGDL with the same set of the algorithm parameters, including t min , t max , batch size, and K , for all the experiments. Optimal parameters were selected based on the lowest VaRSE value for Sections 3.1 and 3.2, the highest PSNR value for Section 3.3, and the lowest validation loss for Section 3.4, within the range of parameters to be described for each example.

0.5

0.4

Amplitude

.3

0.1

0.0

25

50

75 100 125 150 175 200

Frequency [Hz]

0.5

0.4

Amplitude

0.3

0.2

0.1

0.0

0.5

0.41

0.3

0.2

Amplitude

0.1

0.0-

0.5-

0.4-

0.1

0.0-1

<!-- image -->

25

50

75 100 125 150 175 200

25

50

75 100 125 150 175 200

25

50 75 100 125 150 175 200

Figure 7: Amplitude versus one-side frequency plots for λ across settings 1-4.

## B.1 Section 3.1

For settings 1, 2, and 3, the network structure for SGDL is

<!-- formula-not-decoded -->

where [ n ] × N indicates N hidden layers, each with n neurons. The network structure for each grade of MGDL is given by

<!-- formula-not-decoded -->

Here, [ n ] F indicates a layer having n neurons with parameters, trained in the previous grades, remaining fixed during the training of the current grade. We employ this notation across the numerical experiment section without further mentioning. For setting 4, the most challenging case, we employ a deeper structure for SGDL, with two more hidden layers in addition to (11). Correspondingly, for MGDL we add one more grade:

<!-- formula-not-decoded -->

We now describe the search range of the parameters for all the four settings. We let I 1 := { 10 -4 , 10 -5 , 10 -6 } , I 2 := { 10 -3 , 5 × 10 -4 , 1 × 10 -4 } , I 3 := { 10 -5 , 10 -6 , 10 -7 } and I 4 := { 5 × 10 -5 , 10 -5 } . Then for both SGDL and MGDL, we test the pair ( t min , t max ) from all possible cases in the set ( I 1 × I 2 ) ∪ ( I 3 × I 4 ) . The batch size is chosen from 256 , 512 or the full gradient (denoted by 'Full') for each epoch. The total epoch number K is set to be 30,000.

The supporting figures for this experiment include:

- Figure 7, presenting amplitude versus one-side frequency plots for λ across settings 1-4;
- Figure 8, comparing the amplitude of one-side frequency for SGDL and MGDL across settings 1-4;
- Figure 9, illustrating the evolution of spectrum comparison between SGDL and MGDL for settings 1-4.

In Figure 9, the colors in these subfigures show the measured amplitude of the network spectrum at the corresponding frequency, normalized by the amplitude of λ at the same frequency. The colorbar is clipped between 0 and 1 , indicating approximation accuracy from the worst to the best when changing from 0 to 1 .

## B.2 Section 3.2

The network structure that we use for SGDL is

<!-- formula-not-decoded -->

and the grade network structure for MGDL is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

0.5

0.41

30000

0.3

0.2

20000

Amplitude

0.1

0.0-

10000

Epochs (SGDL)

0.5

0.4-

G 4

0.3

G 3

0.2

Amplitude

0.1

MGDL

G 2

0.0

G 1

- 1.0

- 0.8

- 0.6

• 0.4

0.2

25

50 75 100 125 150 175 200|

Frequency [Hz)

0.0

50

150

100

Frequency [Hz]

200

- 1.0

- 0.8

0.4

25

Frequency [Hz]

0.2

50 75 100 125 150 175 200

0.0

50

100

150

Frequency [Hz]

200

0.5

0.4

30000

Amplitude

0.3

20000

0.1-

0.0

Epochs (SGDL)

Amplitude

10000

- 10

- 0.8

- 0.6

0.4

0.2

75 100 125 150 175 200

30000

0.5

0.4-

0.3

0.2

20000

0.1-

0.0-

Epochs (SGDL)

10000

- 1.0

- 0.8

• 0.2

50 75 100 125 150 175 200

0.5

0.4-

30000

403-

20000

0.1 -

10000

Epochs (SGDL)

Figure 8: Comparison of SGDL (1st row) and MGDL (2nd row): Amplitude across settings 1-4.

<!-- image -->

Figure 9: Comparison of SGDL (1st row) and MGDL (2nd row) for settings 1-4 of Section 3.1: The evolution of spectrum.

<!-- image -->

For choices of t min and t max , we let I 1 := { 10 -4 , 10 -5 , 10 -6 } and I 2 := { 1 × 10 -3 , 5 × 10 -4 , 1 × 10 -4 } . For both SGDL and MGDL, we test the pair ( t min , t max ) from all possible cases in the set I 1 × I 2 , the batch size is chosen from 512 , 1024 , or the full gradient for each epoch, and the total number of the epochs K is set to be 30,000.

The supporting figures for this experiment include Figure 10, illustrating the evolution of spectrum comparison between SGDL and MGDL for settings 1-2 of Section 3.2. The mean of colorbar in Figure 10 is consistent with that in Figure 9.

## B.3 Section 3.3

The network structure for SGDL is

<!-- formula-not-decoded -->

and that for MGDL is

<!-- formula-not-decoded -->

For SGDL and all grades of MGDL, we select the learning rate from the set { 10 -2 , 5 × 10 -3 , 10 -3 , 5 × 10 -4 , 10 -4 } , choose the full gradient for each epoch, and set the total epoch number K to be 10 , 000 .

Amplitude

50

25

- 1.0

- 0.8

- 0.6

0.4

25 50 75 100 125 150 175 200

30000

100 -

20000

10000

Epochs (SGDL)

200 -

300 -

G 4

400 -

G 3

MGDL

G

2

500 1

G 1

100 -

200-

300 -

400 -

500

100

200

300

Frequency (Hz)

100

200

100

200

300

•SUI

- 0.8

0.2

0.0

- 1.0

0.8

MGDL

400

0.2

0.0

(a) G 1: 20.41

Frequency [Hz]

100

200

400

300

400

300

400

500

100

200

300

400

500

100

200

300

400

500

Figure 11: Comparison of MGDL and SGDL for image Cat. (a)-(d): Predictions of MGDL for grades 1-4, with the corresponding testing PSNR values indicated in the subtitles. (e): Prediction of SGDL with testing PSNR displayed in the subtitle. (f): Ground truth image

<!-- image -->

The quality of the reconstructed image in Section 3.3 is evaluated by the peak signal-to-noise ratio (PSNR) defined by

<!-- formula-not-decoded -->

where v is the ground truth image, ˆ v is reconstructed image, n is the number of pixels in v , and ∥·∥ F denotes the Frobenius norm of a matrix.

The supporting figures for this experiment include Figures 11-13, which display the predicted image for MGDL and SGDL corresponding to the Cat, Sea, and Building images, respectively.

30000|

20000|

10000

Epochs (SGDL)

100 -

200-

300

100-

200 -

300 -

• 10

• 0.8

• 0.0

- 10

- 0.8

- 0.6

0.2

- 0.0

- 1.0

- 0.8

• 0.6

• 0.4

• 0.2

- 0.0

Figure 10: Comparsion of SGDL (1st row) and MGDL (2nd row) for settings 1 and 2 of Section 3.2: The evolution of spectrum (the first and second columns for the learned functions on manifolds γ q with q = 4 and q = 0 , respectively, for setting 1, while the third and fourth columns for setting 2).

<!-- image -->

500

500

30000

20000

10000

Epochs (SGDL)

Epochs (SGDL)

30000

20000

10000

200

200 1

400

400

600

600

800

800

1000

1000 -

1200

1200

200

200 -

400

400

600

600

800

800-

1000

1000

1200

1200

250

250

250

250

500

750

1000

1250

1500

500

1250

1000

750

1500

(a) G 1: 18.62

(a) G 1: 17.29

500

1000

1250

750

1500

500

750

1000

1250

1500

500

750

1000

1250

1500

500

1000

1250

750

1500

(b) G 2: 21.50

(b) G 2: 18.95

500

750

1000

1250

1500

500

750

1000

1250

1500

(c) G 3: 23.42

(c) G 3: 20.67

Figure 12: Comparison of MGDL and SGDL for image Sea. (a)-(d): Predictions of MGDL for grades 1-4, with the corresponding testing PSNR values indicated in the subtitles. (e): Prediction of SGDL with testing PSNR displayed in the subtitle. (f): Ground truth image

<!-- image -->

Figure 13: Comparison of MGDL and SGDL for image Building. (a)-(d): Predictions of MGDL for grades 1-4, with the corresponding testing PSNR values indicated in the subtitles. (e): Prediction of SGDL with testing PSNR displayed in the subtitle. (f): Ground truth image

<!-- image -->

1750

1750

2000

2000

200

200 -

400

400

600

600

800

800

1000

1000

1200

1200

250

250

1750

1750

2000

2000

200 -

200 1

400 -

400

600

600

800

800

1000

1000

1200

1200

250

250

1750

1750

2000

2000

9

0.20

0.181

0.16

0.14-

0.12 -

0.10

0.08 -

0.06

3.0

2.8 1

2.6

2.4 -

2.2

2.01

1.8 1

1.6

SGDL

1000

2000

3000

Training time [s]

SGDL

1000

Training time (S)

0.20

0.18

0.161

0.141

93013-

0.10 -

0.08 -

0.06 -

(a)

3.0

2.8

2.6

฿.-

2.0 -

1.81

1.6

MGDL

K = 1

= 1

K = 3

K = 3

K = 5

K = 5

K = 7

0.50

0.45 -

0.40 1

0.35

0.30 -

0.50

0.451

0.35

0.30 -

K = 1

= 1

= 3

K =

= 5

5

=

K =

7

Figure 14: Comparison of SGDL and MGDL with structure (14): training (dash curve) and validation (solid curve) loss versus training time for various values of β and κ : (a) β = 0 . 5 , (b) β = 1 , (c) β = 3 , (d) β = 5 .

<!-- image -->

## B.4 Section 3.4

The network structure for SGDL is

Loss

SGDL

Loss

MGDL

<!-- formula-not-decoded -->

For MGDL, we consider two grade splittings. In the first splitting, we split network (13) into three grades, each with two hidden layers. The structure of MGDL for this splitting is as follows:

<!-- formula-not-decoded -->

In the second splitting, we split network (13) into six grades, with each grade containing one hidden layer. The structure for MGDL for this case is as follows:

<!-- formula-not-decoded -->

For choices of parameters t min and t max , we let I 1 := { 10 -4 , 10 -5 } and I 2 := { 10 -3 , 10 -4 } . For both SGDL and MGDL, we test ( t min , t max ) from all possible cases in the set I 1 × I 2 , choose the batch size from 512, 1024, or the full gradient for each epoch, and set the total number of the epochs K to be 2,000.

The supporting figures for this experiment include Figure 14, which compares the training and validation loss for SGDL and MGDL with structure (14); Figure 15, which compares the training and validation loss for SGDL and MGDL with structure (15).

0.20

0.18 -

0.16 -

0.141

9, 0.121

0.10

0.08 -

0.06

3.0

2.8 1

2.6

2.41

2.2-

2.01

1.81

1.6

SGDL

1000

2000

3000

Training time [s]

SGDL

1000

2000

3000

Training time [s]

Loss

0.20

0.18

0.16 -

0.14

8012-

0.10-

0.08 -

0.06

(a)

3.0

2.8

2.6

2.41

2.2

2.0-

1.8

1.6

MGDL

K = 1

= 1

K = 3

K = 3

= 5

K = 5

K = 7

0.50

0.45 -

0.40 1

0.35

0.30

0.50

0.451

0.40 1

0.35

0.30-

K = 1

K = 1

K = 3

3

K =

= 5

K = 5

K =

7

Figure 15: Comparison of SGDL and MGDL with structure (15): training (dash curve) and validation loss (solid curve) versus training time for varies values of β and κ : (a) β = 0 . 5 , (b) β = 1 , (c) β = 3 , (d) β = 5 .

<!-- image -->

##