## ROTATION INVARIANT QUANTIZATION FOR MODEL COMPRESSION

Anonymous authors Paper under double-blind review

## ABSTRACT

Post-training Neural Network (NN) model compression is an attractive approach for deploying large, memory-consuming models on devices with limited memory resources. In this study, we investigate the rate-distortion tradeoff for NN model compression. First, we suggest a Rotation-Invariant Quantization (RIQ) technique that utilizes a single parameter to quantize the entire NN model, yielding a different rate at each layer, i.e., mixed-precision quantization. Then, we prove that our rotation-invariant approach is optimal in terms of compression. We rigorously evaluate RIQ and demonstrate its capabilities on various models and tasks. For example, RIQ facilitates × 19 . 4 and × 52 . 9 compression ratios on pre-trained VGG dense and pruned models, respectively, with &lt; 0 . 4% accuracy degradation. The code is available in the supplementary material.

## 1 INTRODUCTION

Deep neural networks are widely used for various tasks, such as computer vision, Natural Language Processing (NLP), and recommendation systems. Nevertheless, while performance continuously improves, the models become larger with a massive increase in the number of parameters. In fact, modern Neural Network (NN) models may have trillions of parameters, which makes the deployment of these models a challenging task (Chang et al., 2020). One way to mitigate this issue is compressing the model's parameters to reduce its overall memory footprint while satisfying an accuracy constraint. Namely, obtaining a smaller model that is (almost) as capable as the original model.

The most common model compression techniques are weight pruning, quantization, knowledge distillation, and low-rank decomposition. Such optimizations strive to find a smaller model while keeping the original model's accuracy, overlooking the potential inherent in its entropy limit. In the context of NN models, the entropy value is of particular interest as it provides the (theoretical) number of bits required for representing the model parameters. The optimal compression asymptotically attains this entropy limit.

In this context, lossy compression gives considerable merit as it facilitates reducing the NN size significantly with negligible accuracy degradation. The key steps of this approach are the quantization and the encoding (Polyanskiy &amp; Wu, 2022, Ch. 24). In the quantization phase, the number of unique weight values (symbols) is reduced, consequently reducing the model's entropy. Modifying the symbols' statistics, however, introduces distortion (i.e., quantization error) with respect to the original model. Hence, proper quantization methods are substantial as they determine both the resulting entropy and the distortion of the model's output. In the encoding phase, redundant information is removed, reaching the most compact representation possible without introducing further errors. Finding a solution that quantizes the model at the lowest possible bit rate while satisfying a certain distortion requirement is at the heart of quantization optimization problems, and is known as the rate-distortion problem (Cover &amp; Thomas, 2006).

Typically, NN model quantization optimizes a certain distortion (or accuracy) for a given target rate, which is suitable for reducing the computational complexity of the NN model to a lower precision. However, it is reasonable to consider the dual problem, which optimizes the compression rate while satisfying a certain distortion (or accuracy) requirement. The latter has attained very limited attention in the literature, yet, holds many fascinating challenges in its design, which motivated this study to understand the compression limits in the context of NN.

In this study, we investigate the rate distortion for NN models compression, where the distortion is measured by a cosine distance between the outputs of the original and the quantized model (i.e., angular deviation). In particular, we formulate the model compression as an optimization problem, where the goal is maximizing the model compression ratio subject to a given deviation constraint. Our focus is mixed-precision solutions , where each layer gets quantized at a possibly different rate. The advantages of mixed-precision models can be manifested in NLP models, which typically have high redundancy and require extensive memory and loading times. Specifically, the main contribution is as follows.

- We design a post-training Rotation-Invariant Quantization (RIQ) which compresses NN models subject to a deviation constraint. The main theme of our approach is picking the quantization bin width to be proportional to the layers' norm. Since both norm and cosine distance are invariant to rotations, this yields the optimal solution in terms of rate distortion. To find the optimal solution efficiently, we first derive the scale in which the rate increases with the deviation and then suggest a searching paradigm that leverages our finding.
- To analyze the optimality of the RIQ algorithm, we introduce a universal surrogate model that depicts quantization in terms of rotation of the model weights. Its analysis reveals that the rate-distortion minimizing distribution for NN models is a spherical (rotation invariant) distribution constructed by the product of layers' spherical distribution. Due to convexity, the rate achieved under this product distribution is bounded by a rate achieved under the layers' average spherical distribution. Since RIQ optimizes the rate only over spherical distributions, it reaches the optimal solution efficiently.
- We rigorously evaluate the RIQ and demonstrate its capabilities on various models and tasks. RIQ attains a remarkable compression ratio with a negligible accuracy loss in all assessments, surpassing recent results in the literature.

## 2 RELATED WORK

This section is devoted to prior work on model compression that is most relevant to this study. Roughly speaking, typical model compression methods can be classified into four categories. Weight pruning, quantization, knowledge distillation (Sarfraz et al., 2021; Walawalkar et al., 2020), and low-rank decomposition (Idelbayev &amp; Carreira-Perpinán, 2020; Lin et al., 2018; Lee et al., 2019). Even though such methods strive to find a smaller model while retaining the model's accuracy, they often tend to neglect the potential inherent in the entropy limit. In this study, we seek to minimize the model entropy by quantization and then attain this entropy limit by compression while satisfying a distortion requirement at the model's output.

Quantization is a prominent method for compressing NN models. In Wu et al. (2020); Banner et al. (2019); Idelbayev et al. (2021), the authors considered fixed-bit quantization methods, where all layers are quantized at the same integer bit rate. In this paper, on the other hand, we consider post-training mixed-precision solutions. Bhalgat et al. (2020); Wang et al. (2019) focused on quantization-aware training, where the weights quantization is performed during the training. To attain lower quantization rates, Fan et al. (2020); Baskin et al. (2021); Défossez et al. (2022) suggested training the models with noise. Although quantization-aware training methods may achieve better results than the post-training approaches, they are time-consuming, and requires ample representative data and training resources. Nevertheless, these works optimize the distortion, while RIQ optimizes the rate for a given distortion requirement.

The idea of pruning NN connections based on information-theoretic ideas was explored already in the seminal works (LeCun et al., 1989; Hassibi et al., 1993). Later, Han et al. (2015) used magnitude threshold to remove redundant parameters and then utilized Huffman's entropy coding to compress these parameters. Since then, pruning techniques gained popularity, searching for effective methods to prune parameters (Zhang et al., 2021; Frankle &amp; Carbin, 2018; Lee et al., 2019), showing that entropy reduction during training is beneficial, as low-entropy models are more amenable to compression (Oktay et al., 2019; Baskin et al., 2019). Yet, the first rate-distortion formulation of model compression was suggested by Gao et al. (2019); Isik et al. (2022), providing significant insights about the optimal model pruning. In particular, Gao et al. (2019) studied the two-layer networks, providing tractable analysis when assuming the Gaussian distribution of the model weights. This assumption, however, may not hold in general. Isik et al. (2022) observed empirically that

the weights of a pretrained model tend to follow the Laplacian distribution, for which the optimal compression algorithm that resides on the rate-distortion curve must output a sparse model. This study analyzes the rate-distortion of rotation-invariant quantization solutions, which generalizes these findings, as the Gaussian and Laplacian distributions are rotation-invariant distributions.

## 3 PRELIMINARIES

In this section, we rigorously define the model compression optimization problem and the relevant known results on quantization and the rate-distortion theory. Throughout, bold w denote weight vectors, unless stated otherwise. ∥ · ∥ and ⟨· , ·⟩ denotes the standard ℓ 2 -norm and the inner product, respectively. We use p w ( w ) to denote the probability distribution of a random variable w. Hereafter, w [1: L ] = { w 1 , ..., w L } ∈ R N , denotes the concatenation of the weights of a pretrained model with L layers, where w ℓ ∈ R n ℓ are n ℓ weights of layer ℓ and N = ∑ L ℓ =1 n ℓ . The quantized representation of those weights is denoted by ˆ w ℓ .

## 3.1 PROBLEM STATEMENT

Let f : R n x → R n y be a pretrained model that characterizes the prediction of input x to an output y ∈ R n y . The model comprises L intermediate mappings called layers, such that f ( x ) = f w L ( f w L -1 ( · · · f w 1 ( x )) ) , where w ℓ ∈ R n ℓ are the weights of layer ℓ . We further assume that each layer performs an affine operation (e.g., convolution) followed by a nonlinear operation (e.g., ReLU).

Our goal is to obtain the smallest (quantized and compressed) version of this model ˆ f , whose output is as close as possible to the output of f . To assess the fidelity of the quantization, a sample x is sent through f and ˆ f , and the deviation between the outputs f ( x ) and ˆ f ( x ) is measured. In this study, we focus on the cosine distance as distortion measure. That is,

<!-- formula-not-decoded -->

This distortion reflects the rotation angle that is required to align ˆ f ( x ) with f ( x ) . Noticeably, eq. (1) is a rotation-invariant measure, which means that its value does not change when an arbitrary rotation is applied to its arguments. That is, the cosine value does not change when rotating together ˆ f ( x ) and f ( x ) . Other distortion measures, such as the l 2 and l 1 distances, can also be considered for analyzing model compression (Gao et al., 2019; Isik et al., 2022). However, due to the curse of dimensionality, they are mainly useful when addressing low-dimensional optimization (Aggarwal et al., 2001). On the other hand, cosine distance is beneficial to measure the similarity between vectors in high-dimensional spaces, which naturally occurs in models. Further, the correctness of the models in many tasks is determined by the orientation of the output vector rather than the magnitude of the output vector. This motivates us to optimize the quantization rate under the cosine distance measure. Hereafter, the term deviation is used to depict d f, x ( w [1: L ] , ˆ w [1: L ] ) , which is the cosine distance between the outputs, whereas the term distortion refers to the cosine distance d ( w ℓ , ˆ w ℓ ) between the weights w ℓ and their quantized representation ˆ w ℓ .

As x passes the first layer of f and ˆ f , it is rotated (and scaled) by w 1 and ˆ w 1 , respectively. Due to the quantization, w 1 and ˆ w 1 acts differently on x , and thus, yields unequal outputs. These unequal outputs are rotated (and scaled) by the next layer's weights, and so on, reaching the output of f and ˆ f . The resulting deviation in eq. (1) relates to the distortions gathered through the layers. In particular, each quantized layer produces a rotation distortion in its output, and this distortion keeps propagating and accumulating through the layers until reaching the model's output.

Thus, characterizing the connection between the deviation to the distortion in each layer is a key to optimizing the model quantization. Though finding the exact link is intricate in general, intuitively, as the layers' quantization rate jointly decrease, the deviation increases monotonically with the layers' distortion. Since these distortions are invariant to rotations, the rate-distortion theory tells that the optimal quantization must be rotation invariant as well, as we show in the sequel. This motivates a searching paradigm over rotation-invariant solutions, where the layers' rate are jointly adjusted.

Formally, given a trained model f , and a sample x , we wish to find a quantized model ˆ f whose weights ˆ w [1: L ] solves the following optimization problem.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some deviation requirement D , where the rate is the average bits per symbol over the entire quantized model ˆ f . In this work, we characterize the properties of the minimum rate and devise a searching method that finds the minimum rate that satisfies D efficiently.

In this formulation, we consider mixed-precision quantization solutions, where the weights w ℓ of each layer ℓ are quantized at a different rate R ℓ . In other words, each layer utilizes a different number of symbols. Typically, using fewer symbols induces lower entropy, and hence, we strive to minimize the number of symbols in each layer. Though the entropy indicates the shortest representation of a layer, one must encode the layers' weights to reach this entropy limit. In this study, we utilize Asymmetric Numeral Systems (ANS) arithmetic encoder for this purpose (Duda, 2013). Accordingly, the resulting compression ratio, assuming 32 bits representation of the source symbols, is

<!-- formula-not-decoded -->

Where | T ℓ | denotes the coding table size of layer ℓ and H ( ˆ w ℓ ) is the empirical entropy.

## 3.2 RATE-DISTORTION THEORY

The rate-distortion theory determines the minimum number of bits per symbol , or simply the minimum bit rate, required for describing a random variable with a certain (average) distortion. In particular, to quantize a sequence of n independent realizations w = ( w 1 , · · · , w n ) , generated by a source w with distribution p w ( w ) , w ∈ W n into R bits, encoding and decoding functions are utilized. The encoder E : W n → { 0 , 1 } R maps the sequence to one of 2 R possible indices, and the decoder D : { 0 , 1 } R → ˆ W n maps the given index into an estimated (quantized) sequence ˆ w = ( ˆ w 1 , · · · , ˆ w n ) . Thus, the rate-distortion pair ( R,D ) are the resulting rate R and distance D = d ( w, ˆ w ) between the original sequence and the quantized sequence.

In general, we wish to minimize both the rate and the distortion, however, there is an inherent tradeoff, characterized by rate-distortion function as (Cover &amp; Thomas, 2006, Ch. 10)

<!-- formula-not-decoded -->

where I ( w ; ˆ w ) = H ( w ) -H ( w | ˆ w ) is the mutual information between the source vector w and its reconstruction ˆ w (Cover &amp; Thomas, 2006, Ch. 2.4), and d ( · , · ) is a predefined distortion metric, such as the cosine distance in eq. (1). Thus, the rate-distortion function determines the infimum rate R that achieves a given distortion D . This infimum is attained by minimizing overall conditional distributions p (ˆ w | w ) for which distortion D is satisfied under p ( w ) .

## 3.3 UNIFORM SCALAR QUANTIZATION

The rate-distortion theory tells that it is optimal to describe the whole sequence jointly, using one of 2 R indices, even when the variables are i.i.d. Yet, in terms of entropy, Koshelev (1963) showed that uniform scalar quantization is (asymptotically) optimal when one intends to further compress the quantized data losslessly. Since this paper considers the latter approach, this section briefly discusses uniform scalar quantization and its analysis.

For a random variable w ∈ [ -A/ 2 , A/ 2] , where A ∈ R , uniform quantization partitions the range [ -A/ 2 , A/ 2] into N bins uniformly, such that each bin has width ∆ = A/N . Thus, any realization of w is encoded (rounded) into an integer value , ⌈ w / ∆ ⌋ , that corresponds to its bin index. The decoder then reconstructs its value by

<!-- formula-not-decoded -->

The fidelity of this quantization is measured by a distortion measure, such as the Mean Squared Error (MSE) criterion, defined as D ( N ) = E | w -ˆ w | 2 . To analyze, it is more convenient to examine the

quantization in terms of rate R = log 2 N . In high-rate regime (i.e., R ≫ 1 ), the probability density in each bin is nearly flat, and consequently, the distortion is (Polyanskiy &amp; Wu, 2022, Ch. 24.1)

<!-- formula-not-decoded -->

Further, the resulting entropy of the quantized symbol is (Cover &amp; Thomas, 2006, Theorem 8.3.1)

<!-- formula-not-decoded -->

where H ( w ) = -∑ w ∈W p ( w ) log p ( w ) is the entropy function (Cover &amp; Thomas, 2006, eq. (2.1)). In other words, quantization reduces the entropy by log ∆ , and thus, a larger ∆ yields a lower entropy, and hence, potentially, a higher compression ratio by eq. (2).

## 4 ROTATION-INVARIANT MIXED-PRECISION QUANTIZATION

In this section, we present the RIQ method, which yields a different quantization rate in each layer (i.e., mixed-precision solution) while satisfying the deviation requirement in eq. (1). Then, we use the rate-distortion theory to analyze its performance.

## 4.1 THE RIQ ALGORITHM

Given a model and a deviation requirement, it is sufficient to optimize the rate over rotation-invariant solutions. Typical quantization methods which optimize the distortion for a given rate, on the other hand, determine ∆ ℓ according to the range max( w ℓ ) -min( w ℓ ) , whose value depends on the orientation of w ℓ , and hence, are not rotation invariant, which is substantial for optimality.

RIQ designs the bin width, ∆ ℓ , in proportion to the norm ∥ w ℓ ∥ in each layer. Since norm is invariant to rotations, the resulting ∆ ℓ is indifferent to the orientation of w ℓ . Consequently, the resulting distortion is also indifferent to the orientation of w ℓ , as the bin width dictates the distortion.

Let θ ℓ be the rotation angle from w ℓ to ˆ w ℓ such that ⟨ w ℓ ∥ w ℓ ∥ , ˆ w ℓ ∥ ˆ w ℓ ∥ ⟩ ≜ cos( θ ℓ ) . The following lemma examines the relation between ∆ ℓ , the norm ∥ w ℓ ∥ , and the distortion d ( w ℓ , ˆ w ℓ ) = 1 -cos( θ ℓ ) .

Lemma 1. Let ϵ ℓ ≜ 1 -cos( θ ℓ ) be the distortion of layer ℓ . Then, in the high-rate regime, the quantization bin width asymptotically satisfies

<!-- formula-not-decoded -->

The proof of Lemma 1 is elaborated in Appendix A.1. Lemma 1 links between angular distortion, bin width, norm, and dimension. For instance, from the lemma follows that 1 -cos( θ ℓ ) = n ℓ · ∆ 2 ℓ 24 ·∥ w ℓ ∥ 2 , i.e., the distortion scales linearly with the dimension n ℓ . Interestingly, this connection expands to the entire model as follows.

Corollary 1. Let w [1: L ] be a vector representation of the weights and ˆ w [1: L ] denotes its quantized representation, and let θ [1: L ] be the rotation angle from w [1: L ] to ˆ w [1: L ] such that ⟨ w [1: L ] ∥ w [1: L ] ∥ , ˆ w [1: L ] ∥ ˆ w [1: L ] ∥ ⟩ ≜ cos( θ [1: L ] ) . Assuming ∥ ˆ w ℓ ∥ = ∥ w ℓ ∥ + o ( ∥ w ℓ ∥ ) , then,

<!-- formula-not-decoded -->

The proof is deferred to Appendix A.2. In words, a rotation of w [1: L ] translates to a convex combination of the layers' rotation, and vice versa. Interestingly, due to convexity of the rate-distortion, it is beneficial to average over as many rotations as possible, which means considering partitioning the model parameters into shorter vectors as we show in the sequel. Still, the most natural partition of the model is simply to it layers, which is considered herein.

The connection of ∆ ℓ to ∥ w ℓ ∥ in Lemma 1 hints at the rotation-invariant nature of the optimization. To focus on rotation-invariant solutions, RIQ introduces a search parameter k that maintains proportion with ∥ w ℓ ∥ , allowing efficient search over these solutions. Specifically, when ∆ ℓ ( k ) = ∥ w ℓ ∥ /k where k to be optimized, the bin-width is indifferent to the orientation of w ℓ (i.e., rotation invariant). Further, letting the bin width grow linearly with ∥ w ℓ ∥ results in distortion ϵ ℓ = 1 -cos( θ ℓ ) = n ℓ 24 · k 2 , and hence by Corollary 1, 1 -cos( θ [1: L ] ) = 1 24 · k 2 ∑ L ℓ =1 n ℓ ·∥ w ℓ ∥ 2 ∥ w [1: L ] ∥ 2 . Namely, the distortion of the entire parameters scales as O (1 /k 2 ) . Remarkably, the deviation also scales as O (1 /k 2 ) .

## Algorithm 1 The RIQ algorithm

```
Input: model weights w [1: L ] , deviation requirement D , minimum error ϵ 0 Initialize k min = √ n ℓ ∗ / 24 1 -ϵ 0 , k max = √ n ℓ ∗ / 24 √ ϵ 0 · ϵ 0 , k = k min , step = √ k max -k min . while k ≤ k max do for ℓ = 1 to L do set ∆ ℓ = ∥ w ℓ ∥ · ( 1 k + ϵ 0 · √ 24 n ℓ ) and quantize by ˆ w ℓ = ⌈ w ℓ ∆ ℓ ⌋ · ∆ ℓ end for if d f,x ( w [1: L ] , ˆ w [1: L ] ) ≤ D then if step ≤ 3 then compress to H ( ˆ w [1: L ] ) with entropy encoder end if k max = k ; step = √ step; k = k -step · ⌊ step ⌋ end if k = k + step end while
```

Proposition 1. In the high rate regime, the deviation in eq. (1) under RIQ scales as O (1 /k 2 ) .

The proof is deferred to Appendix A.3. Essentially, since ∆ ℓ ( k ) and ϵ ℓ are monotonically decreasing with k , then by eq. (6), the entropy increases with k . This allows RIQ is to reach the smallest k (i.e., minimum entropy) solution that satisfies the deviation requirement in eq. (1).

Next, we introduce an efficient iterative searching algorithm for finding the optimal k . In each iteration, the algorithm refines the searching range until reaching the smallest k (up to a small constant) that satisfies the deviation requirement. Practically, however, as k increases, ∆ ℓ ( k ) → 0 . To prevent this, we add a small constant ϵ 0 to √ ϵ ℓ , which bounds the value of the smallest ∆ ℓ ( k ) . In this case, setting √ ϵ ℓ = 1 k √ n ℓ 24 + ϵ 0 , yields,

<!-- formula-not-decoded -->

Even though the search of k is unbounded in general, practically it is sufficient to search in bounded space since the weights' norm is finite (Idelbayev et al., 2021). In the following proposition, we derive searching bounds for the optimal k . Let k ∗ be the optimal (smallest) k that satisfies constraint D , and let ℓ ∗ be the index of the layer with the largest n ℓ in f .

<!-- formula-not-decoded -->

The proof is deferred to Appendix A.4. To further improve the search time, a nested refinement is utilized. Specifically, at each stage, only √ | O ( k ) | values of k in ascending order are considered. Once a certain value of k satisfies the deviation requirement D , this k becomes the new upper limit for the search, and the search region is refined within a smaller region of k to consider, again with only √ | O ( k ) | potential values to inspect. This repeatedly continues until the search step is sufficiently small (e.g., step ≤ 3 ), and the compression gain becomes negligible. These refinements enable fast convergence in relatively few iterations. See Algorithm 1 for a detailed description of RIQ.

Remark. The additional degree of freedom that ϵ 0 gives is substantial. For example, it facilitates enforcing quantization to a maximum of R bits (e.g., R = 8 bits) for low precision runtime, by setting the limit k →∞ at ϵ 0 ( ℓ ) = max( w ℓ ) -min( w ℓ ) 2 R -1 / √ 24 ·∥ w ℓ ∥ 2 n ℓ .

For simplicity, in the sequel we apply the same small common constant value ϵ 0 to all layers.

## 4.2 RIQ RATE-DISTORTION ANALYSIS

In this section, we provide theoretical justification for the optimality of RIQ. We introduce a surrogate model for which the rate-distortion analysis with cosine distance is tractable, showing that the minimizing distribution of the mutual information is indifferent to the orientation of w ℓ , and is characterized by a single parameter k , as RIQ suggests.

First, extending eq. (4) to NN model quantization, where layer ℓ is encoded uniformly, yields

<!-- formula-not-decoded -->

For tractability, it is common to analyze the rate-distortion for eq. (8) by a surrogate model in which the distortion is modeled as a random additive noise (Kipnis &amp; Reeves, 2021; Marco &amp; Neuhoff, 2005). Yet, when considering angular deviation such representation hinders the rate-distortion analysis since the relation between additive noise to cosine distance that we wish to examine is intricate. Accordingly, we suggest a reparameterization to the additive noise model, which represents quantization as random rotation (and scale) of the weights in each layer. This enables to analyze the rate-distortion of eq. (8) for the deviation in eq. (1) in the sequel.

Surrogate Model. Let w ℓ be the weights of layer ℓ , and let ˆ w ℓ denote their quantized representation. Let θ ℓ be a random rotation angle from w ℓ to ˆ w ℓ , such that ⟨ w ℓ ∥ w ℓ ∥ , ˆ w ℓ ∥ ˆ w ℓ ∥ ⟩ = cos( θ ℓ ) , and let U ( θ ℓ ∣ ∣ w ℓ ) ∈ SO ( n ℓ ) be a random orthogonal transformation corresponding to a random rotation that is θ ℓ away from w ℓ . Then,

<!-- formula-not-decoded -->

models the quantized weights ˆ w ℓ .

Intuitively, U ( θ ℓ ∣ ∣ w ℓ ) randomly rotates any given vector uniformly on a sphere, where one degree of freedom is lost due to the requirement of being θ ℓ away from w ℓ . To obtain ˆ w ℓ in eq. (8), the realization of U ( θ ℓ ∣ ∣ w ℓ ) should rotate the unit vector w ℓ / ∥ w ℓ ∥ in the plane generated by w ℓ and ˆ w ℓ , and then, stretches it into the length ∥ ˆ w ℓ ∥ (see Figure 3(a) for illustration).

In other words, this model describes a random vector ˜ w ℓ that is uniformly distributed on a cone that is θ ℓ away from w ℓ , for which a single realization matches eq. (8). The merit of this model is its tractable analysis, from which spherically symmetric distribution emerges to depict the quantized weights (Fang et al., 2018, Definition 2.1). Accordingly, each layer obtains a randomly rotated version of w ℓ , which translates to a joint rotation at angle θ [1: L ] of all the parameters by Corollary 1. Consequently, in the high rate regime, where the support of θ ℓ is sufficiently small, the distortion and the deviation decrease at same scale with k by Proposition 1.

Proposition 3. Let w ℓ be the weights of layer ℓ , and let ˜ w ℓ model the quantized representation of those weights, modeled by eq. (9) . Then, ˜ w ℓ ∣ ∣ w ℓ has a spherical (rotation-invariant) distribution.

A detailed proof is given in Appendix A.5. Essentially, the strength of Proposition 3 is twofold. First, it proves that the distribution of ˜ w ℓ ∣ ∣ w ℓ in each layer does not change when arbitrary rotations are applied to it. Second, it holds for any distribution of w ℓ and θ ℓ . This universality is substantial when considering various models and tasks. The following theorem extends the results of Proposition 3 to multiple layers, showing that spherical distribution is also the rate-distortion minimizing distribution.

Theorem 1. Let f be a NN model with L statistically independent layers whose weights are w [1: L ] , and let ˜ w [1: L ] be their quantized representation. Then, the unique minimizing distribution p (˜ w [1: L ] ∣ ∣ w [1: L ] ) of the rate-distortion function

<!-- formula-not-decoded -->

is a product distribution constructed as the product of the layers' spherical distribution. Consequently, the infimum rate is characterized by a single parameter.

The detailed proof is given in Appendix A.6. The key steps of the proof are, first, showing that the minimizing distribution is a product distribution. Then, due to the convexity of the rate-distortion in eq. (10), we bound the mutual information with a convex combination of θ ℓ distributions, given in Corollary 1. Consequently, the problem is simplified to a single (average) layer optimization, which is governed by a single quantization parameter.

Remarkably, the joint minimizing distribution of the model's weights p ( ˜ w [1: L ] ∣ ∣ w [1: L ] ) is also spherical since any partitioning of spherical distribution (naturally occurring by the model's layers) remains spherical (Fang et al., 2018, Theorem 2.6). Further, due to the convexity of the mutual information, it

Figure 1: (a) Cosine distance of the validation dataset as a function of the deviation constraint (on the calibration dataset). Models include VGG (green), ResNet-50 (red), and ViT (blue). (b) Accuracy vs Compression for ResNet-50 model. RIQ (green) vs. RAP (orange), OCS (red), linear quantization (cyan), and HAQ (purple). HAQ, however, requires training after the quantization.

<!-- image -->

is beneficial to consider as many partitions as possible, which can only reduce the mutual information. Practically, however, running the model layer-by-layer, where each layer is quantized by a single scalar, has merit due to its simplicity. We point out that the minimizing distribution cannot be found explicitly without any assumption on the distribution of the weights. Interestingly, the optimal solutions presented by Gao et al. (2019); Isik et al. (2022) for the Gaussian and Laplace distributions coincide with our rotation-invariant observation as these distributions are spherical distributions as well. Characterizing a universal rate-distortion problem is substantial as it reveals that it is sufficient to consider only rotation-invariant solutions for any model and task.

## 5 EMPIRICAL RESULTS

In this section, we evaluate the compression ratio and model accuracy of RIQ with ANS and compare them to relevant baseline results. In all experiments, we use pre-trained models for their relevant tasks. Specifically, for classification tasks we use VGG, Simonyan &amp; Zisserman (2014), ResNet-50, He et al. (2016), and ViT, Dosovitskiy et al. (2020) models, from the torchvision library 1 on the ImageNet data-set (I1k). For detection tasks, we use YOLOv5, Jocher et al. (2022), from Neural-Magic library 2 on the COCO dataset (Lin et al., 2014). For NLP task, we use the DistilBERT model, Sanh et al. (2019), on SQuAD dataset, (Rajpurkar et al., 2016). Following RIQ, an ANS encoder encodes each quantized layer to its entropy limit. An efficient implementation of ANS on GPU was demonstrated by Weißenberger &amp; Schmidt (2019), reaching impressive decoding rates. Consequently, RIQ facilitates optimizing and compressing models in a few minutes. For reproduction purposes, we provide a Python code of our algorithm as a supplementary material which includes both the quantization phase (RIQ) and compression phase (ANS). Additional results are given in Appendix A.8.

To measure the resulting deviation by eq. (1) on a validation set as a function of a deviation requirement D , we use two types of calibration data: (a) three real images, sampled from the training, and (b) randomly generated data that follows the Gaussian distribution. In Figure 1(a) we present the deviation measurements on three models: ResNet-50, VGG, and ViT, where the identity line (black-dotted) is given for reference. As we see, even a small calibration set of three images (solid lines) is sufficient to predict the deviation on the validation set. Further, we see that the randomly generated data may not predict well the resulting deviation on the validation set, leading to either a less compressed model (ResNet-50 and VGG) or a higher deviation (ViT).

In Figure 1(b), we evaluate the effect of RIQ on the accuracy and the inverse compression ratio (i.e., the reciprocal of eq. (2)) for a pre-trained ResNet-50 model. Interestingly, the rate-distortion curve reflects well the accuracy-compression tradeoff. For comparison, we depict the accuracy-compression results of the Relaxed Advanced Pipeline (RAP) method, Hubara et al. (2021), Outlier Channel Splitting (OCS), Zhao et al. (2019), and the Hardware-aware Automated Quantization (HAQ), Wang et al. (2019), which requires further training for fine-tuning. Indeed, RIQ surpasses other post-training

1 pytorch.org/vision/stable/models.html 2 sparsezoo.neuralmagic.com

quantization methods, yet, falls short compared to HAQ. This is since retraining the quantized model yields a different rate-distortion curve, which is out of the scope of this paper.

Table 1: Comparison of Top-1 accuracy on ImageNet for various baselines.

| Model     | Compression       | Method          | Acc. (%)    | Ref. (%)    | Drop (%)   |
|----------|----------|----------|----------|----------|----------|
| VGG-16    | × 10 . 6 (3 bits) | GPFQ Zhang et al. (2022a) RIQ (Ours) | 70.24 71.58 | 71.59 71.59 | 1.35 0.01  |
| VGG-16    |          | MSE Banner et al.          | 70.50       |          |          |
| VGG-16    |          | (2019)          |          | 71.60       | 1.10       |
| VGG-16    |          | OMSE Choukroun et al. (2019)         | 71.48       | 73.48       | 2.00       |
| VGG-16    | × 8 (4 bits)      | GPFQ Zhang et al. (2022a)          | 70.90       | 71.59       | 0.69       |
| VGG-16    |          | RIQ (Ours)          | 71.55       | 71.59       | 0.04       |
| VGG-16    | × 6 . 4 (5 bits)  | GPFQ Zhang et al. (2022a)          | 71.05       | 71.59       | 0.54       |
| VGG-16    |          | RIQ (Ours)          | 71.58       | 71.59       | 0.01       |
|          | × 16 (2 bits)     | SuRP with tuning Isik et al. (2022)  | 76.4        | 76.6        | 0.2        |
|          |          | RIQ (Ours)          | 69.47       | 76.14       | 6.67       |
|          | × 10 . 6 (3 bits) | GPFQ Zhang et al. (2022a)          | 70.63       | 76.13       | 5.50       |
|          |          | RIQ (Ours)          | 74.76       | 76.14       | 1.38       |
|          |          | MSE Banner et al. (2019)          | 73.80       | 76.10       | 2.30       |
|          |          | OMSE Choukroun et al. (2019)         | 73.39       | 76.01       | 2.62       |
| ResNet-50 |          | AdaRound Nagel et al. (2020)         | 75.23       | 76.07       | 0.84       |
|          | × 8 (4 bits)      | S-AdaQuant Hubara et al. (2021)      | 75.10       | 77.20       | 2.10       |
|          |          | BRECQ Li et al. (2021)          | 76.29       | 77.00       | 0.71       |
|          |          | GPFQ Zhang et al. (2022a)          | 74.35       | 76.13       | 1.78       |
|          |          | RIQ (Ours)          | 75.61       | 76.14       | 0.53       |
|          | × 6 . 4 (5 bits)  | GPFQ Zhang et al. (2022a)          | 75.26       | 76.13       | 0.87       |
|          |          | RIQ (Ours)          | 75.95       | 76.14       | 0.19       |

In Table 1, we compare RIQ to relevant baseline methods on the VGG-16 and ResNet-50 models. In this table, we modified RIQ in Algorithm 1 to minimize the deviation (accuracy drop) for a given rate requirement. Noticeably, RIQ outperforms most of the baselines, as it reaches the entropy limit by the ANS. Applying ANS to other baselines, however, can only degrade their compression since they consider per-channel quantization, for which the encoding table overhead becomes significant. Further, applying ANS to their layers still degrades compression as the number of unique symbols per-layer is much larger than per-channel. Moreover, some baselines quantized both weights and activation to further accelerate the models. The contribution of ANS to RIQ is discussed in Appendix A.8.

Typical compression ratio and score achieved by RIQ are presented in Table 2 for a variety of models and tasks. Note that the deviation does not translate immediately to the drop in each score, as the latter is a task-specific measure. Yet, in general, the scores improve as the deviation decreases. To further assess the potential of RIQ, we evaluate our method on sparse models taken from the Neural-Magic 2 . Notably, the compression ratio of sparse models is significantly higher. This coincides with the conclusion that pruning is substantial for good compression (Isik et al., 2022).

Table 2: Compression and accuracy achieved by RIQ. Models denoted by asterisk ( ∗ ) were pruned during training, before quantization.

| Model / Dataset          | Metric        | Deviation constraint   | Compression   |   Quant. Score |   Ref. Score |   Drop |
|----------|----------|----------|----------|----------|----------|--------|
| VGG / I1k          | Top-1 Acc (%) | 0.5%          | × 19 . 4      |          71.3  |        71.59 |   0.29 |
| ResNet-50 / I1k          | Top-1 Acc (%) | 0.5%          | × 7 . 31      |          75.88 |        76.14 |   0.26 |
| ViT / I1k          | Top-1 Acc (%) | 0.5%          | × 6 . 98      |          81    |        81.07 |   0.07 |
| YOLO / COCO          | mAP@.5        | 0.3%          | × 8 . 34      |          54.7  |        55.7  |   1    |
| DistilBERT / SQuAD         | F1          | 0.025%          | × 7 . 96      |          85    |        85.08 |   0.08 |
| VGG ∗ (75%) / I1k          | Top-1 Acc (%) | 0.5%          | × 52 . 9      |          69.34 |        69.73 |   0.39 |
| ResNet-50 ∗ (95%) / I1k    | Top-1 Acc (%) | 0.5%          | × 41 . 5      |          75.72 |        76.14 |   0.42 |
| YOLO ∗ (75%) / COCO        | mAP@.5        | 0.3%          | × 16 . 48     |          52.6  |        53.5  |   0.9  |
| DistilBERT ∗ (58%) / SQuAD | F1          | 0.025%          | × 19 . 4      |          84.7  |        84.92 |   0.22 |

'

## 6 CONCLUSION

In this paper, we have investigated a post-training quantization method that strives to minimize the rate of the model's parameters subject to a deviation constraint. A rotation-invariant quantization scheme (RIQ) was introduced, which quantizes each layer in proportion to the layer's norm, searching for the optimal solution over the family of spherical distributions. To find the solution efficiently, we derived the scale in which the rate increases with the deviation and then suggest a searching paradigm that bounds the search space based on our findings. The rate-distortion curve was thoroughly analyzed, showing that the minimizing distribution is a product distribution, constructed as the product of the layer's spherical distribution, which coincides with the RIQ approach.

## REFERENCES

- Charu C Aggarwal, Alexander Hinneburg, and Daniel A Keim. On the surprising behavior of distance metrics in high dimensional space. In Database Theory-ICDT 2001: 8th International Conference London, UK, January 4-6, 2001 Proceedings 8 , pp. 420-434. Springer, 2001.
- Ron Banner, Yury Nahshan, and Daniel Soudry. Post training 4-bit quantization of convolutional networks for rapid-deployment. Advances in Neural Information Processing Systems , 32, 2019.
- Chaim Baskin, Brian Chmiel, Evgenii Zheltonozhskii, Ron Banner, Alex M. Bronstein, and Avi Mendelson. Cat: Compression-aware training for bandwidth reduction, 2019. URL https: //arxiv.org/abs/1909.11481 .
- Chaim Baskin, Natan Liss, Eli Schwartz, Evgenii Zheltonozhskii, Raja Giryes, Alex M Bronstein, and Avi Mendelson. Uniq: Uniform noise injection for non-uniform quantization of neural networks. ACM Transactions on Computer Systems (TOCS) , 37(1-4):1-15, 2021.
- Yash Bhalgat, Jinwon Lee, Markus Nagel, Tijmen Blankevoort, and Nojun Kwak. Lsq+: Improving low-bit quantization through learnable offsets and better initialization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops , June 2020.
- Gabriel Caffarena, Carlos Carreras, Juan A López, and Ángel Fernández. Sqnr estimation of fixedpoint dsp algorithms. EURASIP Journal on Advances in Signal Processing , 2010:1-12, 2010.
- Michael Alan Chang, Domenic Bottini, Lisa Jian, Pranay Kumar, Aurojit Panda, and Scott Shenker. How to train your dnn: The network operator edition. arXiv preprint arXiv:2004.10275 , 2020.
- Yoni Choukroun, Eli Kravchik, Fan Yang, and Pavel Kisilev. Low-bit quantization of neural networks for efficient inference. In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW) , pp. 3009-3018. IEEE, 2019.
- Thomas M. Cover and Joy A. Thomas. Elements of information theory . Wiley-Interscience, 2006.
- Sam Daulton, Maximilian Balandat, and Eytan Bakshy. Parallel bayesian optimization of multiple noisy objectives with expected hypervolume improvement. In NeurIPS , 2021a.
- Samuel Daulton, Maximilian Balandat, and Eytan Bakshy. Differentiable expected hypervolume improvement for parallel multi-objective bayesian optimization. Advances in Neural Information Processing Systems , 33:9851-9864, 2020.
- Samuel Daulton, David Eriksson, Maximilian Balandat, and Eytan Bakshy. Multi-objective bayesian optimization over high-dimensional search spaces. arXiv preprint arXiv:2109.10964 , 2021b.
- Alexandre Défossez, Yossi Adi, and Gabriel Synnaeve. Differentiable model compression via pseudo quantization noise. TMLR , 2022.
- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 , 2020.
- Jarek Duda. Asymmetric numeral systems: entropy coding combining speed of huffman coding with compression rate of arithmetic coding. arXiv preprint arXiv:1311.2540 , 2013.
- David Eriksson and Martin Jankowiak. High-dimensional bayesian optimization with sparse axisaligned subspaces. In UAI , 2021.
- Angela Fan, Pierre Stock, Benjamin Graham, Edouard Grave, Rémi Gribonval, Hervé Jégou, and Armand Joulin. Training with quantization noise for extreme model compression. CoRR , abs/2004.07320, 2020. URL https://arxiv.org/abs/2004.07320 .
- Kai-Tai Fang, Samuel Kotz, and Kai Wang Ng. Symmetric multivariate and related distributions . Chapman and Hall/CRC, 2018.

- Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635 , 2018.
- Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, September 2021. URL https://doi.org/10.5281/zenodo.5371628 .
- Weihao Gao, Yu-Han Liu, Chong Wang, and Sewoong Oh. Rate distortion for model compression: From theory to practice. In International Conference on Machine Learning , pp. 2102-2111. PMLR, 2019.
- Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding, 2015. URL https://arxiv.org/ abs/1510.00149 .
- Babak Hassibi, David G Stork, and Gregory J Wolff. Optimal brain surgeon and general network pruning. In IEEE international conference on neural networks , pp. 293-299. IEEE, 1993.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 770-778, 2016.
- Stephen Montgomery-Smith (https://stephenmontgomerysmith.github.io/). Finding the rotation matrix in n-dimensions. Mathematics Stack Exchange, 2016. URL https://math. stackexchange.com/q/598782 . URL:https://math.stackexchange.com/q/598782 (version: 2016-06-19).
- Itay Hubara, Yury Nahshan, Yair Hanani, Ron Banner, and Daniel Soudry. Accurate post training quantization with small calibration sets. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning , volume 139 of Proceedings of Machine Learning Research , pp. 4466-4475. PMLR, 18-24 Jul 2021. URL https://proceedings. mlr.press/v139/hubara21a.html .
- Michael F. Hutchinson. A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines. Communications in Statistics - Simulation and Computation , 18:1059-1076, 1989.
- Yerlan Idelbayev and Miguel A Carreira-Perpinán. Low-rank compression of neural nets: Learning the rank of each layer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 8049-8059, 2020.
- Yerlan Idelbayev, Pavlo Molchanov, Maying Shen, Hongxu Yin, Miguel A Carreira-Perpinán, and Jose M Alvarez. Optimal quantization using scaled codebook. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 12095-12104, 2021.
- Berivan Isik, Tsachy Weissman, and Albert No. An information-theoretic justification for model pruning. In International Conference on Artificial Intelligence and Statistics , pp. 3821-3846. PMLR, 2022.
- Glenn Jocher, Ayush Chaurasia, Alex Stoken, Jirka Borovec, NanoCode012, Yonghye Kwon, TaoXie, Jiacong Fang, imyhxy, Kalen Michael, Lorna, Abhiram V, Diego Montes, Jebastin Nadar, Laughing, tkianai, yxNONG, Piotr Skalski, Zhiqiang Wang, Adam Hogan, Cristi Fati, Lorenzo Mammana, AlexWang1900, Deep Patel, Ding Yiwei, Felix You, Jan Hajek, Laurentiu Diaconu, and Mai Thanh Minh. ultralytics/yolov5: v6.1 - TensorRT, TensorFlow Edge TPU and OpenVINO Export and Inference, February 2022. URL https://doi.org/10.5281/zenodo.6222936 .
- Alon Kipnis and Galen Reeves. Gaussian approximation of quantization error for estimation from compressed data. IEEE Transactions on Information Theory , 67(8):5562-5579, 2021. doi: 10.1109/TIT.2021.3083271.
- VN Koshelev. Quantization with minimal entropy. Probl. Pered. Inform , 14:151-156, 1963.

- Raghuraman Krishnamoorthi. Quantizing deep convolutional networks for efficient inference: A whitepaper. ArXiv , abs/1806.08342, 2018.
- Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. Advances in neural information processing systems , 2, 1989.
- Dongsoo Lee, Se Jung Kwon, Byeongwook Kim, and Gu-Yeon Wei. Learning low-rank approximation for CNNs. arXiv preprint arXiv:1905.10145 , 2019.
- Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. Brecq: Pushing the limit of post-training quantization by block reconstruction. arXiv preprint arXiv:2102.05426 , 2021.
- Shaohui Lin, Rongrong Ji, Chao Chen, Dacheng Tao, and Jiebo Luo. Holistic cnn compression via low-rank decomposition with knowledge transfer. IEEE transactions on pattern analysis and machine intelligence , 41(12):2889-2905, 2018.
- Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars (eds.), Computer Vision - ECCV 2014 , pp. 740-755, Cham, 2014. Springer International Publishing. ISBN 978-3-319-10602-1.
- D. Marco and D.L. Neuhoff. The validity of the additive noise model for uniform scalar quantizers. IEEE Transactions on Information Theory , 51(5):1739-1755, 2005. doi: 10.1109/TIT.2005. 846397.
- Springer Melvin Dale. The algebra of random variables . Wiley, 1979.
- Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. Up or down? adaptive rounding for post-training quantization. In International Conference on Machine Learning , pp. 7197-7206. PMLR, 2020.
- NVIDIA. Pytorch quantization - functionalities. docs.nvidia.com, 2021. URL https://docs. nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/ docs/userguide.html .
- Deniz Oktay, Johannes Ballé, Saurabh Singh, and Abhinav Shrivastava. Scalable model compression by entropy penalized reparameterization. arXiv preprint arXiv:1906.06624 , 2019.
- Yury Polyanskiy and Yihong Wu. Lecture notes on information theory. Lecture Notes for ECE563 (UIUC) and , 6(2012-2016):7, 2016.
- Yury Polyanskiy and Yihong Wu. Information theory: From coding to learning. Book draft , 2022.
- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250 , 2016.
- Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 , 2019.
- Fahad Sarfraz, Elahe Arani, and Bahram Zonooz. Knowledge distillation beyond model compression. In 2020 25th International Conference on Pattern Recognition (ICPR) , pp. 6136-6143. IEEE, 2021.
- Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 , 2014.
- Devesh Walawalkar, Zhiqiang Shen, and Marios Savvides. Online ensemble model compression using knowledge distillation. In European Conference on Computer Vision , pp. 18-35. Springer, 2020.
- Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin, and Song Han. Haq: Hardware-aware automated quantization with mixed precision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 8612-8620, 2019.

- André Weißenberger and Bertil Schmidt. Massively parallel ans decoding on gpus. In Proceedings of the 48th International Conference on Parallel Processing , pp. 1-10, 2019.
- Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev, and Paulius Micikevicius. Integer quantization for deep learning inference: Principles and empirical evaluation. arXiv preprint arXiv:2004.09602 , 2020.
- Qiaojun Wu, Yuan Li, Song Chen, and Yi Kang. DRGS: Low-precision full quantization of deep neural network with dynamic rounding and gradient scaling for object detection. In International Conference on Data Mining and Big Data , pp. 137-151. Springer, 2022.
- Ofir Zafrir, Ariel Larey, Guy Boudoukh, Haihao Shen, and Moshe Wasserblat. Prune once for all: Sparse pre-trained language models. arXiv preprint arXiv:2111.05754 , 2021.
- Jinjie Zhang, Yixuan Zhou, and Rayan Saab. Post-training quantization for neural networks with provable guarantees. arXiv preprint arXiv:2201.11113 , 2022a.
- Jinnian Zhang, Houwen Peng, Kan Wu, Mengchen Liu, Bin Xiao, Jianlong Fu, and Lu Yuan. Minivit: Compressing vision transformers with weight multiplexing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 12145-12154, 2022b.
- Yanfu Zhang, Shangqian Gao, and Heng Huang. Exploration and estimation for model compression. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pp. 487-496, 2021.
- Ritchie Zhao, Yuwei Hu, Jordan Dotzel, Chris De Sa, and Zhiru Zhang. Improving neural network quantization without retraining using outlier channel splitting. In International conference on machine learning , pp. 7543-7552. PMLR, 2019.

## A APPENDIX

In this section, we provide rigorous proofs for the theorems and the statements herein. Further, we present additional results for RIQ.

## A.1 PROOF OF LEMMA 1

Proof. Let w ℓ be the realization of the weights vector of layer ℓ , and ˆ w ℓ be the quantized representation of those weights, where θ ℓ denotes the angle between those vectors. Before diving into the cosine distance analysis, let us revisit the mean squared error analysis of the uniform quantizer in Section 3.3, and extend it to the multivariate case. In this case, the distortion is

<!-- formula-not-decoded -->

where (a) follows from the law of large numbers, and (b) follows by the analysis of the scalar uniform quantizer, given in (Polyanskiy &amp; Wu, 2022, Ch. 24.1).

Note that the quantization error in each layer can be bounded by ∥ w ℓ -ˆ w ℓ ∥ ≤ √ n ℓ ∆ ℓ 2 . Assuming that the range of w ℓ is symmetric around zero for simplicity, then, ∆ ℓ ≈ 2 max( w ℓ ) 2 R , where R is the rate of quantization. Hence, the error can be bounded by ∥ w ℓ -ˆ w ℓ ∥ ≤ √ n ℓ max( w ℓ ) 2 R ≈ ∥ w ℓ ∥ 2 R , which is O ( ∥ w ℓ ∥ ) . Yet, this O ( ∥ w ∥ ) does not reflect well the high-rate regime that this paper considers. In particular, to focus on the high-rate regime, we let R = O (log c ( ∥ w ℓ ∥ )) for c &gt; 1 , which results in error bound of o ( ∥ w ℓ ∥ ) .

For analyzing the cosine distance between w ℓ and ˆ w ℓ in the high-rate regime, we notice that

<!-- formula-not-decoded -->

Hence, normalizing both sides by 2 ∥ w ℓ ∥ 2 , we obtain that

<!-- formula-not-decoded -->

Combining the analysis of eq. (11) with eq. (12), we obtain

<!-- formula-not-decoded -->

By denoting ϵ ℓ = 1 -cos( θ ℓ ) , and omitting the little order o (1) , the lemma follows.

Note that even when R = O (log( ∥ w ℓ ∥ )) , where the error bound is relaxed to O (1) still provide meaningful results. In this case, the resulting approximation error would be O (1 / ∥ w ℓ ∥ ) .

## A.2 PROOF OF COROLLARY 1

Proof. Let θ [1: L ] be the rotation angle from w [1: L ] to ˆ w [1: L ] such that ⟨ w [1: L ] ∥ w [1: L ] ∥ , ˆ w [1: L ] ∥ ˆ w [1: L ] ∥ ⟩ ≜ cos( θ [1: L ] ) . Assuming ∥ ˆ w ℓ ∥ = ∥ w ℓ ∥ + o ( ∥ w ℓ ∥ ) , then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ∑ L ℓ =1 ∥ w ℓ ∥ 2 ∥ w [1: L ] ∥ 2 = 1 , the parameters' distortion is simply a convex combination of the layers' distortion.

## A.3 PROOF OF PROPOSITION 1

Proof. By eq. (12),

<!-- formula-not-decoded -->

where θ f is the deviation angle obtained by eq. (1).

To make the dependence of f ( x ) on its weights explicit, let us denote f ( x ) = f x ( w [1: L ] ) and ˆ f ( x ) = f x ( ˆ w [1: L ] ) . Since the denominator is independent of the quantization, it is sufficient to focus on the enumerator. Accordingly, we wish to examine

<!-- formula-not-decoded -->

where ϵ is (random) quantization errors. Our goal is to prove that E ϵ [ g ( ϵ )] is monotonically decreasing in k (i.e., higher rate must reduce the quantized model deviation). Using Taylor expansion at ϵ =0 , we have

<!-- formula-not-decoded -->

where the last step follows since g (0) = 0 , and noting that ∇ g (0) = 0 . Finally, omitting the little order o ( ϵ 3 ) , which is negligible in the high rate regime.

Recall that in the high rate regime the error in entry i , ϵ i ∼ U [ -∆ / 2 , ∆ / 2] is i.i.d. uniformly distributed. Accordingly, in each layer ℓ the corresponding sub-vector ϵ ℓ satisfies E [ ϵ ℓ ϵ T ℓ ] = ∆ 2 ℓ / 12 · I n ℓ , where I n ℓ is the n ℓ × n ℓ identity matrix. Since the errors are independent with zero mean and variance ∆ 2 ℓ / 12 , we can utilize the Hutchinson (1989) trick. Accordingly, let N = ∑ L ℓ =1 n ℓ , then

<!-- formula-not-decoded -->

̸

where the last step follows since E ϵ [ ϵ i · ϵ j ] = 0 ∀ i = j . Finally, letting

<!-- formula-not-decoded -->

we can present the last step vectorially as

<!-- formula-not-decoded -->

Since g (0) is a quadratic function with minimum at ϵ =(0) , thus H ( g (0)) is positive definite, which means that its diagonal entries are real and non-negative. Further, the entries of ⃗ ∆ are positive, hence, this dot product is monotonically decreasing in k , as O (1 /k 2 ) , which completes the proof. Note that this proposition holds for any x .

Figure 2: (a) Validation of Proposition 1 on ResNet-50. (b) The impact of ϵ 0 on the performance of RIQ on ResNet50. Higher values of ϵ 0 attains higher compression, yet, higher deviation.

<!-- image -->

Figure 2(a) validates the analysis of Proposition 1. In particular, we measure the deviation for ResNet-50 as a function of k using both random data and a calibration set. Then, we compare the resulting deviations with a fitted curve of the form y = a/k 2 . Clearly, the monotonicity allows to search efficiently the optimal solution.

## A.4 PROOF OF PROPOSITION 2

Proof. The layer whose quantization error converges last to ϵ 0 dictates when to stop the search. Specifically, when k is sufficiently large in eq. (7), the error in layer ℓ ∗ reaches √ ϵ ℓ ∗ = o ( ϵ 0 ) + ϵ 0 , where o ( · ) denotes little order of magnitude. That is where ϵ 0 becomes dominant. At this point, we say that the error has converged for all layers (as it converged even at the largest layer ℓ ∗ ). Since ϵ 0 ≤ 1 in the cosine distance criterion, we choose the little order of magnitude to be o ( ϵ 0 ) = ϵ 0 · √ ϵ 0 , and hence, k can be bounded from above by

<!-- formula-not-decoded -->

which happens when k ≤ √ n ℓ ∗ / 24 / ( ϵ 0 · √ ϵ 0 ) .

In our experiments, we let ϵ 0 = 0 . 01 , hence, the upper limit is simply k ≤ 1000 · √ n ℓ ∗ / 24 3

For a lower bound, we use again the fact that ϵ ℓ ≤ 1 . Thus, focusing on layer ℓ ∗ , we observe that

<!-- formula-not-decoded -->

which happens as long as k ≥ √ n ℓ ∗ / 24 / (1 -ϵ 0 ) . This completes the proof.

As mentioned, the power of ϵ 0 is substantial. On the one hand, ϵ 0 shifts the bin-width ∆ ℓ from the exact rotation-invariant solution. On the other hand, without this term, the quantization rate

3 When the original weights are represented with R bit symbols, then, choosing ϵ 0 = 0 , yields a trivial upper bound k max , which is the largest number that can be represented with R bits, e.g., ⌈ k max ⌋ ≤ 2 31 , when using 32 bits integer.

Figure 3: (a) Illustration of the surrogate model in Section 4.2. Is this illustration, the quantized weights are modeled by the result of an orthogonal transformation U ( θ ℓ | w ℓ ) which rotates the vector w ℓ randomly onto a ring that is θ ℓ away from w ℓ . Note that the true quantization results also lies in this ring. (b) Illustration of the projection of ˜ w ℓ onto the arbitrary perpendicular vectors v 1 and v 2 .

<!-- image -->

<!-- image -->

might get too high, and hence, it has practical merit. To examine its influence, we used RIQ on ResNet50 with different values of ϵ 0 , recording the deviation as a function of the k parameter and the achieved compression (see Figure 2(b)). Interestingly, large values of ϵ 0 did not reach the target deviation (which is 0.005 in our experiments). Yet, since enforces lower rates, then, it attains a good compression ratio. On the other hand, picking small ϵ 0 is beneficial in terms of model deviation, which is crucial for accuracy. The best value tradeoffs between these options. That is, picking the largest value of ϵ 0 that satisfies the deviation requirement.

## A.5 PROOF OF PROPOSITION 3

Proof. By Fang et al. (2018, Theorem 4.3), a necessary and sufficient condition for ˜ w ℓ ∣ ∣ w ℓ to have a spherical rotation-invariant distribution on a cone that is θ ℓ away from w ℓ is when

<!-- formula-not-decoded -->

̸

for any pair of perpendicular vectors v 1 = 0 and v 2 = 0 that are orthogonal to w ℓ .

̸

Consider the model in eq. (9), any orthogonal transformation U ( θ ℓ ∣ ∣ w ℓ ) can be represented by an orthonormal basis, obtained by the Gram-Schmidt process. That is, finding two orthonormal vectors u 1 and u 2 that span the plane of rotation generated by w ℓ and some w ′ ℓ that is θ ℓ away from w ℓ , and then, extend this basis to R n ℓ . This allows us to consider the rotation in the plane generated by those vectors, with respect to the extended basis (https://stephenmontgomerysmith.github.io/).

Accordingly, let u 1 = w ℓ ∥ w ℓ ∥ and u 2 = w ′ ℓ -⟨ u 1 , w ′ ℓ ⟩ u 1 ∥ w ′ ℓ -⟨ u 1 , w ′ ℓ ⟩ u 1 ∥ , then

<!-- formula-not-decoded -->

where I n ℓ is the n ℓ × n ℓ identity matrix and R θ ℓ is the rotation matrix

<!-- formula-not-decoded -->

that rotates at a scalar angle θ ℓ , and [ u 1 , u 2 ] is n ℓ × 2 matrix whose columns are u 1 and u 2 , respectively. Plugging eq. (13) to eq. (9), and noting that w ℓ ∥ w ℓ ∥ = u 1 , we obtain

<!-- formula-not-decoded -->

To simplify notation, let u ≜ (cos( θ ℓ ) u 1 +sin( θ ℓ ) u 2 ) , and note that for any perpendicular pair v 1 , v 2 that are orthogonal to w ℓ , the vector u can be decomposed to u = u ∥ + u ⊥ , where u ∥ resides in the plane generated by v 1 and v 2 , and u ⊥ resides in the null-space of this plane. For illustration, see Figure 3(b). Hence,

<!-- formula-not-decoded -->

Accordingly, we have

<!-- formula-not-decoded -->

where (a) follows by eq. (15). (b) follows by the linearity of the inner product and since u ⊥ is perpendicular to both v 1 and v 2 (c) follows due to the orthogonality of the basis v 1 and v 2 , where ϕ is the angle between u ∥ and v 1 . (d) follows since the angle between u ∥ and an arbitrary v 1 is arbitrary, and hence, every angle has the same distribution. (e) follows by trigonometric identities for the cosine and sine function.

Thus, Proposition 3 follows.

## A.6 PROOF OF THEOREM 1

Proof. Consider the rate-distortion function

<!-- formula-not-decoded -->

Assuming the weights of each layer w ℓ are statistically independent of the weights of the other layers, then, by the properties of the mutual information, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where eq. (20) follows since conditioning reduces entropy. Note, however, that eq. (20) can be attained with equality by letting p ( w [1: L ] ∣ ∣ ˜ w [1: L ] ) = ∏ L ℓ =1 p ( w ℓ ∣ ∣ ˜ w ℓ ) . Consequently, the minimizing distribution in eq. (10) is a product distribution (Polyanskiy &amp; Wu, 2022, Theorem 6.1 (2)).

Interestingly, eq. (22) implies that the optimal solution has a simple form of a layer-by-layer solution , which significantly simplifies the problem at hand. Finding the solution, of course, requires formulating the relation between the deviations D and D ℓ of each layer ℓ , and hence, the resulting rate R ( D ℓ ) for each layer. Moreover, since each layer obtains a different rate, it implies that the optimal solution is indeed a mixed-precision solution , where each layer can be considered independently, and hence, the minimizing distribution is product distribution.

First, let us consider the deviation. Let p ( ˜ w 1 ∣ ∣ w 1 ) · p ( ˜ w 2 ∣ ∣ w 2 ) · · · p ( ˜ w L ∣ ∣ w L ) be a distribution that satisfies the deviation requirement D , for which the induced cosine distance (distortion) in each layer

ℓ is at most ϵ ℓ , for ϵ ℓ = 1 -cos( θ ℓ ) . By Corollary 1, the distortion over the entire parameters is hence ϵ [1: L ] ≜ 1 -cos( θ [1: L ] ) = ∑ L ℓ =1 ∥ w ℓ ∥ 2 ∥ w [1: L ] ∥ 2 ϵ ℓ . Note that ϵ [1: L ] is a convex combination of ϵ ℓ as ∑ L ℓ =1 ∥ w ℓ ∥ 2 ∥ w [1: L ] ∥ 2 = 1 . Assuming ϵ [1: L ] ≤ D , then due to the convexity of the cosine distance for | θ ℓ | ≤ π/ 2 , by Jensen inequality ( 1 -cos ( ∑ L ℓ =1 ∥ w ℓ ∥ 2 ∥ w [1: L ] ∥ 2 θ ℓ )) ≤ ∑ L ℓ =1 ∥ w ℓ ∥ 2 ∥ w [1: L ] ∥ 2 (1 -cos( θ ℓ )) . In words, convex combination of the angles also satisfies the deviation constraint D . Generally, cosine distance is a rotation-invariant distance, as the angle between vectors does not change when they are rotated together. This further hints that the minimizing distribution should also be a rotation-invariant distribution as follows.

Next, let us address p ( ˜ w ℓ | w ℓ ) . By eq. (14), given w ℓ (and hence, u 1 ), the probability of ˜ w ℓ is determined by the probability of the rotation angle θ ℓ and the norm ∥ ˆ w ℓ ∥ . Specifically, for any vector s ℓ ∈ R n ℓ , the density function of this product, if exists, is (Melvin Dale, 1979, Ch. 4.1)

<!-- formula-not-decoded -->

where the rotation θ ℓ occurs on R 2 , rotating about ( n ℓ -2) -dimensional subspace. Further, note that the dimension n ℓ is dictated only by the given w ℓ . Apparently, since each layer ℓ resides at a different dimension n ℓ , it is impossible to consider the convex combination of the layers' distribution directly, as done for the vector case, e.g., as considered in Polyanskiy &amp; Wu (2016, Ch. 5). Nevertheless, since the rotation of θ ℓ is done on R 2 in each layer ℓ , which is described by the rotation matrix R θ ℓ in eq. (13), it is still beneficial to consider a convex combination of θ ℓ distributions over the layers, to allow a similar treatment to Polyanskiy &amp; Wu (2016, Ch. 5), as follows.

To bound the mutual information, the density of (cos( θ ℓ ) u 1 +sin( θ ℓ ) u 2 ) | w ℓ should be expressed first in terms of the density of cos( θ ℓ ) | w ℓ . Examining eq. (14), we note that by the transformation of random variables formula,

<!-- formula-not-decoded -->

Considering the high rate regime, where ∆ ℓ is sufficiently small, and thus, ∥ ˆ w ℓ ∥ = ∥ w ℓ ∥ + o (1) , then, the density function in eq. (23) becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (a) follows since the norm ∥ w ℓ ∥ is a function of the given w ℓ . (b) follows since the uncertainty about ∥ ˆ w ℓ ∥ given ∥ w ℓ ∥ is negligible, and hence, p ∥ ˆ w ℓ ∥|∥ w ℓ ∥ ( h ) ≈ δ ( h -∥ w ℓ ∥ ) , i.e., the conditional density is approximately the Dirac delta function. (c) follows by eq. (24). (d) follows by the characteristics of the Dirac delta function.

Hence, it is possible to consider a convex combination of p ( ˜ w ℓ ∣ ∣ w ℓ ) over the layers, where each layer is embedded in possibly different n ℓ , by considering a convex combination of the rotations' probability p cos( θ ℓ ) | w ℓ , since all rotations are done in R 2 . Accordingly, let

<!-- formula-not-decoded -->

Then, by the convexity of the rate-distortion function Cover &amp; Thomas (2006, Theorem 2.7.4), ¯ p ( ˜ w [1: L ] ∣ ∣ w [1: L ] ) can only reduce the mutual information in eq. (21). Specifically,

<!-- formula-not-decoded -->

where I p ( · ; · ) denotes explicitly the mutual information under probability p . Thus, the infimum rate has a form of a scalar (single-letter) rate .

Moreover, since averaging over more rotations should further reduce the mutual information by its convexity, then, the minimizing p ( ˜ w [1: L ] ∣ ∣ w [1: L ] ) can be chosen to be rotation-invariant (Polyanskiy &amp;Wu, 2022, Ch. 6.2). Consequently, the unique minimizing distribution p ( ˜ w [1: L ] ) is also rotationinvariant. Remarkably, Fang et al. (2018, Theorem 2.6) states that when partitioning a spherical rotation-invariant distribution (naturally, according to the layers ˜ w ℓ ), then its components also have a spherical rotation-invariant distribution. This coincides with Proposition 3, which proves that the partitioning satisfies this property.

Accordingly, the unique minimizing distribution p ( ˜ w [1: L ] ∣ ∣ w [1: L ] ) of the rate-distortion function is a product distribution over the layers, where each term ℓ is a spherical rotation-invariant distribution. This completes the proof.

## A.7 RELATION TO OTHER ERROR CRITERIA

Remark. The proof of Lemma 1 in Appendix A.1 may serve as a proxy to other error criteria such as the Signal to Quantization Noise Ratio (SQNR), Caffarena et al. (2010). Specifically, similar to the proof of Lemma 1, the resulting connection between the scale ∆ ℓ and the SQNR ϵ ′ ℓ in each layer ℓ is

Or, equivalently,

## A.8 ADDITIONAL RESULTS

## A.8.1 DECOMPOSING THE RATE-DISTORTION CURVE

The key steps of lossy compression are quantization and compression. In the quantization phase, the RIQ approach is minimizing the overall model's entropy by allocating a small number of unique symbols for large-norm layers. To achieve (asymptotically) this entropy limit, we utilize the ANS (lossless) entropy encoder. In this section, we evaluate the contribution of each step to the ratedistortion tradeoff. Namely, the average rate per (quantized) symbol before and after ANS. At run-time, when a certain layer is required, it is decoded and represented at a rate according to RIQ. If this rate is below 8 bits/symbol, it enables significant acceleration by performing 8 bits integer operations, as discussed in Appendix A.8.2.

Figure 4(a) depicts the rate-distortion curve for ResNet-50, decomposed to the quantization step (dashed lines) and the resulting compression step, following the quantization step (solid lines). As a baseline, the uniform scalar quantization (red color) is given for comparison with RIQ (green color). Interestingly, RIQ (dashed green line) outperforms the uniform quantization (dashed red line) by about ∼ 4 bits/symbol and even its resulting compressed size by about ∼ 1 bit/symbol. Indeed, the latter indicates that uniform quantization does not minimize the model's entropy. Applying the ANS compression following RIQ reduces additional ∼ 3 bits/symbol (solid green line), which according to our analysis is the minimum entropy possible for a given distortion. Moreover, our method achieves a reduction of about ∼ 8 bits/symbol compared to uniform scalar quantization alone, and an additional ∼ 4 bits/symbol when ANS is applied to the uniformly quantized weights. For completeness, in Figure 4(b), we depict the rate per layer statistics for the ResNet-50 model with a deviation constraint of 0 . 5% , using ϵ 0 = 0 . 01 . Noticeably, most of the layers require less than 8 bits. Moreover, the

<!-- formula-not-decoded -->

Figure 4: (a) Rate-distortion curve for ResNet-50 model obtained for RIQ (green circles) as well as Uniform linear quantization (red squares). Rates are presented for both the quantized model (dashed) as well as following an ANS compression. (b) ResNet-50 rate per layer statistics with ϵ 0 = 0 . 01 in all layers. (c) Rate distortion curves obtained by RIQ + ANS, for a variety of models: VGG (green circles), ResNet-50 (red squares), ViT (blue diamonds), and DistilBERT (orange triangles).

<!-- image -->

average rate is 6.9 bits per symbol, as larger layers get quantized more aggressively. Hence, those layers' run time can be accelerated significantly when their input is quantized as well. Yet, if one wishes to enforce a rate that is less than 8 bit to all layers, it should pick an ϵ 0 according to Section 4.1.

The rate-distortion curves for various models, in particular, the VGG (green circles), ResNet-50 (red squares), ViT (blue diamonds), and DistilBERT (orange triangles) are given in Figure 4(c). As expected, the curves decrease monotonously, reaching an impressive compression rate of less than 8 bits/symbol on average even for extremely low cosine distance in all presented models. Table 3 depicts the compression ratio attained for various models 4 and tasks, with 1% deviation constraint. Interestingly, from this table, we can infer that the MobileNet-v2 and ArcFace models are relatively efficient since their compression ratio is lower than the other (over-parameterized) models.

One of the strengths of RIQ is its efficiency. In fact, for ResNet-50 (whose size is 102 MB), it takes less than 1 minute to find k and compress the model on a CPU. Note, however, that finding the optimal k is typically an offline procedure, and thus it is not time critical.

Table 3: Compression and deviation attained by RIQ for various models and tasks under deviation constraint of 1%.

| Model          | Deviation constraint   | Compression ratio   |
|----------|----------|----------|
| MobileNet-v2 (image classification) | 1%          | × 5 . 79          |
| ArcFace (face detection)          | 1%          | × 7 . 75          |
| AlexNet (image classification)      | 1%          | × 19 . 3          |
| Candy (style transfer)          | 1%          | × 20 . 8          |
| AgeGoogleNet (gender and age)       | 1%          | × 26 . 95          |

Table 4 describes the performance of RIQ on large language models. In particular, we used RIQ to compress the llama-7b × 4 , and then evaluated the quantized model using Gao et al. (2021).

Table 4: Llama-7b performance with × 4 compression.

| Task          | Metric       | Baseline   | RIQ    |
|----------|----------|----------|--------|
| ARC challenge | acc (%)      | 41.89%     | 41.98% |
| SWAG          | acc norm (%) | 76.60%     | 75.95% |
| XWinograd     | acc (%)      | 87.87%     | 87.18% |
| piqa          | acc (%)      | 78.67%     | 77.97% |

Table 5 compares the accuracy and compression ratio attained by RIQ with various baseline techniques. This tables shows that creating a smaller model by knowledge distillation attains impressive accuracy

4 github.com/onnx/models

(sometimes even higher than the baseline), yet, falls short in terms of compression ratio. Other baselines that quantizes both the activations and weights suffer from high accuracy drop.

Table 5: Comparison of RIQ with various baseline compression techniques on various tasks.

| Model          | Method          | Comp.          | Acc. (%)   | Ref. (%)   | Drop (%)   |
|----------|----------|----------|----------|----------|----------|
| ViT I1k          | MiniViT (KD) Zhang et al. (2022b) RIQ (Ours) | × 2 × 6.98      | 84.7 81.0  | 77.9 81.07 | -6.8 0.07  |
| YOLO/COCO        | DRGS Wu et al. (2022) RIQ (Ours)          | × 8 × 8.34      | 33.4 54.7  | 55.7 55.7  | 22.3 1.0   |
| DistilBERT/SQuAD | OFA (KD) Zafrir et al. (2021) RIQ (Ours)     | × 6 . 67 × 7.96 | 88.82 85.0 | 85.8 85.08 | -0.02 0.08 |

## A.8.2 ROTATION INVARIANT QUANTIZATION WITH QUANTIZED ACTIVATIONS

Quantizing both the NN model's weights and its activations can further accelerate the inference, Wu et al. (2020); Nagel et al. (2020); Krishnamoorthi (2018). Nevertheless, in this case, the quantization error of both the weights and the activation affects the model's output. In the seminal work of Wu et al. (2020), the authors utilized the KL distance for quantizing the activations to minimize the information loss at the output. In this section, we examine the RIQ approach, combining it with activation quantization.

To demonstrate, we use the NVIDIA (2021) quantization library for the ResNet-50 model with a 'mini ImageNet' validation set, which comprises one image per class and a total of 1000 images. We evaluate this library's performance as a baseline, where the activations are quantized by the KL-distance criterion, and the weights are quantized to 8-bit linearly. The resulting cosine distance at the output of this baseline is 0 . 69% . For comparison, this reference value is given as the deviation requirement to RIQ. In particular, to integrate RIQ, the activations are quantized as the baseline, and then, we run RIQ according to Algorithm 1. This way, RIQ is aware of the activations' quantization error during its search for the single-letter solution. Note that to facilitate the acceleration of int8 operations, RIQ must yield a quantization rate of up to 8 bits/symbol. In case a certain layer requires a higher rate, we simply perform linear uniform quantization to 8 bits (without clipping), as the baseline does.

Figure 5: (a) The compression ratio as a function of cosine distance. The left-bottom red triangle depicts the resulting distance of 0.0069 achieved by the baseline with a compression ratio of × 4 . The orange upside-down triangles depict the cosine distance and compression ratio attained by RIQ with the ANS compression. The orange line depicts the trend line. (b) MOBO optimization process. Interestingly, MOBO converges at the few last iterations to × 12 compression, with a highest value of × 12 . 61 . On the other hand, RIQ reaches practically the same compression ratio in a few seconds.

<!-- image -->

Figure 5(a) characterizes the compression ratio as a function of cosine distance. The leftmost point reflects a cosine distance of 0 . 0069 achieved by the baseline of NVIDIA (2021). Remarkably, the RIQ attains superior compression with relatively low deviation even when the activations are quantized.

In run-time, of course, the reconstructed values are represented again by 8-bit value, and hence, the significant acceleration of Wu et al. (2020) is still valid.

## A.8.3 COMPARISON WITH MULTI-OBJECTIVE BAYESIAN OPTIMIZATION

In this section, we utilize the Multi-Objective Bayesian Optimization (MOBO) tool, described in Daulton et al. (2020) to compress NN models, and compare results with RIQ. To compress models with MOBO, we set two objective functions for it. The first objective is minimizing the cosine distance in eq. (1). The second objective is maximizing the compression ratio in eq. (2). Then, we let MOBO optimize the rate-distortion tradeoff (i.e., the Pareto frontier surface).

Nonetheless, MOBO is quite complex and requires strong computing capabilities for exploration and exploitation. Particularly, reaching the optimal solution may take days and even weeks, using multiple GPUs. Even on small NN models, to address the high-dimensional search spaces, we apply sparse axis-aligned subspace priors for Bayesian optimization (qNEHVI + SAASBO), with the batch Noisy Expected Improvement (qNEI) acquisition function, as suggested by Eriksson &amp; Jankowiak (2021); Daulton et al. (2021b;a). Moreover, since the two objectives are not within the same range the cosine similarity objective had to be scaled accordingly to converge to the optimal solution, where a calibration set of 4 images are used during 30 iterations of exploration/exploitation.

Accordingly, we pick a (relatively) small model for comparison (with a size of 112 KB), letting MOBO to find for each layer its optimal bin width and quantize accordingly. We emphasize that the MOBO solution does not rely on the rotation invariant insights. In Figure 5(b), the optimization process of MOBO is presented, where each dot depicts experiment results, and its color indicates the iteration in which this result was attained. The compression results of RIQ are presented for comparison. Remarkably, RIQ and MOBO attained almost identical results of × 12 . 6 and × 12 . 61 , respectively, with a cosine distance of 0 . 005 . This indicates that RIQ reaches the optimal solution.