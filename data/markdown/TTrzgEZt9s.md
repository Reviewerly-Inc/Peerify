## DISTRIBUTIONALLY ROBUST OPTIMIZATION WITH BIAS AND VARIANCE REDUCTION

Ronak Mehta ∗

Vincent Roulet †

Krishna Pillutla ‡

Zaid Harchaoui ∗

## ABSTRACT

We consider the distributionally robust optimization (DRO) problem, wherein a learner optimizes the worst-case empirical risk achievable by reweighing the observed training examples. We present Prospect, a stochastic gradient-based algorithm that only requires tuning a single learning rate hyperparameter, and prove that it enjoys linear convergence for smooth regularized losses. This contrasts with previous algorithms that either require tuning multiple hyperparameters or potentially fail to converge due to biased gradient estimates or inadequate regularization. Empirically, we show that Prospect can converge 2-3x faster than baselines such as SGD and stochastic saddle-point methods on distribution shift and fairness benchmarks spanning tabular, vision, and language domains.

## 1 INTRODUCTION

The ingredients of empirical risk minimization (ERM) are generally considered to be: a model with parameters w ∈ R d (e.g. a neural network), a training set z 1 , . . . , z n ∈ Z of independent and identically distributed realizations of a random variable Z ∼ P , a loss function ℓ : R d × Z → R , and an optimization algorithm that solves

<!-- formula-not-decoded -->

where P n is the empirical distribution of { z i } n i =1 . The fourth ingredient-often taken for granted-is the choice of risk functional , which aggregates the distribution of ℓ ( w,Z ) into a univariate summary to be minimized. The objective (1) (the expected loss under P n ) is an unbiased estimate of the expected loss under an underlying data-generating distribution P ; however, a deployed model often observes data from distributions other than P . Motivated by this practical phenomenon, we consider instead an objective that explicitly captures sensitivity to such distribution shifts:

<!-- formula-not-decoded -->

in which Q is an uncertainty set of probability measures, ν ≥ 0 is a hyperparameter, and D ( Q ∥ P n ) represents the divergence of Q from the original training distribution P n (e.g. the χ 2 or Kullback Leibler divergence). The objective (2) emulates a game in which nature pays a price of ν per unit D ( Q ∥ P n ) to replace the expected loss under P n with the expected loss E Z ∼ Q [ ℓ ( w,Z )] associated with the shifted distribution Q . Since ν penalizes these shifts, we shall refer to it as the shift cost .

Objectives of the form (2), known as distributionally robust optimization (DRO) problems, have seen a wave of recent interest in machine learning theory and practice (Chen &amp; Paschalidis, 2020). Historically used in quantitative finance, a popular such objective is the conditional value-at-risk (CVaR, a.k.a. superquantile/expected shortfall/average topk loss). In terms of methods, the CVaR has been used as a canonical DRO objective (Fan et al., 2017; Kawaguchi &amp; Lu, 2020; Rahimian &amp; Mehrotra, 2022), as well as in unsupervised learning (Maurer et al., 2021), reinforcement learning (Singh et al., 2020), and federated learning (Pillutla et al., 2023). In applications, it has also been employed for robust language modeling (Liu et al., 2021) and robotics (Sharma et al., 2020). The superquantile/CVaR falls into the broader category of spectral risk measures (SRMs) , a class of DRO objectives that includes the extremile and exponential spectral risk measure (ESRM) (Acerbi &amp;Tasche, 2002; Cotter &amp; Dowd, 2006; Daouia et al., 2019).

∗ University of Washington, † Google DeepMind, ‡ Google Research.

Motivated by 1) the success of the superquantile in applications and 2) the importance of stochastic optimization in machine learning, the principal goal of this paper is to develop stochastic 1 algorithms for spectral risk minimization .

Contributions. We propose Prospect, a stochastic algorithm for optimizing spectral risk measures with only one tunable hyperparameter: a constant learning rate. Theoretically, Prospect converges linearly for any positive shift cost on regularized convex losses. This contrasts with previous stochastic methods that fail to converge due to bias (Levy et al., 2020; Kawaguchi &amp; Lu, 2020), may not converge for small shift costs (Mehta et al., 2023), or have multiple hyperparameters (Palaniappan &amp; Bach, 2016). Experimentally, Prospect demonstrates equal or faster convergence than competitors on the training objective on nearly all objectives and datasets considered, and exhibits higher stability with respect to external metrics on fairness and distribution shift benchmarks.

Related Work. Examples of DRO formulations range throughout diverse contexts such as reinforcement (Liu et al., 2022b; Kallus et al., 2022; Liu et al., 2022c; Xu et al., 2023; Wang et al., 2023; Lotidis et al., 2023; Kallus et al., 2022; Ren &amp; Majumdar, 2022; Clement &amp; Kroer, 2021), continual (Wang et al., 2022), interactive (Yang et al., 2023; Mu et al., 2022; Inatsu et al., 2021; Sinha et al., 2020), Bayesian (Tay et al., 2022; Inatsu et al., 2022), and federated (Deng et al., 2020; Pillutla et al., 2023) learning, along with dimension reduction (Vu et al., 2022), computer vision (Samuel &amp; Chechik, 2021; Sapkota et al., 2021), and structured prediction (Li et al., 2022; Fathony et al., 2018). Various forms of these objectives are parameterized by the uncertainty set Q , including those based on f -divergences (Levy et al., 2020; Ben-Tal et al., 2013), the Wasserstein metric (Blanchet et al., 2019b; Kuhn et al., 2019), maximum mean discrepancy (Kirschner et al., 2020; Staib &amp; Jegelka, 2019; Nemmour et al., 2021), or more general classes of metrics (Husain, 2020; Shapiro, 2017). We focus on SRM objectives, as motivated in detail in Sec. 2 and Appx. B.

These objectives also yield connections to other areas in modern machine learning. They are a special case of subpopulation shift , wherein the data-generating distribution is modeled as a mixture of subpopulations, and the distribution shift stems from changes in the mixture. In our case, the subpopulations are point masses at the observed data points. In the context of algorithmic fairness , the subpopulations may represent data conditioned on some protected attribute (e.g. race, gender, age range), and common notations of fairness such as demographic/statistical parity (Agarwal et al., 2018; 2019) impose (informally) that model performance with respect to each subpopulation should be roughly equal. As such, robustness to reweighting and algorithmic fairness are often aligned notions (Williamson &amp; Menon, 2019), with recent research arguing that distributionally robust models are more fair (Hashimoto et al., 2018; Vu et al., 2022) and that fair models are more distributionally robust (Mukherjee et al., 2022). In supervised learning, the data distribution is modeled as P = P X,Y for a feature-label pair ( X,Y ) and related settings of covariate shift (changes in P X and not P Y | X ) (Sugiyama et al., 2007) as well as label shift (changes in P Y and not P X | Y ) (Lipton et al., 2018) may also modeled with distributional robustness (Zhang et al., 2021) as illustrated in Fig. 1. In these settings, distributional robustness may be described as a property of learned representations that are transferable to multiple tasks (Słowik &amp; Bottou, 2022).

In comparisons, we include stochastic algorithms that either are single-hyperparameter 'out-of-thebox' methods such as stochastic gradient descent and stochastic regularized dual averaging (Xiao, 2009), or multi-hyperparameter methods that converge linearly on strongly convex SRM-based objectives, such as LSVRG (Mehta et al., 2023) and stochastic saddle-point SAGA (Palaniappan &amp; Bach, 2016). Note that LSVRG may not converge for small shift costs. Other methods may achieve sublinear convergence rates, even with multiple hyperparameters (Yu et al., 2022; Ghosh et al., 2021; Carmon &amp; Hausler, 2022; Li et al., 2019; Shen et al., 2022; Yazdandoost Hamedani &amp; Jalilzadeh, 2023; Namkoong &amp; Duchi, 2016). Non-convex settings have also been studied (Jin et al., 2021; Jiao et al., 2022; Sagawa et al., 2020; Luo et al., 2020), as well as statistical aspects of resulting minimizes of DRO objectives (Liu et al., 2022a; Blanchet et al., 2019a; Zeng &amp; Lam, 2022; Maurer et al., 2021; Lee et al., 2020; Khim et al., 2020; Zhou &amp; Liu, 2023; Zhou et al., 2021; Cranko et al., 2021; Słowik &amp; Bottou, 2022). Our goal is to achieve unconditional linear convergence for smooth, strongly convex (regularized) losses with a single hyperparameter.

1 Weuse stochastic interchangeably with incremental , meaning algorithms that make O (1) calls per iteration to a fixed set of oracles { ( ℓ i , ∇ ℓ i ) } n i =1 , and not streaming algorithms that sample fresh data at each iteration.

Figure 1: Notions of Distribution Shift . Illustration of various forms of distribution shift that are characterized by maintaining the same training data but changing the weight of each example.

<!-- image -->

## 2 MINIMIZING SPECTRAL RISK WITH BIAS AND VARIANCE REDUCTION

We first describe the key technical challenges in constructing a stochastic optimizer for spectral risk measures and how Prospect tackles them. Because we consider a fixed training set of size n and use f -divergences for D (reviewed in Appx. B), we may parameterize probability measures Q by a set of weights q ∈ P ⊆ { p ∈ R n : p i ≥ 0 , ∑ n i =1 p i = 1 } indicating the probability mass allotted to training examples ( z 1 , . . . , z n ) . Similarly, the empirical distribution P n is represented by the uniform weights 1 n /n = (1 /n, . . . , 1 /n ) . We can then rewrite (2) in the convenient form

<!-- formula-not-decoded -->

where ℓ ( w ) = ( ℓ 1 ( w ) , . . . , ℓ n ( w )) ∈ R n and ℓ i ( w ) = ℓ ( w,z i ) is the loss on training example i . In order to build a convergent stochastic algorithm, we will construct an estimate v i for the gradient of (3) based on a single data index i , such that v i →∇ R P ( ℓ ( w )) as the iteration counter approaches infinity. Precisely, we require that for i ∼ Unif[ n ] ,

<!-- formula-not-decoded -->

decreases to zero asymptotically. In the remainder of this section, we first identify concretely our target estimand (i.e. ∇ R P ( ℓ ( w )) for the spectral risk uncertainty set), construct an estimate, and then describe individual procedures to ensure that the bias and variance terms in (4) vanish.

Spectral Risk Measures and their Gradients. The conventional p -superquantile/CVaR (Rockafellar &amp; Royset, 2013) of a loss vector ( ℓ 1 ( w ) , . . . , ℓ n ( w )) is defined as ∑ n i =1 σ i ℓ ( i ) ( w ) , where ℓ (1) ( w ) ≤ . . . ≤ ℓ ( n ) ( w ) are the ordered losses and σ i = 1 /k for the largest k = np values of i and σ i = 0 otherwise (see Equation (21)). Other spectral risk measures (SRMs) are generated by constructing any vector σ = ( σ 1 , . . . , σ n ) of non-negative weights satisfying σ 1 ≤ ··· ≤ σ n and ∑ n i =1 σ i = 1 , called the spectrum . Traditionally used in economics and finance, SRMs strike a user-defined balance between measuring average and tail loss. Recognizing that ∑ n i =1 σ i ℓ ( i ) ( w ) is the maximum of ∑ n i =1 q i ℓ i ( w ) for q a permutation of ( σ 1 , . . . , σ n ) , or equivalently, a convex combination of such permutations, we define the uncertainty set P = P ( σ ) = ConvexHull { permutations of σ } to achieve the regularized formulation in (3). See Appx. B for further details on SRMs including the extremile (Daouia et al., 2019) in Equation (22) and ESRM (Cotter &amp; Dowd, 2006) in Equation (23). SRMs are motivated not only by their historical use but also the efficiency with which we can compute their exact gradient, as we now describe.

Define R σ := R P ( σ ) . When ν &gt; 0 and the map q ↦→ D ( q ∥ 1 n /n ) is strongly convex over P ( σ ) , we have that (Lem. 5, Appx. B) R σ is in fact differentiable with gradient given by

<!-- formula-not-decoded -->

This means the full-batch gradient w ↦→ ∇ R σ ( ℓ ( w )) ∈ R d can be computed by solving the inner maximization to retrieve l ↦→ ∇ R σ ( l ) ∈ R n , calling the oracles to retrieve w ↦→ ∇ ℓ ( w ) ∈ R n × d ,

Figure 2: Prediction Error Reduction . Optimization trajectories on the DR objective. Darker shades indicate lower objective value. Left (Bias): Average trajectory of Prospect over 20 seeds compared to full-batch gradient descent. Right (Variance): Single trajectory of Prospect with/without adding a control variate.

<!-- image -->

<!-- image -->

and multiplying them by the chain rule. To solve for the maximizer, we prove by standard convex duality arguments (Prop. 3, Appx. B) that when D = D f is an f -divergence, the maximum over q can be expressed as a minimization problem that reduces to isotonic regression problem involving f ∗ , the convex conjugate of f . Isotonic regression can be solved exactly by the Pool Adjacent Violators algorithm (Best et al., 2000), which runs in O ( n ) time when the losses are sorted; see Appx. C.

Bias Reduction via Loss Estimation. With a formula for the gradient in hand, we proceed to estimation. Denote by q l := ∇ R σ ( l ) from (5), and observe that by the chain rule, ∇ R σ ( ℓ ( w )) = ∑ n i =1 q ℓ ( w ) i ∇ ℓ ( w ) . In words, we compute the 'most adversarial' reweighting q ℓ ( w ) ∈ P ( σ ) for a given set of losses ℓ ( w ) , and then take a convex combination of the gradients ∇ ℓ 1 ( w ) , . . . , ∇ ℓ n ( w ) weighted by the probability mass function q ℓ ( w ) . While the gradient is computable, however, accessing ℓ ( w ) and ∇ ℓ ( w ) requires n calls to the function/gradient oracles { ℓ i , ∇ ℓ i } n i =1 , which can be prohibitive. While using a plug-in estimate with a mini-batch of size m &lt; n is a natural choice in ERM(making the first term in (4) zero), this will be biased for our objective due to the maximization over q . For example, for m = 1 , we have that R σ ( ℓ i ( w )) = ℓ i ( w ) and ∇ w R σ ( ℓ i ( w )) = ∇ w ℓ i ( w ) , which are unbiased estimates of the ERM objective and gradient, respectively (not the SRM objective). However, note that if the optimal weights q ℓ ( w ) were known, then for i sampled uniformly on [ n ] , that nq ℓ ( w ) i ∇ ℓ i ( w ) is an unbiased estimate for ∑ n i =1 q ℓ ( w ) i ∇ ℓ ( w ) = ∇ R σ ( ℓ ( w )) . While computing q ℓ ( w ) again requires computing ℓ ( w ) , the key ingredient of bias reduction in Prospect is maintaining a table l ∈ R n of losses such that l ≈ ℓ ( w ) for the current iterate w ∈ R d , and using q l as a running estimate of q ℓ ( w ) . This is justified as when q ↦→ D ( q ∥ 1 n ) is strongly convex, we have by Lem. 6 the map l ↦→ q l is Lipschitz continuous in l with respect to ∥·∥ 2 . Thus,

<!-- formula-not-decoded -->

We prove in Sec. 3 that l -ℓ ( w ) → 0 as the iterate counter goes to infinity for our particular choice of l , yielding an asymptotically unbiased gradient estimate as illustrated in Fig. 2 (left).

Variance Reduction via Control Variates. The final ingredient of our stochastic gradient estimate is a variance reduction scheme. Given any estimator ˆ a of a ∈ R d , a control variate is another estimator ˆ b over the same probability space with a known expectation E [ ˆ b ] = b ∈ R d , such that E [(ˆ a -a ) ⊤ ( ˆ b -b )] &gt; 0 . We can exploit this positive correlation to construct an estimator with strictly smaller variance than ˆ a . Indeed, for γ &gt; 0 , we have that

<!-- formula-not-decoded -->

demonstrating the improvement of ˆ a -γ ( ˆ b -b ) over ˆ a . Note that ˆ a need not be unbiased. In our case, we have ˆ a = nq l i ∇ ℓ i ( w ) , where l is the table of losses approximating ℓ ( w ) . We also keep approximations g ∈ R n × d of ∇ ℓ ( w ) and ρ of q ℓ ( w ) , and define ˆ b = nρ i g i and b = E i ∼ Unif[ n ] [ nρ i g i ] = ∑ n j =1 ρ j g j . In the unrealistic case in which ˆ b = ˆ a , the optimal multi-

| Algorithm 1 Prospect          | Algorithm 1 Prospect          |          |
|----------|----------|----------|
| Inputs: Initial w 0 , spectrum σ , number of iterations T , Hyperparameter: Stepsize η > 0 . | Inputs: Initial w 0 , spectrum σ , number of iterations T , Hyperparameter: Stepsize η > 0 .          | regularization µ > 0 , shift cost ν > 0 . |
| 1:          | Initialize l ← ℓ ( w 0 ) and g i ←∇ ℓ i ( w 0 )+ µw 0 for i = 1 , . .          | .,n .          |
| 2:          | Set q ← argmax ¯ q ∈P ( σ ) ¯ q ⊤ l - νD ( q ∥ 1 n /n ) and ρ ← Set ¯ g ← ∑ n ρ g ∈ R d .          | q .          |
| 3: 4: 5:          | i =1 i i Set w ← w 0 . for T iterations do 6: Sample i, j ∼ Unif[ n ] independently. 7: v ← nq i ( ∇ ℓ i ( w )+ µw ) - nρ i g i t +¯ g . 8: w ← w - ηv . 9: l j ← ℓ j ( w ) . | ▷ Variance Reducing          |
| 13:          | ←∇ ρ i q i . Output: Final point w .          | Update          |
| 10: 11: 12:          | q ← argmax ¯ q ∈P ( σ ) ¯ q ⊤ l - νD (¯ q ∥ 1 n /n ) . ¯ g ← ¯ g - ρ i g i + q i ( ∇ ℓ i ( w )+ µw ) . g i ℓ i ( w )+ µw . ←          | ▷ Iterate Update ▷ Bias Reducing Update   |

plier is γ = 1 , trivially achieving zero variance. Similar to l , we prove in Sec. 3 that g -∇ ℓ ( w ) → 0 and ρ -q ℓ ( w ) → 0 , so we have in the notation of (6) that ˆ b -ˆ a → 0 . Thus, by using γ = 1 , our final stochastic gradient estimate is

<!-- formula-not-decoded -->

with asymptotic variance reduction factor (with respect to i ∼ Unif[ n ] ):

<!-- formula-not-decoded -->

This results in asymptotically vanishing variance without decreasing the learning rate, as illustrated in Fig. 2 (right). This scheme generalizes (and is inspired by) the one employed in the SAGA optimizer (Defazio et al., 2014) for ERM, in which ρ = 1 n /n . Finally, while ignored in this section for ease of presentation, each g i will actually store the gradients of the regularized losses ℓ i + µ ∥·∥ 2 2 .

## 3 THE PROSPECT ALGORITHM

By combining the bias reduction and variance reduction schemes from the previous section, we build an algorithm that achieves overall prediction error reduction . Thus, we now present the P rediction ErrorR educed O ptimizer for Spect ral Risk Measures (Prospect) algorithm to solve

<!-- formula-not-decoded -->

where µ &gt; 0 is a regularization constant. The full algorithm is given in Algorithm 1.

Instantiating Bias and Variance Reduction. Consider a current iterate w ∈ R d . As mentioned in Sec. 2, bias and variance reduction relies on the three approximations: the losses l for ℓ ( w ) ∈ R n , each gradient g i for ∇ ℓ i ( w ) + µw ∈ R d , and the weights ρ for q ℓ ( w ) ∈ P . Given initial point w 0 ∈ R d , we initialize l = ℓ ( w 0 ) , g = ∇ ℓ ( w 0 ) + µ 1 n w ⊤ 0 , and ρ = q ℓ ( w 0 ) (including ¯ g := g ⊤ ρ ).

At each iterate, we sample indices i, j ∼ Unif[ n ] independently. The index i is used to compute the stochastic gradient estimate (7), yielding the update direction v in line 7 at the cost of a call to a ( ℓ i , ∇ ℓ i ) oracle. Then, l is updated by replacing l j with ℓ j ( w ) costing another call to ( ℓ j , ∇ ℓ j ) , and we reset q (the variable that stores q l ). Next, we use i again to make the replacements of g i with ∇ ℓ i ( w ) + µw and ρ i with q i = q l i . In summary, each approximation is updated every iteration by changing one component based on the current iterate w . The indices i, j are 'decoupled' for theoretical convenience, but in practice using only i works similarly, which we use in Sec. 4.

Computational Aspects. The weight update in Line 10 is solved exactly by (i) sorting the vector of losses in O ( n log n ) , (ii) plugging the sorted loss table l into the Pool Adjacent Violators (PAV) algorithm running in O ( n ) time, as mentioned in Sec. 2. Because only one element of l changes every iterate, we may simply bubble sort l starting from the index that was changed. While in the worst case, this cost is O ( n ) , it is exactly O ( s ) where s is the number of swaps needed to resort l . We find in experiments that the sorted order of l stabilizes quickly. The storage of the gradient table g requires O ( nd ) space in general, but it can be reduced to O ( n ) for generalized linear models and nonlinear additive models. For losses of the form ℓ i ( w ) = h ( x ⊤ i w,y i ) , for a differentiable loss h and scalar output y i , we have ∇ ℓ i ( w ) = x i h ′ ( x ⊤ i w,y i ) . We only need to store the scalar h ′ ( x ⊤ i w,y i ) , so Prospect requires O ( n + d ) memory. In terms of computational complexity, Lines 8 and 13 require O ( d ) operations and Line 10 requires at most O ( n ) operations, so that in total the iteration complexity is O ( n + d ) . In comparison, a full batch gradient descent requires O ( nd ) operations so Prospect decouples efficiently the cost of computing the losses, gradients, and weights.

Convergence Analysis. We assume throughout that each ℓ i is convex, G -Lipschitz, and L -smooth. We also assume that the D = D f is an f -divergence with the generator f being α n -strongly convex on the interval [0 , n ] (e.g. α n = 2 n for the χ 2 -divergence and α n = 1 for the KL-divergence).

The convergence guarantees depend on the condition numbers κ = 1 + L/µ of the individual regularized losses, as well as a measure κ σ = nσ n of the skewness of the spectrum. Note that both κ and κ σ are necessarily larger than or equal to one. Define w ⋆ := arg min w F σ ( w ) , which exists and is unique due to the strong convexity of F σ . The proof is given in Appx. D.6.

Theorem 1. Prospect with a small enough step size is guaranteed to converge linearly for all ν &gt; 0 . If, in addition, the shift cost is ν ≥ Ω( G 2 /µα n ) , then the sequence of iterates ( w ( t ) ) t ≥ 1 generated by Prospect and learning rate η = (12 µ (1 + κ ) κ σ ) -1 converges linearly at a rate τ = 2max { n, 24 κ σ ( κ +1) } , i.e.,

<!-- formula-not-decoded -->

The number of iterations t required by Prospect to achieve E ∥ w ( t ) -w ⋆ ∥ 2 2 ≤ ε (provided that ν is large enough) is t = O (( n + κκ σ ) ln(1 /ε )) . This exactly matches the rate of the LSVRG (Mehta et al., 2023), the only primal stochastic optimizer that converges linearly for spectral risk measures. However, unlike LSVRG, Prospect is guaranteed to converge linearly for any shift cost and has a single hyperparameter, the stepsize η . Similarly, compared to primal-dual stochastic saddle-point methods, our algorithm requires only one learning rate, streamlining its implementation.

Prospect Variants for Non-Smooth Objectives. We may wonder about the convergence behavior of Prospect when either the shift cost ν = 0 , or the underlying losses ℓ i are non-smooth. While the smoothness of the objective is then lost, Prospect can still converge to the minimizer w ⋆ 0 as we prove below. The first setting is relevant as historically, SRMs such as the superquantile have been employed as coherent risk measures for loss distributions (Acerbi &amp; Tasche, 2002) in the form of an L -estimator ∑ n i =1 σ i l ( i ) (as seen in Sec. 2). If these losses are separated at the optimum, however, we may achieve linear convergence with Prospect even with ν = 0 , due to 'hidden smoothness' of the objective, i.e. differentiability at points for which ℓ ( w ) has distinct components. Assume that each ℓ i is convex and that µ &gt; 0 .

Proposition 2. Let w ⋆ ν be the unique minimizer of (8) with shift cost ν ≥ 0 . Assume that the values ℓ 1 ( w ⋆ 0 ) , . . . , ℓ n ( w ⋆ 0 ) are all distinct. Then, there exists a constant ν 0 &gt; 0 such that w ⋆ 0 = w ⋆ ν exactly for all ν ≤ ν 0 . Thus, running Prospect with ν ∈ (0 , ν 0 ] converges to the minimizer w ⋆ 0 .

In particular, ν 0 is chosen so that 2 nν 0 ( σ i +1 -σ i ) &lt; ℓ ( i +1) ( w ⋆ 0 ) -ℓ ( i ) ( w ⋆ 0 ) for each i , or as the multiplicative factor that relates gaps in the spectrum to gaps in the optimal losses (see Appx. B). When ℓ i itself may be non-smooth, we generalize Prospect by applying it to the Moreau envelope of each loss ℓ i and its gradient (Bauschke &amp; Combettes, 2011; Rockafellar, 1976), allowing for losses such as those containing an ℓ 1 penalty. Specifically, we consider oracles returning ∇ env( ℓ i )( w ) where env( ℓ i )( w ) := inf v ∈ R d ℓ i ( v ) + ∥ w -v ∥ 2 2 ; the update steps can be expressed in terms of the proximal operators of the losses (Bauschke &amp; Combettes, 2011). These oracles can easily be accessed either in closed form or by efficient subroutines in common machine learning settings (Defazio, 2016; Frerix et al., 2018; Roulet &amp; Harchaoui, 2022). The resulting algorithm enjoys a linear convergence guarantee similar to Thm. 1 with a more liberal condition on the shift cost ν while providing competitive performance in practice (see Appx. E).

Figure 3: Regression benchmarks . The y -axis measures suboptimality as given by (9), while the x -axis measures the number of calls to the function value/gradient oracle divided by n (i.e. passes through the training set). Rows indicate different SRM objectives while columns indicate datasets.

<!-- image -->

## 4 EXPERIMENTS

We compare Prospect against baselines in a variety of learning tasks. While we focus attention on its performance as an optimizer of its training objective, we also highlight metrics of interest on the test set in fairness and distribution shift benchmarks. The algorithm implementation and data preparation code is made publicly available online: https://github.com/ronakdm/prospect.

Setting, Baselines, Evaluation. We consider supervised learning tasks with input-label example ( x i , y i ) . Losses are of the form ℓ i ( w ) := h ( y i , w ⊤ ϕ ( x i )) , with a fixed feature embedding ϕ , and h measuring prediction loss. Uncertainty sets considered are the CVaR, extremile, and ESRM. We compare against four baselines: minibatch stochastic gradient descent (SGD), stochastic regularized dual averaging (SRDA) (Xiao, 2009), Saddle-SAGA (Palaniappan &amp; Bach, 2016), and LSVRG (Mehta et al., 2023). For SGD and SRDA, we use a batch size of 64, and for LSVRG we use an epoch length of n . For Saddle-SAGA, we find that allowing different learning rates for the primal and dual variables improves theoretically and experimentally (Appx. F) and compare against an improved heuristic (setting the dual stepsize as 10 n times smaller than the primal stepsize). We plot

<!-- formula-not-decoded -->

where w ⋆ is approximated by running LBFGS (Nocedal &amp; Wright, 1999) on the objective until convergence. The x -axis displays the number of calls to any first-order oracle w ↦→ ( ℓ i ( w ) , ∇ ℓ i ( w )) divided by n , i.e. the number of passes through the training set. We fix the shift cost ν = 1 and regularization parameter µ = 1 /n . Further details of the setup such as hyperparameter tuning, and additional results are given in Appxs H and I respectively.

## 4.1 TABULAR LEAST-SQUARES REGRESSION

We consider five tabular regression benchmarks under square loss. The datasets used are yacht ( n = 244 ) (Tsanas &amp; Xifara, 2012), energy ( n = 614 ) (Baressi Segota et al., 2020), concrete ( n = 824 ) (Yeh, 2006), kin8nm ( n = 6553 ) (Akujuobi &amp; Zhang, 2017), and power ( n = 7654 ) (T¨ ufekci, 2014). The training curves are shown in Fig. 3.

Results. Across datasets and objectives, we find that Prospect exhibits linear convergence at a rate no worse than SaddleSAGA and LSVRG but that is often much better. For example, Prospect

Figure 4: Fairness benchmarks . Top: Training curves for optimizers on the CVaR and extremile for diabetes ( left ) and CVaR and extremile for acsincome ( right ). Bottom: Statistical parity scores for the two classification objectives on diabetes ( left ) and regression objectives on acsincome . Smaller values indicate better performance for all metrics.

<!-- image -->

converges to precision 10 -8 for the CVaR on concrete and the extremile on power within half the number of passes that LSVRG takes for the same suboptimality. Similarly, for the ESRM on yacht , SaddleSAGA requires 64 epochs to reach the same precision as Prospect at 40 epochs. The direct stochastic methods, SGD and SRDA, are biased and fail to converge for any learning rate.

## 4.2 FAIR CLASSIFICATION AND REGRESSION

Inspired by Williamson &amp; Menon (2019), we explore the relationship between distributional robustness and group fairness on 2 common tabular benchmarks. Diabetes 130-Hospitals ( diabetes ) is a classification task of predicting readmission for diabetes patients based on clinical data from US hospitals (Rizvi et al., 2014). Adult Census ( acsincome ) is a regression task of predicting income of US adults from data compiled by the American Community Survey (Ding et al., 2021).

Evaluation. We evaluate fairness with the statistical parity score , which compares predictive distributions of a model given different values of a particular protected attribute Agarwal et al. (2018; 2019). Letting Z = ( X,Y,A ) denote a random (input, label, metadata attribute) triplet, a model g is said to satisfy statistical parity (SP) if the conditional distribution of g ( X ) over predictions given A = a is equal for any value a . Intuitively, SP scores measure the maximum deviation between these distributions for any over a , so values close to zero indicate SP-fairness. In diabetes , we use gender as the protected attribute A , whereas in acsincome we use race as the protected attribute. Note that the protected attributes are not supplied to the models. Results are given in Fig. 4.

Results. Firstly, we note that Prospect converges rapidly on both datasets while LSVRG fails to converge on diabetes and SaddleSAGA fails to converge on acsincome . Secondly, LSVRG does not stabilize with respect to classification SP, showing a mean/std SP score of 1 . 38 ± 0 . 25% within the final ten passes on the diabetes CVaR, whereas Prospect gives 0 . 82 ± 0 . 00% , i.e., a 40% relative improvement with greater stability. While SaddleSAGA does stabilize in SP on diabetes , it fails to qualitatively decrease at all on the acsincome . Interestingly, while suboptimality and SP-fairness are correlated for Prospect, SGD (reaching only 10 -1 suboptimality with respect to the CVaR objectives on acsincome ) achieves a lower fairness score. Again, across both suboptimality and fairness, Prospect is either the best or close to the best.

## 4.3 IMAGE AND TEXT CLASSIFICATION UNDER DISTRIBUTION SHIFT

We consider two tasks from the WILDS distribution shift benchmark (Koh et al., 2021). The Amazon Reviews ( amazon ) task (Ni et al., 2019) consists of classifying text reviews of products to a

Figure 5: Distribution shift benchmarks . Top row: Training curves and worst group misclassification error on amazon test. Bottom row: Training curves and median group misclassification error on the iwildcam test set. Smaller values indicate better performance for all metrics.

<!-- image -->

rating of 1-5, with disjoint train and test reviewers. The iWildCam ( iwildcam ) image classification challenge (Beery et al., 2020) contains labeled images of animals, flora, and backgrounds from cameras placed in wilderness sites. Shifts are due to changes in camera angles, locations, lighting... We use n = 10000 and n = 20000 examples respectively. For both datasets, we train a linear probe classifier , i.e., a linear model over a frozen deep representation. For amazon , we use a pretrained BERT model (Devlin et al., 2019) fine-tuned on a held-out subset of the Amazon Reviews training set for 2 epochs. For iwildcam , we use a ResNet50 pretrained on ImageNet (without fine-tuning).

Evaluation. Apart from the training suboptimality, we evaluate the spectral risk objectives on their robustness to subpopulation shifts. We define each subpopulation group based on the true label. For amazon , we use the worst group misclassification error on the test set (Sagawa et al., 2020). For iwildcam , we use the median group error owing to its larger number of classes.

Results. For both amazon and iwildcam , Prospect and SaddleSAGA (with our heuristic) outperform LSVRG in training suboptimality. We hypothesize that this phenomenon is due to checkpoints of LSVRG getting stale over the n -length epochs for these datasets with large n (leading to a slow reduction of bias). In contrast, Prospect and SaddleSAGA avoid this issue by dynamically updating the running estimates of the importance weights. For the worst group error for amazon , Prospect and SaddleSAGA outperform LSVRG. Prospect has a mean/std worst group error of 77 . 38 ± 0 . 00% over the last ten passes on the extremile, whereas SaddleSAGA has a slightly worse 77 . 53 ± 1 . 57% . Interestingly, on iwildcam , LSVRG and Prospect give stronger generalization performance, nearly 1 pp better, than SaddleSAGA in terms of median group misclassification rate. In summary, across tasks and objectives, Prospect demonstrates best or close to best performance.

## 5 DISCUSSION

Weintroduced Prospect, a distributionally robust optimization algorithm for minimizing spectral risk measures that has a linear convergence guarantee. Prospect demonstrates rapid linear convergence on benchmark examples and has the practical benefits of converging for any shift cost while only having a single hyperparameter. Promising avenues for future work include extensions to the nonconvex setting by considering the regular subdifferential, variations using other uncertainty sets, and further exploring connections to algorithmic fairness.

Acknowledgements. This work was supported by NSF DMS-2023166, CCF-2019844, DMS2134012, NIH, and ODNI's IARPA program via 2022-22072200003. Part of this work was done while Z. Harchaoui was visiting the Simons Institute for the Theory of Computing.

## REFERENCES

- Carlo Acerbi and Dirk Tasche. On the Coherence of Expected Shortfall. Journal of Banking &amp; Finance , 26, 2002.
- Alekh Agarwal, Alina Beygelzimer, Miroslav Dudik, John Langford, and Hanna Wallach. A Reductions Approach to Fair Classification. In ICML , 2018.
- Alekh Agarwal, Miroslav Dudik, and Zhiwei Steven Wu. Fair Regression: Quantitative Definitions and Reduction-Based Algorithms. In ICML , 2019.
- Uchenna Akujuobi and Xiangliang Zhang. Delve: A Dataset-Driven Scholarly Search and Analysis System. SIGKDD Explor. Newsl. , 19, 2017.
- Sandi Baressi Segota, Nikola Andelic, Jan Kudlacek, and Robert Cep. Artificial Neural Network for Predicting Values of Residuary Resistance per Unit Weight of Displacement. Journal of Maritime &amp;Transportation Science , 57, 2020.
- Heinz H Bauschke and Patrick L Combettes. Convex Analysis and Monotone Operator Theory in Hilbert Spaces . Springer, 2011.
- Sara Beery, Elijah Cole, and Arvi Gjoka. The iWildCam 2020 Competition Dataset. arXiv preprint arXiv:2004.10340 , 2020.
- Aharon Ben-Tal, Dick den Hertog, Anja De Waegenaere, Bertrand Melenberg, and Gijs Rennen. Robust Solutions of Optimization Problems Affected by Uncertain Probabilities. Management Science , 59, 2013.
- Dimitri P Bertsekas. Nonlinear Programming. Journal of the Operational Research Society , 1997.
- Michael J Best, Nilotpal Chakravarti, and Vasant A Ubhaya. Minimizing Separable Convex Functions Subject to Simple Chain Constraints. SIAM Journal on Optimization , 10, 2000.
- Jose Blanchet, Peter W Glynn, Jun Yan, and Zhengqing Zhou. Multivariate Distributionally Robust Convex Regression under Absolute Error Loss. In NeurIPS , 2019a.
- Jose Blanchet, Yang Kang, and Karthyek Murthy. Robust Wasserstein Profile Inference and Applications to Machine Learning. Journal of Applied Probability , 56, 2019b.
- Yair Carmon and Danielle Hausler. Distributionally Robust Optimization via Ball Oracle Acceleration. In NeurIPS , 2022.
- Ruidi Chen and Ioannis Ch. Paschalidis. Distributionally robust learning. Foundations and Trends® in Optimization , 2020.
- Julien Grand Clement and Christian Kroer. First-Order Methods for Wasserstein Distributionally Robust MDP. In ICML , 2021.
- John Cotter and Kevin Dowd. Extreme Spectral Risk Measures: an Application to Futures Clearinghouse Margin Requirements. Journal of Banking &amp; Finance , 30, 2006.
- Zac Cranko, Zhan Shi, Xinhua Zhang, Richard Nock, and Simon Kornblith. Generalised Lipschitz Regularisation Equals Distributional Robustness. In ICML , 2021.
- Abdelaati Daouia, Ir` ene Gijbels, and Gilles Stupfler. Extremiles: A New Perspective on Asymmetric Least Squares. Journal of the American Statistical Association , 114, 2019.
- Aaron Defazio. A Simple Practical Accelerated Method for Finite Sums. In NeurIPS , 2016.
- Aaron Defazio, Francis Bach, and Simon Lacoste-Julien. SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives. NeurIPS , 27, 2014.

- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A Large-scale Hierarchical Image Database. In CVPR , 2009.
- Yuyang Deng, Mohammad Mahdi Kamani, and Mehrdad Mahdavi. Distributionally Robust Federated Averaging. In NeurIPS , 2020.
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In NAACL , 2019.
- Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring Adult: New Datasets for Fair Machine Learning. In NeurIPS , volume 34. Curran Associates, Inc., 2021.
- Yanbo Fan, Siwei Lyu, Yiming Ying, and Baogang Hu. Learning with Average Topk Loss. In NeurIPS , volume 30, 2017.
- Rizal Fathony, Ashkan Rezaei, Mohammad Ali Bashiri, Xinhua Zhang, and Brian Ziebart. Distributionally Robust Graphical Models. In NeurIPS , 2018.
- Thomas Frerix, Thomas M¨ ollenhoff, Michael M¨ oller, and Daniel Cremers. Proximal Backpropagation. In ICLR , 2018.
- Soumyadip Ghosh, Mark Squillante, and Ebisa Wollega. Efficient Generalization with Distributionally Robust Learning. In NeurIPS , 2021.
- Tatsunori Hashimoto, Megha Srivastava, Hongseok Namkoong, and Percy Liang. Fairness without Demographics in Repeated Loss Minimization. In ICML , 2018.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. In CVPR , 2016.
- Jean-Baptiste Hiriart-Urruty and Claude Lemar´ echal. Fundamentals of Convex Analysis . Springer Science &amp; Business Media, 2004.
- Hisham Husain. Distributional Robustness with IPMs and links to Regularization and GANs. In NeurIPS , 2020.
- Yu Inatsu, Shogo Iwazaki, and Ichiro Takeuchi. Active Learning for Distributionally Robust LevelSet Estimation. In ICML , 2021.
- Yu Inatsu, Shion Takeno, Masayuki Karasuyama, and Ichiro Takeuchi. Bayesian Optimization for Distributionally Robust Chance-constrained Problem. In ICML , 2022.
- Yang Jiao, Kai Yang, and Dongjin Song. Distributed Distributionally Robust Optimization with Non-Convex Objectives. In NeurIPS , 2022.
- Jikai Jin, Bohang Zhang, Haiyang Wang, and Liwei Wang. Non-convex Distributionally Robust Optimization: Non-asymptotic Analysis. In NeurIPS , 2021.
- Nathan Kallus, Xiaojie Mao, Kaiwen Wang, and Zhengyuan Zhou. Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning. In ICML , 2022.
- Kenji Kawaguchi and Haihao Lu. Ordered SGD: A New Stochastic Optimization Framework for Empirical Risk Minimization. In AISTATS , volume 108, 2020.
- Justin Khim, Liu Leqi, Adarsh Prasad, and Pradeep Ravikumar. Uniform Convergence of Rankweighted Learning. In ICML , 2020.
- Diederik Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In ICLR , 2015.
- Johannes Kirschner, Ilija Bogunovic, Stefanie Jegelka, and Andreas Krause. Distributionally Robust Bayesian Optimization. In AISTATS , 2020.

- Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton A. Earnshaw, Imran S. Haque, Sara Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. WILDS: A Benchmark of in-the-Wild Distribution Shifts, 2021.
- Daniel Kuhn, Peyman Mohajerin Esfahani, Viet Anh Nguyen, and Soroosh Shafieezadeh-Abadeh. Wasserstein Distributionally Robust optimization: Theory and Applications in Machine Learning. Operations research &amp; management science in the age of analytics , 2019.
- Yassine Laguel, Krishna Pillutla, J´ erˆ ome Malick, and Zaid Harchaoui. Superquantiles at Work: Machine Learning Applications and Efficient Subgradient Computation. Set-Valued and Variational Analysis , 2021.
- Jaeho Lee, Sejun Park, and Jinwoo Shin. Learning Bounds for Risk-sensitive Learning. In NeurIPS , volume 33, 2020.
- Daniel Levy, Yair Carmon, John Duchi, and Aaron Sidford. Large-Scale Methods for Distributionally Robust Optimization. In NeurIPS , 2020.
- Jiajin Li, Sen Huang, and Anthony Man-Cho So. A First-Order Algorithmic Framework for Distributionally Robust Logistic Regression. In NeutIPS , 2019.
- Yeshu Li, Danyal Saeed, Xinhua Zhang, Brian Ziebart, and Kevin Gimpel. Moment Distributionally Robust Tree Structured Prediction. In NeurIPS , volume 35. Curran Associates, Inc., 2022.
- Zachary C. Lipton, Yu-Xiang Wang, and Alexander J. Smola. Detecting and Correcting for Label Shift with Black Box Predictors. In ICML , 2018.
- Evan Z Liu, Behzad Haghgoo, Annie S Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, and Chelsea Finn. Just Train Twice: Improving Group Robustness without Training Group Information. In ICML , 2021.
- Jiashuo Liu, Zheyan Shen, Peng Cui, Linjun Zhou, Kun Kuang, and Bo Li. Distributionally Robust Learning With Stable Adversarial Training. IEEE TKDE , 2022a.
- Jiashuo Liu, Jiayun Wu, Bo Li, and Peng Cui. Distributionally Robust Optimization with Data Geometry. In NeurIPS , volume 35. Curran Associates, Inc., 2022b.
- Zijian Liu, Qinxun Bai, Jose Blanchet, Perry Dong, Wei Xu, Zhengqing Zhou, and Zhengyuan Zhou. Distributionally Robust Q -Learning. In ICML , 2022c.
- Kyriakos Lotidis, Nicholas Bambos, Jose Blanchet, and Jiajin Li. Wasserstein Distributionally Robust Linear-Quadratic Estimation under Martingale Constraints. In AISTATS , 2023.
- Luo Luo, Haishan Ye, Zhichao Huang, and Tong Zhang. Stochastic recursive gradient descent ascent for stochastic nonconvex-strongly-concave minimax problems. In NeurIPS , 2020.
- Andreas Maurer, Daniela Angela Parletta, Andrea Paudice, and Massimiliano Pontil. Robust Unsupervised Learning via L-statistic Minimization. In ICML , 2021.
- Ronak Mehta, Vincent Roulet, Krishna Pillutla, Lang Liu, and Zaid Harchaoui. Stochastic Optimization for Spectral Risk Measures. In AISTATS , 2023.
- Tong Mu, Yash Chandak, Tatsunori B Hashimoto, and Emma Brunskill. Factored DRO: Factored Distributionally Robust Policies for Contextual Bandits. In NeurIPS , 2022.
- Debarghya Mukherjee, Felix Petersen, Mikhail Yurochkin, and Yuekai Sun. Domain Adaptation meets Individual Fairness. And they get along. In NeurIPS , 2022.
- Hongseok Namkoong and John C Duchi. Stochastic Gradient Methods for Distributionally Robust Optimization with f-divergences. NeurIPS , 2016.

- Yassine Nemmour, Bernhard Sch¨ olkopf, and Jia-Jie Zhu. Approximate Distributionally Robust Nonlinear Optimization with Application to Model Predictive Control: A Functional Approach. In L4DC , 2021.
- Yurii Nesterov. Smooth Minimization of Non-Smooth Functions. Mathematical Programming , 2005.
- Yurii Nesterov. Lectures on Convex Optimization . Springer Publishing Company, Incorporated, 2nd edition, 2018.
- Jianmo Ni, Jiacheng Li, and Julian McAuley. Justifying Recommendations using Distantly-Labeled Reviews and Fine-grained Aspects. In EMNLP , 2019.
- Jorge Nocedal and Stephen J Wright. Numerical optimization . Springer, 1999.
- Balamurugan Palaniappan and Francis Bach. Stochastic Variance Reduction Methods for SaddlePoint Problems. NeurIPS , 29, 2016.
- Krishna Pillutla, Yassine Laguel, J´ erˆ ome Malick, and Zaid Harchaoui. Federated Learning with Superquantile Aggregation for Heterogeneous Data. Mach. Learn. , 2023.
- B.T. Polyak. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics , 1964.
- Hamed Rahimian and Sanjay Mehrotra. Frameworks and Results in Distributionally Robust Optimization. Open Journal of Mathematical Optimization , 3, 2022.
- Allen Z. Ren and Anirudha Majumdar. Distributionally Robust Policy Learning via Adversarial Environment Generation. IEEE RAL , 2022.
- Ali Rizvi, Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore. Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. BioMed Research International , 2014, 2014.
- R Tyrrell Rockafellar. Monotone Operators and the Proximal Point Algorithm. SIAM Journal on Control and Optimization , 14, 1976.
- R Tyrrell Rockafellar and Johannes O Royset. Superquantiles and Their Applications to Risk, Random Variables, and Regression. In Theory Driven by Influential Applications . Informs, 2013.
- Vincent Roulet and Zaid Harchaoui. Differentiable Programming ` a la Moreau. In ICASSP . IEEE, 2022.
- Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. Distributionally Robust Neural Networks. In ICLR , 2020.
- Dvir Samuel and Gal Chechik. Distributional Robustness Loss for Long-Tail Learning. In ICCV , 2021.
- Hitesh Sapkota, Yiming Ying, Feng Chen, and Qi Yu. Distributionally Robust Optimization for Deep Kernel Multiple Instance Learning. In AISTATS , 2021.
- Alexander Shapiro. Distributionally Robust Stochastic Programming. SIAM Journal on Optimization , 2017.
- Vishnu D. Sharma, Maymoonah Toubeh, Lifeng Zhou, and Pratap Tokekar. Risk-Aware Planning and Assignment for Ground Vehicles using Uncertain Perception from Aerial Vehicles. In 2020 IEEE/RSJ IROS , 2020.
- Xuli Shen, Xiaomei Wang, Qing Xu, Weifeng Ge, and Xiangyang Xue. Towards Scalable and Fast Distributionally Robust Optimization for Data-Driven Deep Learning. In IEEE ICDM , 2022.
- Galen Shorack. Probability for Statisticians . Springer Texts in Statistics, 2017.

- Rahul Singh, Qinsheng Zhang, and Yongxin Chen. Improving Robustness via Risk Averse Distributional Reinforcement Learning. In L4DC , 2020.
- Aman Sinha, Matthew O'Kelly, Hongrui Zheng, Rahul Mangharam, John Duchi, and Russ Tedrake. FormulaZero: Distributionally Robust Online Adaptation via Offline Population Synthesis. In ICML , 2020.
- Agnieszka Słowik and L´ eon Bottou. On Distributionally Robust Optimization and Data Rebalancing . In AISTATS , 2022.
- Matthew Staib and Stefanie Jegelka. Distributionally Robust Optimization and Generalization in Kernel Methods. In NeurIPS , 2019.
- Masashi Sugiyama, Shinichi Nakajima, Hisashi Kashima, Paul Buenau, and Motoaki Kawanabe. Direct Importance Estimation with Model Selection and Its Application to Covariate Shift Adaptation. In NeurIPS , 2007.
- Sebastian Shenghong Tay, Chuan Sheng Foo, Urano Daisuke, Richalynn Leong, and Bryan Kian Hsiang Low. Efficient Distributionally Robust Bayesian Optimization with Worst-case Sensitivity. In ICML , 2022.
- Athanasios Tsanas and Angeliki Xifara. Accurate Quantitative Estimation of Energy Performance of Residential Buildings Using Statistical Machine Learning Tools. Energy and Buildings , 49, 2012.
- Pınar T¨ ufekci. Prediction of Full Load Electrical Power Output of a Base Load Operated Combined Cycle Power Plant using Machine Learning Methods. International Journal of Electrical Power &amp;Energy Systems , 60, 2014.
- Hieu Vu, Toan Tran, Man-Chung Yue, and Viet Anh Nguyen. Distributionally Robust Fair Principal Components via Geodesic Descents. In ICLR , 2022.
- Shengbo Wang, Nian Si, Jose Blanchet, and Zhengyuan Zhou. A Finite Sample Complexity Bound for Distributionally Robust Q-learning. In AISTATS , 2023.
- Zhenyi Wang, Li Shen, Le Fang, Qiuling Suo, Tiehang Duan, and Mingchen Gao. Improving Taskfree Continual Learning by Distributionally Robust Memory Evolution. In ICML , 2022.
- Robert Williamson and Aditya Menon. Fairness Risk Measures. In ICML , 2019.
- Lin Xiao. Dual Averaging Method for Regularized Stochastic Learning and Online Optimization. In NeurIPS , volume 22, 2009.
- Mengdi Xu, Peide Huang, Yaru Niu, Visak Kumar, Jielin Qiu, Chao Fang, Kuan-Hui Lee, Xuewei Qi, Henry Lam, Bo Li, and Ding Zhao. Group Distributionally Robust Reinforcement Learning with Hierarchical Latent Variables. In AISTATS , 2023.
- Zhouhao Yang, Yihong Guo, Pan Xu, Anqi Liu, and Animashree Anandkumar. Distributionally Robust Policy Gradient for Offline Contextual Bandits. In AISTATS , 2023.
- Erfan Yazdandoost Hamedani and Afrooz Jalilzadeh. A Stochastic Variance-Reduced Accelerated Primal-Dual Method for Finite-Sum Saddle-Point Problems. Comput. Optim. Appl. , 2023.
- Ivy Yeh. Analysis of Strength of Concrete Using Design of Experiments and Neural Networks. Journal of Materials in Civil Engineering , 18, 2006.
- Yaodong Yu, Tianyi Lin, Eric V Mazumdar, and Michael Jordan. Fast Distributionally Robust Learning with Variance-Reduced Min-Max Optimization. In AISTATS , 2022.
- Yibo Zeng and Henry Lam. Generalization Bounds with Minimal Dependency on Hypothesis Class via Distributionally Robust Optimization. In NeurIPS , 2022.
- Jingzhao Zhang, Aditya Menon, Andreas Veit, Srinadh Bhojanapalli, Sanjiv Kumar, and Suvrit Sra. Coping with Label Shift via Distributionally Robust Optimisation. In ICLR , 2021.

- Zhengqing Zhou, Zhengyuan Zhou, Qinxun Bai, Linhai Qiu, Jose Blanchet, and Peter Glynn. FiniteSample Regret Bound for Distributionally Robust Offline Tabular Reinforcement Learning. In AISTATS , 2021.
- Zhengyu Zhou and Weiwei Liu. Sample Complexity for Distributionally Robust Learning under chi-square divergence. JMLR , 2023.

## Appendix

The Appendix sections are organized as follows. We summarize notation in Appx. A and provide intuition and results regarding the primal/dual objective function in Appx. B. We describe in detail efficient implementations of the proposed algorithm in Appx. C. In Appx. D, we describe the convergence analyses of the main algorithm. In Appx. E and Appx. F, we describe an Moreau envelope-based variant of our method and an improved version of an existing saddle-point method, respectively. Appx. G contains technical results shared to multiple proofs. We then describe the experimental setup in detail in Appx. H and give additional results in Appx. I.

## Table of Contents

| A   | Summary of Notation          | Summary of Notation          |   17 |
|-----|----------|----------|------|
| B   | Properties of the Primal and Dual Objectives | Properties of the Primal and Dual Objectives          |   17 |
| C   | Efficient Implementation of Prospect         | Efficient Implementation of Prospect          |   25 |
| D   | Convergence Analysis of Prospect          | Convergence Analysis of Prospect          |   30 |
|     | D.1          | Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . .   |   30 |
|     | D.2          | Key Technical Lemma: Bias of the Prospect Gradient Estimate .      |   31 |
|     | D.3          | Bounding the Distance-to-Optimum . . . . . . . . . . . . . . .     |   33 |
|     | D.4          | Bounding the Evolution of the Lyapunov Function Terms . . . .      |   35 |
|     | D.5          | Tuning Constants and Final Rate . . . . . . . . . . . . . . . .    |   39 |
|     | D.6          | Proof of Main Result . . . . . . . . . . . . . . . . . . . . . .   |   46 |
| E   | Improving Prospect with Moreau Envelopes     | Improving Prospect with Moreau Envelopes          |   46 |
|     | E.1          | Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . .   |   46 |
|     | E.2          | Convergence Analysis . . . . . . . . . . . . . . . . . . . . . .   |   48 |
| F   | Improving the Direct Saddle-Point Baseline   | Improving the Direct Saddle-Point Baseline          |   51 |
|     | F.1          | Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . .   |   51 |
|     | F.2          | Convergence Analysis . . . . . . . . . . . . . . . . . . . . . .   |   52 |
| G   | Technical Results from Convex Analysis       | Technical Results from Convex Analysis          |   57 |
| H   | Experimental Details          | Experimental Details          |   59 |
|     | H.1          | Tasks &Objectives . . . . . . . . . . . . . . . . . . . . . . .    |   59 |
|     | H.2          | Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   60 |
|     | H.3          | Hyperparameter Selection . . . . . . . . . . . . . . . . . . . .   |   61 |
|     | H.4          | Compute Environment . . . . . . . . . . . . . . . . . . . . . .    |   61 |
| I   | Additional Experiments          | Additional Experiments          |   61 |

## A SUMMARY OF NOTATION

We collect the notation used throughout the paper in Tab. 1.

Table 1: Notation used throughout the paper.

| Symbol          | Description          |
|----------|----------|
| µ ≥ 0          | Standard regularization constant.          |
| ν ≥ 0          | Shift cost.          |
| α n          | Strong convexity constant for any f generating an f -divergence.          |
| ¯ ν          | Shorthand ¯ ν = nα n ν (used in the convergence proofs).          |
| ℓ 1 ( w ) , . . . , ℓ n ( w ) | Loss functions ℓ i : R d → R .          |
| ℓ ( w )          | Vector of losses ℓ ( w ) = ( ℓ 1 ( w ) , . . . , ℓ n ( w )) for w ∈ R d .          |
| r i ( w )          | Regularized loss r i ( w ) = ℓ i ( w )+ µ 2 ∥ w ∥ 2 2 .          |
| r ( w )          | Vector of regularized losses r ( w ) = ( r 1 ( w ) , . . . , r n ( w )) .          |
| ∇ ℓ ( w )          | Jacobian matrix of ℓ : R d → R n at w (shape = n × d ).          |
| σ          | The vector σ = ( σ 1 , . . .,σ n ) ∈ [0 , 1] n where each σ 1 ≤ . . . ≤ σ n and they sum to 1.          |
| P ( σ )          | The set { Π σ : Π ∈ [0 , 1] n × n , Π 1 n = 1 n , Π ⊤ 1 n = 1 n } , known as the permutahedron.          |
| f          | Convex function f : [0 , ∞ ) → R ∪ { + ∞} generating an f -divergence.          |
| f ∗          | Convex conjugate f ∗ ( y ) := sup x ∈ R { xy - f ( x ) } .          |
| Ω f or Ω          | Shift penalty function Ω f : P ( σ ) ↦→ [0 , ∞ ) . We consider f -divergence penalties Ω f ( q ) = D f ( q ∥ 1 n /n ) .          |
| F σ          | Main objective F σ ( w ) = max q ∈P ( σ ) { q ⊤ ℓ ( w ) - νD f ( q ∥ 1 n /n ) } + µ 2 ∥ w ∥ 2 2 .          |
| q opt ( l ) or q l          | Most unfavorable reweighting for a given vector l of losses, i.e., q opt ( l ) = argmax q ∈P ( σ ) q ⊤ l - νD ( q ∥ 1 n /n ) . q l used only in main text for readability. |
| w ⋆          | Optimal weights argmin w ∈ R d max q ∈P ( σ ) q ⊤ l - νD ( q ∥ 1 n /n )+( µ/ 2) ∥ w ∥ 2 2 .          |
| q ⋆          | Most unfavorable reweighting of ℓ ( w ⋆ ) , i.e., q ⋆ = q opt ( ℓ ( w ⋆ ))          |
| G          | Lipschitz constant of each ℓ i w.r.t. ∥·∥ 2 .          |
| L          | Lipschitz constant of each ∇ ℓ i w.r.t. ∥·∥ 2 .          |
| M          | M = L + µ , the Lipschitz constant of each ∇ r i w.r.t. ∥·∥ 2 .          |
| E t [ · ]          | Shorthand for E [ · &#124; w ( t ) ] , i.e., expectation conditioned on w ( t ) .          |

## B PROPERTIES OF THE PRIMAL AND DUAL OBJECTIVES

Recall that we are interested in the optimization problem

<!-- formula-not-decoded -->

where D f ( q ∥ 1 n /n ) denotes an f -divergence between the distribution associated to the reweighting q and the discrete uniform weights 1 n /n = (1 /n, . . . , 1 /n ) and P ( σ ) is the spectral risk measure uncertainty set.

The first goal for this section will be to derive properties of the function F σ ( w ) , or the primal objective , as well as the inner maximization problem, which we refer to as the dual objective . Both will be useful in motivating and analyzing Prospect (used for the primal minimization) and various subroutines used to compute the maximally unfavorable reweighting (i.e., the maximizer over q in

the inner maximization). These properties do not depend on the structure of the P ( σ ) itself, only that it is a closed, convex set. The second goal of the section is to provide additional background on the choice of P ( σ ) from a statistical modeling perspective. The uncertainty set P ( σ ) is then described further from a computational perspective in Appx. C.

Review of f -Divergences. Let q and p be any two probability mass functions defined on atoms { 1 , . . . , n } . Consider a convex function f : [0 , ∞ ) ↦→ R ∪{ + ∞} such that f (1) = 0 , f ( x ) is finite for x &gt; 0 , and lim x → 0 + f ( x ) = 0 . The f -divergence from q to p generated by this function f is

<!-- formula-not-decoded -->

where we define 0 f (0 / 0) := 0 in the formula above. If there is an i such that p i = 0 but q i &gt; 0 , we say D f ( q ∥ p ) = ∞ . The χ 2 -divergence is generated by f χ 2 ( x ) = x 2 -1 and the KL divergence is generated by f KL ( x ) = x ln x + ι + ( x ) where ι + denotes the convex indicator that is zero for x ≥ 0 and + ∞ otherwise, and we define x ln x = 0 for all x &lt; 0 .

The Dual Problem. We describe the inner maximization first, that is

<!-- formula-not-decoded -->

Its properties will inform the algorithmic implementation for the minimization over w in (10). In the case of an f -divergence between q and the uniform weights 1 n /n , we have

<!-- formula-not-decoded -->

We now derive the dual problem to Equation (11). This will lead to an algorithm to solve the optimization problem efficiently. Throughout, we denote f ∗ ( y ) := sup x ∈ R { xy -f ( x ) } as the convex conjugate of f .

Weconsider the following functions whose conjugates are strictly convex. Recall that if f is smooth, i.e., with Lispchtiz continuous gradients, then its conjugate is strongly convex, hence strictly convex. More generally f ∗ is strictly convex if f is convex and essentially smooth, that is, with gradient norm tending to + ∞ at its boundaries, see e.g. (Rockafellar, 1976) for a detailed presentation. For simple cases such as the χ 2 or KL divergence presented, strict convexity of the convex conjugate is naturally satisfied:

<!-- formula-not-decoded -->

Proposition 3. Let l ∈ R n be a vector and π be a permutation that sorts its entries in non-decreasing order, i.e., l π (1) ≤ . . . ≤ l π ( n ) . Consider a function f strictly convex with strictly convex conjugate defining a divergence D f . Then, the maximization over the permutahedron subject to the shift penalty can be expressed as

<!-- formula-not-decoded -->

where we define g i ( c i ; l ) := σ i c i + ν n f ∗ ( l π ( i ) -c i ν ) . The optima of both problems, denoted

<!-- formula-not-decoded -->

are related as q opt ( l ) = ∇ ( νD f ( ·∥ 1 n /n )) ∗ ( l -c opt π -1 ( l )) , that is,

<!-- formula-not-decoded -->

Proof. Let ι P ( σ ) denote the indicator function of the permutahedron P ( σ ) , which is 0 inside P ( σ ) and + ∞ outside of P ( σ ) . Its convex conjugate is the support function of the permutahedron, i.e.,

<!-- formula-not-decoded -->

For two closed convex functions h 1 and h 2 that are bounded from below, the convex conjugate of their sum is the infimal convolution of their conjugate (Hiriart-Urruty &amp; Lemar´ echal, 2004, Proposition 6.3.1):

<!-- formula-not-decoded -->

Provided that h 1 + h 2 is strictly convex, we have that the maximizer defining the conjugate is unique and equal to the gradient, that is,

<!-- formula-not-decoded -->

If, in addition, h ∗ 1 + h ∗ 2 is strictly convex and h ∗ 2 is differentiable, we have, by Danskin's theorem (Bertsekas, 1997),

<!-- formula-not-decoded -->

Consider then h 1 ( q ) = ι P ( σ ) ( q ) and h 2 ( q ) = Ω f ( q ) := νD f ( q ∥ 1 n /n ) . Provided that f is strictly convex with f ∗ strictly convex, D f is also strictly convex with D ∗ f strictly convex since D f just decomposes as a sum of f on independent variables. We have then

<!-- formula-not-decoded -->

where y (1) ≤ . . . ≤ y ( n ) are the ordered values of y ∈ R n . Moreover we have that

<!-- formula-not-decoded -->

Since for any x ∈ R n , Ω f is decomposable into a sum of identical functions evaluated at the coordinates ( x 1 , . . . , x n ) , that is, Ω f ( x ) = ∑ n i =1 ω ( x i ) , its convex conjugate is Ω ∗ f ( y ) = ∑ n i =1 ω ∗ ( y i ) . In our case, ω ( x i ) = ν n f ( nx i ) from Equation (12), so ω ∗ ( y i ) = ( ν/n ) f ∗ ( y i /ν ) .

Next, by convexity of ω ∗ , we have that if for scalars l i , l j , y i , y j such that l i ≤ l j and y i ≥ y j , then using Lem. 33, we have that

<!-- formula-not-decoded -->

Hence for y to minimize Ω ∗ f ( l -y ) = ∑ n i =1 ω ∗ ( l i -y i ) , the coordinates of y must be ordered as l . That is, if π is an argsort for l , s.t. l π (1) ≤ . . . ≤ l π ( n ) , then y π (1) ≤ . . . ≤ y π ( n ) . Since ι ∗ P ( σ ) ( y ) = ∑ n i =1 σ i y ( i ) does not depend on the ordering of y , the solution of (15) must also be

ordered as l such that the dual problem (15) can be written as

<!-- formula-not-decoded -->

where we used a change of variables such that the solutions of the left and right hand sides are related as y ⋆ ( l ) = c ⋆ π -1 ( l ) .

While strict convexity of the function f ensures naturally the existence of the maximizer q opt defined above, this does not help quantify the continuity of the maximizer with respect to the given vector of losses. For that, we need to consider strong convexity of the f -divergence on the maximization set. The following proposition simply links the strong convexity of f to the strong convexity of the associated weighted divergence in ℓ 2 norm.

Proposition 4. Assume that f : R → R is α n -strongly convex on [0 , n ] . Then, q ↦→ νD f ( q ∥ 1 n /n ) is ( νnα n ) -strongly convex with respect to ∥·∥ 2 .

Proof. Due to the α n -strong convexity of f , for any q, ρ ∈ [0 , 1] n and any θ ∈ (0 , 1) and any i ∈ [ n ] ,

<!-- formula-not-decoded -->

We average this inequality over i , yielding

<!-- formula-not-decoded -->

Defining Ω f ( q ) := D f ( q ∥ 1 n /n ) , the statement above can be succinctly written as

<!-- formula-not-decoded -->

Therefore, Ω f is ( α n n ) -strongly convex with respect to ∥·∥ 2 on [0 , 1] n , so q ↦→ νD f ( q ∥ 1 n /n ) is ( νnα n ) -strongly convex.

The Pool Adjacent Violators (PAV) algorithm is designed exactly for the minimization (13). The algorithm is described for the χ 2 -divergence and KL-divergence with implementation steps in Appx. C. Both the argsort π and the inverse argort π -1 are mappings from [ n ] = { 1 , . . . , n } onto itself, but the interpretation of these indices are different for the input and output spaces [ n ] . The argsort π can be thought of as an index finder , in the sense that for a vector l ∈ R n , because l π (1) ≤ . . . ≤ l π ( n ) , π ( i ) can be interpreted as the index of an element of l which achieves the rank i in the sorted vector. On the other hand, π -1 ( i ) can be thought of as a rank finder , in that π -1 ( i ) = rank( i ) is the position that l i takes in the sorted form of l . To summarize:

<!-- formula-not-decoded -->

We may equivalently write (14) as

<!-- formula-not-decoded -->

Finally, as seen in Appx. C, it will be helpful to compute q opt in sorted order. Because the f -divergence is agnostic to the ordering of the q vector (as it is being compared to the uniform weights),

1.

•n

n

•

9u3

9v2

9v1

900 =

Uncertainty Set

Figure 6: Visualization of dual solution for varying shift cost . The hexagon represents a threedimensional uncertainty set (see Fig. 7) whereas the white dots signify solutions of (11) for a fixed loss vector and varying shift cost ν . As ν increases, the solution tends closer to the uniform vector 1 n /n , whereas as ν decreases, the shifted distribution is allowed to drift closer to the boundary of the uncertainty set.

<!-- image -->

q can also be sorted by π . Thus, we may also write

<!-- formula-not-decoded -->

The effect of varying the shift cost ν on the solution of (11) is visualized in Fig. 6.

The Primal Function. When the divergence generator f is strongly convex and the loss function ℓ : R d → R n is convex and differentiable, we have that (10) is differentiable, as we show next.

Lemma 5. When the map q ↦→ νD ( q ∥ 1 n /n ) is strongly convex over P ( σ ) , then R σ is continuously differentiable with gradient given by

<!-- formula-not-decoded -->

Proof. Because P ( σ ) as defined in (20) is closed and convex, the strongly concave function q ↦→ q ⊤ l -νD ( q ∥ 1 n /n ) has a unique maximizer. Because P ( σ ) is closed subset of the compact set ∆ n , it is also compact. By Danskin's theorem (Bertsekas, 1997, Proposition B.25), we have that F σ is continuously differentiable with the given gradient formula.

Lemma 6. Let ℓ : R d → R n be differentiable with Jacobian w ↦→ ∇ ℓ ( w ) ∈ R n × d . Let each ℓ i : R d → R be convex. Assume ν &gt; 0 . Let f be α n -strongly convex on the interval [0 , n ] . Then, the function F σ from Equation (10) is differentiable with its gradient equal to

<!-- formula-not-decoded -->

Furthermore l ↦→ q opt ( l ) is ( α n nν ) -1 -Lipschitz continuous w.r.t. ∥·∥ 2 .

Proof. First, by Prop. 4, we have that q ↦→ νD f ( q ∥ 1 n ) is ( να n n ) -strongly convex with respect to ∥·∥ 2 on [0 , 1] n . Next, due to the convexity of each ℓ i and the non-negativity of any q ∈ P ( σ ) , we have that

<!-- formula-not-decoded -->

is convex, as is its pointwise maximum (over q ) of a family of convex functions q ⊤ ℓ ( w ) . We have by Lem. 5 that F σ is continuously differentiable with

<!-- formula-not-decoded -->

Moreover, by Nesterov (2005, Theorem 1), we have that l ↦→ q opt ( l ) is Lipschitz continuous with Lipschitz constant equal to the inverse of the strong convexity constant of ν Ω f , which is να n n .

Returning to our canonical examples, we have that for the χ 2 , f χ 2 ( x ) = x 2 -1 is 2 -strongly convex on R and that f KL ( x ) = x ln x is (1 /n ) -strongly convex on [0 , n ] . Thus, the function l ↦→ q opt ( l ) will have Lipschitz constant 2 nν and ν , respectively.

Smoothness Properties. By applying Lem. 6 to Lipschitz continuous losses, we may achieve the following guarantee regarding the changes in q opt with respect to w .

Lemma 7. Let f be α n -strongly convex on the interval [0 , n ] . For any w 1 , . . . , w n , w ′ 1 , . . . , w ′ n ∈ R d construct ¯ ℓ ( w 1 , . . . , w n ) = ( ℓ i ( w i ) ) n i =1 ∈ R n , as well as ¯ ℓ ( w ′ 1 , . . . , w ′ n ) where each ℓ i is G -Lipschitz w.r.t. ∥·∥ 2 . Then, we have

<!-- formula-not-decoded -->

Proof. By the Lipschitz property of q opt (Lem. 6), we have,

<!-- formula-not-decoded -->

As a special case of Lem. 7, we may consider w 1 = · · · = w n = w ∈ R d and w ′ 1 = · · · = w ′ n = w ′ ∈ R d , in which case the result reads

<!-- formula-not-decoded -->

Properties under No Shift Penalty. Next, we use the smoothness properties above to prove Prop. 2 by virtue of the following proposition, which states the equivalence of the minimizers of 'no-cost' and 'low-cost' objectives.

Proposition 8. Let w ⋆ ν be the unique minimizer of (8) with shift cost ν ≥ 0 and χ 2 -divergence penalty. Define ℓ (1) ( w ⋆ 0 ) &lt; . . . , &lt; ℓ ( n ) ( w ⋆ 0 ) to be the order statistics of ℓ 1 ( w ⋆ 0 ) , . . . , ℓ n ( w ⋆ 0 ) , which are assumed to be distinct. Consider ν 0 such that

<!-- formula-not-decoded -->

We have that w ⋆ 0 = w ⋆ ν for all ν ≤ ν 0 .

Proof. For a vector l ∈ R n and ν ≥ 0 , consider

<!-- formula-not-decoded -->

where we used that q ⊤ 1 = 1 for all q ∈ P ( σ ) . For ν &gt; 0 , by Danskin's theorem (Bertsekas, 1997, Proposition B.25),

<!-- formula-not-decoded -->

Without loss of generality, assume that l is sorted, so that by applying the duality given by Prop. 3 and that f ∗ ( t ) = t 2 / 4 + 1 , we have that

<!-- formula-not-decoded -->

By differentiating with respect to c , we have that if unconstrained, c opt i ( l ) = l i +2 nνσ i is the primal solution and the dual solution is given by q opt i ( l ) = σ i , which is equal to the gradient of l ↦→ g 0 ( l ) when l has distinct elements. Thus, we derive a condition under which c opt i ( l ) = l i + 2 nνσ i is monotonically non-decreasing (i.e. has the same sorted order as l ), which will be true if ν is small enough. Specifically, we have that if

<!-- formula-not-decoded -->

for some ν 0 &gt; 0 , then c opt i ( l ) = l i +2 nνσ i is monotonically non-decreasing. Consequently, for any ν ≤ ν 0 ,

Denote our objective as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we explicitly show the dependence on the shift cost ν ≥ 0 . For ν = 0 , since the losses are differentiable and ℓ ( w ⋆ 0 ) is composed of distinct coordinates, L σ, 0 is differentiable at w ⋆ 0 with gradient ∇ ℓ ( w ⋆ 0 ) ⊤ ∇ h 0 ( ℓ ( w ⋆ 0 )) + µw ⋆ 0 (Mehta et al., 2023, Proposition 2), where ∇ ℓ ( w ⋆ 0 ) ∈ R n × d denotes the Jacobian of ℓ at w ⋆ 0 . Using the chain rule, we successively deduce

<!-- formula-not-decoded -->

Applying the first-order optimality conditions of L σ, 0 and L σ,ν , as well as the uniqueness of w ⋆ 0 completes the proof.

Prop. 2 of the main paper then follows by combining Prop. 8 above with the convergence guarantee Thm. 1 of Prospect. Indeed, Thm. 1 shows that Prospect is able to converge linearly for arbitrarily small ν &gt; 0 and as long as ν ≤ ν 0 . Under Prop. 8, the minimizer will be equal to w ⋆ 0 .

1.0

0.8

0.6

0.4

0.0

0.6

(VaR P(o)

1.0

0.0

0.2

0.4

1.0

Figure 7: Visualization of uncertainty sets . Top: Illustration of the permutahedra P ( σ ) within the three-dimensional probability simplex for the CVaR (21) ( top left ) p = 0 . 25 (light purple) and p = 0 . 5 (light purple), the extremile (22) ( top center ) with b = 2 . 5 (light yellow) and b = 1 . 5 (dark yellow), and the ESRM (23) ( top right ) with γ = 3 (light pink) and γ = 1 (dark pink). The size of P ( σ ) increases for more non-uniform spectra σ . Each point within the colored shapes represents a vector of probabilities q over n = 3 samples. The vertices of the polytopes are various permutations of the σ vectors, whereas the interior of the polytopes are elements of the convex hull of these permutations. Bottom: The σ associated to each colored shape is given by the same-colored bars in the bottom rows, i.e. the CVaR ( bottom left ) p = 0 . 25 (light purple) and p = 0 . 5 (light purple) from Equation (21), the extremile ( bottom center ) with b = 2 . 5 (light yellow) and b = 1 . 5 (dark yellow) from Equation (22), and the ESRM ( bottom right ) with γ = 3 (light pink) and γ = 1 (dark pink) from Equation (23). The parameters ( p, b, γ ) are risk-sensitivity parameters that increase or decrease the size of the uncertainty sets.

<!-- image -->

̸

We interpret this phenomenon as the 'hidden smoothness' of F σ , in that the non-differentiable points of the map w ↦→ max q ∈P ( σ ) q ⊤ ℓ ( w ) are precisely the points at which ℓ i ( w ) = ℓ j ( w ) for some i = j , as the subdifferential may contain multiple elements (Mehta et al., 2023, Proposition 2). Thus, if the losses are well-separated enough (in comparison to the spectrum σ ) at the minimizer w ⋆ 0 , the objective for the non-smooth setting ν = 0 and regularized setting ν &gt; 0 result in the same minimizer.

Spectral Risk Measures. Recall that for a spectral risk measure with spectrum σ = ( σ 1 , . . . , σ n ) , the uncertainty set P ( σ ) is given by:

<!-- formula-not-decoded -->

As for particular examples: for p ∈ [0 , 1] , the p -CVaR (a.k.a. superquantile) (Rockafellar &amp; Royset, 2013; Kawaguchi &amp; Lu, 2020; Laguel et al., 2021) requires that k = np elements of σ be non-zero with equal probability and that the remaining n -k are zero. The b -extremile (Daouia et al., 2019) and γ -exponential spectral risk measure (Cotter &amp; Dowd, 2006) define their spectra the equations below.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

1.0

0.8

0.6

0.4

02

0.0

0.2

1.0

0.8

0.6

0.4

0.2

0.0

0.2

The multiple cases in the CVaR definition account for the instance in which np is not an integer. In light of Lem. 7, when ν &gt; 0 , we have that the objective based on the uncertainty set (20) is differentiable with Lipschitz continuous gradient (i.e. smooth). To be more specific, we may call this the regularized or smoothed spectral risk measure. On the other hand, as described in Sec. 2, SRMs were historically computed for any loss vector l ∈ R n as

<!-- formula-not-decoded -->

where l (1) ≤ . . . ≤ l ( n ) are the order elements of l . For this reason, SRMs may also be called L -risks (Maurer et al., 2021), based on classical L -estimators (linear combinations of order statistics) from the statistics literature (Shorack, 2017). In fact, they were originally introduced as functionals L s on arbitrary real-valued probability measures (Acerbi &amp; Tasche, 2002) given by a weighted integral of the quantile function

<!-- formula-not-decoded -->

where F -1 ( p ) = inf { x ∈ R : F ( x ) &gt; p } is the quantile function of cumulative distribution function (CDF) F and s is a non-negative, non-decreasing function satisfying ∫ 1 0 s ( p ) d p = 1 . To recover (24) we view l 1 , . . . , l n as a random sample and define F n as the empirical distribution induced by the sample. Then, the quantile function F -1 n ( p ) is given by the order statistic l ( i ) when p ∈ ( i -1 n , i n ) and the discontinuity points of F n are defined to make it left-continuous. Applying L s to the empirical CDF yields

<!-- formula-not-decoded -->

for σ i = ∫ i/n ( i -1) /n s ( p ) d p . Thus, the formulae for σ i for each SRM are defined by taking an n -bin discretization of a continuous spectrum s over [0 , 1) . Notably, the permutation-based description of spectral risk measures in (24) is a unique feature of the their discrete versions. For a visualization of the feasible set P ( σ ) for the CVaR, extremile, and ESRM, see Fig. 7.

## C EFFICIENT IMPLEMENTATION OF PROSPECT

In this section, we describe Prospect including computational details, in a way that is amenable to implementation. For convenience, the conceptual description of the algorithm from Sec. 3 is restated in Algorithm 1.

Efficient Implementation. We exactly solve the maximization problem

<!-- formula-not-decoded -->

by a sequence of three steps:

- Sorting: Find π such that l π (1) ≤ . . . ≤ l π ( n ) .
- Isotonic regression: Apply Pool Adjacent Violators (PAV) (Subroutine 1) to solve the isotonic regression minimization problem (13), yielding solution c = c opt ( l ) .
- Conversion: Use (14) to convert c back to q = q opt ( l ) .

The sorting step runs in O ( n ln n ) elementary operations whereas the isotonic regression and conversion steps run in O ( n ) operations. Crucially, retrieving q from the output c = c opt ( l ) in the third

| Algorithm 1 Prospect (restated from Sec. 3)   | Algorithm 1 Prospect (restated from Sec. 3)          | Algorithm 1 Prospect (restated from Sec. 3)          |
|----------|----------|----------|
|          | Inputs: Initial w 0 , spectrum σ , number of iterations T , regularization µ > 0 , shift cost ν > 0 . Hyperparameter: Stepsize η > 0 . | Inputs: Initial w 0 , spectrum σ , number of iterations T , regularization µ > 0 , shift cost ν > 0 . Hyperparameter: Stepsize η > 0 . |
| 1:          | Initialize l ← ℓ ( w 0 ) and g i ←∇ ℓ i ( w 0 )+ µw 0 for i = 1 , . . .,n .          | Initialize l ← ℓ ( w 0 ) and g i ←∇ ℓ i ( w 0 )+ µw 0 for i = 1 , . . .,n .          |
| 2:          | Set q ← argmax ¯ q ∈P ( σ ) ¯ q ⊤ l - νD ( q ∥ 1 n /n ) and ρ ← q .          | Set q ← argmax ¯ q ∈P ( σ ) ¯ q ⊤ l - νD ( q ∥ 1 n /n ) and ρ ← q .          |
| 3:          | Set ¯ g ← ∑ n i =1 ρ i g i ∈ R d .          | Set ¯ g ← ∑ n i =1 ρ i g i ∈ R d .          |
| 4:          | Set w ← w 0 .          | Set w ← w 0 .          |
| 5:          | for T iterations do          | for T iterations do          |
| 6: 7:          | Sample i, j ∼ Unif[ n ] independently. v nq i ( ℓ i ( w )+ µw ) nρ i g i +¯ g          | ▷ Iterate Update          |
| 8:          | ← ∇ - t w ← w - ηv .          | ← ∇ - t w ← w - ηv .          |
| 9:          | l j ← ℓ j ( w ) .          | ▷ Bias Reducing Update          |
| 10:          | q ← argmax ¯ q ∈P ( σ ) ¯ q ⊤ l - νD (¯ q ∥ 1 n /n ) .          | q ← argmax ¯ q ∈P ( σ ) ¯ q ⊤ l - νD (¯ q ∥ 1 n /n ) .          |
| 11:          | ¯ g ← ¯ g - ρ i g i + q i ( ∇ ℓ i ( w )+ µw ) .          | ▷ Variance Reducing Update          |
| 12:          | g i ←∇ ℓ i ( w )+ µw .          | g i ←∇ ℓ i ( w )+ µw .          |
| 13:          | ρ i ← q i .          | ρ i ← q i .          |
|          | Output: Final point w .          | Output: Final point w .          |

step can be done by a single O ( n ) -time pass by setting

<!-- formula-not-decoded -->

for i = 1 , . . . , n , as opposed to computing the inverse π -1 and use (14) directly, which in fact requires another sorting operation and can be avoided. Because only one element of l changes on every iteration, we may sort it by simply bubbling the value of the index that changed into its correct position to generate the newly sorted version. The full algorithm is given Algorithm 2. We give a brief explanation on the PA V algorithm for general f -divergences below.

Pool Adjacent Violators (PAV) Algorithm. First, recall the optimization problem we wish to solve:

<!-- formula-not-decoded -->

The objective can be thought of as fitting a real-valued monotonic function to the points (1 , l π (1) ) , . . . , ( n, l π ( n ) ) , which would require specifying its values ( c 1 , . . . , c n ) on (1 , . . . , n ) and defining the function as any x ∈ [ c j , c j +1 ] on ( j, j +1) . Because l π (1) ≤ . . . ≤ l π ( n ) , if we evaluated our function ( c 1 , . . . , c n ) on a loss such as ∑ n i =1 ( l π ( i ) -c i ) 2 , we may easily solve the problem by returning c 1 = ℓ π (1) , . . . , c n = l π ( n ) . However, by specifying functions g 1 , . . . , g n we allow our loss function to change in different regions of the inputs space { 1 , . . . , n } . In such cases, the monotonicity constraint c 1 ≤ . . . ≤ c n is often violated because individually minimizing g i ( c i ) for each c i has no guarantee of yielding a function that is monotonic.

The idea behind the PAV algorithm is to attempt a pass at minimizing each g i individually, and correcting violations as they appear. To provide intuition, define c ∗ i ∈ arg min c i ∈ R g i ( c i ) , and consider i &lt; j such that c ∗ i &gt; c ∗ j . If f ∗ is strictly convex, then g i ( x ) &gt; g i ( c ∗ i ) for any x &lt; c ∗ i and similarly g j ( x ) &gt; g j ( c ∗ j ) for any x &gt; c ∗ j . Thus, to correct the violation, we decrease c ∗ i to ¯ c i and increase c ∗ j to ¯ c j until ¯ c i = ¯ c j . We determine this midpoint precisely by

<!-- formula-not-decoded -->

as these are exactly the contributions made by these terms in the overall objective. The computation above is called pooling the indices i and j . We may generalize this viewpoint to violating chains , that is collections of contiguous indices ( i, i + 1 , . . . , i + m ) such that c ∗ j &lt; c ∗ i for all j &lt; i and c ∗ j &gt; c ∗ i + m for all j &gt; i + m , but c ∗ i &gt; c ∗ i + m . One approach is use dynamic programming to identify

such chains and then compute the pooled quantities

<!-- formula-not-decoded -->

This requires two passes through the vector: one for identifying violators and the other for pooling. The Pool Adjacent Violators algorithm, on the other hand, is able to perform both operations in one pass by greedily pooling violators as they appear. This can be viewed as a meta-algorithm, as it hinges on the notion that the solution of 'larger' pooling problems can be easily computed from solutions of 'smaller' pooling problems. Precisely, for indices S ⊆ [ n ] = { 1 , . . . , n } define

<!-- formula-not-decoded -->

We rely on the existence of an operation Pool , such that for any S, T ⊆ [ n ] such that S ∩ T = ∅ , we have that

<!-- formula-not-decoded -->

where m ( S ) denotes 'metadata' associated to S , and that the number of elementary operations in the Pool function is O (1) with respect to | S | + | T | . We review our running examples.

For the χ 2 -divergence, we have that

<!-- formula-not-decoded -->

Thus, the metadata m ( S ) = | S | used in the pooling step eq. (27) is the size of each subset.

For the KL divergence, f KL ( x ) = x ln x and f ∗ KL ( y ) = e -1 exp( y ) , so so

<!-- formula-not-decoded -->

Here, we carry the metadata m ( S ) = (ln ∑ i ∈ S exp ( l π ( i ) /ν ) , ln ∑ i ∈ S σ i ) , which can easily be combined and plugged into the function

<!-- formula-not-decoded -->

for two instances of metadata ( m 1 , m 2 ) and ( m ′ 1 , m ′ 2 ) . We carry the 'logsumexp' instead of just the sum of exponential quantities for numerical stability, and Equation (28) applies this operation as well. It might be that ∑ i ∈ S σ i = 0 , e.g. for the superquantile. In this case, we may interpret

## Algorithm 2 Prospect (with exact implementation details)

```
Inputs: Initial points w 0 , spectrum σ , stepsize η > 0 , number of iterations T , regularization parameter µ > 0 , shift cost ν > 0 , loss/gradient oracles ℓ 1 , . . . , ℓ n and ∇ ℓ 1 , . . . , ∇ ℓ n . 1: l ← ℓ ( w 0 ) R n . 2: g ← ( ∇ ℓ i ( w 0 ) + µw 0 ) n i =1 ∈ R n × d . 3: π ← argsort( l ) . 4: c ← PAV( l, π, σ ) . ▷ Subroutine 1 or Subroutine 2 5: q ← Convert( c, l, π, ν, 0 n ) . ▷ Subroutine 3 6: ρ ← q . 7: ¯ g ← ∑ n i =1 ρ i g i ∈ R d . 8: for T iterations do 9: Sample i, j ∼ Unif[ n ] . 10: v ← nq i ( ∇ ℓ i ( w ) + µw ) -nρ i g i -¯ g . ▷ Iterate Update 11: w ← w -ηv ( t ) . 12: l j ← ℓ j ( w ) . ▷ Bias Reducing Update 13: π ← Bubble( π, l ) . ▷ Subroutine 4 14: c ← PAV( l, π, σ ) . 15: q ← Convert( c, l, π, ν, q ) . 16: ¯ g ← ¯ g -ρ i g i + q i ( ∇ ℓ i ( w ) + µw ) . ▷ Variance Reducing Update 17: g i ←∇ ℓ i ( w ) + µw . 18: ρ i ← q i . Output: Final point w .
```

Sol( S ) = -∞ and evaluate exp( -∞ ) = 0 in the conversion formula (26). Two examples of the PAV algorithm are given in Subroutine 1 and Subroutine 2, respectively. These operate by selecting the unique values of the optimizer and partitions of indices that achieve that value.

Hardware Acceleration. Finally, note that all of the subroutines in Algorithm 2 (Subroutine 1/Subroutine 2, Subroutine 3, and Subroutine 4) all require primitive operations such as control flow and linear scans through vectors. Because these steps are outside of the purview of oracle calls or matrix multiplications, they benefit from just-in-time compilation on the CPU. We accelerate these subroutines using the Numba package in Python and are able to achieve an approximate 50%-60% decrease in runtime across benchmarks.

```
Subroutine 1 Pool Adjacent Violators (PAV) Algorithm for χ 2 divergence Inputs: Losses ( ℓ i ) i ∈ [ n ] , argsort π , and spectrum ( σ i ) i ∈ [ n ] . 1: Initialize partition endpoints ( b 0 , b 1 ) = (0 , 1) , partition value v 1 = l π (1) -2 nνσ 1 , number of parts k = 1 . 2: for i = 2 , . . . , n do 3: Add part k = k +1 . 4: Compute v k = l π ( i ) -2 nνσ i . 5: while k ≥ 2 and v k -1 ≥ v k do 6: v k -1 = ( b k -b k -1 ) v k -1 +( i -b k ) v k i -b k -1 . 7: Set k = k -1 . 8: b k = i . Output: Vector c containing c i = v k for b k -1 < i ≤ b k .
```

## Subroutine 2 Pool Adjacent Violators (PAV) Algorithm for KL divergence

```
Inputs: Losses ( ℓ i ) i ∈ [ n ] , argsort π , and spectrum ( σ i ) i ∈ [ n ] . 1: Initialize partition endpoints ( b 0 , b 1 ) = (0 , 1) , number of parts k = 1 . 2: Initialize partition value v 1 = ν ( l π (1) /ν -ln σ 1 -ln n -1 ) . 3: Initialize metadata m 1 = ℓ π (1) /ν and t 1 = ln σ 1 . 4: for i = 2 , . . . , n do 5: Add part k = k +1 . 6: Compute v k = ν ( l π ( i ) /ν -ln σ i -ln n -1 ) . 7: Compute m k = ℓ π ( i ) /ν and t k = ln σ i 8: while k ≥ 2 and v k -1 ≥ v k do 9: m k -1 = logsumexp( m k -1 , m k ) and t k -1 = logsumexp( t k -1 , t k ) . v k -1 = ν ( m k -1 -t k -1 -ln n -1) . Set k = k -1 . b k = i .
```

10: 11: 12: Output: Vector c containing c i = v k for b k -1 &lt; i ≤ b k .

## Subroutine 3 Convert

Require: Sorted vector c ∈ R , vector l ∈ R n , argsort π of l , shift cost ν ≥ 0 , vector q ∈ R n .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 3: return q .

## Subroutine 4 Bubble

Require: Index j init, sorting permutation π , loss table l .

- 1: j = j init .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 4: j = j init .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 7: return π

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Algorithm 3 Prospect (with iteration counters specified to accompany convergence analysis)

̸

Inputs: Initial points w (0) , stepsize η &gt; 0 , number of iterations T . 1: Set z (0) i = ζ (0) i = w (0) for all i ∈ [ n ] . 2: q (0) = arg max q ∈P ( σ ) q ⊤ ℓ ( w (0) ) -¯ ν Ω( q ) , ρ (0) = q (0) . 3: Set l (0) = ( ℓ i ( ζ (0) i )) n i =1 ∈ R n , g (0) = ( ∇ r i ( z (0) i )) n i =1 ∈ R d × n , ¯ g (0) = ∑ n i =1 ρ (0) i g (0) i ∈ R d . 4: for t = 0 , . . . , T -1 do 5: i t ∼ Unif([ n ]) , j t ∼ Unif([ n ]) . 6: 7: v ( t ) = nq ( t ) i t ∇ r i t ( w ( t ) ) -( nρ ( t ) i t ∇ r i t ( z ( t ) i t ) -¯ g ( t ) ) . ▷ Iterate Update 8: w ( t +1) = w ( t ) -ηv ( t ) . 9: 10: ζ ( t +1) j t = w ( t ) and ζ ( t +1) j = ζ ( t ) j for j = j t . ▷ Bias Reducing Update 11: l ( t +1) = ℓ ( ζ ( t +1) ) . 12: q ( t +1) = arg max q ∈P ( σ ) q ⊤ l ( t +1) -¯ ν Ω( q ) . 13: 14: z ( t +1) i t = w ( t ) and z ( t +1) i = z ( t ) i for i = i t . ▷ Variance Reducing Update 15: g ( t +1) = ( ∇ r i ( z ( t +1) )) n i =1 . 16: ρ ( t +1) i t = q ( t ) i t and ρ ( t +1) i = ρ ( t ) i for i = i t . 17: ¯ g ( t +1) = ∑ n i =1 ρ ( t +1) i g ( t +1) i . Output: Final point w ( T )

̸

## D CONVERGENCE ANALYSIS OF PROSPECT

This section provides the main convergence analysis for Prospect. For readability, we reference the version of the algorithm presented in Alg. 3, which is written to match quantities appearing in the proof. We begin with a high-level overview, whereas the remaining subsections contain technical lemmas of interest along with key steps in the proof.

## D.1 OVERVIEW

Notation used throughout the proof is collected in Tab. 1, and is also introduced as it appears. In the following, we denote M = L + µ the smoothness constant of the regularized losses r i . We denote E t the expectation w.r.t to the randomness induced by picking i t , j t given w ( t ) , i.e. the conditional expectation given w ( t ) . The optimum of (8) is denoted w ⋆ and satisfies

<!-- formula-not-decoded -->

Denote in addition l ⋆ = ℓ ( w ⋆ ) . For simplicity, we use the shorthand

<!-- formula-not-decoded -->

for any f -divergence D f , where α n is the strong convexity constant of the generator f over the interval [0 , n ] . By Prop. 4, this gives that Ω a 1 -strongly convex function over the probability simplex. All forthcoming statements will reference the setting of Algorithm 3. Note that when implementing the algorithm, storing the iterates { z ( t ) i } n i =1 and { ζ ( t ) i } n i =1 is not necessary.

Proof Outline. We argue convergence by way of defining a Lyapunov function V ( t ) , which will upper bound the quantity ∥ w ( t ) -w ⋆ ∥ 2 2 , which will be called the 'main term'. Specifically, define

<!-- formula-not-decoded -->

̸

where c 1 , c 2 , c 3 , and c 4 are constants to be determined later, and

<!-- formula-not-decoded -->

Though not included in the Lyapunov function, we will also introduce

<!-- formula-not-decoded -->

When the shift cost ν is large, we will be able to simplify the analysis by excluding the terms U ( t ) and R ( t ) . The colors are used for the convenience of the reader so that the quantities above are easy to track from result to result. Each term in the Lyapunov function is motivated by terms that appear when bounding the main term ∥ w ( t ) -w ⋆ ∥ 2 2 , which appears in Step 2 below. The outline of the proof is as follows.

1. We introduce a lemma of general interest which is the key technical step in the analysis: bounding the bias of the gradient estimate v ( t ) given in line 7 of Algorithm 3.
2. We expand the main term and identify 'descent' and 'noise' terms, as in a standard analysis of stochastic gradient methods. The descent term will be treated using the technical lemma from the previous step, whereas the noise term will be upper bounded using standard techniques. In either case, additional non-negative terms are introduced that will be incorporated into the Lyapunov function.
3. We the bound the evolution of the Lyapunov terms that are not the main term. For the large shift cost setting, only S ( t ) and T ( t ) are needed, while U ( t ) and R ( t ) can be ignored.
4. In the final step, we tune any free constants that are encountered in previous steps. At this point, we split the proof into two subsections, one for the large shift cost and one for any shift cost.

## D.2 KEY TECHNICAL LEMMA: BIAS OF THE PROSPECT GRADIENT ESTIMATE

In this subsection, we present the key technical step that allows for the unconditional linear convergence of Prospect. When analyzing stochastic gradient methods in the smooth and strongly convex setting, we typically expand

<!-- formula-not-decoded -->

First, note that the expectation of the primal gradient estimate v ( t ) is ∇ r ( w ( t ) ) ⊤ q ( t ) , where q ( t ) = arg max q ∈P ( σ ) q ⊤ l ( t ) -¯ ν Ω( q ) , where l ( t ) ∈ R n denotes the estimate of the full loss vector. Applying standard convex inequalities to the descent term yields

<!-- formula-not-decoded -->

The first two terms on the right-hand side are negative, which provide decrease in the expected distance-to-optimum value on every iterate. In the empirical risk minimization setting, the final term on the right-hand side would be zero due to the first-order optimality conditions on w ⋆ , as q ( t ) = 1 n /n , implying the decrease of E t ∥ w ( t +1) -w ⋆ ∥ 2 2 for small enough η . However, because

q ( t ) is a potentially non-uniform vector estimated using the table of losses l ( t ) (as opposed to the loss vector ℓ ( w ⋆ ) at optimum), the term -〈 ∇ r ( w ⋆ ) ⊤ q ( t ) , w ( t ) -w ⋆ 〉 is non-zero. Additionally, this term is multiplied only by the learning rate η , instead of the noise terms which are multiplied by η 2 . Thus, this bias term must be bounded carefully in order to achieve the convergence guarantee under this regime. This is the subject of Lem. 9, the main result of this subsection.

Lemma 9 (Bias Bound) . Consider any w ∈ R d , l ∈ R n , and ¯ q ∈ P ( σ ) . Define

<!-- formula-not-decoded -->

For any β 1 ∈ [0 , 1] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. First, for any q i &gt; 0 , we have that w ↦→ q i r i ( w ) is ( q i M ) -smooth and ( q i µ ) -strongly convex, so by applying standard convex inequalities (Lem. 30) we have that

<!-- formula-not-decoded -->

as q i ≤ σ n . The second inequality holds for q i = 0 as well, so by summing the inequality over i and using that ∑ i q i = 1 , we have that

<!-- formula-not-decoded -->

Applying the same argument replacing q by ¯ q and swapping w and w ⋆ yields

<!-- formula-not-decoded -->

Summing the two inequalities yields

<!-- formula-not-decoded -->

Dropping the ∑ n i =1 ∥ ¯ q i ∇ r i ( w ) -¯ q i ∇ r i ( w ⋆ ) ∥ 2 2 term and applying a weight of β 1 ∈ [0 , 1] to ∑ n i =1 ∥ q i ∇ r i ( w ) -q i ∇ r i ( w ⋆ ) ∥ 2 2 still satisfies the inequality, which can equivalently be written as

<!-- formula-not-decoded -->

Next, because

<!-- formula-not-decoded -->

we have that (by summing over i ) that

<!-- formula-not-decoded -->

where we used that each ∥∇ r i ( w ⋆ ) ∥ 2 ≤ 2 G . To see this, use that ∇ r ( w ⋆ ) ⊤ q ⋆ = 0 and ∇ r ( w ⋆ ) = ∇ ℓ ( w ⋆ ) + µw ⋆ , so

<!-- formula-not-decoded -->

Because the map q opt is the gradient of a convex and (1 / ¯ ν ) -smooth map, we also have that

<!-- formula-not-decoded -->

so we apply the above to (32) to achieve

<!-- formula-not-decoded -->

Wealso use (33) to claim non-negativity of ( q -q ⋆ ) ⊤ ( l -ℓ ( w ⋆ )) . Finally, because ∑ i q i = ∑ i q ⋆ i = 1 , we have that

<!-- formula-not-decoded -->

Combine (31), (34), and (35) along with κ σ = nσ n to achieve the claim.

In the next step, we apply Lem. 9, as well as a noise bound to give the initial per-iterate progress bound on the distance-to-optimum quantity E t ∥ w ( t +1) -w ⋆ ∥ 2 2 .

## D.3 BOUNDING THE DISTANCE-TO-OPTIMUM

To outline the remainder of the proof, observe the expansion (30). With the bias bound for the descent term in hand, we now upper bound the noise term.

Lemma 10 (Noise Bound) . Consider the notations of Alg. 3, we have for any β &gt; 0 ,

<!-- formula-not-decoded -->

Proof. In the following, we use the identity E ∥ X -E [ X ] ∥ 2 2 = E ∥ X ∥ 2 2 - ∥ E [ X ] ∥ 2 2 in equations denoted with ( ⋆ ) . We denote by β an arbitrary positive number stemming from using Young's inequality ∥ a + b ∥ 2 2 ≤ (1+ β ) ∥ a ∥ 2 2 +(1+ β -1 ) ∥ b ∥ 2 2 in equation ( ◦ ) . Noting that ∑ n i =1 q ⋆ i ∇ r i ( w ⋆ ) = 0 , we get,

<!-- formula-not-decoded -->

We then combine the analyses of the first and second-order terms to yield the main result of this subsection.

Lemma 11 (Analysis of distance-to-optimum term) . For any constants β 1 ∈ [0 , 1] and β 2 &gt; 0 , and any ¯ q ∈ P ( σ ) , we have that

<!-- formula-not-decoded -->

Proof. Recall the expansion given in (30), which is:

<!-- formula-not-decoded -->

Observe that

<!-- formula-not-decoded -->

By Lem. 9 with l = l ( t ) , q = q ( t ) , w = w ( t ) , and multiplying by 2 η , we have that

<!-- formula-not-decoded -->

The noise term is bounded by applying Lem. 10, so that for some β 2 &gt; 0 ,

<!-- formula-not-decoded -->

Combine the two displays above to get the desired result.

Now, we see the appearance of S ( t ) , T ( t ) , U ( t ) , and R ( t ) in Lem. 11. We incorporate them into the Lyapunov function and describe their evolution in the next section.

## D.4 BOUNDING THE EVOLUTION OF THE LYAPUNOV FUNCTION TERMS

We now describe the evolution of the terms S ( t ) , T ( t ) , U ( t ) , R ( t ) from iterate t to iterate t +1 .

The first two terms simply measure the closeness of the iterates { z ( t ) i t } n i =1 and { ζ ( t ) i t } n i =1 within the table to the optimum w ⋆ , measured either in closeness in weighted gradients ( S ( t ) = 1 n ∑ n i =1 ∥ nρ ( t ) i t ∇ r i t ( z ( t ) i t ) -nq ⋆ i t ∇ r i t ( w ⋆ ) ∥ 2 2 ) or directly ( T ( t ) = ∑ n i =1 ∥ ζ ( t ) i -w ⋆ ∥ 2 2 ). Recall that Q ( t ) = 1 n ∑ n i =1 ∥ nq ( t ) i ∇ r i ( w ( t ) ) -nq ⋆ i ∇ r i ( w ⋆ ) ∥ 2 2 .

Lemma 12. The following hold.

<!-- formula-not-decoded -->

Proof. Write

<!-- formula-not-decoded -->

Similarly,

<!-- formula-not-decoded -->

Next, we handle U ( t ) = 1 n ∑ n j =1 ∥ w ( t ) -ζ ( t ) j ∥ 2 2 , which can be ignored if we assume a particular lower bound on ¯ ν .

Lemma 13. For any value of β 2 &gt; 0 , we have that

<!-- formula-not-decoded -->

Proof. First, note that a separate index j t (independent of i t ) is used to update { ζ ( t ) j } n j =1 , so we may first take the expected value with respect to j t conditioned on i t :

̸

<!-- formula-not-decoded -->

Next, we expand the second term.

<!-- formula-not-decoded -->

The first term is simply the noise term that appears in Lem. 10, whereas the last term is U ( t ) . Next, we have

<!-- formula-not-decoded -->

where the last term is introduced because ∇ ( q ⋆ ⊤ r )( w ⋆ ) = 0 . We bound each of the three terms. First,

<!-- formula-not-decoded -->

because q ( t ) ⊤ r is µ -strongly convex (Nesterov, 2018, Theorem 2.1.9). Second,

<!-- formula-not-decoded -->

by Young's inequality with parameter β 4 and the M -Lipschitz continuity of ∇ ( q ( t ) ⊤ r ) . Third,

<!-- formula-not-decoded -->

by Young's inequality with parameter β 5 and the G -Lipschitz continuity of each ℓ i . Combining with the above, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

when we set β 4 = β 5 = µ -1 . Hence, we get

<!-- formula-not-decoded -->

where the two last steps follow from Lem. 10 and Lem. 34 to claim ∥ q ( t ) -q ⋆ ∥ 2 2 ≤ 1 ¯ ν ( q ( t ) -q ⋆ )( l ( t ) -l ⋆ ) .

Finally, we consider R ( t ) = 2 ηn ( q ( t ) -q ⋆ ) ⊤ ( l ( t ) -l ⋆ ) . This can be viewed as a measurement of orthogonality between the vectors q ( t ) -q ⋆ and l ( t ) -l ⋆ , which in turn can be viewed as the directions to the optimal gradient and optimal solution of a constrained optimization problem. Indeed, we may define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality is taken element-wise. The set L is a convexification of the set ℓ ( R d ) which shares a minimizer and has the same minimum value. Indeed, letting ¯ l be any minimizer of h , select ¯ w such that ¯ l = ℓ ( ¯ w ) . Define ¯ q = ∇ h ( ¯ l ) = arg max q ∈P ( σ ) q ⊤ ¯ l -¯ ν Ω(¯ q ) , and due to the non-negativity of ¯ q , we have that

<!-- formula-not-decoded -->

and

Taking the maximum over ¯ q shows that min l ∈L h ( l ) = h ( ℓ ( ¯ w )) . For convexity, for any l 1 , l 2 ∈ L satisfying l 1 ≥ ℓ ( w 1 ) and l 2 ≥ ℓ ( w 2 ) , and any θ ∈ (0 , 1) , apply the following inequalities elementwise:

<!-- formula-not-decoded -->

with θw 1 + (1 -θ ) w 2 ∈ R d . By convexity, ( q ( t ) -q ⋆ ) ⊤ ( l ( t ) -l ⋆ ) ≥ 0 . Finally, this term is of particular importance because the term -( q -q ⋆ ) ⊤ ( ℓ ( w ) -ℓ ( w ⋆ )) that appears in Lem. 9 can be used for cancellation in this case. The next result describes its evolution.

Lemma 14. For any β 3 &gt; 0 , it holds that

<!-- formula-not-decoded -->

Proof. First, decompose

<!-- formula-not-decoded -->

Because q ( t ) = q opt ( l ( t ) ) for all t , and q opt ( · ) is the gradient of a convex and (1 / ¯ ν ) -smooth function, we have for the second term of (38) that

<!-- formula-not-decoded -->

Next, using Young's inequality, that is, a ⊤ b ≤ β 3 2 ∥ a ∥ 2 2 + β -1 3 2 ∥ b ∥ 2 2 for any β 3 &gt; 0 , we have for the third term term of (38) that

<!-- formula-not-decoded -->

Note that we have

Hence, we get,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, apply the G -Lipschitz continuity of each ℓ i to achieve

<!-- formula-not-decoded -->

Replacing β 3 by 2¯ νβ 3 gives the claim.

## D.5 TUNING CONSTANTS AND FINAL RATE

Recall that our Lyapunov function is given by

<!-- formula-not-decoded -->

Recall in addition the definitions

<!-- formula-not-decoded -->

We will derive a value τ &gt; 0 such that for all t ≥ 0 ,

<!-- formula-not-decoded -->

In order to describe our rates, we define the condition number κ := M/µ and recall that κ σ = nσ n .

## D.5.1 STEP 3A: ANALYZE LARGE SHIFT COST SETTING.

The following gives the convergence rate for large shift cost.

Theorem 15. Suppose the shift cost satisfies

<!-- formula-not-decoded -->

Then, the sequence of iterates produced by Algorithm 3 with η = 1 / (12( µ + M ) κ σ ) achieves

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

Proof. First, invoke Lem. 11 with q ′ = q ( t ) and β 1 = 1 to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We will first bound (40), by using that ∇ r ( w ⋆ ) q ⋆ = 0 and Young's inequality with parameter a &gt; 0 to write

<!-- formula-not-decoded -->

where we used in the second inequality that:

<!-- formula-not-decoded -->

We also have by Cauchy-Schwartz and Lipschitz continuity that

<!-- formula-not-decoded -->

Combining the above displays yields

<!-- formula-not-decoded -->

We take β 2 = 2 , c 3 = c 4 = 0 , and apply Lem. 12 to achieve

<!-- formula-not-decoded -->

where τ &gt; 0 is a to-be-specified rate constant. We now need to set the various free parameters a , c 1 , c 2 , and η to make each of the squared bracketed terms be non-positive. We enforce τ ≥ 2 n throughout. By setting

<!-- formula-not-decoded -->

we have that the bracketed constants before c 1 S ( t ) and Q ( t ) vanish. Then, setting

<!-- formula-not-decoded -->

make the bracketed constant before ∥ w ( t ) -w ⋆ ∥ 2 2 , assuming that we enforce

<!-- formula-not-decoded -->

We turn to the final constant after substituting the values of a , c 2 , and η . We need that

<!-- formula-not-decoded -->

which occurs when

<!-- formula-not-decoded -->

Because γ 2 ∗ ≤ nG 2 ≤ 2 nG 2 , this is achieved when

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To complete the proof, we bound the initial terms. Because c 3 = c 4 = 0 , we need only to bound S (0) and T (0) .

<!-- formula-not-decoded -->

This means ultimately that

Next, we have

Thus, we can write

<!-- formula-not-decoded -->

completing the proof.

## D.5.2 STEP 3B: ANALYZE SMALL SHIFT COST SETTING.

To describe the rate, define δ := nG 2 / ( µ ¯ ν ) . The quantity δ captures the effect of the primal regularizer µ and dual regularizer ¯ ν as compared to the inherent continuity properties of the underlying losses.

completing the proof of the claim

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 16. Assume that n ≥ 2 and that the shift cost ¯ ν ≤ 8 nG 2 /µ . The sequence of iterates produced by Algorithm 3 with

<!-- formula-not-decoded -->

we have for

<!-- formula-not-decoded -->

Proof. First, we apply Lem. 11 with q ′ = q ⋆ , as well as Lem. 14, Lem. 12, and Lem. 13, set c 4 = 1 , and consolidate all constants to write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first set c 1 = nηβ 1 4( M + µ ) κ σ and c 2 = ηµ/ 2 to clean up (42) and (47). We also drop the terms (1 -1 /n ) ≤ 1 . Then, we notice in (43) that to achieve

<!-- formula-not-decoded -->

we need that β 1 ≤ (( M + µ ) κ σ ) / (8 nG 2 / ¯ ν ) . Combined with the requirement that β 1 ∈ [0 , 1] , we set β 1 = (( M + µ ) κ σ ) / (8 nG 2 / ¯ ν +( M + µ ) κ σ ) . We set β 2 = 2 , and can rewrite the expression

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

above.

<!-- formula-not-decoded -->

Next, set the learning rate to be to cancel out Q ( t ) and achieve

<!-- formula-not-decoded -->

Requiring now that τ ≥ 2 n , we may also cancel the S ( t ) term. We substitute δ = nG 2 / ( µ ¯ ν ) to achieve

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It remains to select c 3 and β 3 . As such, we set β 3 = 4 nδ and use that 1 + 4 nδ ≤ 8 nδ when n ≥ 2 and δ ≥ 1 / 8 as assumed, and so

<!-- formula-not-decoded -->

We require now that which cancels T ( t ) and R ( t ) , leaving

<!-- formula-not-decoded -->

From the above, we retrieve the requirement that

<!-- formula-not-decoded -->

It now remains to set η . By substituting in the values for β 1 and c 3 into (48), we have that

<!-- formula-not-decoded -->

The combination of (49) and the above display yields

<!-- formula-not-decoded -->

We need finally that τ ≥ 2 / ( µη ) , resulting in the requirement

<!-- formula-not-decoded -->

This is achieved by setting

<!-- formula-not-decoded -->

completing the proof of the claim

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we bound the initial terms to achieve the final rate. First, we bound η which is used in all of the terms. Because δ ≥ 1 / 8 ,

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

Continuing with β 1 ≤ 1 and (50),

<!-- formula-not-decoded -->

Next, we have T (0) = n ∥ ∥ w (0) -w ⋆ ∥ ∥ 2 2 and by (50),

<!-- formula-not-decoded -->

Because U (0) = 0 , it is bounded trivially. For R (0) , with c 4 = 1 we have

<!-- formula-not-decoded -->

Combining each of these terms together, we have that

<!-- formula-not-decoded -->

completing the proof.

## D.6 PROOF OF MAIN RESULT

The objective is once again

<!-- formula-not-decoded -->

where Ω( q ) = D f ( q ∥ 1 n /n ) /nα n is the penalty scaled to be 1 -strongly convex and we simply notate ¯ ν = nα n ν . The previous subsections give the convergence analysis in the cases of large and small values of ¯ ν . They are combined below.

Theorem 1. Prospect with a small enough step size is guaranteed to converge linearly for all ν &gt; 0 . If, in addition, the shift cost is ν ≥ Ω( G 2 /µα n ) , then the sequence of iterates ( w ( t ) ) t ≥ 1 generated by Prospect and learning rate η = (12 µ (1 + κ ) κ σ ) -1 converges linearly at a rate τ = 2max { n, 24 κ σ ( κ +1) } , i.e.,

<!-- formula-not-decoded -->

Proof. Combine Thm. 15 (the analysis for ¯ ν ≥ 8 nG 2 /µ ) and Thm. 16 (the analysis for ¯ ν ≤ 8 nG 2 /µ ) to achieve convergence for any value of ¯ ν . Substitute ¯ ν = nα n ν so that the condition ¯ ν ≥ 8 nG 2 /µ reads as ν ≥ G 2 / ( µα n ) .

## E IMPROVING PROSPECT WITH MOREAU ENVELOPES

As mentioned in Sec. 3, we may want to generalize Prospect to non-smooth settings which arise either when the shift cost ν = 0 or when the underlying losses ( ℓ i ) are non-smooth. The former case is already addressed by Prop. 8 in Appx. B. The latter case can be handled by considering a variant of Prospect applied to the Moreau envelope of the losses, as defined below. Not only does this extend the algorithm to non-smooth losses, it also allows even in the smooth setting for a less stringent lower bound on ν required for the O (( n + κ σ κ ) ln(1 /ε )) rate. The rest of this section contains necessary background, implementation details, and the adjustments to the analysis.

## E.1 OVERVIEW

We first describe the method and practical details of the implementation, followed by the convergence analysis in the next section.

Notation. The Moreau envelope and the proximal (prox) operator of a convex function f : R d → R are respectively defined for a constant η &gt; 0 as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A fundamental property is that the gradient of the Moreau envelope is related to the proximal operator:

<!-- formula-not-decoded -->

For simplicity, we denote ¯ ν = 2 nν .

Algorithm Description. The algorithm is nearly equivalent to Algorithm 3, but makes the following changes. We sample i t ∼ q ( t ) non-uniformly in the sense that P [ i t = i ] = q ( t ) i . We do not store an additional vector of weights ρ ( t ) , and use q ( t ) in all associated steps. This does not change the expectation of the update direction or the control variate, but creates minor changes in the analysis of the variance term. In the iterate update, we replace the gradient descent-like update with

<!-- formula-not-decoded -->

The vector u ( t ) adds the control variate to w ( t ) before passing it to the proximal operator. The second change is that the elements of the gradient table are updated using j t and the gradients of the Moreau envelope. That is,

<!-- formula-not-decoded -->

̸

Plugging these changes into Algorithm 3 produces the Prospect-Moreau variant.

Implementation Details. The proximal operators can be computed in closed form or algorithmically for common losses. We list here the implementations for some losses of interest. The proximal operators for the binary or multiclass logistic losses cannot be obtained in closed form, we approximate them by one Newton step.

Squared loss. For the squared loss, defined as ℓ ( w ) = 1 2 ( w ⊤ x -y ) 2 for x ∈ R d , y ∈ R , then

<!-- formula-not-decoded -->

Binary logistic loss. For the binary logistic loss defined for x ∈ R d , y ∈ { 0 , 1 } , w ∈ R d as ℓ ( w ) = -y ln( σ ( x ⊤ w )) -(1 -y ) ln(1 -σ ( x ⊤ w )) = -yx ⊤ w +ln(1 + e x ⊤ w ) , we approximate the proximal operator by one Newton step, whose formulation reduces to

<!-- formula-not-decoded -->

Multinomial logistic loss. For the multinomial logistic loss of a linear model defined by W on a sample ( x, y ) as ℓ ( W ) = -y ⊤ Wx +ln(exp( Wx ) ⊤ 1 ) . for x ∈ R d , y ∈ { 0 , 1 } k , y ⊤ 1 = 1 , W ∈ R k × d , we consider approximating the proximal operator by one Newton-step, whose formulation reduces to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Regularized losses. For a convex ℓ : R d → R , define r ( w ) = ℓ ( w ) + ( µ/ 2) ∥ w ∥ 2 . Then, we have,

<!-- formula-not-decoded -->

## E.2 CONVERGENCE ANALYSIS

Prospect-Moreau satisfies the following convergence bound. Recall that γ ⋆ = ∥∇ ℓ ( w ⋆ ) ∥ 2 . Theorem 17. Suppose the smoothing parameter ¯ ν is set large enough as

<!-- formula-not-decoded -->

and define a constant

<!-- formula-not-decoded -->

for κ ∗ σ = σ n /σ 1 . Then, the sequence of iterates ( w ( t ) ) generated by Prospect-Moreau with learning rate η = M -1 min { 1 / (4 κ ∗ σ -1) , κ/ ( n -1) } satisfies

<!-- formula-not-decoded -->

We now prove Thm. 17, using similar techniques as in Appx. D.

Additional Notation. Further, we define w ∗ i = w ∗ + η ∇ r i ( w ∗ ) . By analyzing the first-order conditions of the prox, it is easy to see that

<!-- formula-not-decoded -->

We will use the Lyapunov function

<!-- formula-not-decoded -->

The first step is to analyze the effect of the update on w ( t ) as the first term of the Lyapunov function.

Proposition 18. The iterates of Prospect-Moreau satisfy

<!-- formula-not-decoded -->

Proof. We use the co-coercivity of the prox operator (Thm. 31) to get

<!-- formula-not-decoded -->

where we added and subtracted w ( t ) in the last step.

For the first term, we observe that E t [ u ( t ) ] = w ( t ) and E t [ w ∗ i t ] = w ∗ + η E t [ ∇ r i t ( w ∗ )] so that

<!-- formula-not-decoded -->

For T 2 , note that

<!-- formula-not-decoded -->

We manipulate T 2 to set ourselves up to apply co-coercivity of prox-gradient by adding and subtracting ∇M η [ r i t ]( w ∗ i t ) as follows:

<!-- formula-not-decoded -->

Now, co-coercivity of the prox-gradient (Thm. 32) of the M -smooth function r i t gives

<!-- formula-not-decoded -->

Next, we use u ( t ) = w ( t ) + η ( g ( t ) i t -¯ g ( t ) ) , and w ∗ i = w ∗ + η ∇ r i ( w ∗ ) and ∇M η [ r i ]( w ∗ i ) = ∇ r i ( w ∗ ) to get

<!-- formula-not-decoded -->

where we used that E t [ g ( t ) i t ] = ¯ g ( t ) . Next, we use ∥ x + y ∥ 2 ≤ 2 ∥ x ∥ 2 +2 ∥ y ∥ 2 for any vectors x, y and E ∥ X -E [ X ] ∥ 2 ≤ E ∥ X ∥ 2 for any random vector X to get

<!-- formula-not-decoded -->

Plugging (59), (58), and (59) into (56) gives us

<!-- formula-not-decoded -->

Next, we note that P ( σ ) ⊂ [ σ 1 , σ n ] n to get,

<!-- formula-not-decoded -->

Moreover, we also have that

<!-- formula-not-decoded -->

Plugging these back into (60) completes the proof.

Next, we analyze the other two terms of the Lyapunov function. The proof is trivial, so we omit it.

Proposition 19. We have,

<!-- formula-not-decoded -->

We are now ready to prove Thm. 17.

Proof of Thm. 17. Let τ &gt; 1 be a constant to be determined later and let Γ := γ 2 ∗ G 2 / ( M 2 ¯ ν 2 ) denote the effect of the smoothing. Combining Props. 18 and 19, we can write

<!-- formula-not-decoded -->

Let η = b/M . Our goal is to set the constants b, c 1 , c 2 , τ &gt; 0 so that the right side above is nonpositive and τ is as small as possible. We will require τ ≥ 2 n so that n -1 -τ -1 ≥ (2 n ) -1 . Thus, we can have the right side nonpositive with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us set c 1 = τ -1 . By setting c 2 = 4 κnσ n b 2 / ( b + κ ) , we ensure that (62d) is satisfied. Next, we satisfy (62a) with

<!-- formula-not-decoded -->

Now, (62b) is an inequality only in τ . It is satisfied with τ ≥ τ ∗ := 2 + 2 κ (4 κ ∗ σ -1) . This lets us fix τ = max { 2 n, τ ∗ } throughout, which leads to the value of η as claimed in the theorem statement. Finally, (62c) requires

<!-- formula-not-decoded -->

Thus, under these conditions, the right-hand side of (61) is non-negative. Iterating (61) over t updates, we get

<!-- formula-not-decoded -->

To complete the proof, we note that c 1 ≤ 1 / (2 n ) and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This lets us use the fact that ∇ r i is M -Lipschitz to bound

## F IMPROVING THE DIRECT SADDLE-POINT BASELINE

In Sec. 4 we compared against the existing method of Palaniappan &amp; Bach (2016), which views our objective (8) in its min-max form directly and applies variance reduction techniques to both the primal and dual sequences. In order to make the comparison more convincing, we also improve this method both theoretically and empirically by utilizing different learning rates for the primal and dual sequences. In this section, we provide a new convergence analysis for this improved twohyperparameter variant, which we dub SaddleSAGA , under the χ 2 -divergence penalty.

## F.1 OVERVIEW

As in Appx. E, we begin with the additional notation and description of the algorithm, followed by the convergence analysis.

Notation. For simplicity, we denote ¯ ν = 2 nν , and consider directly the min-max problem

<!-- formula-not-decoded -->

Note that the function Ψ is strongly convex in its first argument and strongly concave in its second argument. A pair ( w ⋆ , q ⋆ ) is called a saddle point of the convex-concave function Ψ if

<!-- formula-not-decoded -->

In our setting, we can verify that the pair w ⋆ = arg min F σ and q ⋆ = q opt ( ℓ ( w ⋆ )) is the unique saddle point of Ψ .

Algorithm Description. The algorithm makes use of proximal operators (as described in Appx. E in addition), which is defined for a convex function f : R d → R , and x ∈ R d as

<!-- formula-not-decoded -->

The method is nearly equivalent to Algorithm 3, but applies the update

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

to the dual iterates, where δ &gt; 0 is the dual learning rate. The vector π ( t ) plays the role of an update direction, and the proximal update on q ( t +1) can be solved with the PAV algorithm, as seen in Appx. C. Overall, the time and space complexity is identical to that of Prospect.

Rate of Convergence. We prove the following rate of convergence for SaddleSAGA.

Theorem 20. The iterates ( w ( t ) , q ( t ) ) of SaddleSAGA with learning rates

<!-- formula-not-decoded -->

converge linearly to the saddle point of (63) . In particular, for non-trivial regularization µ ¯ ν ≤ 8 n 2 G 2 and µ ≤ 6( Lκ σ +2 G 2 n/ ¯ ν ) , the number of iterations t to get ∥ w ( t ) -w ⋆ ∥ 2 2 + c ∥ q ( t ) -q ⋆ ∥ 2 2 ≤ ε (for some constant c ) is at most

<!-- formula-not-decoded -->

The proof of this statement is given as Cor. 26 later in this section. To compare to the original variant, the rate obtained by Palaniappan &amp; Bach (2016) in terms of our problem's constants is

<!-- formula-not-decoded -->

Compared to this, the rate we prove for SaddleSAGA improves κ 2 to κκ σ while suffering an additional factor of n in the n 2 G 2 / ( µ ¯ ν ) term. Empirical comparisons between SaddleSAGA and the original algorithm are given in Appx. I. As compared to Prospect, SaddleSAGA matches the rate of Thm. 1 only when the shift cost ¯ ν is large enough.

## F.2 CONVERGENCE ANALYSIS

In the following, we denote by E t [ · ] the expecation of a quantity according to the randomness of i t conditioned on w ( t ) , q ( t ) . Throughout the proof, we consider that the losses are L -smooth and G -Lipschitz continuous.

Evolution of Distances-to-Optimum. Westart by using the contraction properties of the proximal operator to bound the evolution of the distances to the saddle point ( w ⋆ , q ⋆ ) .

to the primal iterates and

Lemma 21. We have,

<!-- formula-not-decoded -->

Proof. By considering the first-order optimality conditions of the problem one verifies that w ⋆ , q ⋆ satisfy for any η, δ ,

<!-- formula-not-decoded -->

Recall that the proximal operator of a c -strongly convex function h is contractive such that ∥ prox h ( z ) -prox h ( z ′ ) ∥ 2 ≤ 1 1+ c ∥ z -z ′ ∥ 2 . In our case, it means that

<!-- formula-not-decoded -->

By taking the squared norm, the expectation, expanding the squared norms and using that E t [ v ( t ) ] = ∇ ( q ( t ) ⊤ ℓ )( w ( t ) ) , E t [ π ( t ) ] = ℓ ( w ( t ) ) , we get the result.

Evolution of Variance Term. We consider the evolution of the additional variance term added to the dual variables.

Lemma 22. We have for any β 2 &gt; 0 ,

<!-- formula-not-decoded -->

Proof. As in the proof of Lem. 10, we have for some β 2 &gt; 0 ,

<!-- formula-not-decoded -->

Incorporating Smoothness and Convexity of the Losses. The improved algorithm we developed here, for the purpose of a fair comparison to our own algorithm, differs from the original one from Palaniappan &amp; Bach (2016) by Cor. 24 stemming from Lem. 23. We exploit the smoothness and convexity of the losses to get a negative term -E t [ ∥ nq i 1 ∇ ℓ i t ( w ( t ) ) -nq ∗ i t ∇ ℓ i t ( w ⋆ ) ∥ 2 2 ] used to temper the variance of the primal updates at the price of an additional positive term ∥ q ( t ) -q ⋆ ∥ 2 2 . The sum of both being positive we can dampen the effect of the additional positive term ∥ q ( t ) -q ⋆ ∥ 2 2 at the price of getting a less negative term -E t [ ∥ nq i 1 ∇ ℓ i t ( w ( t ) ) -nq ∗ i t ∇ ℓ i t ( w ⋆ ) ∥ 2 2 ] .

<!-- formula-not-decoded -->

Proof. For any q ∈ P ( σ ) and any w,v ∈ R d , we have by smoothness and convexity of q i ℓ i , for q i &gt; 0

<!-- formula-not-decoded -->

Note that the second inequality is then true even if q i = 0 , since in that case all terms are 0. Therefore, for any q 1 , q 2 ∈ P ( σ ) , and any w 1 , w 2 , we have

<!-- formula-not-decoded -->

Combining these inequalities, we get

<!-- formula-not-decoded -->

For any 4 vectors a, b, c, d ,

<!-- formula-not-decoded -->

Reorganizing the terms and bounding ℓ ( w ) ⊤ ℓ ( w ) by G 2 we get the result.

∇ i 1 ∇ i 2

Corollary 24. We have for any α ∈ [0 , 1]

<!-- formula-not-decoded -->

Proof. Follows from Lem. 23

Lyapunov Function and Overall Convergence. Thm. 25 shows that an appropriately defined Lyapunov function incorporating the distances to the optima, decrease exponentially.

Theorem 25. Define the Lyapunov function

<!-- formula-not-decoded -->

with c 1 = n 2( Lκ σ +2 G 2 n/ ¯ ν ) and c 2 = µ 2 with κ σ = nσ max . By taking

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

for some τ &gt; 1 . In particular, for small regularizations, i.e., µ ¯ ν ≤ 8 n 2 G 2 and µ ≤ 6( Lκ σ + 2 G 2 n/ ¯ ν ) , we have

<!-- formula-not-decoded -->

Proof. Let us denote

<!-- formula-not-decoded -->

we have,

<!-- formula-not-decoded -->

By combining Cor. 24, Lem. 10, Lem. 22 we have, denoting κ σ = nσ max ,

<!-- formula-not-decoded -->

Therefore for some τ &gt; 1 , we have

<!-- formula-not-decoded -->

with,

<!-- formula-not-decoded -->

Fix β 1 = 2 , β 2 = 1 . Denote also ¯ η = ηµ 1+ ηµ ∈ (0 , 1) and ¯ δ = δ ¯ ν 1+ δ ¯ ν ∈ (0 , 1) with e.g. η = ¯ η µ (1+¯ η ) . We have then for c 1 /n = α/ (2 Lκ σ ) and c 2 = µ/ 2 ,

<!-- formula-not-decoded -->

We can further take 3 η ≤ α/ (2 Lκ σ ) and δ ≤ µ/ (8 n 2 G 2 ) . By imposing the constraint τ ≥ 2 n , we can simplify

<!-- formula-not-decoded -->

Recall that α must be chosen in [0 , 1] . Taking then

<!-- formula-not-decoded -->

we get

<!-- formula-not-decoded -->

By taking η ≤ 1 /µ , δ ≤ 1 / ¯ ν , we get ¯ η ≤ 1 / 2 , ¯ δ ≤ 1 / 2 and so ¯ η 2 -3 4 ¯ η ≤ -1 4 ¯ η and ¯ δ 2 -¯ δ ≤ -1 2 ¯ δ . Therefore taking

<!-- formula-not-decoded -->

we get K i ≤ 0 for all i as long as τ ≥ max { 2 n, 4 / ¯ η, 2 / ¯ δ } . In our case,

<!-- formula-not-decoded -->

The result follows.

Corollary 26. Under the setting of Thm. 25, after t iterations of SaddleSAGA, we have

<!-- formula-not-decoded -->

## G TECHNICAL RESULTS FROM CONVEX ANALYSIS

Herein, we collect several results, mostly from Nesterov (2018), used throughout the paper. In the following, let ∥·∥ denote an arbitrary norm on R d and let ∥·∥ ∗ denote its associated dual norm.

The first concerns L -smooth function, or those with L -Lipschitz continuous gradient.

Theorem 27. (Nesterov, 2018, Theorem 2.1.5) The conditions below are considered for any x, y ∈ R d and α ∈ [0 , 1] . The following are equivalent for a differentiable function f : R d → R .

1. f is convex and L -smooth with respect to ∥·∥ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we detail the properties of strongly convex functions.

Theorem 28. (Nesterov, 2018, Theorem 2.1.10) If f : R d → R is µ -strongly convex and differentiable, then for any x, y ∈ R d ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, functions that are both smooth and strongly convex enjoy a number of relevant primal-dual properties.

Theorem 29. (Nesterov, 2018, Theorem 2.1.12) If f is both L -smooth and µ -strongly convex, then for any x, y ∈ R d ,

<!-- formula-not-decoded -->

Lemma 30. Let f : R d → R be µ -strongly convex and M -smooth. Then, we have for any w,v ∈ R d ,

<!-- formula-not-decoded -->

Proof. The function g = f -µ ∥ · ∥ 2 2 / 2 is convex and M -µ smooth. Hence, we have by line 3 of Thm. 27 for any w,v ∈ R d ,

<!-- formula-not-decoded -->

Expanding g and ∇ g , we get

<!-- formula-not-decoded -->

Using Young's inequality, that is, a ⊤ b ≤ α 2 ∥ a ∥ 2 2 + α -1 2 ∥ b ∥ 2 2 , we have

<!-- formula-not-decoded -->

Taking α = 2 µ + M

<!-- formula-not-decoded -->

We state a few properties of the prox operator.

Theorem 31 (Co-coercivity of the prox) . If f : R d → R is µ -strongly convex, then we have for any constant η &gt; 0 that

<!-- formula-not-decoded -->

The same result applied to the convex conjugate f ⋆ of f and noting that ∇M η [ f ]( x ) = prox f ⋆ /η ( x/η ) gives the following result:

Table 2: Dataset attributes and dimensionality d , train sample size n train, and test sample size n test .

| Dataset   |    d | n train   | n test   | Task          | Source    |
|----------|------|----------|----------|----------|----------|
| yacht     |    6 | 244       | 62       | Regression          | UCI       |
| energy    |    8 | 614       | 154      | Regression          | UCI       |
| concrete  |    8 | 824       | 206      | Regression          | UCI       |
| kin8nm    |    8 | 6,553     | 1,639    | Regression          | OpenML    |
| power     |    4 | 7,654     | 1,914    | Regression          | UCI       |
| diabetes  |   33 | 4,000     | 1,000    | Binary Classification     | Fairlearn |
| acsincome |  202 | 4,000     | 1,000    | Regression          | Fairlearn |
| amazon    |  535 | 10,000    | 10,000   | Multiclass Classification | WILDS     |
| iwildcam  | 9420 | 20,000    | 5,000    | Multiclass Classification | WILDS     |

Theorem 32 (Co-coercivity of the prox) . If f : R d → R is L -smooth, then we have for any constant η &gt; 0 that

<!-- formula-not-decoded -->

Lemma 33. For a convex function f : R → R ∪ { + ∞} , if x 1 ≥ x 2 and y 2 ≥ y 1 , then

<!-- formula-not-decoded -->

Proof. First, observe that

<!-- formula-not-decoded -->

Thus, y 2 -x 1 and y 1 -x 2 both lie between y 2 -x 2 and y 1 -x 1 and can be expressed as a convex combination of the two endpoints, that is

<!-- formula-not-decoded -->

for some α, β ∈ [0 , 1] . By solving for α we get α = 1 -β . Apply the definition of convexity to get

<!-- formula-not-decoded -->

Sum both inequalities to achieve the desired result.

Lemma 34. Define for l ∈ R n ,

<!-- formula-not-decoded -->

The function h is 1 / ¯ ν -smooth and convex such that for any l, l ′ ∈ R n ,

<!-- formula-not-decoded -->

## H EXPERIMENTAL DETAILS

## H.1 TASKS &amp; OBJECTIVES

In all settings, we consider supervised learning tasks specified by losses of the form

<!-- formula-not-decoded -->

where we consider an input x i ∈ X , a feature map φ : X → R d , and a label y i ∈ Y . The function h : Y × R → R measures the error between the true label and another value which is the prediction in regression and the logit probabilities of the associated classes in classification. In the regression tasks, Y = R and we used the squared loss

<!-- formula-not-decoded -->

For binary classification, we have Y = {-1 , 1 } , denoting a negative and positive class. We used the binary logistic loss

<!-- formula-not-decoded -->

For multiclass classification, Y = { 1 , . . . , C } where C is the number of classes. We used the multinomial logistic loss:

<!-- formula-not-decoded -->

The design matrix ( φ ( x 1 ) , . . . , φ ( x n )) ∈ R n × d is standardized to have columns with zero mean and unit variance, and the estimated mean and variance from the training set is used to standardize the test sets as well. Our final objectives are of the form

<!-- formula-not-decoded -->

for shift cost ν ≥ 0 and regularization constant µ ≥ 0 .

## H.2 DATASETS

We detail the datasets used in the experiments. If not specified below, the input space X = R d and φ is the identity map. The sample sizes, dimensions, and source of the datasets are summarized in Tab. 2, where d refers to the dimension of each φ ( x i ) .

- (a) yacht : prediction of the residuary resistance of a sailing yacht based on its physical attributes Tsanas &amp; Xifara (2012).
- (b) energy : prediction of the cooling load of a building based on its physical attributes Baressi Segota et al. (2020).
- (c) concrete : prediction of the compressive strength of a concrete type based on its physical and chemical attributes Yeh (2006).
- (d) kin8nm : prediction of the distance of an 8 link all-revolute robot arm to a spatial endpoint (Akujuobi &amp; Zhang, 2017).
- (e) power : prediction of net hourly electrical energy output of a power plant given environmental factors (T¨ ufekci, 2014).
- (f) diabetes : prediction of readmission for diabetes patients based on 10 years worth of clinical care data at 130 US hospitals (Rizvi et al., 2014).
- (g) acsincome : prediction of income of US adults given features compiled from the American Community Survey (ACS) Public Use Microdata Sample (PUMS) (Ding et al., 2021).
- (h) amazon : prediction of the review score of a sentence taken from Amazon products. Each input x ∈ X is a sentence in natural language and the feature map φ ( x ) ∈ R d is generated by the following steps:
- A BERT neural network Devlin et al. (2019) (fine-tuned on 10 , 000 held-out examples) is applied to the text x i , resulting in vector x ′ i .
- The x ′ 1 , . . . , x ′ n are normalized to have unit norm.
- Principle Components Analysis (PCA) is applied, resulting in 105 components that explain 99% of the variance, resulting in vectors x ′′ i ∈ R 105 . The d in Tab. 2 refers to the total dimension of the parameter vectors for all 5 classes.
- (i) iwildcam : prediction of an animal or flora in an image from wilderness camera traps, with heterogeneity in illumination, camera angle, background, vegetation, color, and relative animal frequencies Beery et al. (2020). Each input x ∈ X is an image the feature map φ ( x ) ∈ R d is generated by the following steps:

- A ResNet50 neural network He et al. (2016) that is pretrained on ImageNet Deng et al. (2009) is applied to the image x i , resulting in vector x ′ i .
- The x ′ 1 , . . . , x ′ n are normalized to have unit norm.
- Principle Components Analysis (PCA) is applied, resulting in d = 157 components that explain 99% of the variance. The d in Tab. 2 refers to the total dimension of the parameter vectors for all 60 classes.

## H.3 HYPERPARAMETER SELECTION

We fix a minibatch size of 64 SGD and SRDA and an epoch length of N = n for LSVRG. For SaddleSAGA we consider three schemes for selecting the primal and dual learning rates that reduce to searching for a single parameter η &gt; 0 , as described in Appx. I. In practice, the regularization parameter µ and shift cost ν are tuned by a statistical metric, i.e. generalization error as measured on a validation set. We study the optimization performance of the methods for multiple values of each in Appx. I.

For the tuned hyperparameters, we use the following method. Let k ∈ { 1 , . . . , K } be a seed that determines algorithmic randomness. This corresponds to sampling a minibatch without replacement for SGD and SRDA and a single sampled index for SaddleSAGA, LSVRG, and Prospect. Letting L k ( η ) denote the average value of the training loss of the last ten passes using learning rate η and seed k , the quantity L ( η ) = 1 K ∑ K k =1 L k ( η ) was minimized to select η . The learning rate η is chosen in the set { 1 × 10 -4 , 3 × 10 -4 , 1 × 10 -3 , 3 × 10 -3 , 1 × 10 -2 , 3 × 10 -2 , 1 × 10 -1 , 3 × 10 -1 , 1 × 10 0 , 3 × 10 0 } , with two orders of magnitude lower numbers used in acsincome due to its sparsity. We discard any learning rates that cause the optimizer to diverge for any seed.

## H.4 COMPUTE ENVIRONMENT

No GPUs were used in the study; Experiments were run on a CPU workstation with an Intel i9 processor, a clock speed of 2.80GHz, 32 virtual cores, and 126G of memory. The code used in this project was written in Python 3 using the PyTorch and Numba packages for automatic differentiation and just-in-time compilation, respectively.

## I ADDITIONAL EXPERIMENTS

Varying Risk Parameters. We study the effect of varying the risk parameters, that is ( p, b, γ ) for the p -CVaR (Equation (21)), b -extremile (Equation (22)), γ -ESRM (Equation (23)), choosing the spectrum to increase the condition number κ σ = nσ n compared to the experiments in the main text. We use p = 0 . 25 , b = 2 . 5 , and γ = 1 /e -2 to generate 'hard' version of the superquantile, extremile, and ESRM. Fig. 8 plots the corresponding training curves for four datasets of varying sample sizes: yacht , energy , concrete , and iwildcam . We see that the comparison of methods is the same as the original methods, that is that Prospect performs the best or close to best in terms of optimization trajectories. Except on concrete , SaddleSAGA generally matches the performance of Prospect. The trajectory of LSVRG is noticeably noisier than on the original settings; we hypothesize that the bias accrued by this epoch-based algorithm is exacerbated by the skewness in the spectrum, as mentioned in Mehta et al. (2023, Proposition 1).

Lowering or Removing Shift Cost. A relevant setting is the low or no shift cost regime, as this allows the adversary to make arbitrary distribution shifts (while still constrained to P ( σ ) ). These settings correspond to ν = 10 -3 and ν = 0 , respectively. The low-cost experiment is displayed in Fig. 9 while Fig. 10 displays these curves for the no-cost experiment. When ν = 0 , the optimization problem can equivalently be written as

<!-- formula-not-decoded -->

In this case, we always have that q opt ( l ) = ( σ π -1 (1) , . . . , σ π -1 ( n ) ) , where π sorts l . Here, w is chosen to optimize a linear combination of order statistics of the losses. In the low shift cost settings, performance trends are qualitatively similar to those seen from ν = 1 . Interestingly, for the no-cost

Figure 8: Harder risk parameter settings. Each row represents a different 'hard' variant of the superquantile, extremile, and ESRM spectra. Columns represent different datasets. Suboptimality (9) is measured on the y -axis while the x -axis measures the total number of gradient evaluations made divided by n , i.e. the number of passes through the training set.

<!-- image -->

Figure 9: Low shift cost settings. Each row represents a different spectral risk objective with ν = 10 -3 (instead of ν = 1 ) while each column represents a different datasets. Suboptimality (9) is measured on the y -axis while the x -axis measures the total number of gradient evaluations made divided by n , i.e. the number of passes through the training set.

<!-- image -->

setting, LSVRG, SaddleSAGA, and Prospect seem to converge linearly empirically even without smoothness of the objective.

Lowering Regularization. Next, we decrease the ℓ 2 -regularization from µ = 1 /n to µ = 1 / (10 n ) and µ = 1 / (100 n ) . These settings are plotted in Fig. 11 and Fig. 12, respectively. Performance rankings among methods reflect those of the original parameters. For five of the six datasets, that is yacht , energy , concrete , kin8nm , and power , the regression tasks involve optimizing the squared error. This function is already strongly convex, with a constant depending on the smallest eigenvalue of the empirical second-moment matrix. When assuming that the input data vectors are

Figure 10: No shift cost settings. Each row represents a different spectral risk objective with ν = 0 (instead of ν = 1 ) while each column represents a different datasets. Suboptimality (9) is measured on the y -axis while the x -axis measures the total number of gradient evaluations made divided by n , i.e. the number of passes through the training set.

<!-- image -->

Figure 11: Reduced ℓ 2 -regularization settings ( µ = 1 / (10 n ) . Each row represents a different spectral risk objective with µ = 1 / (10 n ) (instead of µ = 1 /n ) while each column represents a different dataset. Suboptimality (9) is measured on the y -axis while the x -axis measures the total number of gradient evaluations made divided by n , i.e. the number of passes through the training set.

<!-- image -->

bounded, this function is also G -Lipschitz. Thus, if the problem is already well-conditioned, we may observe similar behavior even at negligible regularization ( µ = 5 · 10 -7 for iwildcam , for example).

Comparison of Saddle-Point and Moreau Variants. Finally, observe in Fig. 13 the comparison of SaddleSAGA variants (Appx. F), as well as the Moreau version of Prospect using Moreau envelopebased oracles (Appx. E). There are variants shown.

- Primal LR = Dual LR: The original variant of Palaniappan &amp; Bach (2016), in which the primal and dual learning rates are set to be equal and searched as a single hyperparameter.
- Search Dual LR: Here, the primal learning rate is fixed as the optimal one for Prospect, and the dual learning rate is searched as a single hyperparameter.

Figure 12: Low ℓ 2 -regularization settings ( µ = 1 / (100 n ) . Each row represents a different spectral risk objective with µ = 1 / (100 n ) (instead of µ = 1 /n ) while each column represents a different dataset. Suboptimality (9) is measured on the y -axis while the x -axis measures the total number of gradient evaluations made divided by n , i.e. the number of passes through the training set.

<!-- image -->

- Primal-Dual Heuristic: In this version, used as the 'SaddleSAGA' baseline in the main text, the dual learning rate is set to be 10 n times smaller than the primal learning rate.
- Prospect-Moreau: The Moreau-envelope version of Prospect using proximal oracles.

We find that all methods besides the original variant (primal LR = dual LR) perform comparably on yacht , energy , concrete , kin8nm , and power . Notably, the ProxSAGA method performs similarly to Prospect and the saddle point-based baselines. While using the Moreau envelope results in accelerated rates in the ERM setting Defazio (2016), we find that the convergence rate is the same empirically. This phenomenon is in agreement with Thm. 17, which states that ProxSAGA will achieve the same linear convergence rate as Prospect, but will require a much less stringent condition on the shift cost ν than in the case of Prospect.

Measuring Wall Time. In Fig. 14, we measure the suboptimality for each method as a function of the wall time taken for a training run on the compute environment described in Appx. H. Across tasks, SGD is approximately 2 orders of magnitude faster than Prospect (and SaddleSAGA) but fails to converge due to bias/variance. In low-to-moderate sample size settings, LSVRG achieves a similar suboptimality to Prospect in a 2-3x smaller wall time. However, in the large sample size setting, LSVRG achieves essentially the same suboptimality as SGD compared to for Prospect, indicating that while fast, LSVRG does not demonstrate adequate accuracy. Also, at the large sample size setting, Prospect performs about 3x faster than SaddleSAGA but achieves the same suboptimality. Finally, a wall time cost that is not represented by the attached Figure 2 is the hyperparameter tuning cost. Both LSVRG and SaddleSAGA involve a grid search for tuning two hyperparameters, whereas Prospect has a single hyperparameter that can be tuned with binary search.

Comparison to General-Purpose Learning Algorithms. In Fig. 15, we compare the Prospect and baselines against optimization algorithms with widespread usage in generic learning problems, namely stochastic gradient descent (SGD) with momentum (Polyak, 1964; Nesterov, 2018) and Adam (Kingma &amp; Ba, 2015). We use the default hyperparameters such as momentum 0 . 9 for SGD and ( β 1 , β 2 ) = (0 . 9 , 0 . 999) for Adam. For both methods, we use a batch size of 64 and tune the learning rate. Because of the bias and variance issues identified in Sec. 2, Adam and SGD with momentum perform comparably to standard SGD and are unable to converge on the objectives considered. Across objectives/datasets, final suboptimality often caps at 10 -2 for both methods.

Figure 13: SaddleSAGA and Prospect-Moreau method comparisons. Each row represents a different spectral risk objective while each column represents a different dataset. Suboptimality (9) is measured on the y -axis while the x -axis measures the total number of gradient evaluations made divided by n , i.e. the number of passes through the training set.

<!-- image -->

Figure 14: Wall time comparisons. Each row represents a different spectral risk objective while each column represents a different dataset. Suboptimality (9) is measured on the y -axis while the x -axis measures the total wall time in seconds on the environment described in Appx. H.

<!-- image -->

Figure 15: Comparison to SGD w/ Momentum and Adam. Each row represents a different spectral risk objective while each column represents a different dataset. Suboptimality (9) is measured on the y -axis while the x -axis measures the total number of gradient evaluations made divided by n , i.e. the number of passes through the training set.

<!-- image -->