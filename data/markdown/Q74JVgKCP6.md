## Near-Optimality of Contrastive Divergence Algorithms

Pierre Glaser

Kevin Han Huang

Arthur Gretton

Gatsby Computational Neuroscience Unit, University College London pierreglaser@gmail.com, han.huang.20@ucl.ac.uk, arthur.gretton@gmail.com

## Abstract

Weperform a non-asymptotic analysis of the contrastive divergence (CD) algorithm, a training method for unnormalized models. While prior work has established that (for exponential family distributions) the CD iterates asymptotically converge at an O ( n -1 / 3 ) rate to the true parameter of the data distribution, we show, under some regularity assumptions, that CD can achieve the parametric rate O ( n -1 / 2 ) . Our analysis provides results for various data batching schemes, including the fully online and minibatch ones. We additionally show that CD can be near-optimal, in the sense that its asymptotic variance is close to the Cramér-Rao lower bound.

## 1 Introduction

Describing data using probability distributions is a central task in multiple scientific and industrial disciplines [1, 2, 3]. Since the true distribution of the data is generally unknown, such a task requires finding an estimator of the true distribution among a model class that best describes the available data. An estimator can be characterized at multiple levels of granularity: at the highest level lies consistency [4], a property which states that as the number of available data points increases, a given estimator will converge to the one best describing the data distribution. At a lower level, a consistent estimator can be further characterized by its convergence rate, a quantity upper-bounding its distance to the true distribution as a function of the number of samples. A convergence rate can be either asymptotic , e.g. hold only in the limit of an infinite sample size, or non-asymptotic , in which case the rate also holds for finite sample sizes. In their simplest form, convergence rates are provided in bigO notation, discarding finer grained information such as asymptotically dominated quantities as well as multiplicative constants. These constants play a role in the so-called asymptotic variance of the estimator, which is a precise descriptor of an estimator's statistical efficiency. Convergence rates and asymptotic variances have been the subject of extensive research in the statistical literature; in particular, well-known lower bounds exists regarding both the best possible (asymptotic) convergence rate of an estimator and its best possible asymptotic variance. These results set a clear frame of reference to interpret individual convergence rates, and are routinely present in the analysis of modern statistical algorithms such as noise-contrastive estimation [5, 6] or score matching [7, 8, 9].

In this work, we focus on cases where (1) the true data distribution admits a density with respect to some known base measure, and (2) the model class is parametrized by a finite-dimensional parameter. In this setting, provided that the true distribution belongs to the model class, a celebrated result in statistical estimation states that the model maximizing the average log-likelihood both achieves the best possible asymptotic convergence rate (called the parametric rate ) and the best possible asymptotic variance, called the Cramér-Rao bound (see, e.g. [10]). While this result shows that Maximum Likelihood Estimators (MLE) are asymptotically optimal, fitting them is complicated by computational hurdles when using models with intractable normalizing constants. Such unnormalized models are common in the Machine Learning literature due to their high flexibility [11, 12]; their weakness however lies in the fact that expectations under these models have no unbiased approximation. For this reason, popular approximation algorithms such as unbiased gradient-based

stochastic optimization of the empirical log-likelihood cannot a priori be used, as the gradient of the normalizing constant is given by an expectation under the model distribution.

The Contrastive Divergence (CD) algorithm [13] is a popular approach that circumvents this issue by using a Markov Chain Monte Carlo (MCMC) algorithm to approximate the gradient of the loglikelihood. Unnormalized models trained with Contrastive Divergence have been shown to reach competitive performance in high-dimensional tasks such as image [14, 15, 16], text [17], and protein modeling [18, 19], or neuroscience [20]. A consistency analysis of the Contrastive Divergence algorithm is delicate, however: indeed, the optimization error e.g. the difference between the estimate returned by CD and the MLE, is likely to be non-negligible as compared with statistical error - the distance between the MLE and the true distribution - and thus cannot be discarded, as often done when analyzing estimators that minimize tractable objectives [8, 5]. Recent work [21] elegantly established asymptotic O ( n -1 / 3 ) -consistency of the CD estimator for unnormalized exponential families when using only a finite number of MCMC steps. Key to their argument is the fact that the bias of the CD gradient estimate decreases as iterates approach the data distribution. However, as noted by the authors, their work left open the question of whether and under what conditions CD might achieve O ( n -1 / 2 ) -consistency.

Contributions In this work, we answer this question by providing a non-asymptotic analysis of the CD algorithm for unnormalized exponential families. While existing convergence bounds [21] were derived for the 'full batch' setting, where the CD gradient is estimated using the full dataset at each iteration, our analysis covers both the online setting (where data points are processed one at a time without replacement), and the offline setting with multiple data reuse strategies (including full batch).

In the online case (Section 3), we show, under a restricted set of assumptions compared to Jiang et al. [21], that the CD iterates can converge to the true distribution at the parametric O ( n -1 / 2 ) rate. Our analysis reveals that CD contains two sources of approximation: a bias term, and a variance term. These sources are almost independent of each other, in the sense that decreasing the bias by increasing the number of MCMC steps will not decrease the variance. The impact of these two sources of approximation transparently propagates in our resulting bounds: in particular, as the bias of the CD algorithm goes to 0, our bounds recover well-known results in online stochastic optimization [22]. Finally, we study the asymptotic variance of an estimator obtained by averaging the CD iterates, a classic acceleration technique in stochastic optimization [23]. We show that provided that the number of steps m is sufficiently large, the asymptotic variance of this estimator matches (up to a factor 4) the Cramér-Rao bound.

Next, we study the offline setting (Section 4), where the CD gradient is estimated by reusing (potentially random) subsets of a finite dataset. We show that a similar result to the online setup holds, up to an additional correlation term that arises from data reuse, and present several approaches to control this term. We improve over the results of [21] by showing a non-asymptotic and nearparametric rate at O ((log n ) 1 / 2 n -1 / 2 ) under their conditions, and also illustrate how different rates can be obtained under a variety of conditions. Our results also show an interesting tradeoff between the effect of initialization and the statistical error as a function of batch size.

In summary, we establish the near-optimality of a variety of Contrastive Divergence algorithms for unnormalized exponential families in the so called 'long-run' regime, where the number of MCMC steps is high enough to ensure that the CD gradient bias is sufficiently offset by the convexity of the negative log-likelihood.

## 2 Contrastive Divergence in Unnormalized Exponential Families

Unnormalized Exponential Families Exponential families (EF) [24, 25] form a well-studied class of probability distributions, given by

<!-- formula-not-decoded -->

Here, X ∋ x is the data or sample space , which we set to be a subset of R d for some d ∈ N ∗ , although our results are readily extendable to more general measurable spaces. c is a measure on X called the base or carrier measure. When X ⊆ R d , c is often set to be the corresponding Lebesgue measure. ψ ∈ Ψ ⊆ R p is a finite-dimensional parameter called the natural parameter , and ϕ : R d ↦-→ R p is a function called the sufficient statistics , which, alongside with the base measure, fully describes an exponential family. Finally, log Z ( ψ ) , the log-normalizing (or cumulant ) function, is a quantity

ensuring that p ψ integrates to 1 over X . Crucially, we will not assume that log Z ( ψ ) admits a closed form expression for all ψ . The latter fact provides the practitioner with a great deal of flexibility in designing the model class: indeed, the only requirement that should be satisfied prior to performing statistical estimation is to have Z ( ψ ) &lt; + ∞ for all ψ , something that can be readily verified and is often the case in practice. The drawback of unnormalized EFs is the fact that sampling (and thus approximating expectations under the model) cannot usually be performed in an unbiased manner. Instead, inference in unnormalized EFs is often performed using tools from the Bayesian Inference literature, such as MCMC [26]. Unnormalized EFs belong to the larger class of unnormalized models [27, 28, 7, 6], of the form e -E ψ ( x ) -log Z ( ψ ) c ( d x ) , Z ( ψ ) = ∫ e -E ψ ( x ) c ( d x ) , for some parametrized function E ψ : R d ↦-→ R referred to as the energy . Unnormalized models thus take the flexibility of unnormalized EFs one step further by allowing the (negative) unnormalized log-density to be an arbitrary function E ψ of x and ψ , instead of requiring a linear dependence on ψ as in Equation 1. We focus in this work on unnormalized EFs due to the multiple computational benefits they provide, as explained in the next section, but we believe that extending our analysis to more general unnormalized models is an interesting avenue for future work.

Statistical Estimation in Unnormalized Exponential Families using Contrastive Divergence We now review the Contrastive Divergence algorithm, an algorithm used to fit unnormalized models, and our main object of study in this work. The general setting is the following: we assume access to n i.i.d. samples ( X 1 , . . . , X n ) drawn from some unknown distribution p ⋆ , which we assume belongs to P ψ , e.g. p ⋆ = p ψ ⋆ for some ψ ⋆ ∈ Ψ . Given these samples, we aim to perform statistical estimation , e.g. find a parameter ψ n within Ψ that should approach ψ ⋆ as n grows.

The starting point of the Contrastive Divergence algorithm is the unfortunate realization that Maximum Likelihood Estimation, which corresponds to minimizing the cross-entropy L ( ψ ) := -E p n log d p ψ / d c between the model p ψ and the empirical data distribution p n := 1 /n ∑ n i =1 δ X i , cannot be performed using exact (possibly stochastic) gradient-based optimization, as the gradient ∇ ψ L ( ψ ) of L with respect to the parameter ψ contains an expectation under the model distribution p ψ . Indeed, the cross entropy and its gradient are given by

<!-- formula-not-decoded -->

The second line follows from the well known identity ∇ ψ log Z ( ψ ) := E p ψ ϕ ; we refer to [25, Proposition 3.1] for a proof. The Contrastive Divergence algorithm circumvents this issue by running approximate stochastic gradient descent (SGD) on L , where the intractable expectation in ∇ ψ log Z is estimated using an MCMC algorithm initialized at the empirical data distribution. In more details, given a number of epochs T , a sequence of data batches B t,j of size B (e.g. B t,j ∈ [ [1 , n ] ] B , 1 ≤ t ≤ T, 1 ≤ j ≤ N ⌈ n/B ⌉ ), and a family of Markov kernels { k ψ , ψ ∈ Ψ } each with invariant distribution p ψ , at the j th minibatch of epoch t , ∇ ψ log Z ( ψ t, j -1 ) is approximated by 1 B ∑ i ∈ B t,j ϕ ( ˜ X i m ) , where ˜ X i m is produced by running the recursion ˜ X k i ∼ k ψ t ( ˜ X k -1 i , · ) , ˜ X 0 i = X i up to k = m . Throughout the paper, we will refer to the conditional distribution of ˜ X i m given X i as k m ψ ( X i , · ) . The resulting gradient estimate arising from combining this approximation with the other (tractable) sum over the data samples present in ∇ ψ L ( ψ ) , which we refer to as the CD gradient and denote as h t , is thus

<!-- formula-not-decoded -->

Key to the behavior and analysis of the CD algorithm is the strategy employed to generate minibatches B t,j . The case where T = 1 , B = 1 , and B 1 ,j = { j } will be referred to as online CD, while the variant where T &gt; 1 , and each batch B t,j draws B indices (with or without replacement) from [ [1 , n ] ] will be referred to as offline CD. In online CD, each data point is present in one and one batch only, while in offline CD, data points are reused across batches. From a statistical perspective, we will see that online CD can be analyzed in a remarkably simple way, while offline CD introduces additional correlations that require care to be controlled. Both settings come with their advantages and drawbacks, as we will see in the next section. The CD algorithms we study will employ decreasing step size schedules ( η t ) t ≥ 0 of the form η t = Ct -β , where C &gt; 0 is the initial leaning rate and β ∈ [0 , 1] . We lay out online CD and offline CD in Algorithms 1 and 2. Note that our algorithms include a projection step on the parameter space Ψ to account for the case where Ψ is compact. In the case Ψ = R p , this step can be omitted. Next we depart from the setting of [21] and start by analyzing online CD.

## Algorithm 1 Online CD

,

```
Input: ( X 1 , . . . , X n ) i.i.d. ∼ p ψ ⋆ Parameters: Model class { p ψ , ψ ∈ Ψ } Markov kernels { k ψ , ψ ∈ Ψ } , number of MCMC steps m , learning rate schedule η t := Ct -β , β ∈ [0 , 1] , C > 0 , initial parameter ψ 0 for t = 1 , . . . , n do //Approx. sample from p ψ t -1 ˜ X t m ∼ k m ψ t -1 ( X t , · ) h t := ϕ ( X t ) -ϕ ( ˜ X t m ) ψ t ← ψ t -1 -η t h t ψ t ← Proj Ψ ( ψ t ) end for return ψ n
```

## 3 Non-asymptotic analysis of Online CD

## 3.1 Preliminaries and Assumptions

Recall that the chi-squared divergence between two probability measures p and q is defined as: χ 2 ( p, q ) := ∫ ( d p d q ( x ) -1) 2 q (d x ) if p ≪ q , and + ∞ otherwise. Here, p ≪ q denotes that p is absolutely continuous with respect to q and d p/ d q is the Radon-Nikodym derivative [29] of p with respect to q . Let L 2 ( p ψ ) be the space of square-integrable functions with respect to p ψ . For a function f ∈ L 2 ( p ψ ) , we define

<!-- formula-not-decoded -->

which is a measure of how quick a Markov chain with kernel k ψ mixes, relative to the function f [30]. With these definitions in hand, we now state the assumptions required by our analysis of online CD. These assumptions form a strict subset of the assumptions considered in prior work [21], which required additional regularity and tail conditions on the Markov kernels k ψ .

Assumption A1. P ψ is a subset of a regular and minimal [25, Section 3.2] exponential family with natural parameter domain D ⊆ R p , Ψ is a convex and compact subset of D , and ψ ∗ lies in the interior of Ψ .

Assumption A2. There exists a constant C χ &gt; 0 such that χ 2 ( p ψ ⋆ , p ψ ) ≤ C 2 ∥ ψ -ψ ⋆ ∥ 2

Assumption A3. α := sup { α ( f, ψ ) , f ∈ { ϕ i } i =1 ∪{ ϕ i ϕ j } i,j =1 , ψ ∈ Ψ } &lt; 1 , where ϕ i is the i -th component of the function ϕ , and ϕ 2 is the i -th component of the function x ↦-→ ϕ ( x ) 2 .

χ p p i

A well known property of EFs [25, Proposition 3.1] is that their negative cross-entropy (against any other measure) is C ∞ , convex, and strictly so if the exponential family is minimal (meaning that the set of sufficient statistic functions ϕ i are not linearly dependent). Leaving aside the issue of intractable expectations, this convexity suggests that L can be efficiently minimized using stochastic approximation algorithms [31, 22]. The compactness of Ψ provided by Assumption A1 thus ensures, by the extreme value theorem [32], the existence of finite positive constants µ and L defined as:

<!-- formula-not-decoded -->

where ∇ 2 ψ L is the Hessian of L with respect to ψ . µ (called the strong convexity constant) and L (a bound controlling the smoothness of the problem) play a critical role in the analysis of convex optimization algorithms [31]. While it is possible to obtain convergence rates in non-smooth or non-strongly-convex settings, our analysis follows the spirit of [21] by leveraging the strong convexity of the problem to compensate for the bias introduced by using CD gradients instead of unbiased stochastic gradients.

Assumption A2 allows link variations in distribution space to variations in parameter space, and will be instrumental to control the bias of the CD gradient. Note that since χ 2 ( p ψ ⋆ , p ψ ) =

## Algorithm 2 Offline CD

```
Input: ( X 1 , . . . , X n ) i.i.d. ∼ p ψ ⋆ Parameters: Same as Algorithm 1, plus number of epochs T , batch size B , batching schedule B t,j , initial parameter ψ 0 , 0 for t = 1 , . . . , T do for j = 1 , . . . ⌈ n/B ⌉ do [ ˜ X m t,i,j ∼ k m ψ t -1 ( X i , · ) for i in B t,j ] h t,j := 1 B ∑ i ∈ B t,j ( ϕ ( X i ) -ϕ ( ˜ X m t,i,j )) ψ t,j ← ψ t,j -1 -η t h t,j ψ t,j ← Proj Ψ ( ψ t,j ) end for end for return ψ T
```

e log Z (2 ψ -ψ ⋆ ) -(2 log Z ( ψ ) -log Z ( ψ ⋆ )) -1 provided that 2 ψ -ψ ⋆ ∈ D (see [33, Lemma 1]), we expect Assumption A2 to hold in many cases of interests. On the other hand, the possible exponential scaling of C χ w.r.t log Z suggests that this constant may be large in some instances.

Assumption A3 is a restricted spectral gap condition: it guarantees that the time required by the MCMCalgorithm to estimate expectations of ϕ and ϕ 2 under p ψ will be uniformly bounded. This assumption is weaker than the (unrestricted) uniform spectral gap condition of [21], which requires that α controls the convergence rate of all functions in L 2 ( p ψ ) . Note that standard results in stochastic analysis [34] guarantee that α ≤ 1 : thus, it only remains to ensure that α is strictly less than 1 . Spectral gaps are strongly dependent on two properties of distribution: their tail behavior and their multimodality. While multimodality poses the risk of pushing the constant α close to 1 , very heavy tails distributions may not verify the spectral gap condition at all.

## 3.2 Results

## 3.2.1 Parametric convergence of online CD

In this section, we show that under the assumptions stated in Section 3.1, the iterates ψ t produced by the online CD algorithm described in Algorithm 1 will converge to the true parameter ψ ⋆ at the parametric rate O ( n -1 / 2 ) . To do so, we follow a well known paradigm in convex optimization [22] by deriving a recursion on the quantity δ t := E ∥ ψ t -ψ ⋆ ∥ 2 , which will allow, after unrolling, to obtain convergence rates for the iterates ψ t . We aim to characterize precisely the impact of performing CD as opposed to performing online SGD on L , which would consist of replacing the CD gradient h t of Algorithm 1 by the unbiased (stochastic) gradient, given by:

<!-- formula-not-decoded -->

which satisfies E g t ( ψ ) = ∇ ψ L ( ψ ) . The only stochasticity in g t comes from the sampling of a single data point x t from the true distribution, which is unavoidable in the online setting, and we have E ∥ g t ( ψ ⋆ ) ∥ 2 = Tr ( Cov p ψ⋆ ϕ ) =: σ 2 ⋆ . σ 2 ⋆ plays a key role in the analysis of Stochastic Gradient Descent [22]. We expect that replacing g t by h t will introduce two sources of approximation: a bias term coming from using a finite number of MCMC steps m , and an additional variance term, coming from using a single sample ˜ x t m to estimate ∇ ψ log Z ( ψ t ) . With that in mind, we derive a recursion on δ t in the following lemma.

Lemma 3.1. Let ( ψ t ) 0 ≤ t ≤ n be the iterates from Algorithm 1. Denote δ t = E ∥ ψ t -ψ ⋆ ∥ 2 , σ ⋆ = ( E p ψ⋆ ∥ ϕ -E p ψ⋆ ϕ ∥ 2 ) 1 / 2 , and σ t = ( E p ψt ∥ ϕ -E p ψt ϕ ∥ 2 ) 1 / 2 . Then, under A1, A2 and A3, for all t ≥ 1 ,

<!-- formula-not-decoded -->

where ∥ log Z ∥ 3 , ∞ is a constant, ˜ µ m,t := µ -α m σ t C χ , and ˜ σ m,t := ( σ 2 ⋆ + σ 2 t +2 σ 2 t α 2 m ) 1 / 2 .

Lemma 3.1 is proved in Appendix D.2, which details the form of ∥ log Z ∥ 3 , ∞ , a constant that we expect to scale roughly as dL . Loosely speaking, this recursion suggests that as the learning rate η t goes to 0 , the two terms scaling in η 2 t will be negligible, in which case we will have: δ t ≤ (1 -2 η t ˜ µ m,t -1 ) δ t -1 &lt; δ t -1 , yielding convergence of δ t to 0 . We make these arguments formal in the next theorem. The reader familiar with the convex optimization literature will note the similarities between this recursion and the one derived in [22], which would apply as is to online SGD on L using g t . The difference between the two recursions is that the roles of the strong convexity constant µ and the noise σ ⋆ are now played respectively by

<!-- formula-not-decoded -->

These two modifications respectively characterize the impact of the bias and the additional variance introduced by the CD gradient. The last term in Equation 7, scaling in α m/ 2 η 2 t √ δ t , is a residual higher order mixed term coming from relating the variance of the Markov chain sample ˜ x t m to σ 2 t . This term can be easily controlled as done next, and disappears as m →∞ . Investigating the impact of m in the recursion, we notice that as m → ∞ , ˜ µ m,t → µ . As we will see later, this ensures that CD will converge for a sufficiently high m . On the other hand, in that same regime, ˜ σ m,t does not converge to σ ⋆ , but rather to ( σ 2 ⋆ + σ 2 t ) 1 / 2 , showing the irreducible impact of the variance term.

While we precisely investigate the impact of the residual variance term in the next section, we now unify σ ⋆ and σ t by introducing

<!-- formula-not-decoded -->

̸

σ is an upper bound on the noise induced both by the CD gradient and by the online setup, and was used in prior work [21]. Note that by the properties of log Z , σ 2 also equals sup ψ ∈ Ψ tr( ∇ 2 ψ L ( ψ )) , where tr( A ) is the trace of A ∈ R p × p , and thus finite by the extreme value theorem. The following theorem is obtained by invoking standard unrolling arguments in the convex optimization literature. In the next result, we use the function φ γ ( t ) , defined as φ γ ( t ) = t γ -1 γ if γ = 0 , and log t if γ = 0 .

Theorem 3.2. Fix n ≥ 1 . Let ( ψ t ) 0 ≤ t ≤ n be the iterates produced by Algorithm 1, and define δ t := E ∥ ψ t -ψ ⋆ ∥ 2 . Moreover, assume that m &gt; log( σC χ /µ ) log | α | , i.e. ˜ µ m := µ -α m σC χ &gt; 0 . Then under Assumptions A1, A2 and A3, for η t = Ct -β with C &gt; 0 , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 3.2 is proved in Appendix D.3. It shows that the iterates produced by online CD will converge to the true parameter ψ ⋆ at the rate O ( n -1 / 2 ) provided that the number of steps m is sufficiently large, improving over the asymptotic O ( n -1 / 3 ) rate of [21], while imposing slightly weaker conditions on the number of steps m (see [21, Theorem 2.1]). This proves that online CD can be asymptotically competitive with other methods for training unnormalized models, such as Noise Contrastive Estimation [6], or Score Matching [7]. However, the asymptotic variance of ψ t (e.g. the multiplicative factor in front of the O ( n -1 / 2 ) term) is likely to be suboptimal, e.g. much larger than the Crámer-Rao bound, given by the trace of the inverse of the Fisher information matrix [25]. Given the statistical optimality of MLE, and the fact that CD in an approximate MLE method, this motivates the further goal or obtaining a CD estimator with near-optimal statistical properties. In the next section, we achieve this goal by showing that averaging the iterates ψ t will produce a near statistically-optimal estimator, in a sense that we will make precise.

## 3.2.2 Towards statistical optimality with averaging

Polyak-Ruppert averaging [23] is a simple yet surprisingly effective way to construct an asymptotically optimal estimator ¯ ψ n := 1 n ∑ n t =1 ψ i from a sequence of iterates ( ψ t ) 0 ≤ t ≤ n obtained by running a standard online SGD algorithm [22]. As shown in [22], when the objective is the cross-entropy of a model, and assuming the unbiased stochastic gradients are available, averaging yields an estimator ψ with the asymptotic variance tr( I ( ψ ⋆ ) -1 ) /n , where I ( ψ ) := Cov p ψ⋆ ϕ is the Fisher information matrix of the data distribution p ψ ⋆ . I ( ψ ⋆ ) -1 being the Cramér-Rao lower bound on asymptotic variances of statistical estimators [10], this estimator ¯ ψ n is asymptotically optimal. The following theorem shows conditions under which averaging CD iterates can give rise to a near-optimal estimator.

Theorem 3.3 (Contrastive Divergence with Polyak-Ruppert averaging) . Let ( ψ t ) t ≥ 0 the sequence of iterates obtained by running the CD algorithm with a learning rate η t = Ct -β for β ∈ ( 1 2 , 1) . Define ¯ ψ n := 1 n ∑ n i =1 ψ i . Then, under the same assumptions as Theorem 3.2, and assuming additionally that m := m ( n ) &gt; (1 -β ) log n 2 | log α | , we have, for all n ≥ 1 ,

<!-- formula-not-decoded -->

Consequently, we have that lim sup n →∞ n E ( ∥ ∥ ψ n -ψ ⋆ ∥ ∥ 2 ) ≤ 4tr( I ( ψ ⋆ ) -1 ) .

Theorem 3.3, alongside with a statement which includes the asymptotic order of the residual term, is proved in Appendix D.4. It shows that at the cost of an increase in computational complexity of the entire algorithm from O ( n ) to O ( n log n ) , ¯ ψ n will be a near-optimal statistical estimator of ψ ⋆ .

While this increase in complexity emerges from the bias of CD, the additional variance of CD results in an asymptotic variance inflated by a factor of 4 compared to the Cramér-Rao bound.

Theorem 3.3 concludes our analysis of online CD. Despite their asymptotic near-optimality, the bounds provided for online CD and its averaged version have weaknesses: the online CD iterates are not robust to choices of C. On the other hand, as shown in Appendix D.4, the bound of the averaged iterates contain higher-order terms that could be large in intermediate sample regimes. Next, we show that offline CD, which processes data points multiple times, can alleviate these issues.

## 4 Non-asymptotic analysis of offline CD

In practice, CD gradient approximation schemes are commonly used within an offline stochastic gradient descent (SGD) algorithm, where one is given the full sizen dataset upfront and each update uses some stochastic subset of the data. We study CD under offline SGD with replacement (SGDw), i.e. Algorithm 2 with batches B t,j being i.i.d. uniform draws of sizeB subsets of [ n ] , and include SGD without replacement in Appendix B.2. To do so, we follow the setting of prior work on offline CD [21], which established its asymptotic O ( n -1 3 ) consistency. We show that by slightly strengthening a moment assumption used in [21], the offline CD iterates converge to the true parameter at a near-parametric O ((log n ) 1 2 n -1 2 ) rate. Our proof proceeds by controlling a 'tail probability' term specific to the offline setting which characterizes the strength of the correlations between the offline CD iterates and the training data. While, as we show, the assumptions of [21] provide a tail control sufficient to obtain a near-parametric rate, other strategies are possible to obtain convergence guarantees. In particular, we show that non-asymptotic convergence can be obtained by either (1) relaxing assumptions on the Markov kernel required by prior work, or (2) making a specific mixing assumption the Markov chain.

## 4.1 Background: Asymptotic consistency of offline CD in subexponential settings

Prior work [21] has established asymptotic O ( n -1 3 ) consistency of the (averaged) offline CD iterates in the full-batch case. We summarize their results and assumptions below.

Assumption A4. There exists ν ≥ 2 s.t. for all m ∈ ◆ , there is κ ν ; m &lt; ∞ s.t.

<!-- formula-not-decoded -->

Assumption A5. There exists some C m &gt; 0 such that, for all ψ 1 , ψ 2 ∈ Ψ , sup x ∈X ∥ E [ ϕ ( K m ψ 1 ( x ))] -E [ ϕ ( K m ψ 2 ( x ))] ∥ ≤ C m ∥ ψ 1 -ψ 2 ∥ .

Assumption A6. There exist some σ m , ζ m &gt; 0 such that, for any z ∈ ❘ p with ∥ z ∥ ≤ ζ m , E [ e z ⊤ ( ϕ ( K m ψ ∗ ( X 1 )) -E [ ϕ ( K m ψ ∗ ( X 1 ))]) ] ≤ e σ 2 m ∥ z ∥ 2 / 2 .

Theorem 4.1 (Theorem 2.1 of [21]) . Assume assumptions A1,A2, A3, A4 (for ν = 2 ), A5 and A6. Let ψ t, 1 be the t -th iterate of offline CD with full-batch gradient descent and constant stepsize η t = C , i.e the iterates produced by Algorithm 2 using B t, 1 = [ [1 , n ] ] . Then for any learning rate C and number of Markov kernel steps m satisfying µ -α m σC χ -C 2 ( L + α m σC χ ) 2 &gt; 0 , we have, for some A m &gt; 0 ,

<!-- formula-not-decoded -->

This result shows convergence of the averaged full-batch CD iterates to the true parameter in the large n and T limit. As discussed, this result is asymptotic both in n and T : the probability of the error exceeding A m n -1 3 goes to 0 as n →∞ and T →∞ , but at an unknown rate. Moreover, the O ( n -1 3 ) does not match the optimal O ( n -1 2 ) rate.

## 4.2 Sharpening offline CD bounds in subexponential settings

## 4.2.1 Non-asymptotic ˜ O ( n -1 / 2 ) -consistency

As a first result, we show that under the assumptions of [21] (except for a slightly stronger ν &gt; 2 moment assumption in A4), ψ SGDw T,N in fact achieves a near-parametric rate. The most general version of our result holds for any learning rate schedule of the form Ct -β , β ∈ [0 , 1] , and for offline SGD

with arbitrary batch sizes B , with data drawn either with or without replacement across batches. For simplicity, we first present our result assuming full batch ( B = n, N = 1 , ψ SGDw t,j = ψ SGDw t, 1 for t ≥ 1 ) SGD with constant step sizes η t = C , which is the setting of [21]. Analogue bounds holding for the other mentioned batching and step sizes schedules can be found in Appendix B.

Theorem 4.2. Assume the setup of Theorem 4.1, except that Assumption A4 holds for some ν &gt; 2 , and that ˜ µ m = µ -α m σC χ &gt; 4 CL 2 . Let δ SGDw t,j := E ∥ ψ SGDw t,j -ψ ∗ ∥ 2 . Then, we have:

<!-- formula-not-decoded -->

where E T, 1 1 , E T, 1 2 are functions decreasing exponentially in T , and C ′ ( p, ν, m, Ψ) is a constant in n, T . Consequently,

<!-- formula-not-decoded -->

The precise values of all the constants can be found in Theorem B.1 (for E T, 1 1 , E T, 1 2 ) and Lemma B.3 (for C ′ ( p, ν, m, Ψ , β ) ), including their expressions for N &gt; 1 and β ∈ [0 , 1] . We comment on the main differences between our result and the one of [21]. First our bound holds for any epoch T and number of samples n . Second, fixing n but taking T →∞ , the final bound matches the parametric O ( √ n ) up to a √ log( n ) factor, a significant improvement over the O ( n -1 3 ) rate of [21]. Finally, we control an L 2 error, which is a stronger control than a high probability bound by Markov's inequality; we hypothesize this is the reason why a slightly stronger moment assumption is required for our setup, compared to the one used for the high probability bound in [21].

Inspecting Equation 9, we notice the presence of two transient terms, and a stationary term , reminiscent of the structure of upper bound of Theorem 3.2. The transient terms (i.e. the ones containing E T, 1 1 and E T, 1 2 ) vanish exponentially fast in the total number of CD updates T . However, unlike in online CD where the number of updates and the number of samples are tied (e.g. T = n ), these two values are now decoupled , and these terms can be made arbitrarily small by increasing the number of gradient steps T without having to collect more samples n . The stationary term, which is the only one remaining in the limit of T →∞ , decreases with n at a rate that is independent of hyperparameters like the step size C or the learning rate schedule β (see Lemma B.3). In that sense, offline CD compares favorably to online CD, whose rate is sensitive to β and C , and averaged online CD, whose bound contains higher-order (in n ) terms which can be large in the moderate n regime. On the other hand, the stationary term in offline CD is asymptotically suboptimal: its rate is larger (while only up to a log factor) than the best-case O ( √ n ) one achieved by online CD algorithms, and the leading constant does not match the optimal one.

## 4.2.2 Proof of Theorem 4.2

The high-level proof of Theorem 4.2 follows a similar strategy as the online one: first, derive a recursion for the quantity δ SGDw t, 1 := E ∥ ψ SGDw t, 1 -ψ ∗ ∥ 2 , then unroll it explicitly to obtain a final bound on δ SGDw T, 1 . The main difference to online CD is the presence of an additional offline-specific correlation between the iterates and the data. We thus break down the proof into three steps: (1) deriving a controllable, uniform-in-time upper bound of the data-iterate correlations, (2) deriving and unrolling a recursion on δ SGDw t, 1 containing this new term, and (3) controlling that term to obtain a final bound on δ SGDw T, 1 .

̸

Step 1:characterizing the data-iterate correlations in offline CD In offline CD, at each epoch t ≥ 1 , the iterate ψ SGDw t -1 , 1 and the data samples X i are correlated: this is because these samples may have been used in previous epochs t ′ &lt; t -1 to obtain the ψ SGDw t ′ , 1 , which themselves influenced ψ t -1 , 1 . With such correlations, we now have P ( X i | ψ SGDw t -1 , 1 ) = P ( X i ) , preventing us from obtaining an unrollable recursion on δ SGDw t, 1 by first marginalizing X i out to obtain an upper bound of E [∥ ∥ ψ SGDw t, 1 -ψ ∗ ∥ ∥ 2 | ψ SGDw t -1 , 1 ] that only depends on ∥ ∥ ψ SGDw t -1 , 1 -ψ ⋆ ∥ ∥ , and then marginalizing over ψ SGDw t -1 , 1 to obtain a recursion as in Lemma 3.1. As this problem would not have occurred had we used 'fresh samples' (e.g. i.i.d copies of X i not present in the training data) to perform our update, the

core of the proof lies in controlling the following quantity:

<!-- formula-not-decoded -->

where X ′ 1 is an i.i.d. copy of X 1 . ∆( ψ SGDw t, 1 ) is the expected (over the data and iterates) error between a quantity that allows to obtain a recursion (the rightmost term) and the one actually used by offline CD (the leftmost term). To control it, we upper-bound it using a tail decomposition:

<!-- formula-not-decoded -->

We invoke an additional assumption to ensure that ( E [∆( ψ SGDw t, 1 ) ν ]) 2 /ν is finite; in the results of [21], this is automatically implied by assumptions A4 and A6. For simplicity we assume the same bounding constant κ ν ; m .

Assumption A7. There exists ν ≥ 2 s.t. for all m ∈ ◆ , κ from A4 moreover verifies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note the similarity of this assumption with assumption A4: the only difference is that X 1 is now a random training point instead of an deterministic (arbitrary) one. Ensuring assumption A7 in addition to assumption A4 thus requires controlling a ν -th order moment, instead of all moments as implied by assumption A6.

Step 2: Deriving and unrolling the recursion on δ SGDw t, 1 The right-hand side of Equation (10) does not depend on t , allowing for the derivation of an 'unrollable' recursion on δ SGDw t, 1 and its subsequent unrolling, which is performed in the following theorem. For simplicity, we again assume β = 0 and N = 1 and defer the general case to Theorem B.1 in appendix.

Theorem 4.3 (Convergence up to a tail control) . Assume A1, A2, A3, A4 and A7. Let η t = C for some C &gt; 0 , and assume that ˜ µ m = µ -α m σC χ &gt; 4 CL 2 . Then for any ϵ &gt; 0 ,

<!-- formula-not-decoded -->

where ε SGDw n,m,T ; ν ( ϵ ) is defined in Equation (10) .

Note that in the general, non-full batch B ≤ n case, 5 σ +5 κ ν,m √ n is replaced by 5 σ +5 κ ν,m √ B (see Theorem B.1). Under our bounds, obtaining consistency thus requires setting B ≡ B ( n ) → n →∞ + ∞ .

Step 3: Controlling the tail probability term Theorem 4.3 is just one step away from the final bound of Theorem 4.2: it remains to control the tail term ε SGDw n,m,T ; ν ( ϵ ) . Under the assumptions of [21], minimizing ε SGDw n,m,T ; ν ( ϵ ) over ϵ yields the following result:

Lemma 4.4. Assume the setup of Theorem 4.2. Let n ∈ ◆ be sufficiently large s.t. log n n &lt; σ 2 m ζ 2 m p + ν -2 . Denote r Ψ as the radius of the smallest sphere in ❘ p that contains Ψ , which is finite under A1. Then inf ϵ&gt; 0 ε SGDw ν ; n,m,T ( ϵ ) ≤

<!-- formula-not-decoded -->

To obtain this result, we control the moment term (sup t E [∆( ψ SGDw t, 1 ) ν ]) using A7, and we control the tail probability term sup t /a80 (∆( ψ SGDw t, 1 ) &gt; ϵ ) as in [21, Lemma 3.1] using an union bound, a covering argument and A6. Theorem 4.2 then follows by plugging Lemma 4.4 into Theorem 4.3.

## 4.3 Consistency of offline CD: beyond subexponential tails.

As discussed above, the general unrolling result of Theorem 4.3 holds without the subexponentiality assumption A6; this assumption was only used in Lemma 4.4 to control ε SGDw n,m,T ; ν ( ϵ ) . We now discuss two alternative ways to control this quantity without requiring subexponential tails. The first generalizes the idea of Jiang et al. [21], while the second exploits mixing of the Markov chain K m ψ ( x ) as m → ∞ . As before we only state partial results (full batch, β = 0 ) and defer the full explicit bounds to Appendix B.3.

Control via Markov Inequality Our first alternative uses Markov Inequality to yield the following. Theorem 4.5. Assume the setup of Theorem 4.3 and additionally that A5 holds. Then

<!-- formula-not-decoded -->

where ˜ C and ˜ C ′ are functions whose explicit expressions are given in Lemma B.4 in the appendix.

In the case p = 1 and ν = 3 , the sub-optimal error from Theorem 4.5 reads O ( n -3 / 20 ) . Theorems 4.2 and 4.5 reveal that, depending on the tail condition imposed on the noise introduced by the Markov kernel, the convergence rate of offline CD varies: A subexponential tail, as assumed in prior work, in fact leads to near-parametric rate. Meanwhile, consistency can be obtained without assuming subexponentiality, albeit at a sub-optimal rate.

Control via Markov chain mixing. Alternatively, notice that E [∆( ψ SGDw t, 1 ) 2 ] involves an average of E [ ϕ ( K m ψ SGDw t, 1 ( X i ) 2 )∣ ∣ X i , ψ SGDw t, 1 ] -E [ ϕ ( K m ψ SGDw t, 1 ( X ′ 1 ) )∣ ∣ ψ SGDw t, 1 ] . When m →∞ , the effect of initialization vanishes, and one may expect the difference to converge to zero. We defer to Lemma B.5 in the appendix to show that, under a ϕ -discrepancy mixing condition ([35]) with a mixing coefficient ˜ α ∈ [0 , 1) ,

<!-- formula-not-decoded -->

As m → ∞ , this recovers the parametric rate O ( n -1 / 2 ) . This alternative convergence guarantee comes at the cost of requiring m , the number of Markov chain steps, to grow with the sample size n .

Remark (Examples) . In our main results (Theorems 3.2, 3.3 and 4.3) and the tail condition for offline SGD (Theorem 4.2), we employed a weaker set of assumptions than those in [21] (except for the mild ν &gt; 2 moment assumption in (A4)). Consequently, our results apply to all three examples studied in [21]: A bivariate Gaussian model with unknown mean and random-scan Gibbs sampler, a fully visible Boltzmann machine with random-scan Gibbs sampler, and an exponential-family random graph model with a Metropolis-Hastings sampler.

## 5 Related Work

Central to this paper is the prior work of Jiang et al. [21], which provided a rigorous theoretical foundation to analyze the convergence of full-batch CD, and which we refine. The study of optimization with biased gradient descent has attracted a lot of attention in recent years [36, 37, 38, 39]. These works, while closely connected to ours, analyze algorithms with different implementation choices than the CD algorithm: i.i.d. noise setup [36], or setup where a persistent Markov chain is maintained through the iterations [36, 37, 38, 39]. The latter is akin to a variant of the CD algorithm, called the persistent CD [40]. In contrast, our analysis focus on the CD algorithm that restarts a batch of Markov chains from the data distribution at every iteration. Finally, there is a rich body of work on convergence guarantees for offline multi-pass SGD [41, 42, 43, 44, 45, 46]. A notable difference of our analysis is that we are primarily concerned with statistical errors associated with convergence to the true parameter ψ ∗ in number of samples n , and not the commonly studied convergence rate in number of epochs T . Consequently, most of our work for the offline setup goes into handling the correlations that accumulate by reusing data across epochs.

## 6 Discussion

In this work, we provide a non-asymptotic analysis of the Contrastive Divergence algorithms, showing, in the online setting, their potential to converge at the parametric rate and to have near-optimal asymptotic variance, and proving a near-parametric rates in the offline setting, significantly extending prior results. Our results apply to unnormalized exponential families: despite their flexibility, these models only cover log-densities with linear relationships on the model parameters. We believe that extending our results to more general forms of unnormalized models is an important direction for future work.

## Acknowledgments and Disclosure of Funding

All authors acknowledge support from the Gatsby Charitable Foundation.

## References

- [1] Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference. Proceedings of the National Academy of Sciences , 117(48):30055-30062, 2020.
- [2] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning , pages 8748-8763. PMLR, 2021.
- [3] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog , 1(8):9, 2019.
- [4] Aad W Van der Vaart. Asymptotic statistics , volume 3. Cambridge university press, 2000.
- [5] Holden Lee, Chirag Pabbaraju, Anish Prasad Sevekari, and Andrej Risteski. Pitfalls of gaussians as a noise distribution in nce. In The Eleventh International Conference on Learning Representations , 2023.
- [6] Michael Gutmann and Aapo Hyvärinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the thirteenth international conference on artificial intelligence and statistics , 2010.
- [7] Aapo Hyvärinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research , 2005.
- [8] Frederic Koehler, Alexander Heckett, and Andrej Risteski. Statistical efficiency of score matching: The view from isoperimetry. In The Eleventh International Conference on Learning Representations , 2022.
- [9] Chirag Pabbaraju, Dhruv Rohatgi, Anish Prasad Sevekari, Holden Lee, Ankur Moitra, and Andrej Risteski. Provable benefits of score matching. Advances in Neural Information Processing Systems , 36, 2024.
- [10] George Casella and Roger L Berger. Statistical inference . Cengage Learning, 2021.
- [11] Nicolas Le Roux and Yoshua Bengio. Representational power of restricted boltzmann machines and deep belief networks. Neural computation , 20(6):1631-1649, 2008.
- [12] Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models. Advances in Neural Information Processing Systems , 32, 2019.
- [13] Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural computation , 2002.
- [14] Erik Nijkamp, Mitch Hill, Tian Han, Song-Chun Zhu, and Ying Nian Wu. On the anatomy of mcmc-based maximum likelihood learning of energy-based models. In Proceedings of the AAAI Conference on Artificial Intelligence , 2020.
- [15] Erik Nijkamp, Mitch Hill, Song-Chun Zhu, and Ying Nian Wu. Learning non-convergent non-persistent short-run mcmc toward energy-based model. Advances in Neural Information Processing Systems , 2019.
- [16] Yilun Du, Shuang Li, Joshua Tenenbaum, and Igor Mordatch. Improved contrastive divergence training of energy- based models. arXiv preprint arXiv:2021.01316 , 2021.
- [17] Lianhui Qin, Sean Welleck, Daniel Khashabi, and Yejin Choi. Cold decoding: Energy-based constrained text generation with langevin dynamics. Advances in Neural Information Processing Systems , 35:9538-9551, 2022.
- [18] Deqian Kong, Bo Pang, Tian Han, and Ying Nian Wu. Molecule design by latent space energybased modeling and gradual distribution shifting. In Uncertainty in Artificial Intelligence , pages 1109-1120. PMLR, 2023.

- [19] Csilla Varnai, Nikolas S Burkoff, and David L Wild. Efficient parameter estimation of generalizable coarse-grained protein force fields using contrastive divergence: a maximum likelihood approach. Journal of chemical theory and computation , 9(12):5718-5733, 2013.
- [20] Pierre Glaser, Michael Arbel, Arnaud Doucet, and Arthur Gretton. Maximum likelihood learning of energy-based models for simulation-based inference. 2022.
- [21] Bai Jiang, Tung-Yu Wu, Yifan Jin, and Wing H Wong. Convergence of contrastive divergence algorithm in exponential family. The Annals of Statistics , 46(6A):3067-3098, 2018.
- [22] Eric Moulines and Francis Bach. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. Advances in neural information processing systems , 24, 2011.
- [23] Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization , 30(4):838-855, 1992.
- [24] Lawrence D Brown. Fundamentals of statistical exponential families: with applications in statistical decision theory. Ims, 1986.
- [25] Martin J Wainwright, Michael I Jordan, et al. Graphical models, exponential families, and variational inference. Foundations and Trends® in Machine Learning , 1(1-2):1-305, 2008.
- [26] Charles J Geyer. Introduction to markov chain monte carlo. Handbook of markov chain monte carlo , 20116022:45, 2011.
- [27] Yann LeCun, Sumit Chopra, Raia Hadsell, M Ranzato, and Fujie Huang. A tutorial on energybased learning. Predicting structured data , 1(0), 2006.
- [28] Yang Song and Diederik P Kingma. How to train your energy-based models. arXiv preprint arXiv:2101.03288 , 2021.
- [29] Joseph L Doob. Measure theory , volume 143. Springer Science &amp; Business Media, 2012.
- [30] David A Levin and Yuval Peres. Markov chains and mixing times , volume 107. American Mathematical Soc., 2017.
- [31] Yurii Nesterov. Introductory lectures on convex optimization: A basic course , volume 87. Springer Science &amp; Business Media, 2013.
- [32] Gary Harris and Clyde Martin. Shorter notes: The roots of a polynomial vary continuously as a function of the coefficients. Proceedings of the American Mathematical Society , pages 390-392, 1987.
- [33] Frank Nielsen and Richard Nock. On the chi square and higher-order chi distances for approximating f-divergences. IEEE Signal Processing Letters , 21(1):10-13, 2013.
- [34] Dominique Bakry, Ivan Gentil, Michel Ledoux, et al. Analysis and geometry of Markov diffusion operators , volume 103. Springer, 2014.
- [35] Maxim Rabinovich, Aaditya Ramdas, Michael I Jordan, and Martin J Wainwright. Functionspecific mixing times and concentration away from equilibrium. 2020.
- [36] Belhal Karimi, Blazej Miasojedow, Eric Moulines, and Hoi-To Wai. Non-asymptotic analysis of biased stochastic approximation scheme. In Conference on Learning Theory , pages 1944-1974. PMLR, 2019.
- [37] Tao Sun, Yuejiao Sun, and Wotao Yin. On markov chain gradient descent. Advances in neural information processing systems , 31, 2018.
- [38] Thinh T Doan. Finite-time analysis of markov gradient descent. IEEE Transactions on Automatic Control , 68(4):2140-2153, 2022.
- [39] Yves F Atchadé, Gersende Fort, and Eric Moulines. On perturbed proximal gradient algorithms. Journal of Machine Learning Research , 18(10):1-33, 2017.

- [40] Tijmen Tieleman. Training restricted boltzmann machines using approximations to the likelihood gradient. In Proceedings of the 25th international conference on Machine learning , 2008.
- [41] Andre Elisseeff, Theodoros Evgeniou, Massimiliano Pontil, and Leslie Pack Kaelbing. Stability of randomized learning algorithms. Journal of Machine Learning Research , 6(1), 2005.
- [42] Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In International conference on machine learning , pages 1225-1234. PMLR, 2016.
- [43] Junhong Lin and Lorenzo Rosasco. Optimal rates for multi-pass stochastic gradient methods. Journal of Machine Learning Research , 18(97):1-47, 2017.
- [44] Loucas Pillaud-Vivien, Alessandro Rudi, and Francis Bach. Statistical optimality of stochastic gradient descent on hard learning problems through multiple passes. Advances in Neural Information Processing Systems , 31, 2018.
- [45] Nicole Mücke, Gergely Neu, and Lorenzo Rosasco. Beating sgd saturation with tail-averaging and minibatching. Advances in Neural Information Processing Systems , 32, 2019.
- [46] Difan Zou, Jingfeng Wu, Vladimir Braverman, Quanquan Gu, and Sham Kakade. Risk bounds of multi-pass sgd for least squares in the interpolation regime. Advances in Neural Information Processing Systems , 35:12909-12920, 2022.
- [47] Dheeraj Nagaraj, Prateek Jain, and Praneeth Netrapalli. Sgd without replacement: Sharper rates for general smooth convex functions. In International Conference on Machine Learning , pages 4703-4711. PMLR, 2019.
- [48] Shashank Rajput, Anant Gupta, and Dimitris Papailiopoulos. Closing the convergence gap of sgd without replacement. In International Conference on Machine Learning , pages 7964-7973. PMLR, 2020.
- [49] Dimitri Bertsekas. Convex optimization theory , volume 1. Athena Scientific, 2009.
- [50] Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint , volume 48. Cambridge university press, 2019.
- [51] S. W. Dharmadhikari, V. Fabian, and K. Jogdeo. Bounds on the moments of martingales. Ann. Math. Statist. , pages 1719-1723, 1968.

## Supplementary Material for "Near-Optimality of Contrastive Divergence Algorithms"

The supplementary material provides the proofs of the main results of the paper:

Section B states full explicit bounds for the offline CD algorithm.

Section C collects a list of useful tools for our proofs. These include the properties of φ γ introduced before Theorem 3.2 in the main text, as well as several contraction and integrability results.

Section D provides the proofs for the online CD algorithm.

Sections E, F and G contain the proofs about the offline CD algorithm and the tail control.

## A Notations

Throughout the proofs, we will denote by P m ψ the following operator from L 2 ( p ψ ) to itself:

<!-- formula-not-decoded -->

Here, k m ( x, x ′ ) is the m -iterated version of some Markov transition kernel k ψ , e.g.:

<!-- formula-not-decoded -->

Proj Ψ : R p ↦-→ Ψ denotes the projection operator onto the convex set Ψ , e.g.

<!-- formula-not-decoded -->

We also frequently use the following function, used in standard convex optimization results [22].

̸

<!-- formula-not-decoded -->

which is defined on R + \ { 0 } .

## B Additional results for offline SGD

In this section, we provide the full statements on error bounds for SGD with replacement (SGDw), SGD with reshuffling (SGDo) and tail moment bounds, which complement the results in Section 4. Proofs are deferred to Appendix F, which make use of L 2 approximation by auxiliary gradient updates derived in Appendix E.

Notations Denote the SGDw iterates by ( ψ SGDw t,j ) t ∈ ◆ ,j ≤ N and let X ′ 1 be an i.i.d. copy of X 1 . Throughout the remaining of the appendix, we define given ϵ &gt; 0 and n ∈ ◆

<!-- formula-not-decoded -->

and ϑ SGDo n,m,T ( ϵ ) analogously. Using these notations, we can redefine the quantity ε SGDw n,m,T ; ν ( ϵ ) in the main as ε SGDw n,m,T ; ν ( ϵ ) := √ ϵ 2 + κ 2 ν ; m ( ϑ SGDw n,m,T ( ϵ ) ) ν -2 ν .

## B.1 An explicit finite-sample bound for SGDw

In the result below, we write δ SGDw t,j := E ∥ ∥ ψ SGDw t,j -ψ ∗ ∥ ∥ 2 and, for a fixed ϵ &gt; 0 , the quantity

<!-- formula-not-decoded -->

Theorem B.1. Assume A1 (where Ψ may be non-compact), A2, A3, A4 and A7. Let η t = Ct -β for some β ∈ [0 , 1] and C &gt; 0 , and assume that m &gt; log( σC χ /µ ) log | α | s.t. ˜ µ m = µ -α m σC χ &gt; 0 as in

Theorem 3.2. Then for any ϵ &gt; 0 , √ δ SGDw T,N is upper bounded by

<!-- formula-not-decoded -->

where E T,N 1 and E T,N 2 are two decreasing functions in T defined by

<!-- formula-not-decoded -->

We emphasize that the full result above holds for any β ∈ [0 , 1] , which in particular includes the constant step size β = 0 regime considered by [21]. When β = 0 , for E T,N 1 and E T,N 2 to decay to zero as T →∞ , we additionally need the condition

<!-- formula-not-decoded -->

This is almost identical to the condition used in [21, Equation 2.5, Theorem 2.1], except that 4 L 2 gets replaced by 1 2 ( L + α m σC χ ) 2 . Notably this says that an additional step size condition is needed for our results to hold in the constant step size regime, but not necessary for a decreasing step size.

## B.2 Results for SGDo

SGD with reshuffling (SGDo, also called SGD without replacement) is an optimization scheme that is also widely used in practice compared to SGDo and online SGD. In the context of CD, it corresponds to Algorithm 2 with batches chosen as

<!-- formula-not-decoded -->

where π is a uniform draw of the permutation group on n elements. We denote the iterates of SGDo ( ψ SGDo t,j ) t ∈ ◆ ,j ∈ [ N ] . Analogously to ϑ SGDw n,m,T , we define, for X ′ 1 an i.i.d. copy of X 1 , ϵ &gt; 0 and n ∈ ◆ , the tail probability term

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following result says that ψ SGDo t,j enjoys exactly the same convergence guarantee as ψ SGDo t,j in Theorem 4.3. The statement is identical to that of Theorem B.1 and is stated in full for completeness; see Appendix F.2 for the proof, which is a slight adaptation of the proof for Theorem B.1. As before we write δ SGDo t,j := E ∥ ∥ ψ SGDo t,j -ψ ∗ ∥ ∥ 2 .

Theorem B.2 (Convergence of CD-SGDo) . Assume A1 (where Ψ may be non-compact), A2, A3, A4 and A7. Let η t = Ct -β for some β ∈ [0 , 1] and C &gt; 0 , and assume that m &gt; log( σC χ /µ ) log | α | s.t. ˜ µ m = µ -α m σC χ &gt; 0 as in Theorem 3.2. Then for any ϵ &gt; 0 , √ δ SGDo T,N is upper bounded by

<!-- formula-not-decoded -->

where E T,N 1 and E T,N 2 are two decreasing functions in T defined by

<!-- formula-not-decoded -->

Remark. We also remark that existing works [47, 48] show that the standard SGDo typically gives a faster convergence rate in T than SGDw. An analogous result for the CD setup would involve additional technical hurdles of jointly controlling the correlations across minibatches and from reusing data samples, and we defer this to future work.

## B.3 Explicit tail control

We now provide the full explicit tail control bounds. All results in this section hold directly for ε SGDo ν ; n,m,T ( ϵ ) and δ SGDo T,N , and we omit them here. In the result below, we denote r Ψ as the radius of the smallest sphere in ❘ p that contains Ψ , which is finite under A1.

Lemma B.3. Assume A5 and A6. Let n ∈ ◆ be sufficiently large s.t. log n n &lt; σ 2 m ζ 2 m p + ν -2 . Then inf ϵ&gt; 0 ε SGDw ν ; n,m,T ( ϵ ) ≤

<!-- formula-not-decoded -->

In particular, if we additionally assume the conditions of Theorem B.1, we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Lemma B.4. Assume the conditions of Theorem 4.3 and additionally that A5 holds. Then

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The next result considers a ϕ -discrepancy mixing condition ([35]), which is a mixing assumption on K m ψ but with respect to a specific test function ϕ , and we impose it uniformly over ψ ∈ Ψ . We also recall that X ψ 1 ∼ p ψ .

Lemma B.5. Assume that there exist ˜ α ∈ [0 , 1) and ˜ C K &gt; 0 such that, for all ψ ∈ Ψ and x ∈ X , ∥ E [ ϕ ( K m ψ ( x ))] -E [ ϕ ( X ψ 1 )] ∥ ≤ ˜ C K ˜ α m . Then

<!-- formula-not-decoded -->

In particular, if we additionally assume the conditions of Theorem 4.3, we have

<!-- formula-not-decoded -->

## C Auxiliary Tools

## C.1 Properties of φ γ

The following lemma collects some identities used in [22].

Lemma C.1. φ γ satisfies the following properties:

- (i) φ γ is increasing on R + for all γ ;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The next lemma provides some additional results on φ γ .

Lemma C.2. φ γ satisfies the following properties:

- (i) φ γ is positive on t &gt; 0 and increasing for every γ ∈ ❘ ;
- (ii) For 1 ≤ t 1 ≤ t 2 and β ≥ 0 , we have

<!-- formula-not-decoded -->

If instead β &lt; 0 , we have

<!-- formula-not-decoded -->

̸

(iii) For 1 ≤ t 1 ≤ t 2 and γ = 0 , we have

<!-- formula-not-decoded -->

̸

(iv) Let 1 ≤ t 1 &lt; t 2 and κ, β ≥ 0 . If κ = 1 and a &gt; 0 , we have

<!-- formula-not-decoded -->

̸

and if κ = 1 and a &lt; 0 , we have

<!-- formula-not-decoded -->

Proof of Lemma C.2. (i) follows from checking γ &gt; 0 , γ = 0 and γ &lt; 0 respectively. The first set of bounds in (ii) follow by noting that t ↦→ t -β is decreasing for β ≥ 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The second set of bounds follows from noting that t ↦→ t -β is increasing for β &lt; 0 :

<!-- formula-not-decoded -->

̸

For (iii), we note that for γ = 0 , so by the mean value theorem,

<!-- formula-not-decoded -->

The desired bounds then follow from an explicit computation of the infimum and the maximum in each of the two cases γ ≥ 1 and γ ≤ 1 .

̸

For (iv), we first consider the case κ = 1 and a &gt; 0 . Then

<!-- formula-not-decoded -->

Since, for x ≥ 0 , x ↦→ ( x +1) -κ is decreasing and x ↦→ exp( ax 1 -κ / (1 -κ )) is increasing, we have that for t 1 ≤ t ≤ t 2 and x ∈ [ t, t +1] ,

<!-- formula-not-decoded -->

This implies that

<!-- formula-not-decoded -->

̸

The main difference in the case κ = 1 and a &lt; 0 is that we now use x ↦→ exp( a ( x +1) 1 -κ / (1 -κ )) is decreasing to obtain, for t 1 ≤ t ≤ t 2 and x ∈ [ t, t +1] ,

<!-- formula-not-decoded -->

A similar argument then yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We also need the following lemma, which is useful for controlling the accumulation of errors from the noise terms over iterations.

LemmaC.3. For any a, b ≥ 0 , T, N ∈ ◆ and κ, β ≥ 0 such that bt -β -at -κ ≤ 1 for all 1 ≤ t ≤ T , we have that

<!-- formula-not-decoded -->

Moreover, for any ζ ≥ 0 , we have that

<!-- formula-not-decoded -->

where and

<!-- formula-not-decoded -->

In the special case ζ = β = 1 &lt; κ , we have

<!-- formula-not-decoded -->

Proof of Lemma C.3. By assumption, bt -β -at -κ ≤ 1 for all 1 ≤ t ≤ T . Since 0 ≤ 1 -x ≤ e -x for all x ≤ 1 , we have that for any 1 ≤ t 1 ≤ t 2 ≤ T ,

<!-- formula-not-decoded -->

Applying this to the first quantity of interest followed by noting that a, b ≥ 0 and using Lemma C.2(ii), we obtain the first bound that

<!-- formula-not-decoded -->

For the second bound, we define

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

Then by noting that 1 -bt -β + at -κ ≥ 0 for all 1 ≤ t ≤ T again,we can bound the quantity of interest as

<!-- formula-not-decoded -->

In the last line, we have used that 0 ≤ 1 -b 2 t -β ≤ 1 for t ≥ t 0 +1 and 1 + at -κ ≤ 1 + a . To control the three quantities, we first note that by (13) , we have

<!-- formula-not-decoded -->

In ( a ) above, we have noted that b 2 ≤ as -( κ -β ) for s ≤ t 0 ; in ( b ) , we have used t 0 ≤ T and Lemma C.2(ii) with a, b ≥ 0 . In the special case β = 1 &lt; κ , the above yields

<!-- formula-not-decoded -->

We now control S 2 . By (13) again, we have

<!-- formula-not-decoded -->

̸

In ( c ) above, we have applied Lemma C.2(ii); in ( d ) , we have noted that sup t ∈ ◆ ( t +2) β /t β = 3 β . If κ = 1 and a &gt; 0 , we can apply Lemma C.2(iv) to get that

<!-- formula-not-decoded -->

If κ = 1 or a = 0 , the bound from ( c ) above reads

<!-- formula-not-decoded -->

where we have used Lemma C.2(ii) in the last line. Now consider the special case with ζ = 1 &lt; κ , the bound from ( d ) becomes

<!-- formula-not-decoded -->

We are left with controlling S 1 , which follows from a similar strategy as controlling S 2 :

<!-- formula-not-decoded -->

̸

In ( a ) above, we used Lemma C.2(ii). For β = 1 and b = 0 , we can apply Lemma C.2(iv) with b 2 &gt; 0 to obtain

̸

<!-- formula-not-decoded -->

In ( b ) , we have used Lemma C.2(iii) with 1 -β ≤ 1 . Meanwhile, if β = 1 or b = 0 , we have

<!-- formula-not-decoded -->

For the special case with ζ = β = 1 , the bound in (14) becomes

<!-- formula-not-decoded -->

In ( c ) , we have used Lemma C.2(ii) for both the case 1 -bN 2 ≤ 0 and 1 -bN 2 ≥ 0 .

Combining the bounds for the general cases, we obtain the first desired inequality that

<!-- formula-not-decoded -->

For the special case ζ = β = 1 &lt; κ, γ , combining the earlier bounds gives

<!-- formula-not-decoded -->

## C.2 Contraction and integrability results

The next result is a standard result in convex analysis, needed to handle projections performed in Algorithms 1 and 2.

Lemma C.4. Let Ψ a be convex subset of R p . Let ψ ⋆ ∈ Ψ . Then, for all ψ ∈ R p , we have:

<!-- formula-not-decoded -->

Proof. We have:

<!-- formula-not-decoded -->

Since by [49, Proposition 1.1.9], we have:

<!-- formula-not-decoded -->

for all ψ ′ ∈ Ψ , we can use this inequality at ψ ′ = ψ ⋆ ∈ Ψ to obtain:

<!-- formula-not-decoded -->

and the result follows by taking the square root.

We now state two lemmas that guarantee an amount of integrability sufficient to our analysis.

Lemma C.5. Let p, q ∈ P ( X ) such that d p d q exists, and such that χ 2 ( p ; q ) &lt; + ∞ . Assume that f ∈ L 2 ( q ) Then | E p f | &lt; + ∞ .

Proof. By assumption, f ∈ L 2 ( q ) . Moreover, χ 2 ( p, q ) &lt; + ∞ , and thus we have d p d q -1 ∈ L 2 ( q ) . Thus, the inner product is finite, and

<!-- formula-not-decoded -->

Lemma C.6. For all ψ ∈ Ψ , for all m ≥ 1 , and for all k ≥ 1 , we have:

<!-- formula-not-decoded -->

Proof. By analycity of the log partition function ψ ↦-→ log Z ( ψ ) , we have ∫ ∥ ϕ ∥ k d p ψ ( x ) &lt; + ∞ for all ψ ∈ Ψ , and thus, the function x ↦-→ ∥ ϕ ∥ k ( x ) ∈ L 2 ( p ψ ) for all ψ . Consequently, P m ψ ∥ ϕ ∥ k ∈ L 2 ( p ψ ) . We can apply Lemma C.5 to P m ψ ∥ ϕ ∥ k to obtain E ψ ⋆ P m ∥ ϕ ∥ k &lt; + ∞ for all k ≥ 1 and for all m ≥ 0 . As a by-product, we obtain P m ψ ∥ ϕ ∥ k ∈ L 2 ( p ψ ⋆ ) , and thus so ∥ P m ϕ ∥ k .

The following lemma is used multiple time in our analysis.

Lemma C.7. Assume A3. Let q be a positive integer. Let f := ( f 1 , . . . , f q ) such that f k ∈ { ϕ i } p i =1 ∪ { ϕ i ϕ j } p i,j =1 for k ∈ [ q ] . Then, for all ψ ∈ Ψ , we have

<!-- formula-not-decoded -->

Proof. Let us note first that

<!-- formula-not-decoded -->

Here, we used the fact that P m ψ admits p ψ as an invariant measure in ( a ) [34, Eq. (1.2.2)], the Cauchy-Schwarz inequality in ( b )

## C.3 Miscellaneous

Lemma C.8. Let f : Ψ → ❘ p be a differentiable function in the interior of Ψ ⊆ ❘ p . For ψ ∈ Ψ , define σ min ( ψ ) := inf θ ∈ Ψ , ∥ θ ∥ =1 θ ⊤ ∇ f ( ψ ) θ and σ max ( ψ ) := sup θ ∈ Ψ , ∥ θ ∥ =1 θ ⊤ ∇ f ( ψ ) θ with respect to the Jacobian matrix ∇ f ( ψ ) . Then for any ψ 1 , ψ 2 ∈ Ψ , we have that

<!-- formula-not-decoded -->

Proof of Lemma C.8. By the mean value theorem, there exists some a ∈ (0 , 1) such that

<!-- formula-not-decoded -->

Plugging in the definition of σ max gives the desired upper bound and similarly σ min implies the lower bound.

## D Proofs for Online CD

## D.1 Auxiliary Lemmas for Online CD

We recall the following notations used in the next lemmas, namely σ ψ := E p ψ ∥ ϕ -E p ψ ϕ ∥ 2 , σ ⋆ := σ ψ ⋆ and σ := sup ψ ∈ Ψ σ ψ .

We now provide two intermediary lemmas necessary to analyze the impact of variance in the CD gradient. The strategy is similar in both of them: we change the integration from p ψ ⋆ to p ψ to obtain contraction, at the cost of an additional term scaling with C χ ∥ ψ -ψ ⋆ ∥ . We obtain exact constants that we choose to describe in terms of the smoothness parameters of the problem, e.g. the k th derivatives of the log partition function log Z , which, for k ≥ 2 , equals the k th derivative of the negative cross-entropy model w.r.t p ψ ⋆ .

Second Moment convergence The following lemmas guarantee the second moment of a sample from k m ψ p ψ ⋆ approaches the second moment of a sample from the target distribution p ψ .

Lemma D.1. Under A1, A2 and A3, for all ψ ∈ Ψ , we have:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Proof. Applying Lemma C.7 to each f i := ϕ 2 i , we have

<!-- formula-not-decoded -->

We map the two moments to derivatives of log Z ( ψ ) , since the k th derivative of log Z ( ψ ) is the k th cumulant. It can be shown, using the multivariate moment to cumulant mapping, that

<!-- formula-not-decoded -->

where ∂ k i log Z ( ψ ) denotes the k th derivative of log Z with respect to ψ i . On the other hand,

<!-- formula-not-decoded -->

implying

<!-- formula-not-decoded -->

The result follows by summing over i , since:

<!-- formula-not-decoded -->

Note that ∥ log Z ∥ 1 , ∞ is finite since Ψ is compact and log Z is analytic.

Squared First Moment convergence The next lemma provides convergence (in squared absolute value) of the first moment of the m -iterated Markov kernel k m ψ .

Lemma D.2. Under A1, A2 and A3, for all ψ ∈ Ψ , we have

<!-- formula-not-decoded -->

where and

<!-- formula-not-decoded -->

Proof. We have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

In ( a ) , we used the restricted spectral gap Assumption A3 for ∆ 1 , 1 , and the Cauchy-Schwarz inequality combined with Assumption A2 for ∆ 1 , 2 . In ( b ) , we used the Cauchy-Schwarz once again, and in ( c ) we used the fact that P m ψ is a contraction in L 6 ( p ψ ) and another invocation of the spectral gap assumption A3 As an aside, note that a simpler result can be obtained by making regularity assumption on the mapping ψ ↦-→ P m ψ . Assuming that ψ ↦-→ P m ψ ( x ) is uniformly L m -Lipschitz across x ∈ X for instance (as done in [21, Assumption 5]), the second term ∆ 1 , 2 of ∆ 1 could have been handled using

<!-- formula-not-decoded -->

Although this result does not require possibly large constants related to sixth-order moments, it is less tight in the sense that it does not go to 0 as m →∞ . Back to the main proof, and ∆ 2 in particular. Applying Lemma C.7 to f := ϕ , we have

<!-- formula-not-decoded -->

Putting everything together, we have:

<!-- formula-not-decoded -->

Summing over i , we obtain

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Similarly to the previous lemma, one can upper bound E ( ϕ i -E p ψ ϕ i ) 6 using the centered moment to cumulant formula:

<!-- formula-not-decoded -->

To get a full description of ∥ log Z ∥ 2 , ∞ :

<!-- formula-not-decoded -->

## Lemma D.3. Under A1, A2 and A3, for all ψ ∈ Ψ , we have

<!-- formula-not-decoded -->

where and

<!-- formula-not-decoded -->

Proof. We have:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

In ( a ) , we used the restricted spectral gap Assumption A3 for ∆ 1 , 1 , and the Cauchy-Schwarz inequality combined with Assumption A2 for ∆ 1 , 2 . In ( b ) , we used the Cauchy-Schwarz once again, and in ( c ) we used the fact that P m ψ is a contraction in L 6 ( p ψ ) and another invocation of the spectral gap assumption A3 As an aside, note that a simpler result can be obtained by making regularity assumption on the mapping ψ ↦-→ P m ψ . Assuming that ψ ↦-→ P m ψ ( x ) is uniformly L m -Lipschitz across x ∈ X for instance (as done in [21, Assumption 5]), the second term ∆ 1 , 2 of ∆ 1 could have been handled using

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Although this result does not require possibly large constants related to sixth-order moments, it is less tight in the sense that it does not go to 0 as m →∞ . Back to the main proof, and ∆ 2 in particular. Applying Lemma C.7 to f := ϕ , we have

<!-- formula-not-decoded -->

Putting everything together, we have:

<!-- formula-not-decoded -->

Summing over i , we obtain

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Similarly to the previous lemma, one can upper bound E ( ϕ i -E p ψ ϕ i ) 6 using the centered moment to cumulant formula:

<!-- formula-not-decoded -->

To get a full description of ∥ log Z ∥ 2 , ∞ :

<!-- formula-not-decoded -->

We can now use the previous lemmas to obtain an expression on the second moment of the contrastive divergence gradient estimator, relating it to the one of the stochastic log-likelihood gradient estimator. Lemma D.4. Under A1, A2 and A3, we have:

<!-- formula-not-decoded -->

3 , ∞ 1 , ∞ 2 , ∞

Proof. We rely on the following decomposition:

<!-- formula-not-decoded -->

∆ 1 , 1 +∆ 1 , 2 form the differentiable stochastic gradient g t of Equation 6. Note that ∆ 1 , 1 is meanzero, and ∆ 2 is mean-zero conditionally on X t . Consequently, E ⟨ ∆ 2 , ∆ 3 ⟩ = E ⟨ ∆ 2 , ∆ 1 , 1 ⟩ = E ⟨ ∆ 1 , 1 , ∆ 1 , 2 ⟩ = E ⟨ ∆ 1 , 1 , ∆ 2 ⟩ = 0 , and the only mixed-terms that remain to be controlled are E ⟨ ∆ 1 , 1 , ∆ 3 ⟩ and E ⟨ ∆ 1 , 2 , ∆ 3 ⟩ . We first control the unmixed terms, and the simple ones first: we have E ∥ ∆ 1 , 1 ∥ 2 = σ 2 ⋆ , as well as E ∥ ∆ 1 , 2 ∥ 2 ≤ L 2 ∥ ψ -ψ ⋆ ∥ 2 . For ∆ 2 , we have:

<!-- formula-not-decoded -->

We can invoke Lemmas D.3 and D.1, which guarantee

<!-- formula-not-decoded -->

to obtain

<!-- formula-not-decoded -->

where ∥ log Z ∥ 3 , ∞ := 2 max( ∥ log Z ∥ 1 , ∞ , ∥ log Z ∥ 2 , ∞ ) .

For ∆ 3 , notice that ∆ 3 is precisely the term ∆ 1 in Lemma D.3, and we can thus bound it by

<!-- formula-not-decoded -->

Finally, we simply bound 2 E ⟨ ∆ 1 , 1 , ∆ 3 ⟩ by E ∥ ∆ 1 , 1 ∥ 2 + E ∥ ∆ 3 ∥ 2 , and 2 E ⟨ ∆ 1 , 2 , ∆ 3 ⟩ by E ∥ ∆ 1 , 2 ∥ 2 + E ∥ ∆ 3 ∥ 2 . Putting everything together, we have:

<!-- formula-not-decoded -->

## D.2 Proof of the SGD recursion (Lemma 3.1)

We are now ready to provide an SGD-style recursion for the expected squared distance to the optimum E [ ∥ ψ t -ψ ⋆ ∥ 2 ] .

Lemma (Restatement of Lemma 3.1) . Let ψ t be the iterates produced by Algorithm 1. Denote δ t = E [ ∥ ψ t -ψ ⋆ ∥ 2 ] , σ ⋆ = ( E p ψ⋆ ∥ ϕ -E p ψ⋆ ϕ ∥ 2 ) 1 / 2 , and σ t = ( E p ψt ∥ ϕ -E p ψt ϕ ∥ 2 ) 1 / 2 . Then, under Assumptions A1, A2 and A3, for all t ≥ 1 , we have:

<!-- formula-not-decoded -->

Proof. In this proof, we note ( F t ) t ≥ 0 , the increasing family of σ -algebras generated by the random variables ( X t ) t ≥ 0 ∼ p ψ ⋆ and the Markov chain samples ˜ X t m | X t , ψ t ∼ k m ψ t ( X t , · ) . We decompose the integrand of δ t as follows:

<!-- formula-not-decoded -->

The first term is (up to an averaging operation) the previous iterate. The middle term will ensure (provided m is large enough) contraction of the expected distance to the optimum. Finally, the third term can be described by Lemma D.4, and essentially behaves like the second moment of a log-likelihood stochastic gradient. Indeed, noting g ( ψ ) := -E p ψ⋆ ϕ + E p ψ ϕ the expectation of g t w.r.t x t (which is the gradient of the negative cross-entropy between p ψ and p ψ ⋆ ), we have:

<!-- formula-not-decoded -->

and applying Lemma C.7, we get that

<!-- formula-not-decoded -->

meaning

<!-- formula-not-decoded -->

On the other hand, by applying Lemma C.8 to g , we have:

<!-- formula-not-decoded -->

Combining the above results, we obtain:

<!-- formula-not-decoded -->

And the result follows by integrating over F t -1 .

## D.3 Proof of Online CD convergence

We now prove Theorem 3.2. The recursion of Lemma 3.1 is almost identifiable, up to a cross-term of second order, with the one of an SGD algorithm as presented in the setting of [22, Theorem 1]. To make the identification exact, we use the bound 4 α m/ 2 η 2 t ∥ log Z ∥ 3 , ∞ C χ δ 1 / 2 t -1 ≤ 2 α m/ 2 η 2 t δ t + 2 α m/ 2 η 2 t ∥ log Z ∥ 2 3 , ∞ C 2 χ , yielding the following recursion:

<!-- formula-not-decoded -->

where we used the fact that ˜ σ m,t ≤ σ . This recursion is of the same form as the one studied in [22, Equation 6, Theorem 1] given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can use the same unrolling strategy as theirs (the only condition required to proceed is that ˜ µ m &gt; ˜ L , which automatically holds since µ &lt; L ), and we obtain

<!-- formula-not-decoded -->

by identifying:

## D.4 Proof of online CD with averaging (Theorem 3.3)

We first restate the theorem in its complete form.

Theorem (Contrastive Divergence with Polyak-Ruppert averaging) . Let ( ψ t ) t ≥ 0 the sequence of iterates obtained by running the CD algorithm with a learning rate η t = Ct -β for β ∈ ( 1 2 , 1) . Define ¯ ψ n := 1 n ∑ n i =1 ψ i . Then, under the same assumptions as Theorem 3.2 we have, for all n ≥ 1 ,

<!-- formula-not-decoded -->

Where I ( ψ ) := Cov [ ϕ ] is the Fisher information matrix of the data distribution. Additionally, if

<!-- formula-not-decoded -->

⋆ p ψ⋆ .

Throughout the proof, we will denote by h n the standard online CD gradient defined in Equation 3:

<!-- formula-not-decoded -->

as well as

<!-- formula-not-decoded -->

We start by establishing some intermediate lemmas.

Lemma D.5. Under Assumptions A1, A2, A3, the online CD iterates produced by Algorithm 1 using η t = Ct -β for β ∈ ( 1 2 , 1) verify

<!-- formula-not-decoded -->

Proof. Let us note δ n = E [ ∥ ψ n -ψ ⋆ ∥ 2 ] . Summing the r.h.s of Theorem 3.2, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where A 3 is finite if β &lt; 1 , and A 3 = O ( n ) otherwise [22].

Lemma D.6. Under Assumptions A1, A2, A3, the online CD iterates produced by Algorithm 1 using η t = Ct -β for β ∈ ( 1 2 , 1) verify

<!-- formula-not-decoded -->

Proof. Let us note δ n = E [ ∥ ψ n -ψ ⋆ ∥ 2 ] . Applying √ x + y ≤ √ x + √ y to the r.h.s of Theorem 3.2, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where A 4 is finite if β &lt; 1 , and A 4 = O ( n ) otherwise [22].

Lemma D.7. Under Assumptions A1, A2, A3, the online CD iterates produced by Algorithm 1 using η t = Ct -β for β ∈ ( 1 2 , 1) verify

<!-- formula-not-decoded -->

Proof. The result follows from the fact that h t verifies

<!-- formula-not-decoded -->

A similar quantity was handled in the case of standard SGD [22, Theorem 3], and the only condition needed to reuse their steps is that ( ψ t ) t ≤ n satisfies an upper bound of the same form as the one [22, Theorem 1] derived. This is precisely the nature of our bound of ψ t estblished in Theorem 3.2, with ˜ µ m , ˜ σ m , ˜ L . Borrowing on their result, we have:

<!-- formula-not-decoded -->

where ˜ µ m , ˜ σ m and ˜ L are defined in 3.2, and

<!-- formula-not-decoded -->

Lemma D.8. Under Assumptions A1, A2, A3, the online CD iterates produced by Algorithm 1 using η t = Ct -β for β ∈ ( 1 2 , 1) verify

<!-- formula-not-decoded -->

Proof. We proceed as in the proof of [22, Theorem 3], first establishing a recurrence for E [ ∥ ψ i -ψ ⋆ ∥ 4 ] , and then unrolling it. We have

<!-- formula-not-decoded -->

The second and fourth terms will be controlled using results from our previous sections. For simplicity, we don't attempt to relate the moments of ∥ h n ∥ 4 as precisely as before. Instead we use:

<!-- formula-not-decoded -->

And let us note τ = ( sup ψ ∈ Ψ E p ψ ∥ ϕ ∥ 8 ) 1 / 8 . We have, for k ≥ 4 ,

<!-- formula-not-decoded -->

Where we used the fact that P m ψ n -1 is a contraction, and ( E ∥ ϕ ∥ k ) 1 /k is an increasing function of k . On the other hand, we simply have E p ψ⋆ ∥ ϕ ( X n , · ) ∥ k ≤ τ k . Plugging this into the previous equation, we obtain

<!-- formula-not-decoded -->

To simplify the recursion, we use the four following inequalities:

<!-- formula-not-decoded -->

Injecting them in our recursion, we obtain:

<!-- formula-not-decoded -->

which, after further simplifications, yields

<!-- formula-not-decoded -->

where we defined ˜ τ 1 := 2(1 + C χ ) τ and ˜ L 1 := 2(1 + C χ ) τ + L . This recursion is of the form of the one studied in [22, Equation 32] (note that by design, ˜ L ≥ ˜ µ m .) The steps performed to bound E [ ∥ ψ i -ψ ⋆ ∥ 4 ] thus follow from their derivations, and we obtain:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and we have A (1) &lt; + ∞ if β &lt; 1 , and A (1) = O ( n ) otherwise.

Lemma D.9. For all ψ ∈ Ψ , we have:

<!-- formula-not-decoded -->

Proof. We have

<!-- formula-not-decoded -->

Looking at the second moment first, we have

<!-- formula-not-decoded -->

Applying lemma C.7 to the R d 2 -valued function f given by f ij := ϕ i ϕ j -E p ψ ϕ i ϕ j ,

<!-- formula-not-decoded -->

We now investigate the first moment. We have

<!-- formula-not-decoded -->

and thus, applying Lemma C.7 on ∆ 2 , 1 , we have:

<!-- formula-not-decoded -->

where we used that ∥ ∆ 2 , 1 ∆ ⊤ 2 , 1 ∥ F = ∥ ∆ 2 , 1 ∥ 2 . We can now combine our two matrix moment bounds to obtain

<!-- formula-not-decoded -->

Lemma D.10. Under Assumptions A1, A2, A3 it holds that

<!-- formula-not-decoded -->

where M := sup ψ ∈ Ψ ∥ ∥ ∇ 3 log Z ( ψ ) ∥ ∥ op( ∥·∥ F , ∥·∥ F ) &lt; + ∞ .

<!-- formula-not-decoded -->

Noting ∆ the l.h.s of Lemma D.10, we have, summing over [ n ] , and using Minkowski's inequality,

<!-- formula-not-decoded -->

Note that this step was made possible because we are looking at the square-root of the variance, which is unlike the recursion in Lemma 3.1. This allows to separate the terms and use fewer intermediaries than in the proof of Lemma 3.1.

Since both ∆ 1 ,n and ∆ 2 ,n are martingale differences with respect to the filtration F n -1 , the covariance terms vanish, and we have

<!-- formula-not-decoded -->

In ( a ) we used Lemma D.10, the cyclicity of the trace, tr( A ⊤ B ) ≤ ∥ AB ∥ F ≤ ∥ A ∥ F ∥ B ∥ F , and the fact that since Cov p ψ i -1 [ ϕ ( x i )] = ∇ 2 ψ L ( ψ i -1 ) , by analycity of L , there exists a constant M := sup ψ ∈ Ψ ∥ ∥ ∇ 3 log Z ( ψ ) ∥ ∥ op( ∥·∥ , ∥·∥ F ) such that

<!-- formula-not-decoded -->

In ( b ) we used Jensen's inequality to get E [ ∥ ψ i -1 -ψ ⋆ ∥ ] ≤ √ E [ ∥ ψ i -1 -ψ ⋆ ∥ 2 ] = δ 1 / 2 i -1 .

We are now ready to prove Theorem 3.3.

Proof of Theorem 3.3. It holds that:

<!-- formula-not-decoded -->

Applying on both sides: (a) a summation over i ∈ [ n ] , (b) a multiplication by f ′′ ( ψ ⋆ ) -1 , (c) √ E [ ∥ · ∥ 2 ] , and using Minkowski's inequality on the r.h.s, we obtain

<!-- formula-not-decoded -->

( i ) and ( ii ) have direct analogues in the proofs of prior work [22] on the convergence of (unbiased) SGD with Polyak-Ruppert averaging, and will be bounded similarly. ( iii ) captures the bias of the CD algorithm, while ( iv ) captures the variance.

Bounding ( i ) Using Lemma D.7, we have ( i ) = O ( n β 2 -1 ) .

Bounding ( ii ) Since log Z ( ψ ) is analytic, there exists some constant M ′ such that

<!-- formula-not-decoded -->

Thus, we have:

<!-- formula-not-decoded -->

where the second-to-last inequality used Minkowski's inequality, and the last applied Lemma D.8.

Bounding ( iii ) By Minkowski's inequality, we have:

<!-- formula-not-decoded -->

Moreover, using lemma C.7, we have:

<!-- formula-not-decoded -->

We thus obtain

<!-- formula-not-decoded -->

Recalling that δ i satisfies Theorem 3.2, we have that δ 1 2 n = O ( n -β/ 2 ) , and we thus have ∑ n i =1 δ 1 / 2 i = O ( n 1 -β 2 ) . By squaring the result of Lemma D.6, we have ∑ n i =1 δ 1 / 2 i = O ( n -1 -β 2 ) , and thus, we obtain that ( iii ) = O ( α m n -β/ 2 ) = O ( n -( β 2 + m | log α | log n ) ) .

Bounding ( iv ) We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Where in ( a ) , we used Lemma D.10 and in ( b ) , we used Lemma D.6 and D.5.

Final bound Putting everything together, we have that:

<!-- formula-not-decoded -->

If, furthermore, m&gt; (1 -β ) log n 2 | log α | , we have

<!-- formula-not-decoded -->

which concludes the proof.

## E L 2 approximation by auxiliary gradient updates

In this section, we consider different gradient update schemes starting from some random initialization θ init , and control the L 2 distance between the different updates and the deterministic target ψ ∗ ∈ Ψ .

Notation. Recall the notation that X 1 , . . . , X n i . i . d . ∼ p ψ ∗ , B = n/N and for ψ ∈ Ψ , K ψ ( x ) ∼ k m ψ ( x, · ) . We also write, for m ∈ ◆ ∪ {∞} ,

<!-- formula-not-decoded -->

Let θ init be some Ψ -valued random initialization that is possibly correlated with X 1 , . . . , X n . We capture the effect of correlation through the following quantities: For ϵ &gt; 0 and ν &gt; 2 , let

<!-- formula-not-decoded -->

We also consider the i.i.d. samples, drawn independently of X 1 , . . . , X n and on a given ψ ∈ Ψ , as

<!-- formula-not-decoded -->

For notational clarity, we shall use θ m,B to denote parameters arising from an one-step update, where the subscripts m,B represent performing the one-step update with lengthm Markov chains and with batch size B . This is to be distinguished from ψ t elsewhere in the text, which denotes the parameter from the actual multi-step CD algorithm and the subscript t denotes the t -th CD iterate.

Gradient update schemes. We consider five different updates. Let X ′ 1 be an i.i.d. copy of X 1 drawn independently of all other random variables. The SGD-with-replacement update is given by

<!-- formula-not-decoded -->

and S w is a uniformly drawn sizeB subset of [ n ] . The SGD-without-replacement update, after renormalizing the learning rate, is given by the N -fold function composition

<!-- formula-not-decoded -->

and S o j 's are disjoint sizeB random subsets of [ n ] , defined by ( S o 1 , . . . , S o N ) = π ([ n ]) for a uniformly drawn element π of the permutation group on n objects. The full-batch gradient update is given by

<!-- formula-not-decoded -->

The full-batch gradient update with an infinite-length Markov chain is given by

<!-- formula-not-decoded -->

The population gradient update with an infinite-length Markov chain is given by

<!-- formula-not-decoded -->

where we use the lowercase f to emphasize that f pop is a deterministic function.

The forthcoming results are summarized below:

<!-- formula-not-decoded -->

Lemma E.1. Let F n be the sigma algebra generated by { X i , K m i ; θ init ( X i ) | 1 ≤ i ≤ n } . Then

<!-- formula-not-decoded -->

Moreover, under A1 and A7, we have

<!-- formula-not-decoded -->

Proof of Lemma E.1. Write A := ( A 1 , . . . , A n ) , where

<!-- formula-not-decoded -->

Since S w is uniformly drawn from all sizeB subsets of [ n ] and independently of all other variables, we have that almost surely

<!-- formula-not-decoded -->

To prove the remaining bound, we note that the above relation implies θ SGDw m,B -θ GD m is zero-mean. By the law of total variance, we have

<!-- formula-not-decoded -->

To compute the covariance, recall that S w is a uniformly drawn sizeB subset of [ n ] with B = n/N . Let P N ([ n ]) be the collection of all partitions of [ n ] into N sizeB subsets. We can generate S w by the following two-step process:

- (i) Uniformly draw a partition P ′ = ( P ′ 1 , . . . , P ′ N ) from P N ([ n ]) ;
- (ii) Uniformly sample an index K from [ N ] and set S w = P ′ K .

Then we have, almost surely

<!-- formula-not-decoded -->

By noting that A i 's are exchangeable, we obtain

<!-- formula-not-decoded -->

In ( a ) , we have used a Cauchy-Schwarz inequality; in ( b ) , we have used A1 and A7. Finally we note that if B = n , θ SGDw m,B = θ GD m almost surely, which implies the desired bound.

Lemma E.2. Denote A n as the sigma algebra generated by θ init , X 1 , . . . , X n . Under A1, A2, A3 and A7, we have that for any ϵ &gt; 0 and ν &gt; 2 ,

<!-- formula-not-decoded -->

Proof of Lemma E.2. The main challenge arises from the possible correlation between θ init and X 1 , . . . , X n . First note that for any ϵ &gt; 0 , ν &gt; 2 and a real-valued random variable Y , by Hölder's inequality, we have

<!-- formula-not-decoded -->

Also note the useful inequality that for two real-valued random vectors (possibly correlated) V 1 , V 2 , we have

<!-- formula-not-decoded -->

Now to control the first quantity of interest, by using a triangle inequality, we have

<!-- formula-not-decoded -->

̸

where

<!-- formula-not-decoded -->

and X ′ 1 is an i.i.d. copy of X 1 and in particular independent of θ init . ∆ 1 is controlled via (15):

<!-- formula-not-decoded -->

In ( a ) , we have plugged in the definition of ∆ 1 and applied a Jensen's inequality with respect to the empirical average; in ( b ) , we have used A7 to bound the ν -th moment term as

<!-- formula-not-decoded -->

and recalled the definitions of ϑ init n,m and ε init n,m ; ν . On the other hand,

<!-- formula-not-decoded -->

In ( a ) , we have used that ( Kf )( x ) = ∫ K ( x, y ) f ( y ) dy ; in ( b ) , we have used that the Markov operator leaves the constant function invariant; in ( c ) , we used that K 1; θ init leaves p θ init invariant; in

( d ) , we denoted ( e l ) l ≤ d as the standard basis vectors of ❘ d and multiplied and divided by p θ init ( x ) ; in ( e ) , we have used a Cauchy-Schwarz inequality; in ( f ) , we have used the definition of the spectral gap α in A3; in ( g ) , we have used A1 and A2. Combining the bounds gives the first inequality that

<!-- formula-not-decoded -->

Now we handle the second quantity by conditioning on θ init and perform a bias-variance decomposition:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Note that the covariance terms separate because X θ init i is independent of K m i ; θ init ( X i ) conditioning on θ init . η 2 Q B is exactly the quantity controlled above, so it suffices to bound the variance term Q V . By explicitly computing the second covariance term while noting that X θ init 1 , . . . , X θ init n are conditionally i.i.d. given θ init and K m i ; θ init ( X i ) 's are conditionally independent across 1 ≤ i ≤ n given A n , we have

<!-- formula-not-decoded -->

where we have used A4, A7 and A1 in the last line. Combining the bounds, we obtain that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma E.3. Since both F GD ∞ and f pop involve infinite-length Markov chains, the initializations do not matter and we can decouple the stochasticity of X i and K i ; θ init . In particular,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

In the computations above, we have used the relation ∇ 2 θ log Z ( θ ) = Cov X ∼ p θ [ ϕ ( X )] and the assumption sup θ ∈ Ψ tr( ∇ 2 θ log Z ( θ )) = σ 2 from A1. This implies the desired bound.

Lemma E.4. Under A1, E ∥ θ pop -ψ ∗ ∥ 2 ≤ ( 1 -2 µη + L 2 η 2 ) E ∥ θ init -ψ ∗ ∥ 2 .

Proof of Lemma E.4. Recall that

<!-- formula-not-decoded -->

By construction, f pop is deterministic and f pop ( ψ ∗ ) = ψ ∗ . By plugging in the recursions and expanding the square, we get that

<!-- formula-not-decoded -->

In the last line, we have recalled inf ψ ∈ Ψ λ min ( ∇ 2 ψ log Z ( ψ )) = µ and sup θ ∈ Ψ λ max ( ∇ 2 ψ log Z ( ψ )) = L by A1 and applied Lemma C.8. Combining the coefficients gives the desired statement.

## F Proofs for offline SGD

We prove Theorem B.1 (which directly implies Theorem 4.3) and Theorem B.2 in this section. The key ingredient of both proofs is Lemma F.1 below, which provides an iterative error bound for the SGD-with-replacement scheme by combining different approximation bounds in Appendix E. Throughout this section, we denote δ SGDw t,j := E ∥ ∥ ψ SGDw t,j -ψ ∗ ∥ ∥ 2 .

Lemma F.1. Under A1, A2, A3, A4 and A7, we have that for 1 ≤ j ≤ N -1 ,

<!-- formula-not-decoded -->

where 1 -η t ( µ -α m σC χ -L 2 2 η t ) &gt; 0 .

Proof of Lemma F.1. We first remark that in view of Lemma C.4, the projection step in Algorithm 2 does not increase δ SGD w t,j , so it suffices to bound δ SGDw t,j as if projection is not performed on ψ sGDw t,j . To apply the results from Appendix E, we identify θ init = ψ SGDw t,j -1 and η = η t , which allows us to write ψ SGDw t,j = θ SGDw m,B . This also implies E ∥ θ init -ψ ∗ ∥ 2 = δ SGDw t,j -1 and ε init n,m ; ν ( ϵ ) ≤ ε SGDw ν ; n,m,t ( ϵ ) .

By adding and subtracting the auxiliary gradient updates followed by expanding the square, we obtain

<!-- formula-not-decoded -->

In ( a ) , we have expanded the square, used F n defined in Lemma E.1 and A n defined in Lemma E.2, and noted that θ pop -ψ ∗ is almost surely constant given θ init ; in ( b ) , we have applied Lemmas E.1, E.2, E.3 and E.4 under A1, A2, A3, A4 and A7, and used √ a + b ≤ √ a + √ b for a, b ≥ 0 ; in ( c ) , we have grouped the terms by powers of δ SGDw t,j -1 , bounded the indicator function from above by 1 and used B ≤ n to replace n in the denominator by B . While the computation above is complicated, we remark that the key steps are taking the conditional expectations for the crossterms in ( b ) , which gives us a tighter bound than directly applying a triangle inequality of the form E ∥ Y 1 + Y 2 ∥ 2 ≤ ( √ E ∥ Y 1 ∥ 2 + √ E ∥ Y 2 ∥ 2 ) . To further simplify the bounds, we seek to bound each

coefficient by a square, which yields

<!-- formula-not-decoded -->

This implies

<!-- formula-not-decoded -->

Now note that since µ ≤ L by the definitions in A1, 2 µη t -L 2 η 2 t ≤ 2 Lη t -L 2 η 2 t ≤ 1 . By using √ 1 -x ≤ 1 -x 2 for all x ≤ 1 , we get that

<!-- formula-not-decoded -->

Substituting this into the earlier bound and taking a square-root, we obtain the desired bound that

<!-- formula-not-decoded -->

Moreover, since L ≥ µ by definition from A1, we have

<!-- formula-not-decoded -->

which finishes the proof.

## F.1 Proof of Theorem B.1

Under A1, A2, A3, A4 and A7, Lemma F.1 implies

<!-- formula-not-decoded -->

for 1 ≤ t ≤ T and 1 ≤ j ≤ N , where we have used ˜ µ m = µ -α m σC χ , η t = Ct -β and

<!-- formula-not-decoded -->

By an induction on j = 1 , . . . , N , we have

<!-- formula-not-decoded -->

By noting that δ SGDw t, 0 = δ SGDw t -1 ,N almost surely for t ≥ 2 and using another induction on t = 1 , . . . , T ,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Note that by Lemma F.1, we also have that 1 -˜ µ m Ct -β + L 2 C 2 2 t -2 β &gt; 0 . This allows us to apply Lemma C.3 to obtain

<!-- formula-not-decoded -->

To control ∑ T t =1 Q t A t t -β , recall that β ∈ [0 , 1] , ˜ µ m &gt; 0 , L 2 C 2 2 &gt; 0 , and that

<!-- formula-not-decoded -->

We can now apply Lemma C.3: If β ̸∈ { 1 2 , 1 } , then

<!-- formula-not-decoded -->

If β = 1 2 , i.e. 2 β = 1 , we have

<!-- formula-not-decoded -->

If β = 1 , we get that

<!-- formula-not-decoded -->

Substituting the bounds into (18), we get the desired bound that √ δ SGDw T,N is upper bounded by

<!-- formula-not-decoded -->

## F.2 Proof of Theorem B.2

Recall that δ SGDo t,j := E ∥ ∥ ψ SGDo t,j -ψ ∗ ∥ ∥ 2 . To apply the results from Appendix E to ψ SGDo t,j , we condition on S o t , which in particular fixes S o t,j , the last sizeB subset of [ n ] chosen. We then identify θ init = ψ SGDo t,j -1 , η = η t and the dataset used as D o t,j := ( X i : i ∈ S o t,j ) , which allows us to identify ψ SGDo t,j = θ GD m , the full-batch gradient descent update using D o t,j . Meanwhile, we note that almost surely θ GD m = θ SGDw m,B , the SGD-with-replacement iterate that uses the full dataset D o t,j . Observe that the proof of Lemma F.1 holds with δ SGDo t,j -1 replaced by any random initialization θ init possibly correlated with X 1 , . . . , X n , which allows us to obtain

<!-- formula-not-decoded -->

Since the error recursion for δ SGDo t,j is identical to that of δ SGDw t,j in Lemma F.1, the proof of Theorem 4.3 follows directly, thereby yielding an identical result for δ SGDo T,N as with δ SGDw T,N in Theorem 4.3.

## G Proofs for tail probability bounds in offline SGD

We present the proofs for results in Appendix B.3 that control the tail probability terms ϑ SGDw ν ; n,m,T and ε SGDw ν ; n,m,T .

Proof of Lemma B.3. For δ &gt; 0 , let N δ be the δ -covering number of Ψ , which satisfies N δ ≤ ( r Ψ (1 + 2 /δ ) ) p (Example 5.8, [50]). Note also that by the Jensen's inequality applied to E [ · | X 1 ] and Assumption A6, there exist some σ m , ζ m &gt; 0 such that, for any z ∈ ❘ p with ∥ z ∥ ≤ ζ m ,

<!-- formula-not-decoded -->

Meanwhile under A5, recycling the proof of Lemma 3.1 of [21] shows that if C m δ/ √ p &lt; σ 2 m ζ m ,

<!-- formula-not-decoded -->

Since the probability above is an upper bound to ϑ SGDw n,m,T (3 C m δ ) , we get that if C m δ/ √ p &lt; σ 2 m ζ m ,

<!-- formula-not-decoded -->

Recall that by assumption, log n n &lt; σ 2 m ζ 2 m p + ν -2 . We now choose

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Taking a squareroot across and using √ a + b ≤ √ a + b for a, b &gt; 0 gives the desired bound. The limiting result follows by substituting this bound into Theorem B.1.

Proof of Lemma B.4. Let δ &gt; 0 and N δ be defined as in the proof of Lemma B.3, with N δ ≤ ( r Ψ (1 + 2 /δ ) ) p . Let ( ψ l ) N δ l =1 be the centers of the covering δ -balls. The covering-ball argument of the proof of Lemma 3.1 of [21] shows that

<!-- formula-not-decoded -->

By a Markov's inequality followed by the Burkholder's inequality applied to an average of i.i.d. summands (see e.g. [51] for ν &gt; 2 ), there exists a constant C ν &gt; 0 depending only on ν such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used assumption A4 in the last line. This implies

<!-- formula-not-decoded -->

Choosing δ = n -( ν -2) ν 2( ν 2 +( ν -2) p ) ≤ 1 , we get that

<!-- formula-not-decoded -->

Taking a squareroot across and using √ a + b ≤ √ a + √ b for a, b &gt; 0 gives the desired bound. The limiting result follows by substituting this bound into Theorem B.1.

Proof of Lemma B.5. By a Markov's inequality and a Jensen's inequality with respect to the empirical average, we have

<!-- formula-not-decoded -->

Meanwhile by a triangle inequality and the ergodicity assumption, we have

<!-- formula-not-decoded -->

This implies ϑ SGDw n,m,T ( ϵ ) ≤ 2 ˜ C K ˜ α m ϵ -1 , and therefore

<!-- formula-not-decoded -->

Choosing ϵ = ˜ α ( ν -2) m/ (3 ν -2) gives

<!-- formula-not-decoded -->

Taking a squareroot across and using √ a + b ≤ √ a + √ b for a, b &gt; 0 gives the desired bound. The limiting result follows by substituting this bound into Theorem B.1.

##