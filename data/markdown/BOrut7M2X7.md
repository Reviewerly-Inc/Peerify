## Divide-and-Conquer Posterior Sampling for Denoising Diffusion Priors

Yazid Janati ∗ , 1 Badr Moufad ∗ , 1

Alain Durmus 1

Eric Moulines 1 , 3

Jimmy Olsson 2

1 CMAP, Ecole polytechnique

2 KTH Royal Institute of Technology

3 MBZUAI

## Abstract

Recent advancements in solving Bayesian inverse problems have spotlighted denoising diffusion models (DDMs) as effective priors. Although these have great potential, DDM priors yield complex posterior distributions that are challenging to sample. Existing approaches to posterior sampling in this context address this problem either by retraining model-specific components, leading to stiff and cumbersome methods, or by introducing approximations with uncontrolled errors that affect the accuracy of the produced samples. We present an innovative framework, divide-and-conquer posterior sampling, which leverages the inherent structure of DDMs to construct a sequence of intermediate posteriors that guide the produced samples to the target posterior. Our method significantly reduces the approximation error associated with current techniques without the need for retraining. We demonstrate the versatility and effectiveness of our approach for a wide range of Bayesian inverse problems. The code is available at https://github.com/Badr-MOUFAD/dcps

## 1 Introduction

Many problems in machine learning can be formulated as inverse problems, such as superresolution, deblurring, and inpainting, to name but a few. They all have the same goal, namely to recover a signal of interest from an indirect observation. One line of research addresses these problems through the lens of the Bayesian framework by specifying two components: a prior distribution, which embodies the specification of the signal, and a likelihood that describes the law of the observation conditionally on the signal. Once these elements are specified, the inverse problem is solved by sampling from the posterior distribution, which, after including the observation, contains all available information about the signal and thus about its uncertainty as well [12]. The importance of the specification of the prior in solving Bayesian ill-posed inverse problems is paramount. In the last decade, the success of priors based on deep generative models has fundamentally changed the field of linear inverse problems [40, 55, 19, 36, 24]. Recently, denoising diffusion probabilistic models (DDMs) have received special attention. Thanks to their ability to learn complex and multimodal data distributions, DDM represent the state-of-the-art in many generative modeling tasks, e.g. image generation [45, 20, 50, 52, 15, 46, 49], super-resolution [43, 1], and inpainting [45, 11, 22].

Popular methods to sample from posterior distribution include Markov chain Monte Carlo (MCMC) and variational inference; see [53, 6] and the references therein. These methods are iterative schemes that require an explicit procedure to evaluate pointwise the prior distribution and often its (Stein) score function [21] in order to compute acceptance ratios and construct efficient proposals. While

* Equal contribution

Corresponding authors: { yazid.janati,badr.moufad } @polytechnique.edu

sampling from the DDM priors is straightforward, posterior sampling is usually challenging since the intractability of the posterior density and its score make them computationally prohibitive and thus invalidate all conventional simulation methods. Although approximations exist, their associated iterative sampling schemes can be computationally intensive and exhibit high sensitivity to the choice of hyperparameters; see e.g. [24].

This paper proposes the DIVIDE-AND-CONQUER POSTERIOR SAMPLER (DCPS), a novel approach to posterior sampling in Bayesian inverse problems with DDM priors. Thanks to the Markov property of the data-generating backward diffusion, the posterior can be expressed as the marginal distribution of a Feynman-Kac (FK) path measure [13], whose length corresponds to the number of diffusion steps and whose user-defined potentials serve to bias the dynamics of the data-generating backward diffusion to align with the likelihood of the observation. Besides, for a given choice of potentials, the FK path law becomes Markovian, making it possible to express the posterior as the marginal of a time-reversed inhomogeneous Markov chain.

This approach is tempting, yet, the backward Markov decomposition remains difficult to apply in practice as these specific potential functions are difficult to approximate, especially when the number of diffusion steps is large. We tackle this problem with a divide-and-conquer approach. More precisely, instead of targeting the given posterior by a single simulation run through the full backward decomposition, our proposed scheme targets backward a sequence ( π k ℓ ) L ℓ =0 of distributions along the path measure leading to the target posterior distribution (section 3). These distributions are induced by a sequence of increasingly complex potentials and converge to the target distribution. Starting with a sample from π k ℓ +1 , a draw from π k ℓ is formed by a combination of Langevin iterations and the simulation of an inhomogeneous Markov chain. In other words, π k ℓ is expressed as the final marginal distribution of a time-reversed inhomogeneous Markov chain of moderate length k ℓ +1 -k ℓ ∈ N ∗ with an initial distribution π ℓ k ℓ +1 . This chain, whose transition densities are intractable, is approximately sampled using Gaussian variational inference. The rationale behind our approach stems from the observation that the Gaussian approximation error can be reduced by shortening the length of the intermediate FK path measures ( i.e. , by increasing L ); a result that we show in Proposition A.1. We finally illustrate that our algorithm can provide high-quality solutions to Bayesian inverse problems involving a variety of datasets and tasks.

To sum up our contribution, we

- show that the existing approximations of the Markovian backward decomposition can be improved using a bridge-kernel smoothing technique
- design a novel divide-and-conquer sampling approach that enables efficient bias-reduced sampling from the posterior, and illustrate its performance on several Bayesian inverse problems including inpainting, outpainting, Poisson imaging, and JPEG dequantization,
- propose a new technique to efficiently generate approximate samples from the backward decomposition using Gaussian variational inference.

Notation. For ( m,n ) ∈ N 2 such that m &lt; n , we let J m,n K := { m,... , n } . We use N( x ; µ, Σ) to denote the density at x of a Gaussian distribution with mean µ and covariance matrix Σ . I d is the d -dimensional identity matrix and δ a denotes the Dirac mass at a . W 2 denotes the Wasserstein distance of order 2. We use uppercase for random variables and lowercase for their realizations.

## 2 Posterior sampling with DDM prior

DDM priors. We provide a brief overview of DDMs [45, 50, 20]. Suppose we can access an empirical sample from some data distribution p data defined on R d x . For n ∈ N large enough and k ∈ J 0 , n K , define the distribution q k ( x k ) := ∫ p data ( x 0 ) q k | 0 ( x k | x 0 )d x 0 with q k | 0 ( x k | x 0 ) := N( x k ; √ α k x 0 , (1 -α k ) I d x ) , where ( α k ) n k =0 is a decreasing sequence with α 0 = 1 and α n approximately equals zero. The probability density q k corresponds to the marginal distribution at time k of an auto-regressive process on R d x given by X k +1 = √ α k +1 /α k X k + √ 1 -α k +1 /α k ϵ k +1 , with X 0 ∼ p data and ( ϵ k ) n k =0 being a sequence of i.i.d. d x -dimensional standard Gaussians.

DDMs leverage parametric approximations ˆ x θ 0 | k of the mappings x k ↦→ ∫ x 0 q 0 | k ( x 0 | x k )d x 0 , where q 0 | k ( x 0 | x k ) ∝ p data ( x 0 ) q k | 0 ( x k | x 0 ) is the conditional distribution of X 0 given X k = x k . Each ˆ x θ 0 | k

is defined as ˆ x θ 0 | k ( x k ) := ( x k - √ 1 -α k ˆ ϵ θ k ( x k )) / √ α k , where ˆ ϵ θ k is a noise predictor network trained by minimizing a denoising objective; see [46, Eq. (5)] and Appendix A for details. Following [15, Section 4.2], ˆ ϵ θ k also provides an estimate of the score ∇ log q k ( x k ) given by ˆ s θ k ( x k ) := -( x k -√ α k ˆ x θ 0 | k ( x k ) ) / (1 -α k ) . Wedenote by θ ⋆ the minimizer of the denoising objective. Having access to θ ⋆ , we can define a generative model for p data by adopting the denoising diffusion probabilistic model (DDPM) framework of [20]. As long as n is large enough, q n can be confused with a multivariate standard Gaussian. Define the bridge kernel q k | 0 ,k +1 ( x k | x 0 , x k +1 ) ∝ q k | 0 ( x k | x 0 ) q k +1 | k ( x k +1 | x k ) which is a Gaussian distribution with mean µ k | 0 ,k +1 ( x 0 , x k +1 ) and diagonal covariance σ 2 k | k +1 I d x defined in Appendix A.1. Define the generative model for p data as

<!-- formula-not-decoded -->

where for every k ∈ J 1 , n -1 K , the backward transitions are

<!-- formula-not-decoded -->

with p θ ⋆ 0 | 1 ( ·| x 1 ) := δ ˆ x θ ⋆ 0 | 1 ( x 1 ) and p n ( x n ) = N( x n ; 0 , I d x ) . In the following, we assume that we have access to a pre-trained DDM and omit the superscript θ ⋆ from the notation, writing simply p and ˆ x 0 | k when referring to the generative model and the denoiser, respectively. In addition, we denote by p k the k -th marginal of p 0: n and write, for all ( ℓ, m ) ∈ J 0 , n K 2 such that ℓ &lt; m , p ℓ | m ( x ℓ | x m ) := ∏ m -1 k = ℓ p k | k +1 ( x k | x k +1 ) .

Posterior sampling. Let g 0 be a nonnegative function on R d x . When solving Bayesian inverse problems, g 0 is taken as the likelihood of the signal given the observation specified using the forward model (see the next section). Our objective is to sample from the posterior distribution

<!-- formula-not-decoded -->

where Z := ∫ g 0 ( x 0 ) p 0 ( x 0 )d x 0 is the normalizing constant and the prior p 0 is the marginal of (2.1) w.r.t. x 0 , in which case the posterior (2.3) can be expressed as

<!-- formula-not-decoded -->

Thus, Equation (2.3) can be interpreted as the marginal of a time-reversed FK (Feynman-Kac) model with a non-trivial potential only for k = 0 ; see [13] for a comprehensive introduction to FK models. In this work, we twist, without modifying the law of the FK model, the backward transitions p k | k +1 by artificial positive potentials ( g k ) n k =0 , each being a function on R d x , and write

<!-- formula-not-decoded -->

This allows the posterior of interest to be expressed as the time-zero marginal of an FK model with initial distribution p n , Markov transition kernels ( p k | k +1 ) n -1 k =0 , and ( g k ) n k =0 .

Recent works that aim to sample from the posterior (2.3) generally employ the FK representation (2.4). These studies, however, adopt varying auxiliary potentials [10, 47, 60, 4, 54, 59]. FK models can be effectively sampled using sequential Monte Carlo (SMC) methods; see, e.g. , [13, 9]. SMC methods sequentially propagate weighted samples, whose associated weighted empirical distributions target the flow of the FK marginal distributions. The effectiveness of this technique depends heavily on the choice of intermediate potentials ( g k ) n k =1 , as discussed in [54, 59, 7, 16]. However, SMC methods require a number of samples proportional and often exponential in the dimensionality of the problems hence limiting their application in these setups due to the resulting probabitive memory cost [2]. On the other hand, reducing the number of samples makes them vulnerable to mode collapse.

In the following, we will focus on a particular choice of potential functions ( g k ) n k =1 for which the posterior π 0 can be expressed as the time-zero marginal distribution of a time-reversed Markov chain. The transition densities of this chain are obtained by twisting the transition densities of the generative model with the considered potential functions. More precisely, define, for all k ,

the potentials g ⋆ k ( x k ) := ∫ g 0 ( x 0 ) p 0 | k ( x 0 | x k ) d x 0 . Note that these potentials satisfy the recursion g ⋆ k +1 ( x k +1 ) = ∫ g ⋆ k ( x k ) p k | k +1 ( x k | x k +1 ) d x k . Builing upon that, define the Markov transitions

<!-- formula-not-decoded -->

allowing the posterior (2.4) to be rewritten as

<!-- formula-not-decoded -->

In other words, the distribution π 0 is the time-zero marginal of a Markov model with transition densities ( π k | k +1 ) 0 k = n -1 and initial distribution π n . According to this decomposition, a sample X ⋆ 0 from the posterior (2.3) can be obtained by sampling X ⋆ n ∼ π n and then, recursively sampling X ⋆ k ∼ π k | k +1 ( ·| X ⋆ k +1 ) from k = n -1 till k = 0 . In practice, however, neither the Markov transition densities π k | k +1 nor the probability density function π n are tractable. The main challenge in estimating π k | k +1 stems essentially from the intractability of the potential g ⋆ k ( x k ) as it involves computing an expectation under the high-cost sampling distribution p 0 | k ( ·| x k ) .

Recent works have focused on developing tractable approximations of p 0 | k ( ·| x k ) . For the Diffusion Posterior Sampling (DPS) algorithm [10], the point mass approximation δ ˆ x 0 | k ( x k ) of p 0 | k ( ·| x k ) results in the estimate ∇ x k log g 0 (ˆ x 0 | k ( x k )) of ∇ x k log g ⋆ k ( x k ) . Then, given a sample X k +1 , an approximate sample X k from π k | k +1 ( ·| X k +1 ) is obtained by first sampling ˜ X k ∼ p k | k +1 ( ·| X k +1 ) and then setting

<!-- formula-not-decoded -->

where ζ &gt; 0 is a tuning parameter. As noted in [48, 7, 4], the DPS updates (2.7) do not lead to an accurate approximation of the posterior π 0 even in the simplest examples; see also Section 4. Alternatively, [47] proposed the Pseudoinverse-Guided Diffusion Model ( Π GDM), which uses a Gaussian approximation of p 0 | k ( ·| x k ) with mean ˆ x 0 | k ( x k ) and diagonal covariance matrix set to (1 -α k ) I d x , which corresponds to the covariance of q 0 | k ( ·| x k ) if p data had been a standard Gaussian; see [47, Appendix 1.3]. More recently, [17, 4] proposed to approximate the exact KL projection of p 0 | k ( x 0 | x k ) onto the space of Gaussian distributions by noting that both its mean and covariance matrix can be estimated using ˆ x 0 | k ( x k ) and its Jacobian matrix. We discuss in more depth the related works in Appendix B.

## 3 The DCPS algorithm

Smoothing the DPS approximation. The bias of the DPS updates (2.7) stems from the point mass approximation of the conditional distribution p 0 | k ( ·| x k ) . This approximation becomes more accurate as k tends to zero and is crude otherwise. We aim here to mitigate the resulting approximation errors. A core result that we leverage in this paper is that for any ( k, ℓ ) ∈ J 0 , n K 2 such that ℓ &lt; k , we can construct an estimate ˆ p ℓ | k ( ·| x k ) of p ℓ | k ( ·| x k ) that bears a smaller approximation error than the estimate δ ˆ x 0 | k ( x k ) relatively to p 0 | k ( ·| x k ) . Formally, let ˆ p 0 | k ( ·| x k ) denote any approximation of p 0 | k ( ·| x k ) , such as that of the DPS or Π GDM, and define the approximation of p ℓ | k ( ·| x k )

<!-- formula-not-decoded -->

where q ℓ | 0 ,k ( x ℓ | x 0 , x k ) is defined in (A.4). We then have the following result.

Proposition 3.1 (informal) . Let k ∈ J 1 , n K . For all ℓ ∈ J 0 , k -1 K and x k ∈ R d x ,

<!-- formula-not-decoded -->

The proof is postponed to Appendix A.3. Note that the ratio in the right-hand-side of (3.2) is less than 1 and decreases as ℓ increases. As an illustration, using the DPS approximation of p 0 | k ( ·| x k ) , we find that ˆ p ℓ | k ( x ℓ | x k ) = q ℓ | 0 ,k ( x ℓ | ˆ x 0 | k ( x k ) , x k ) improves upon DPS in terms of approximation error.

This observation prompts to consider DPS-like approximations on shorter time intervals; instead of approximating expectations under p 0 | k ( ·| x k ) , such as the potential g ⋆ k ( x k ) , we should transform our initial sampling problem so that we only have to estimate expectations under p ℓ | k ( ·| x k ) for any ℓ such that the difference k -ℓ is small. This motivates the blocking approach introduced next.

Intermediate posteriors. We approach the original problem of sampling from π 0 via a series of simpler, intermediate posterior sampling problems of increasing difficulty. More precisely, let us consider the intermediate posteriors defined as

<!-- formula-not-decoded -->

where ( g k ℓ ) L ℓ =1 are potential functions designed by the user and ( k ℓ ) L ℓ =0 is an increasing sequence in J 0 , n K such that k 0 = 0 and k L = n . Here, L is typically much smaller than n . To obtain an approximate sample from π 0 = π k 0 , the DCPS algorithm recursively uses an approximate sample X k ℓ +1 from π k ℓ +1 to obtain an approximate sample X k ℓ from π k ℓ . Indeed, mirroring (2.6) it holds

<!-- formula-not-decoded -->

where for m ∈ k ℓ , k ℓ +1 -1 ,

<!-- formula-not-decoded -->

and for m ∈ J k ℓ +1 , k ℓ +1 ,

<!-- formula-not-decoded -->

We emphasize that the initial distribution π ℓ k ℓ +1 in (3.4) is different from the posterior π k ℓ +1 as the former involves the user-defined potential whereas the latter the intractable one. The main advantage of our approach lies in the fact that, unlike the potentials in the transition densities (2.5), which involve expectations under p 0 | k ( ·| x k ) , the potentials (3.5) are given by expectations under the distributions p k ℓ | m ( ·| x m ) , which are easier to approximate in the light of Proposition 3.1. In the sequel, we use this approximation for the estimation of the potentials (3.5); this yields approximate potentials

<!-- formula-not-decoded -->

which serve as a substitute for the intractable g ℓ,⋆ m . Let us now summarize how our algorithm works. Starting from a sample X k ℓ +1 , which is approximately distributed according to π k ℓ +1 , the next sample X k ℓ is generated in the next two steps:

1. Perform Langevin Monte Carlo steps initialized at X k ℓ +1 and targeting π ℓ k ℓ +1 , yielding X ℓ k ℓ +1 .
2. Simulate a Markov chain ( X j ) k ℓ j = k ℓ +1 initialized with X k ℓ +1 = X ℓ k ℓ +1 and whose transition from X j +1 to X j is the minimizer of

<!-- formula-not-decoded -->

where λ φ j | j +1 is a mean-field Gaussian approximation with parameters φ := (ˆ µ, ˆ σ ) ∈ R d x × R d x &gt; 0 . X j is drawn from λ φ j ( X j +1 ) j | j +1 ( ·| X j +1 ) , where φ j ( X j +1 ) is a minimizer of the proxy of (3.7).

In the following, we elaborate more on Step 1 and Step 2 and discuss the choice of the intermediate potentials. The pseudo-code of the DCPS algorithm is in Algorithm 1.

Sampling the initial distribution. In order to perform Step 1 , we use the discretized Langevin dynamics [38] with the estimate ∇ log ˆ g ℓ,⋆ k ℓ +1 + ˆ s k ℓ +1 of the score ∇ log π ℓ k ℓ +1 . This estimate results from the use of ˆ s k ℓ +1 as an approximation of ∇ log p k ℓ +1 in combination with the approximate potential (3.6). We then obtain the approximate sample X ℓ k ℓ +1 of π k ℓ +1 by running M steps of the tamed unadjusted Langevin (TULA) scheme [5]; see Algorithm 1. Here, the intractability of the involved densities hinder the usage of the Metropolis-Hastings corrections to reduce the inherent bias of the Langevin algorithm.

Sampling the transitions. We now turn to Step 2 . Given X j +1 , we optimize the following estimate of Equation (3.7), where we simply replace g ℓ,⋆ j by the approximation (3.6):

<!-- formula-not-decoded -->

Letting λ φ j | j +1 ( x j | x j +1 ) = N( x j ; ˆ µ j , diag(e ˆ υ j )) , where the variational parameters ˆ µ j , ˆ υ j are in R d x , the previous estimate yields the objective

<!-- formula-not-decoded -->

where Z is d x -dimensional standard Gaussian and µ j | j +1 ( x j +1 ) is the mean of (2.2). Note here that we have used the reparameterization trick [26] and the closed-form expression of the KL divergence between two multivariate Gaussian distributions. We optimize the previous objective using a few steps of SGD by estimating the first term on the r.h.s. with a single sample as in [26]. For each j ∈ J k ℓ , k ℓ +1 -1 K , we use µ j | j +1 and log σ 2 j | j +1 as initialization for ˆ µ j and ˆ υ j .

Intermediate potentials. Here, we give general guidelines to choose the user-defined potentials ( g k ℓ ) L ℓ =1 . Our design choice is to rescale the input and then anneal the initial potential g 0 . Therefore, we suggest where γ k ℓ , β k ℓ &gt; 0 are tunable paramerters. This design choice is inspired from the tempering sampling scheme [33] which uses the principle of progressively moving an intial distribution to the targeted one. We provide some examples in the case of Bayesian inverse problems where the unobserved signal and the observation are modelled jointly as a realization of ( X,Y ) ∼ p ( y | x ) p 0 ( x ) , where p ( y | x ) is the conditional density of Y given X = x . In this case, the posterior π 0 of X given Y = y is given by (2.3) with g 0 ( x ) = p ( y | x ) .

<!-- formula-not-decoded -->

Linear inverse problems with Gaussian noise. In this case, g 0 ( x ) = N( y ; Ax,σ 2 y I d y ) , where A ∈ R d y × d x . Popular applications in image processing include super-resolution, inpainting, outpainting, and deblurring. We use (3.9) with ( β k ℓ , γ k ℓ ) = ( √ α k ℓ , α k ℓ ) ,

<!-- formula-not-decoded -->

which corresponds to the likelihood of x given the pseudo observation √ α k ℓ y under the same linear observation model that defines g 0 . This choice of g k ℓ enables exact computation of (3.6) and allows information on the observation y to be taken into account early in the denoising process.

Low-count (or shot-noise) Poisson denoising. In a Poisson model for an image, the grey levels of the image pixels are modelled as Poisson-distributed random variables. More specifically, let A ∈ R d y × d x be a matrix with nonnegative entries and x ∈ [0 , 255] C × H × W , where C is the number of channels and H the height and W the width. For every i ∈ J 1 , d y K , Y i is Poissondistributed with mean ( Ax ) i , and the likelihood of x given the observation is therefore given by x ↦→ ∏ d y j =1 ( λAx ) y j j e -( λAx ) j /y j ! where λ &gt; 0 is the rate. Following [10] we consider as likelihood its normal approximation, i.e. g 0 = ∏ d y j =1 N( y j ; λ ( Ax ) j , y j ) . This model is relevant for many tasks such as low-count photon imaging and computed tomography (CT) reconstruction [35, 39, 31]. We use (3.9) with β k ℓ = γ k ℓ = √ α :

JPEG dequantization. JPEG [57] is a ubiquitous method for lossy compression of images. Use h q to denote the JPEG encoding function with quality factor q ∈ J 0 , 100 K , where a small q is associated with high compression. Denote by h † q the JPEG decoding function that returns an image in RGB space with a certain loss of detail, depending on the degree of compression q , compared to the original image. Since we require the potential to be differentiable almost everywhere, we use the differentiable approximation of JPEG developed in [44], which replaces the rounding function used in the quantization matrix with a differentiable approximation that has non-zero derivatives almost everywhere. In this case, g 0 ( x ) = N( h † q ( y ); h † q ( h q ( x )) , σ 2 y I d y ) , where y is in YCbCr space. Combining this with Equation (3.9) with ( β k ℓ , γ k ℓ ) = ( α k ℓ , α k ℓ ) and assuming that the composition h † q ◦ h q is a homogenious map, the intermediate potentials are g k ℓ ( x ) = N( √ α k ℓ h † q ( y ); h † q ( h q ( x )) , σ 2 y I d x ) .

<!-- formula-not-decoded -->

DCPS

DDRM

DPS

## 4 Experiments

Figure 1: First two dimensions of samples (in red) from each algorithm on the 25 component Gaussian mixture posterior sampling problem with ( d x , d y ) = (100 , 1) . The true posterior samples are given in blue.

<!-- image -->

In this section, we demonstrate the performance of DCPS and compare it with DPS [10], Π GDM [47], DDRM [24], REDDIFF [32], and MCGDIFF [7] on several Bayesian inverse problems. We also benchmark our algorithm against DIFFPIR [62], DDNM [58], FPS [16], and SDA [42] but we defer the results to the Appendix C.5.

First, we consider a simple toy experiment in which the posterior distribution is available in closed form. Next, we apply our algorithm to superresolution (SR 4 × and 16 × ), inpainting and outpainting tasks with Gaussian and Poisson noise, and JPEG dequantization. For these imaging experiments, we use the FFHQ256 [23] and ImageNet256 [14] datasets and the publicly available pre-trained models of [8] and [15]. Finally, we benchmark our method on a trajectory inpainting task using the pedestrian dataset UCY for which we have trained a Diffusion model. All details can be found in Appendix C.1.

Gaussian mixture. We first evaluate the accuracy of DCPS on a linear inverse problem with a Gaussian mixture (GM) prior, for which the posterior can be explicitly computed: it is also a Gaussian mixture whose means, covariance matrices, and weights are in a closed form; see Appendix C.2. In this case, the predictor ˆ x θ ∗ 0 | k is available in a closed form;

Table 1: 95% confidence interval for the SW on the GM experiment.

|          | d x = 10 ,d y = 1   | d x = 100 ,d y = 1   |
|----------|----------|----------|
| DCPS 50  | 2 . 91 ± 0 . 74     | 4 . 04 ± 1 . 00      |
| DCPS 500 | 2 . 19 ± 0 . 68     | 3 . 29 ± 0 . 95      |
| DPS      | 5 . 80 ± 0 . 75     | 5 . 68 ± 0 . 73      |
| DDRM     | 3 . 77 ± 0 . 96     | 5 . 70 ± 0 . 78      |
| Π GDM    | 4 . 23 ± 0 . 90     | 4 . 61 ± 0 . 68      |
| REDDIFF  | 6 . 36 ± 1 . 27     | 7 . 47 ± 0 . 87      |
| MCGDIFF  | 2 . 28 ± 0 . 75     | 2 . 83 ± 0 . 71      |

gradient steps, respectively, and compared two configurations, denoted by DCPS 50 and DCPS 500 , of the algorithm with M = 50 and M = 500 Langevin steps, respectively. See Algorithm 1. The results are reported in Table 1. It is worthwhile to note that DCPS outperforms all baselines except for MCGDIFF. However, by increasing the number of Langevin steps, its performance closely matches that of MCGDIFF.

see Appendix C.2 for more details. We consider a Gaussian mixture prior with 25 components in dimensions d x = 10 and d x = 100 . The potential is g 0 ( x ) = N( y ; Ax,σ 2 y I d y ) with d y = 1 and A is a 1 × d x vector. The results are averaged over 30 randomly generated replicates of the measurement model ( y, A, σ 2 y ) and the mixture weights. Then, for each pair of prior distribution and measurement model, we generate N s = 2000 samples with each algorithm and compare them with N s samples from the true posterior distribution using the sliced Wasserstein (SW) distance. For DCPS, we used L = 3 blocks and K = 2

Imaging experiment. Table 2 reports the results for the linear inverse problems with Gaussian noise with two noise variance levels σ y = 0 . 05 and σ y = 0 . 3 , Table 3 for the JPEG dequantization problem with σ y = 10 -3 , QF ∈ { 2 , 8 } , and Table 6 for the Poisson denoising task with rate λ = 0 . 1 . For all tasks and datasets, we use the same parameters for DCPS and therefore do not perform any task or dataset-specific tuning. We use L = 3 , K = 2 gradient steps, and M = 5 Langevin steps. To ensure a fair comparison with DPS and Π GDM we use 300 DDPM steps for DCPS and 1000 steps for both DPS and Π GDM, which ensures that all the algorithms have the same runtime and memory footprint; see Table 4. For MCGDIFF, which has a large memory requirement, we use N = 32 particles in the SMC sampling step and then randomly draw one sample from the resulting particle approximation of the posterior. Finally, for DDRM we use 200 diffusion steps and for REDDIFF we use 1000 gradient steps and the parameters recommended in the original paper. We provide the implementation details for all algorithms in Appendix C.1.

NIGDM

REDDIFF

MCGDIFF

Observation

DCPS

DDRM

DPS

NGDM

REDDIFF MCGDIFF

Observation

DCPS

DDRM

Catalyst

MCGDIFF

Catalyst

Figure 2: Sample images for inpainting with center, half, expand masks and for Super Resolution with 4 × and 16 × factors. On the left: FFHQ dataset and on the right ImageNet dataset.

<!-- image -->

For the JPEG dequantization task, we use σ y = 10 -3 and λ = 0 . 1 . We only benchmark our method against Π GDM and REDDIFF, since MCGDIFF and DDRM do not handle non-linear inverse problems. We did not include DPS in our benchmark because we have not managed to find a suitable choice of hyperparameters to achieve reasonable results. Finally, for the Poissonshot noise case, we compare against DPS. We use the step size for super-resolution recommended in the original paper [see

Table 2: Mean LPIPS value on different tasks. Lower is better.

| Dataset / σ y   | Task    |   DCPS |   DDRM |   DPS |   Π GDM | REDDIFF   |   MCGDIFF |
|----------|---------|--------|--------|-------|---------|----------|----------|
| FFHQ / 0.05     | Half    |   0.2  |   0.25 |  0.24 |    0.26 | 0.28 0.12 |      0.36 |
| FFHQ / 0.05     | Center  |   0.05 |   0.06 |  0.07 |    0.19 |          |      0.24 |
| FFHQ / 0.05     | SR 4 ×  |   0.09 |   0.18 |  0.09 |    0.33 | 0.36      |      0.15 |
| FFHQ / 0.05     | SR 16 × |   0.23 |   0.36 |  0.24 |    0.44 | 0.51      |      0.32 |
| FFHQ / 0.3      | Half    |   0.25 |   0.3  |  0.31 |    0.64 | 0.76      |      0.8  |
| FFHQ / 0.3      | Center  |   0.1  |   0.13 |  0.11 |    0.62 | 0.75      |      0.55 |
| FFHQ / 0.3      | SR 4 ×  |   0.21 |   0.26 |  0.19 |    0.77 | 0.77      |      0.65 |
| FFHQ / 0.3      | SR 16 × |   0.35 |   0.41 |  0.43 |    0.64 | 0.74      |      0.52 |
| ImageNet / 0.05 | Half    |   0.35 |   0.4  |  0.44 |    0.38 | 0.44      |      0.83 |
| ImageNet / 0.05 | Center  |   0.18 |   0.14 |  0.31 |    0.29 | 0.22      |      0.45 |
| ImageNet / 0.05 | SR 4 ×  |   0.24 |   0.38 |  0.41 |    0.78 | 0.56      |      1.32 |
| ImageNet / 0.05 | SR 16 × |   0.44 |   0.72 |  0.5  |    0.6  | 0.83      |      1.33 |
| ImageNet / 0.3  | Half    |   0.4  |   0.46 |  0.48 |    0.82 | 0.76      |      0.86 |
| ImageNet / 0.3  | Center  |   0.24 |   0.25 |  0.4  |    0.68 | 0.71      |      0.47 |
| ImageNet / 0.3  | SR 4 ×  |   0.43 |   0.5  |  0.47 |    0.87 | 0.83      |      1.31 |
| ImageNet / 0.3  | SR 16 × |   0.72 |   0.77 |  0.57 |    0.72 | 0.92      |      0.67 |
| Average         |         |   0.28 |   0.35 |  0.32 |    0.57 | 0.60      |      0.67 |

10, Appendix D.1], and found, via a grid search, that the same value is also effective for the other tasks.

Evaluation. As shown in Table 2, DCPS outperforms the other baselines on 13 out of 16 tasks and has the best average performance. In particular, it compares favorably with Π GDM and DPS, its closest competitors, while exhibiting the same runtime and memory requirements; see Table 4, where we give the average runtime and memory usage for each algorithm. The memory consumption is measured by how many samples each algorithm can generate in parallel on

Table 3: Mean LPIPS value on JPEG dequantization.

| Dataset   | Task          | DCPS      | Π GDM     | REDDIFF   |
|----------|----------|----------|----------|----------|
| FFHQ      | QF = 2 QF = 8 | 0.20 0.08 | 0.37 0.15 | 0.32 0.18 |
| ImageNet  | QF = 2        | 0.44 0.24 | 0.93 0.95 |          |
|          | QF = 8        |          |          | 0.50 0.31 |

a single 48GB L40S NVIDIA GPU for the Diffusion model trained on FFHQ [15]. We emphasize that DCPS is more robust to larger noise levels than Π GDM and REDDIFF, as evidenced by the large increase in the LPIPS value for these algorithms in the case σ y = 0 . 3 . On the JPEG dequantization task (Table 3), DCPS also shows better performance than these algorithms and even more so for the high compression level (QF = 2 ). On the Poisson-shot noise tasks, DCPS outperforms DPS by a significant margin; see Table 6. Finally, we display various reconstructions obtained with each algorithm. More specifically, we have generated 4 samples each, with the same seed. Figure 2 displays the first sample and the remaining ones are deferred to Appendix D. For MCGDIFF we show 4 random samples of the same particle filter. Due to the collapse of the particle filter in very large dimensions [2], they are all similar. Surprisingly, the samples produced by DDRM and REDDIFF for the outpainting tasks also show striking similarities, although the samples have been drawn independently.

DPS

NGDM

REDDIFF

h° (y)

MCGDIFF

DCPS

NGDM

•

Reconstruction

DCPS (ours)

REDDIFF

•

Observation

Observation

DDRM

DPS

Confidence intervals

Figure 3: Left: JPEG dequantization with QF = 2 . Middle: Poisson denoising. Right: SR 4 × Poisson denoising.

<!-- image -->

Trajectory prediction. We evaluate our algorithm on the UCY dataset consisting of pedestrian trajectories, encoded as 2D time series with 20 time steps [27, 29, 18, 30]. We pre-train a trajectory model on this dataset and then use it for trajectory reconstruction tasks. The model architecture and implementation are detailed in Appendix C.4. We focus on the completion of trajectories where only a few timesteps are observed. The missing steps are filled in based on the observations and the pre-trained prior model, similar to the inpainting task in the previous section. We use MCGDIFF with 5000 particles to obtain approximate samples from the posterior. Indeed, as the dimension of the observation space is low ( d x = 40 ) and MCGDIFF is asymptotically exact as the number of particles tends to infinity, it yields an accurate approximation of the posterior; see [7, Proposition 2.1]. Then, we compute the ℓ 2 distance between the median, quantile 25 , and quantile 75 of the MCGDIFF samples and the reconstructions of each algorithm. We report these results in Table 5. Finally, in Figure 4 we illustrate the reconstructed trajectories on a specific trajectory completion problem.

Table 4: LPIPS metric against the runtime and memory cost of the algorithms.

<!-- image -->

Table 5: ℓ 2 distance quantiles with MCGDIFF as reference.

|         | σ y = 0 . 005   | σ y = 0 . 005   | σ y = 0 . 005   | σ y = 0 . 01   | σ y = 0 . 01   | σ y = 0 . 01   |
|---------|----------|----------|----------|----------|----------|----------|
|         | q 50          | q 25          | q 75          | q 50          | q 25          | q 75          |
| DCPS    | 1.31          | 1.33          | 1.47          | 1.33          | 1.42          | 1.42          |
| DPS     | 1.34          | 1.40          | 1.61          | 1.36          | 1.48          | 1.52          |
| DDRM    | 1.48          | 1.46          | 1.61          | 1.59          | 1.62          | 1.61          |
| Π GDM   | 1.36          | 1.35          | 1.47          | 1.37          | 1.43          | 1.42          |
| REDDIFF | 1.67          | 1.57          | 1.82          | 1.56          | 1.54          | 1.65          |

Figure 4: Trajectory completion where only the middle part of the trajectory is observed. The figures in the 1 st row display 3 reconstructions per algorithm. The 2 nd and 3 rd rows show confidence intervals across different time steps. The Groundtruth is a trajectory taken from the UCY dataset.

<!-- image -->

## 5 Conclusion.

In this paper, we introduce DCPS to handle Bayesian linear inverse problems with DDM priors without the need for problem-specific additional training. Our divide-and-conquer strategy helps to reduce the approximation error of existing approaches, and our variational framework provides a

DCPS

DPS

Groundtruth

NGDM

Observation

REDDIFF

DCPS

DPS

principled method for estimating the backward kernels. DCPS applies to various relevant inverse problems and is competitive with existing methods.

Limitations and future directions. Our method has some limitations that shed light on opportunities for further development and refinement. First, the intermediate potentials that we considered were specifically designed for each problem, meaning our method is not universally applicable to all inverse problems. For instance, our approach can not be applied to for linear inverse problems using latent diffusion models [41] since there is no clear choice of intermediate potentials. Therefore, in our opinion, deriving a learning procedure that is capable to automatically design effective intermediate potentials applicable to any g 0 is an important research direction. Moreover, there is an aspect of the choice of the intermediate potentials and the number of blocks L that remains to be understood properly. Indeed, while our backward approximations reduce the local approximation errors w.r.t. DPS and Π GDM;nonetheless DCPS requires appropriate intermediate potentials in order to perform well. DCPS can still provide decent performance with irrelevant intermediate potentials as long as the number of Langevin steps, in-between the blocks, is large enough. Finally, although our method provides decent results with the same computational cost as DPS and Π GDM, it remains slower than REDDIFF and DDRM which which do not compute vector-jacobian product over the denoiser. Therefore, overcoming this bottleneck when optimizing the KL objective would be a significant improvement for our method.

Acknowledgments. The work of Y.J. and B.M. has been supported by Technology Innovation Institute (TII), project Fed2Learn. The work of Eric Moulines has been partly funded by the European Union (ERC-2022-SYG-OCEAN-101071601). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References

- [1] Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, and Christian Etmann. Conditional image generation with score-based diffusion models. arXiv preprint arXiv:2111.13606 , 2021.
- [2] P. Bickel, B. Li, and T. Bengtsson. Sharp failure rates for the bootstrap particle filter in high dimensions. In B. Clarke and S. Ghosal, editors, Pushing the Limits of Contemporary Statistics: Contributions in Honor of Jayanta K. Ghosh , pages 318-329. Institute of Mathematical Statistics, 2008.
- [3] Christopher M. Bishop. Pattern Recognition and Machine Learning (Information Science and Statistics) . Springer-Verlag, Berlin, Heidelberg, 2006.
- [4] Benjamin Boys, Mark Girolami, Jakiw Pidstrigach, Sebastian Reich, Alan Mosca, and O Deniz Akyildiz. Tweedie moment projected diffusions for inverse problems. arXiv preprint arXiv:2310.06721 , 2023.
- [5] Nicolas Brosse, Alain Durmus, Éric Moulines, and Sotirios Sabanis. The tamed unadjusted langevin algorithm. Stochastic Processes and their Applications , 129(10):3638-3663, 2019.
- [6] Daniela Calvetti and Erkki Somersalo. Inverse problems: From regularization to Bayesian inference. Wiley Interdisciplinary Reviews: Computational Statistics , 10(3):e1427, 2018.
- [7] Gabriel Cardoso, Yazid Janati, Eric Moulines, and Sylvain Le Corff. Monte carlo guided denoising diffusion models for bayesian linear inverse problems. In The Twelfth International Conference on Learning Representations , 2024.
- [8] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models. arXiv preprint arXiv:2108.02938 , 2021.
- [9] Nicolas Chopin, Omiros Papaspiliopoulos, et al. An introduction to sequential Monte Carlo , volume 4. Springer, 2020.
- [10] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In The Eleventh International Conference on Learning Representations , 2023.
- [11] Hyungjin Chung, Byeongsu Sim, and Jong Chul Ye. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 12413-12422, 2022.
- [12] Masoumeh Dashti and Andrew M. Stuart. The Bayesian Approach to Inverse Problems , pages 311-428. Springer International Publishing, Cham, 2017.
- [13] Pierre Del Moral. Feynman-kac formulae. In Feynman-Kac Formulae , pages 47-93. Springer, 2004.
- [14] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition , pages 248-255. Ieee, 2009.
- [15] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems , 34:8780-8794, 2021.
- [16] Zehao Dou and Yang Song. Diffusion posterior sampling for linear inverse problem solving: A filtering perspective. In The Twelfth International Conference on Learning Representations , 2024.
- [17] Marc Anton Finzi, Anudhyan Boral, Andrew Gordon Wilson, Fei Sha, and Leonardo ZepedaNúñez. User-defined event sampling and uncertainty quantification in diffusion models for physical dynamical systems. In International Conference on Machine Learning , pages 1013610152. PMLR, 2023.

- [18] Tianpei Gu, Guangyi Chen, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou, and Jiwen Lu. Stochastic trajectory prediction via motion indeterminacy diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 17113-17122, 2022.
- [19] Bichuan Guo, Yuxing Han, and Jiangtao Wen. Agem: Solving linear inverse problems via deep priors and sampling. Advances in Neural Information Processing Systems , 32, 2019.
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems , 33:6840-6851, 2020.
- [21] Aapo Hyvärinen. Some extensions of score matching. Computational statistics &amp; data analysis , 51(5):2499-2512, 2007.
- [22] Bowen Jing, Gabriele Corso, Renato Berlinghieri, and Tommi Jaakkola. Subspace diffusion generative models. In European Conference on Computer Vision , pages 274-289. Springer, 2022.
- [23] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 4401-4410, 2019.
- [24] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. Advances in Neural Information Processing Systems , 35:23593-23606, 2022.
- [25] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization.
- [26] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 , 2013.
- [27] Alon Lerner, Yiorgos Chrysanthou, and Dani Lischinski. Crowds by example. In Computer graphics forum , 2007.
- [28] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 , 2016.
- [29] Karttikeya Mangalam, Yang An, Harshayu Girase, and Jitendra Malik. From goals, waypoints &amp;paths to long term human trajectory forecasting. In IEEE/CVF , 2021.
- [30] Weibo Mao, Chenxin Xu, Qi Zhu, Siheng Chen, and Yanfeng Wang. Leapfrog diffusion model for stochastic trajectory prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 5517-5526, 2023.
- [31] Willem Marais and Rebecca Willett. Proximal-gradient methods for poisson image reconstruction with bm3d-based regularization. In 2017 IEEE 7th International Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP) , pages 1-5. IEEE, 2017.
- [32] Morteza Mardani, Jiaming Song, Jan Kautz, and Arash Vahdat. A variational perspective on solving inverse problems with diffusion models. In The Twelfth International Conference on Learning Representations , 2024.
- [33] Radford M Neal. Annealed importance sampling. Statistics and computing , 11:125-139, 2001.
- [34] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning , pages 8162-8171. PMLR, 2021.
- [35] Robert D Nowak and Eric D Kolaczyk. A statistical multiscale framework for poisson inverse problems. IEEE Transactions on Information Theory , 46(5):1811-1825, 2000.
- [36] Xingang Pan, Xiaohang Zhan, Bo Dai, Dahua Lin, Chen Change Loy, and Ping Luo. Exploiting deep generative prior for versatile image restoration and manipulation. IEEE Transactions on Pattern Analysis and Machine Intelligence , 44(11):7474-7489, 2021.
- [37] Michael K Pitt and Neil Shephard. Filtering via simulation: Auxiliary particle filters. J. Amer. Statist. Assoc. , 94(446):590-599, 1999.

- [38] Gareth O. Roberts and Richard L. Tweedie. Geometric convergence and central limit theorems for multidimensional Hastings and Metropolis algorithms. Biometrika , 83:95-110, 1996.
- [39] Isabel Rodrigues, Joao Sanches, and Jose Bioucas-Dias. Denoising of medical images corrupted by poisson noise. In 2008 15th IEEE international conference on image processing , pages 1756-1759. IEEE, 2008.
- [40] Yaniv Romano, Michael Elad, and Peyman Milanfar. The little engine that could: Regularization by denoising (red). SIAM Journal on Imaging Sciences , 10(4):1804-1844, 2017.
- [41] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 10684-10695, 2022.
- [42] Franccois Rozet and Gilles Louppe. Score-based data assimilation. Advances in Neural Information Processing Systems , 36:40521-40541, 2023.
- [43] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence , 45(4):4713-4726, 2022.
- [44] Richard Shin and Dawn Song. Jpeg-resistant adversarial images. In NIPS 2017 workshop on machine learning and computer security , volume 1, page 8, 2017.
- [45] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning , pages 2256-2265. PMLR, 2015.
- [46] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations , 2021.
- [47] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations , 2023.
- [48] Jiaming Song, Qinsheng Zhang, Hongxu Yin, Morteza Mardani, Ming-Yu Liu, Jan Kautz, Yongxin Chen, and Arash Vahdat. Loss-guided diffusion models for plug-and-play controllable generation. In International Conference on Machine Learning , pages 32483-32498. PMLR, 2023.
- [49] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. Advances in Neural Information Processing Systems , 34:14151428, 2021.
- [50] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems , 32, 2019.
- [51] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 , 2020.
- [52] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations , 2021.
- [53] Andrew M Stuart. Inverse problems: a Bayesian perspective. Acta numerica , 19:451-559, 2010.
- [54] Brian L. Trippe, Jason Yim, Doug Tischer, David Baker, Tamara Broderick, Regina Barzilay, and Tommi S. Jaakkola. Diffusion probabilistic modeling of protein backbones in 3d for the motifscaffolding problem. In The Eleventh International Conference on Learning Representations , 2023.
- [55] Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Deep image prior. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 9446-9454, 2018.

- [56] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.
- [57] Gregory K Wallace. The jpeg still picture compression standard. IEEE transactions on consumer electronics , 38(1):xviii-xxxiv, 1992.
- [58] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. In The Eleventh International Conference on Learning Representations , 2023.
- [59] Luhuan Wu, Brian L. Trippe, Christian A Naesseth, John Patrick Cunningham, and David Blei. Practical and asymptotically exact conditional sampling in diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- [60] Guanhua Zhang, Jiabao Ji, Yang Zhang, Mo Yu, Tommi Jaakkola, and Shiyu Chang. Towards coherent image inpainting using denoising diffusion implicit models. In International Conference on Machine Learning , pages 41164-41193. PMLR, 2023.
- [61] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 586-595, 2018.
- [62] Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 1219-1229, 2023.

## A Methodology details

## A.1 Denoising Diffusion models

DDMs learn a sequence (ˆ x θ 0 | t ) T t =1 of denoisers by minimizing, using SGD, the objective

<!-- formula-not-decoded -->

w.r.t. the neural network parameter θ , where ( ϵ t ) T t =1 are i.i.d. standard normal vectors and ( w t ) T t =1 are some nonnegative weights. We denote by θ ⋆ an estimator of the minimizer of the previous loss. Having access to θ ⋆ , we can define a generative model for p data . Let ( t k ) n k =0 be an increasing sequence of time instants in J 0 , T K with t 0 = 0 . We assume that t n is large enough so that q t n is approximately multivariate standard normal. For convenience, we assign the index k to any quantity depending on t k ; e.g. , we denote p t k by p k . For ( j, k ) ∈ J 1 , n -1 K 2 such that j &lt; k , define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then the bridge kernel

<!-- formula-not-decoded -->

is a Gaussian distribution with mean µ j | 0 ,k ( x 0 , x k ) and covariance σ 2 j | k I d x . DDPM [20] posits the following variational approximation

<!-- formula-not-decoded -->

where p θ k | k +1 ( x k | x k +1 ) = q k | 0 ,k +1 ( x k | ˆ x θ 0 | k +1 ( x k +1 ) , x k +1 ) and p θ 0 | 1 ( ·| x 1 ) = δ ˆ x θ 0 | 1 ( x 1 ) . An efficient generative model is then obtained by plugging in the parameter θ ⋆ .

## A.2 Further details on DCPS

In this section we provide further details on Steps 1 and 2 detailed in the main paper. The complete algorithm is given in Algorithm 1.

Tamed unadjusted Langevin. For the tamed unadjusted Langevin steps we simulate the Markov chain ( ˜ X j ) j M =0 where

<!-- formula-not-decoded -->

and ( Z j ) M -1 j =0 are i.i.d. d x -dimensional standard normal, X ℓ +1 is an approximate sample from π ℓ +1 obtained from the previous iteration of the algorithm, and for all x ∈ R d x and γ &gt; 0 ,

<!-- formula-not-decoded -->

We then set X ℓ ℓ +1 := ˜ X M , which serves as an initialization of the Markov chain in Step 2 .

Potential computation. In order to perform the tamed Langevin steps and to optimize the variational approximation using the criterion (3.8), it is crucial to be able to compute exactly the potential (3.6). The optimal potentials we have proposed for both linear inverse problems with Gaussian noise (3.10) and low-count Poisson denoising (3.11) (for ℓ &gt; 0 ) are available in a closed form:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

µ ℓ | j ( x j ) := µ ℓ | 0 ,j (ˆ x 0 | j ( x j ) , x j ) , and σ 2 ℓ | j is defined in (A.2). As a result, the first term of the variational criterion L (ˆ µ j , ˆ υ j ; x j +1 ) in (3.8), given by

<!-- formula-not-decoded -->

can be computed exactly. Indeed, as µ ℓ | j is a linear function of x j , this expectation is simply that of a quadratic function under a Gaussian density, given by

<!-- formula-not-decoded -->

Hence, for these cases, (3.8) has a closed-form expression. However, it involves the computation of an inverse matrix which, for many problems, can be prohibitively expensive. To avoid this inversion, we instead optimize a biased estimate of L j (ˆ µ j , ˆ υ j ; x j +1 ) obtained by drawing two noise vectors ( Z, Z ′ ) ∼ N(0 d x , I d x ) and setting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This estimator is computable for any choice choice of potential and we have found in practice that it is sufficient to ensure good enough performance for our algorithm. Regarding the tamed unadjusted Langevin steps, we use the same biased estimate when the matrix inversions are expensive to compute; i.e. at each Langevin step, we approximate G ℓ γ ( ˜ X j ) by

<!-- formula-not-decoded -->

## Algorithm 1 DIVIDE-AND-CONQUER POSTERIOR SAMPLER (DCPS)

Input: timesteps ( k ℓ ) L ℓ =0 , learning-rate ζ , numbers K and M of gradient and Langevin steps, respectively.

X

L

k

to

Initial sample for

-

1

ℓ

=

L

Z

∼

0

N(0

Draw

X

ℓ

k

+1

ℓ

for

i

←

X

= 1

Z

d

k

ℓ

to

∼

N(0

X

ℓ

k

+1

ℓ

end for for

j

=

ˆ

j

←

µ

for

+1

M

do

d

x

, I

←

X

k

ℓ

x

d

ℓ

k

+1

-

µ

j

j

|

r

+1

= 1

+1

ℓ

1

)

∼ N

do

x

x

d

, I

;

+

to

(

X

to

K

(

Draw

[

j

ˆ

µ

]

ˆ

j

υ

end for

ε

∼ N

(0

X

ℓ

j

←

## end for

X

k

ℓ

←

end for

G

(0

)

x

x

d

d

, I

)

;

and compute

ℓ

ℓ

k

ℓ

+1

j

G

γ

X

(

˜

do

γ

k

ℓ

ℓ

j

+1

do

)

∼

Z, Z

′

[

←

d

)

;

N(0

υ

ˆ

←

d

ζ

j

ˆ

µ

ˆ

]

j

υ

x

ˆ

µ

X

j

ℓ

k

ℓ

, I

d

x

)

+diag(e

;

∥∇

x

, I

ˆ

µ

/

2

)

γ

˜

ℓ

) +

log

d

x

j

,

υ

ˆ

ε

;

j

(

√

2

ℓ

k

ℓ

X

γZ

+1

;

σ

2

j

+1

j

|

1

·

d

)

L

and compute

j

˜

(ˆ

µ

j

,

ˆ

υ

j

;

X

ℓ

j

+1

-

ˆ

υ

j

)

(A.9);

x

;

L

˜

j

(ˆ

)

∥

µ

j

-

1

,

υ

ˆ

∇

j

ˆ

µ

j

;

,

X

ℓ

j

ˆ

j

+1

L

)

j

(ˆ

µ

˜

υ

(A.8);

j

,

ˆ

υ

j

;

X

ℓ

j

+1

)

## A.3 Proof of Proposition 3.1

For all k ∈ J 0 , n -1 K we denote by q k | k +1 ( x k | x k +1 ) the exact backward kernel which satisfies

<!-- formula-not-decoded -->

Note that the backward kernels p k | k +1 are to be understood as Gaussian approximations of the true backward kernels q k | k +1 . Below we give a complete statement of the proposition and provide a proof.

Proposition A.1. k ∈ J 1 , n K q k | k +1 ( x k | x k +1 ) = p k | k +1 ( x k | x k +1 ) ( x k , x k +1 ) ∈ ( R d x ) 2 . For all ℓ ∈ 0 , k -1 and x k ∈ R d x ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Proposition A.1. Under the assumptions of the proposition, we have, for all m&gt;ℓ ,

<!-- formula-not-decoded -->

Indeed, by definition of the backward kernel q 0 | k ( x 0 | x k ) and (A.10), it holds that

<!-- formula-not-decoded -->

As a result, we have that

<!-- formula-not-decoded -->

where, by definition, ˆ p 0 | k ( ·| x k ) is a Gaussian approximation of q 0 | k ( ·| x k ) as defined in the main paper.

Next, let Π 0 | k ( ·| x k ) denote a coupling of q 0 | k ( ·| x k ) and ˆ p 0 | k ( ·| x k ) , i.e. , for all A ∈ B ( R d x ) ,

Consider then the random variables

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( X 0 | k , ˆ X 0 | k ) ∼ Π 0 | k ( ·| x k ) and Z ∼ N (0 d x , I d x ) . Then ( X ℓ | k , ˆ X ℓ | k ) is distributed according to a coupling of ˆ p ℓ | k ( ·| x k ) and p ℓ | k ( ·| x k ) , and consequently

<!-- formula-not-decoded -->

The result is obtained by taking the infinimum of the rhs with respect to all couplings of q 0 | k ( ·| x k ) and ˆ p 0 | k ( ·| x k ) .

## B Related works.

In this section we discuss in more details existing works that bear some similarities with DCPS.

SMC based approaches. The MCGDIFF, the Twisted Diffusion sampler (TDS) of [59] using the FK representation (2.4). MCGDIFF is specific to linear inverse problems and the potentials used are g k ( x k ) = N( √ α k y ; Ax k , (1 -α k ) I d y ) when σ y = 0 . TDS applies to any potential g 0 and relies on the DPS approximation for its potentials; i.e. g k ( x k ) = g 0 (ˆ x 0 | k ( x k )) . In either cases, a particle approximation of the posterior of interest π 0 is obtained using the Auxiliary Particle filter framework [37]. [16] also use particle filters for the posterior distribution; the potentials used are g k ( x k ) = N( √ α k y k ; Ax k , α k σ 2 y I d x ) where ( y k ) n k =0 , with y 0 = y is a sequence of observations sampled according to an auto-regressive process; see [16, Equation 7]. The posterior is thus viewed as approximately the time 0 marginal of a Hidden Markov model with transition p k | k +1 and observation likelihood g k , which is different from the FK representation (2.4). Our choice of intermediate potentials for linear inverse problems with Gaussian noise differs from that of MCGDIFF by the standard deviation of the observation model, which we set to be σ y . A major difference of DCPS with these works lies in the fact that we do not rely on particle filters, thus avoiding the collapse in very large dimensions. As we have shown in the experimental section DCPS can achieve comparable performance to MCGDIFF in low dimensions, see Table 1 while also being efficient in very large dimensions, see Table 2. A second and major difference is that we have derived potentials for both the JPEG dequantization and Poisson-shot denoising tasks, which may be used to extend MCGDIFF and FPS-SMC [16] to these problems.

RedDiff. In this work we have also proposed to use Gaussian variational inference to approximate the intractable backward transition π ℓ k | k +1 . One particularity of our approach is that we do not use amortized variational inference [26] and instead optimize the variational distribution at each step of the diffusion. A similar approach is used in REDDIFF [32] but in a different way. Indeed, the authors use a non-amortized Gaussian variational approximation for the posterior π 0 , meaning that in order to draw one sample from REDDIFF, several steps of optimization are performed on a score-matching-like loss. Interestingly, this approach does not require differentiating through the denoising network and is thus faster and more memory efficient. However, we found that this comes at the cost of performance as can be seen in Table 1, 2 and 3.

SDA. In [42], the authors introduce a posterior sampling algorithm for inverse problem where the chosen potential approximation is

<!-- formula-not-decoded -->

with γ &gt; 0 being a tunable parameter. Noteworthy, this potential is similar to one used in [47] with a slightly different choice of variance. Then, the Score-based Data Assimilation (SDA) algorithm proceed following the Predictor-Corrector framework [51]. In the Prediction stage, a sample X k given X k +1 is drawn using the conditional score

<!-- formula-not-decoded -->

In the Correction stage, a Langevin MC targeting the marginal distribution g k ( x k ) p k ( x k ) is simulated starting from the predicted sample following

<!-- formula-not-decoded -->

where δ k is a state-dependent step-size. We emphasise that due to the dependence of the step sizes on the states, these are only Langevin-like updates that do not inherit the theoretical guarantees of the unadjusted Langevin algorithm. While SDA and our algorithm DCPS both use Langevin, the pivotal difference is that its purpose, in our case, is not to correct to ensure that the sample X k ℓ is distributed according to the marginal π k ℓ ( x k ℓ ) ∝ g k ℓ ( x k ℓ ) p k ℓ ( x k ℓ ) , but rather to ensure that the sample is distributed according to the next distribution π k ℓ ( x k ℓ ) , which is the initial distribution of the next block, as per Equation (3.4). Hence, in our case, Langevin MC is used between blocks and not within blocks.

## C Experiments

## C.1 Implementation details

In this section we provide the global implementation details for each algorithm. We provide the specific parameters (when needed) used for each experiment (Gaussian mixture, image restoration and trajectory inpainting) in the dedicated sections below.

DCPS. For all the experiments we implement Algorithm 1. We use the same parameters K = 2 , L = 3 and ζ = 1 for all the experiments. For the number of Langevin steps, we set it to M = 50 and M = 500 (respectively) for the Gaussian mixture experiment and M = 5 for the imaging and trajectory inpainting experiments.

DDRM. We have used the official implementation 1 and used the recommended parameters in the original paper. We use 200 steps for DDRM and found that it works better than when we used 1000 steps.

DPS. We have implemented both Algorithm 1 (for linear inverse problems) and Algorithm 2 (for Poisson-shot restoration) given in [10]. In all the experiments we run DPS with 1000 Diffusion steps.

RedDiff. For RedDiff, we have used the publicly available implementation 2 . We have empirically found that RedDiff works best in the low observation standard deviation regime and produces spatially coherent reconstructions in the larger noise regime but struggles with getting rid of the noise as evidenced by the large increase in LPIPS values in Table 2. Note also that it is not clear how the parameters of the algorithm depend on the inverse problem standard deviation; indeed, looking at Algorithm 1 and then Appendix C.2 where the authors consider a noisy inverse problem 3 there seems to be no clear dependence of λ on σ v ( σ y with our notations). In fact the authors use λ = 0 . 25 similarly to the noiseless experiments in the main paper and we believe that the tuning is performed only on the initial step-size of Adam. As a result, for the experiments with σ y = 0 . 3 , we have tuned it using a grid-search in [0 . 1 , 0 . 25] and retained 0 . 1 .

Π GDM. Regarding Π GDM [47], note that there is no publicly available implementation and we have thus implemented the noisy version of [47, Algorithm 1] in the original paper. However, we did not manage to obtain appropriate results and found it to be quite unstable. We have further investigated the issue and found that Π GDM is implemented in the github repository of RedDiff 4 , which is by the same authors. We have noted that it has a slight difference with Algorithm 1 of the Π GDM paper; the gradient term, coined g in [47, Algorithm 1], is multiplied by √ α t -1 α t instead of simply √ α t . We have found that this stabilizes the algorithm significantly for the linear inverse problem experiment. We use the same rescaling for the Gaussian mixture and trajectory inpainting experiment. However, even with this modification to the algorith we found that Π GDM does not perform well when the noise standard deviation is large; see Table 2. For the JPEG experiment we do not use this rescaling as we found that the algorithm remains stable.

MCGDiff. For MCGDiff we have used the official implementation 5 with N = 32 particles for the imaging experiments. There are no further tuning parameters as far as we can tell.

DIFFPIR We implemented [62, Algorithm 1] and use the hyperparameters recommended in the official, released version 6 .

DDNM. We adapted the implementation in the released code 7 to our code base.

1 https://github.com/bahjat-kawar/ddrm 2 https://github.com/NVlabs/RED-diff 3 https://openreview.net/pdf?id=1YO4EE3SPB 4 https://github.com/NVlabs/RED-diff 5 https://github.com/gabrielvc/mcg\_diff 6 https://github.com/yuanzhi-zhu/DiffPIR 7

https://github.com/wyhuai/DDNM

SDA. We implement the posterior sampling algorithm by combining [42, Algo 3 and 4 in Appendix C]. In the experiments, we use two Langevin corrections steps and found that γ = 0 . 1 works well across problems for the diagonal approximation the same as τ = 0 . 1 for the Langevin correction steps size.

FPS We implement [16, Algorithm 2] provided in the appendix.

## C.2 Gaussian mixtures

For a given dimension d x , we consider p data a mixture of 25 Gaussian random variables. The means of the Gaussian components of the mixture are ( m i ) 25 i =1 := { (8 i, 8 j, · · · , 8 i, 8 j ) ∈ R d x : ( i, j ) ∈ {-2 , -1 , 0 , 1 , 2 } 2 } . The covariance of each component is identity. The mixture (unnormalized) weights w i,j are independently drawn from a Dirichlet distribution.

Metrics. To assess the performance of each algorithm we draw 2000 samples and compare against 2000 samples from the true posterior distribution using the Sliced Wasserstein distance by averaging over 10 4 slices. In Table 1 we report the average SW and the 95% confidence interval over 30 seeds. We found DPS and Π GDM to be sometimes unstable, resulting in NaN values. To account for these unstabilities when computing the average SW distance, we replace NaN with 7 which is the typical value obtained when a stable algorithm fails to sample from the posterior.

Parameters. For DPS we use ζ m = 0 . 1 / ∥ y -A ˆ x θ ⋆ 0 | m ( x m ) ∥ at step m of the Diffusion. As to DCPS we use γ = 10 -2 for the Langevin step-size.

Denoisers. Note that the loss (A.1) can be written as

<!-- formula-not-decoded -->

Hence the minimizer is

<!-- formula-not-decoded -->

which yields ˆ x θ ⋆ 0 | t = E [ X 0 | X t = · ] . Next, by Tweedie's formula we have that

<!-- formula-not-decoded -->

Hence, since q data is a mixture of Gaussians, q t is also a mixture of Gaussians with means ( √ α t m i ) 25 i =1 and unit covariances. Therefore, ∇ x log q t ( x t ) and hence ˆ x θ ⋆ 0 | t ( x t ) can be computed using automatic differentiation libraries.

Measurement model. For a pair of dimensions ( d x , d y ) the measurement model ( y, A, σ y ) is drawn as follows: the elements d x × d y elements of the matrix are drawn i.i.d. from a standard Gaussian distribution, then σ y is drawn uniformly in [0 , 1] and finally we draw x ⋆ ∼ p data and ε ∼ N (0 d y , I d y ) and set y = Ax ⋆ + σ y ε .

Table 6: Mean LPIPS value on low count Poisson restoration.

| Dataset   | Task          | DCPS      | DPS       |
|----------|----------|----------|----------|
| FFHQ      | Denoising SR 4 × | 0.07 0.17 | 0.12 0.31 |
| ImageNet  | Denoising SR 4 × | 0.17 0.36 | 0.24 0.80 |

Posterior. Having drawn both p data and ( y, A, σ y ) , the posterior can be computed exactly using standard Gaussian conjugation formulas [3, Eq. 2.116] and hence the posterior is a Gaussian mixture where all the components have the same covariance matrix Σ := ( I d x + σ -2 y A T A ) -1 and means and weights given by

<!-- formula-not-decoded -->

## C.3 Imaging experiments

Parameters. For DCPS we set γ = 10 -3 for the Langevin step-size. For DPS we use the parameters recommended in the original paper, which we found to work well even on the half and expand masks; see [10, Appendix D.1].

Evaluation. In order to evaluate each algorithm we compute the LPIPS metric [61] on each dataset using 100 samples from the validation sets and report the average in Table 2, 3 and 6.

JPEGdequantization. Weuse the differentiable JPEG framework [44] which replaces the rounding function x ↦→ ⌊ x ⌉ used in the quantization part with x ↦→ ⌊ x ⌉ +( x -⌊ x ⌉ ) 3 which has non-zero derivatives almost everywhere.

## C.4 Trajectory inpainting experiment

Trajectory DDM prior. The denoiser of the diffusion model has a Transformer-like architecture. In the entry of the network, the trajectory is augmented to a higher dimensional space ( 512 ) via dense layer. At this stage a positional encoding [56] is added to account for the diffusion step. Afterward, the output is flowed through a transformer encoder [56] whose feedforward layer dimension is 2048 to learn temporal dependence within the trajectory before being feed to an MLP with 4 layers ( 512 → 1024 → 1024 → 512 ) and in between ReLU activation functions, to output the added noise. A Cosine noise scheduler with 1000 diffusion steps was used [34]. The UCY -student dataset was split int a train and a validation sets with 1450 and 140 trajectories respectively. The batch size was set to 10 times the training set, namely 145 samples The denoiser was trained to minimize the loss of DDPM [20] for 1000 epochs using Adam solver [25] with a Cosine learning rate scheduler [28]. The training was performed on 48GB L40S NVIDIA GPU and took roughly one minute to complete.

Metrics. The trajectory completion experiment was performed on the validation set. Every trajectory was masked randomly. Leveraging MCGDIFF 's asymptotical approximation of the posterior, it was run with 5000 particles to sample 100 samples from the posterior and afterward these were checked against a 100 reconstructions of each other algorithm by computing the timestep wise ℓ 2 distance between the quantile 50 (median), 25 , 75 and also by computing the Sliced Wasserstein distance. This procedure was repeated for all trajectories in the validation set and later the results of each algorithm were aggregated by the mean ℓ 2 distances. Finally, this experiment was performed for two levels of noise σ y = 0 . 005 and σ y = 0 . 01 .

## C.5 Additional experiments

Here, we provide the complete tables of results on imaging and trajectories inpainting experiments that includes in addition DIFFPIR, DDNM, FPS , and SDA. These additional experiments were conducted during the rebuttal phase of our work.

DCPS

DPS

Observation

Ground truth

Observation

Ground truth

Table 7: Mean LPIPS value on different tasks. Lower is better.

| Dataset / σ y   | Task    |   DCPS |   DDRM |   DPS |   Π GDM |   REDDIFF |   MCGDIFF |   DIFF PIR |   DDNM |   SDA |   FPS |
|----------|---------|--------|--------|-------|---------|----------|----------|----------|--------|-------|-------|
| FFHQ / 0.05     | Half    |   0.2  |   0.25 |  0.24 |    0.26 |      0.28 |      0.36 |       0.23 |   0.22 |  0.23 |  0.28 |
| FFHQ / 0.05     | Center  |   0.05 |   0.06 |  0.07 |    0.19 |      0.12 |      0.24 |       0.06 |   0.05 |  0.05 |  0.09 |
| FFHQ / 0.05     | SR 4 ×  |   0.09 |   0.18 |  0.09 |    0.33 |      0.36 |      0.15 |       0.13 |   0.14 |  0.1  |  0.1  |
| FFHQ / 0.05     | SR 16 × |   0.23 |   0.36 |  0.24 |    0.44 |      0.51 |      0.32 |       0.28 |   0.3  |  0.44 |  0.71 |
| FFHQ / 0.3      | Half    |   0.25 |   0.3  |  0.31 |    0.64 |      0.76 |      0.8  |       0.3  |   0.26 |  0.26 |  0.67 |
| FFHQ / 0.3      | Center  |   0.1  |   0.13 |  0.11 |    0.62 |      0.75 |      0.55 |       0.16 |   0.11 |  0.1  |  0.69 |
| FFHQ / 0.3      | SR 4 ×  |   0.21 |   0.26 |  0.19 |    0.77 |      0.77 |      0.65 |       0.28 |   0.23 |  0.19 |  0.75 |
| FFHQ / 0.3      | SR 16 × |   0.35 |   0.41 |  0.43 |    0.64 |      0.74 |      0.52 |       0.42 |   0.39 |  0.49 |  0.71 |
| ImageNet / 0.05 | Half    |   0.35 |   0.4  |  0.44 |    0.38 |      0.44 |      0.83 |       0.35 |   0.38 |  0.54 |  0.39 |
| ImageNet / 0.05 | Center  |   0.18 |   0.14 |  0.31 |    0.29 |      0.22 |      0.45 |       0.14 |   0.13 |  0.14 |  0.19 |
| ImageNet / 0.05 | SR 4 ×  |   0.24 |   0.38 |  0.41 |    0.78 |      0.56 |      1.32 |       0.36 |   0.34 |  0.85 |  0.27 |
| ImageNet / 0.05 | SR 16 × |   0.44 |   0.72 |  0.5  |    0.6  |      0.83 |      1.33 |       0.63 |   0.7  |  1.13 |  0.69 |
| ImageNet / 0.3  | Half    |   0.4  |   0.46 |  0.48 |    0.82 |      0.76 |      0.86 |       0.5  |   0.44 |  0.61 |  0.71 |
| ImageNet / 0.3  | Center  |   0.24 |   0.25 |  0.4  |    0.68 |      0.71 |      0.47 |       0.36 |   0.22 |  0.25 |  0.7  |
| ImageNet / 0.3  | SR 4 ×  |   0.43 |   0.5  |  0.47 |    0.87 |      0.83 |      1.31 |       0.61 |   0.46 |  1.14 |  0.84 |
| ImageNet / 0.3  | SR 16 × |   0.72 |   0.77 |  0.57 |    0.72 |      0.92 |      0.67 |       0.76 |   0.75 |  1.19 |  0.74 |
| Average         |         |   0.28 |   0.35 |  0.32 |    0.57 |      0.6  |      0.67 |       0.35 |   0.32 |  0.48 |  0.53 |
| Median          |         |   0.24 |   0.33 |  0.35 |    0.63 |      0.72 |      0.6  |       0.32 |   0.28 |  0.35 |  0.69 |

Table 8: ℓ 2 distance quantiles with MCGDIFF as reference.

|          | σ y = 0 . 005   | σ y = 0 . 005   | σ y = 0 . 005   | σ y = 0 . 01   | σ y = 0 . 01   | σ y = 0 . 01   |
|----------|----------|----------|----------|----------|----------|----------|
|          | q 50          | q 25          | q 75          | q 50          | q 25          | q 75          |
| DCPS     | 1.31          | 1.33          | 1.47          | 1.33          | 1.42          | 1.42          |
| DPS      | 1.34          | 1.40          | 1.61          | 1.36          | 1.48          | 1.52          |
| DDRM     | 1.48          | 1.46          | 1.61          | 1.59          | 1.62          | 1.61          |
| Π GDM    | 1.36          | 1.35          | 1.47          | 1.37          | 1.43          | 1.42          |
| REDDIFF  | 1.67          | 1.57          | 1.82          | 1.56          | 1.54          | 1.65          |
| DIFF PIR | 1.57          | 1.84          | 1.98          | 1.52          | 1.94          | 1.89          |
| DDNM     | 1.45          | 1.45          | 1.65          | 1.52          | 1.59          | 1.59          |
| FPS      | 2.60          | 2.61          | 2.62          | 2.91          | 2.90          | 2.89          |
| SDA      | 1.52          | 1.55          | 1.69          | 1.54          | 1.59          | 1.61          |

## D Sample reconstructions

In this section we display the remaining samples from the experiments in the main paper. We remind the reader that all algorithms are run with the same seed and we draw in parallel 4 samples from each algorithm and display them in their order of appearance.

Figure 5: Denoising task with Poisson noise on FFHQ .

<!-- image -->

DCPS

DCPS

DPS

DDRM

DPS

HIGDM

MCGDIFF

REDDIFF

Observation

Observation

Ground truth

Ground truth

DCPS

Figure 6: Denoising task with Poisson noise on ImageNet .

<!-- image -->

Figure 7: Outpainting task with half mask on ImageNet .

<!-- image -->

Observation

Ground truth

DCPS

DCPS

DDRM

DDRM

DPS

DPS

HIGDM

TIGDM

MCGDIFF

MCGDIFF

REDDIFF

REDDIFF

Observation

Observation

Ground truth

Ground truth

Observation

Observation

Catalyst

Ground truth

Ground truth

Catalyst

Figure 9: Inpainting task with box mask on ImageNet .

<!-- image -->

DCPS

DCPS

DDRM

DDRM

DPS

DPS

HIGDM

TIGDM

MCGDIFF

MCGDIFF

REDDIFF

REDDIFF

Observation

Observation

Ground truth

Ground truth

Observation

Ground truth

Observation

Ground truth

Figure 11: Outpainting task with half mask on ImageNet .

<!-- image -->

DCPS

DCPS

DDRM

DDRM

DPS

DPS

TIGDM

TIGDM

MCGDIFF

MCGDIFF

REDDIFF

REDDIFF

Observation

Observation

Ground truth

Ground truth

Observation

Ground truth

Observation

Ground truth

Figure 13: Outpainting expend task on ImageNet .

<!-- image -->

DCPS

DCPS

DPS

DDRM

DPS

HIGDM

MCGDIFF

REDDIFF

Observation

Observation

Ground truth

Ground truth

DCPS

DCPS

DPS

DDRM

Figure 14: SR 4 × task with Poisson noise on FFHQ .

<!-- image -->

REDDIFF

Figure 15: SR 4 × task on ImageNet .

<!-- image -->

Observation

Observation

Ground truth

Ground truth

DCPS

DDRM

DPS

IIGDM

MCGDIFF

REDDIFF

DCPS

DDRM

DPS

IIGDM

MCGDIFF

REDDIFF

Observation

Observation

Ground truth

Ground truth

Observation

Ground truth

Figure 16: SR 16 × task on FFHQ .

<!-- image -->

DCPS

DDRM

DPS

IIGDM

MCGDIFF

REDDIFF

Observation

Ground truth

DCPS

DDRM (

DPS

IIGDM

MCGDIFF

REDDIFF

Observation

Ground truth

Figure 17: SR 16 × task on ImageNet .

<!-- image -->

DCPS

DCPS

TIGDM

TIGDM

REDDIFF

REDDIFF

Observation

Observation

Ground truth

Ground truth

DCPS

DCPS

TIGDM

TIGDM

REDDIFF

REDDIFF

Figure 19: JPEG task with QF=2 on FFHQ .

<!-- image -->

Observation

Observation

Ground truth

Ground truth

DCPS

DCPS

TIGDM

HIGDM

REDDIFF

REDDIFF

Observation

Observation

Ground truth

Ground truth

DCPS

DCPS

TIGDM

TIGDM

REDDIFF

REDDIFF

Figure 20: JPEG task with QF=8 on ImageNet .

<!-- image -->

Figure 21: JPEG task with QF=2 on ImageNet .

<!-- image -->

Observation

Observation

Ground truth

Ground truth

##