## ZERO-SHOT IMAGE RESTORATION VIA DIFFUSION INVERSION

Anonymous authors

Paper under double-blind review

## ABSTRACT

Recently, various methods have been proposed to solve Image Restoration (IR) tasks using a pre-trained diffusion models leading to state-of-the-art performance. A common characteristic among these approaches is that they alter the diffusion sampling process in order to satisfy the consistency with the corrupted input image. However, this choice has recently been shown to be sub-optimal and may cause the generated image to deviate from the data manifold. We propose to address this limitation through a novel IR method that not only leverages the power of diffusion but also guarantees that the sample generation path always lies on the data manifold. One choice that satisfies this requirement is not to modify the reverse sampling , i . e ., not to alter all the intermediate latents, once an initial noise is sampled. This is ultimately equivalent to casting the IR task as an optimization problem in the space of the diffusion input noise. To mitigate the substantial computational cost associated with inverting a fully unrolled diffusion model, we leverage the inherent capability of these models to skip ahead in the forward diffusion process using arbitrary large time steps. We experimentally validate our method on several image restoration tasks. Our method SHRED achieves state of the art results on multiple zero-shot IR benchmarks especially in terms of image quality quantified using FID.

## 1 INTRODUCTION

Recent advances in the field of generative learning due to diffusion models (Ho et al., 2020; Song &amp; Ermon, 2019) and better architecture choices (Karras et al., 2021; Zhang et al., 2022) have led to models capable of generating detailed high-resolution images. In addition to their data generation capabilities, these models also provide a smooth representation of the learned data distribution, which can be used for various applications, such as image restoration. Indeed, different approaches (Kawar et al., 2022; Lugmayr et al., 2022; Wang et al., 2023) have emerged to solve inverse problems using pretrained diffusion models. Those approaches start from a random noise vector as the diffusion input and recursively alter the diffusion sampling process in order to guarantee the data fidelity. After each diffusion reverse step, a projection-based measurement consistency step is added to ensure the consistency with the corrupted image. Chung et al. (2022b) shows that this procedure may cause the generated image to step outside the data manifold during the iterative denoising process.

In this work, we devise a new zero-shot image restoration method that we call in short SHRED (zero-SHot image REstoration via Diffusion inversion) which, by construction, ensures that the sample generation path always lies on the data manifold. To this end, we opt for a simple yet effective strategy: we do not alter the reverse sampling , i . e ., all the intermediate latents, once an initial noise is sampled. More specifically, SHRED use a pre-trained Denoising Diffusion Implicit Models (DDIM) (Song et al., 2020) and exploits the existing deterministic correspondence between noise and images in DDIMs by casting the inverse restoration problem as a latent estimation problem, i . e ., where the latent variable is the input noise to the diffusion model.Given a degraded image, the pretrained generative model allows us to retrieve the optimal initial noise that, when applying the reverse diffusion sampling, generates the closest clean image under the learnt image distribution. Since SHRED is an optimization method that works by iteratively optimizing the initial noise, multiple forward passes of the diffusion process are required which can be computationally expensive. A single forward pass of a standard diffusion model (with 1000 steps) takes in the order of minutes on a typical modern GPU. Thus, a direct implementation of the optimization procedure in the case of the Diffusion model as a black box would be computationally demanding and ultimately impractical.

Figure 1: Image inpainting with SHRED. We show the flow during one iteration of our optimization scheme of initial noise x T . The dashed and single line arrows denote the reverse and forward diffusion processes respectively. We optimize the inpainting error L IR only with respect to the initial noise x T . In our notation, x 0 | t is our intermediate estimate of x 0 given x t . The top and bottom row images show x t and x 0 | t for different values of t respectively. The reconstruction of x 0 | t allows us to skip time steps and make the generation process and the gradient back-propagation much more efficient.

<!-- image -->

To mitigate the substantial computational cost associated with inverting a fully unrolled diffusion model, we leverage the inherent capability of these models to skip ahead in the forward diffusion process using arbitrary large time steps. Thanks to the non-Markovian marginal distributions used in DDIM, our method introduces a hyperparameter δt that can be used to prioritize either image quality or computational cost. We show that by optimizing for the best initial noise that corresponds to the unknown restored image, our method is able to achieve competitive image performance across a wide range of image restoration tasks both in the blind and non-blind setting. Our main contributions can be summarized as:

1. We are the first to cast the IR task as a latent optimization problem in the context of diffusion models where only the initial noise is optimized. This design choice does not alter the intermediate diffusion latents and thus provides more guarantees that the generated images lies in the in-distribution manifold.
2. We leverage the capabilty of DDIM to skip ahead in the forward diffusion process by arbitrary time steps and propose an efficient and practical diffusion model inversion in the context of image restoration. We note that in our method, DDIM is neither fine-tuned nor retrained.
3. Weachieve competitive results on both CelebA and ImageNet for different IR tasks including image inpainting, super-resolution, compressed sensing and blind deconvolution.

## 2 RELATED WORK

Zero-shot methods Recently, methods that can be repurposed to different inverse problems have been called zero-shot . Song &amp; Ermon (2019) proposes a method based on guiding the reverse diffusion process with the unmasked region to solve inpainting in a zero-shot manner. Song et al. (2021) proposes the use of gradient guidance to solve inverse problems in the context of medical imaging. Choi et al. (2021) applies low-frequency guidance from a reference image to solve super-resolution. RePaint (Lugmayr et al., 2022) solves the inpainting problem by guiding the diffusion process with the unmasked region. To condition the generation process, the reverse diffusion iterations are altered by sampling the unmasked regions using the given image information. DDRM (Kawar et al., 2022) proposes an inverse problem solver based on posterior sampling by introducing a variational inference objective for learning the posterior distribution of the inverse problem.

DDNM (Wang et al., 2023) proposes a zero-shot framework for IR tasks based on the range-null space decomposition. The method works by refining only the null-space contents during the reverse diffusion process,to satisfy both data consistency and realness. Fei et al. (2023) leverage DDPM and solve inverse problems via hierarchical guidance and a patch-based method. Chung et al. (2022b) argues that relying on an iterative procedure consisting of reverse diffusion steps and a projectionbased consistency step runs the risk of stepping outside of the data manifold, a risk they mitigate using an additional correction term.

Methods based on GAN inversion Different approaches Pan et al. (2021); Yang et al. (2021); Yu et al. (2022) use a pre-trained GAN inversion to solve image restoration. Pan et al. (2021) use GAN as an image prior and estimates the latent that when fed to GAN will generates the clean image through optimization. The authors shows that this procedure alone does not achieve satisfactory results and propose to finetune the GAN while optimizing for the GAN latent. Yu et al. (2022) opt for encoder-based inversion to solve image inpainting. An encoder is trained to projects corrupted images into a latent space with a pre-modulation for learning more discriminative representation to solve the inpainting task. We note that SHRED is separate enough to the GAN inversion literature. In fact, GAN and diffusion models are fundamentally different. Thus, their inversion through optimization requires different techniques. For example, one of the main differences is that GANs generate samples in "one forward pass", while Diffusion models require multiple iterative steps. Moreover, all IR methods based on GAN inversion, either train an auxiliary network or fine-tune the pre-trained GAN network. In our method, the diffusion model is neither fine-tuned nor trained.

## 3 BACKGROUND

## 3.1 DENOISING DIFFUSION PROBABILISTIC MODELS

Denoising Diffusion Probabilistic Models (DDPM) (Ho et al., 2020) leverage diffusion processes in order to generate high quality image samples. The aim is to reverse the forward diffusion process that maps images to noise, either by relying on a stochastic iterative denoising process or by learning the explicit dynamics of the reverse process, e . g ., through an ODE (Song et al., 2020). More precisely the forward diffusion process maps an image x 0 ∼ p ( x ) to a zero-mean Gaussian x T ∼ N (0 , I ) by generating intermediate images x t for t ∈ (0 , T ] which are progressively noisier versions of x 0 . DDPM (Ho et al., 2020) adopts a Markovian diffusion process, where x t only depends on x t -1 . Given a non-increasing sequence α 1: T ∈ (0 , 1] , the joint and marginal distributions of the forward diffusion process are described by

<!-- formula-not-decoded -->

which implies that we can sample x t simply by conditioning on x 0 with ¯ α t = ∏ s ≤ t α s via

<!-- formula-not-decoded -->

To invert the forward process, one can train a model ϵ θ to minimize the objective

<!-- formula-not-decoded -->

Image samples ˆ x 0 given the initial noise x T , are obtained by iterating for t ∈ [1 , T ] the denoising update

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Song et al. (2020) point out that the quality of generated images highly depends on the total number of denoising steps T . Thus, the inference loop using Eq. (4) becomes computationally expensive. To reduce the computational cost, they propose a Denoising Diffusion Implicit Model (DDIM) (Song et al., 2020), which foregoes the Markovian assumption in favor of a diffusion process

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ert ert

<!-- image -->

Original

<!-- image -->

Reconstructed (PSNR= 44 . 10 ± 1 . 06 )

Original

<!-- image -->

Reconstructed (PSNR= 42 . 43 ± 0 . 67 )

<!-- image -->

Figure 2: Inversion results using our optimization scheme on samples from both CelebA and ImageNet validation datasets. The PSNR mean and standard deviation are computed over 10 runs.

When σ t = 0 for all t , the diffusion process is fully deterministic. This means that when we start from the same noisy sample x T we obtain the same generated image sample. Given x t , one can first predict the denoised observation ˆ x 0 , which is a prediction of x 0 given x t

<!-- formula-not-decoded -->

then we can predict x t -1 from x t and ˆ x 0 using Eq. (5) by setting σ t = 0

<!-- formula-not-decoded -->

with ˆ ω = x t - √ ¯ α t ˆ x 0 √ 1 -¯ α t a direction pointing to x t .

Song et al. (2020) shows that this formulation allows DDIM to use fewer time steps at inference by directly predicting x t -k with k &gt; 1 which results in a more computationally efficient process using

<!-- formula-not-decoded -->

## 4 IMAGE RESTORATION VIA SHRED

In this section, we introduce the general setting of inverse problems that we aim to solve and how generative models can be used to address the ambiguities associated with this class of image restoration tasks. We then present our zero-shot image restoration algorithm SHRED.

## 4.1 AN UNBIASED FORMULATION

Image restoration (IR) can be cast as a Maximum a Posteriori (MAP) optimization problem (Zhang et al., 2020)

<!-- formula-not-decoded -->

where ˆ x ∈ R N x × M x is the restored image and y ∈ R N y × M y is the degraded image, and where p ( y | x ) is the so-called likelihood and p ( x ) the prior distribution. Linear inverse problems describe IR tasks with linear degradation operators H : R N x × M x → R N y × M y and often consider zero-mean additive noise η , such that the observed degraded image y can be written as

<!-- formula-not-decoded -->

IR tasks such as image inpainting, denoising and super-resolution can all be described using specific choices of H and η . For instance, if we vectorize the image x and the output of the operator H , we can describe image inpainting via a pixel projection operator H ( x ) i = x k [ i ] , where i = 1 , . . . , N y × M y , k is a permutation of N x × M x &gt; N y × M y indices, and η = 0 . The associated H and η for each image restoration task 1 considered in this paper are provided in the supplementary material. By

1 A more accurate denomination of the operators used in the linear image restoration problems should be affine rather than linear. Nonetheless, we follow the terminology used in the literature for simplicity.

```
Algorithm 1 SHRED: Image Restoration via DDIM inversion t
```

```
Require: Degraded image y , step size δt , learning rate α Ensure: Restored image ˆ x 0 1: Initialize x T with random from N (0 , 1) . If blind, initialize H θ . 2: for k : 1 → N do 3: Initialize t = T 4: while t > 0 do 5: ˆ x 0 | t = ( x t - √ 1 -¯ α t ϵ θ ( x t , t )) / √ ¯ α t 6: x t -δt = √ ¯ α t -δt ˆ x 0 | t + √ 1 -¯ α t -δt . x t - √ ¯ αt ˆ x 0 | √ 1 -¯ αt 7: t = t δt 8: end while 9: if BLIND then 10: x k +1 T = x k T -α ∇ x T L IR ( x k T , H θ k ) 11: θ k +1 = θ k -α ∇ θ L IR ( x k T , H θ k ) 12: else 13: x k +1 T = x k T -α ∇ x T L IR ( x k T ) 14: end if 15: end for 16: Return the restored image ˆ x 0
```

applying the desired observation model to the MAP formulation, one can aim to recover ˆ x given y by using

<!-- formula-not-decoded -->

where ∥ . ∥ is often a norm that depends on the nature of the noise η , e . g ., a zero-mean Gaussian noise η induces a squared L 2 norm, R is a term corresponding to the prior, and λ &gt; 0 is a coefficient that regulates the interplay between the likelihood and the prior.

In plug-and-play methods (Zhang et al., 2020) the prior R is obtained through a pre-training that aims at denoising the images x . However, using the prior p ( x ) also captures the bias of the specific dataset used for pre-training. Since we do not know where the prior will be used, we argue that a bias could impact negatively the restoration process. Thus, we propose to employ a formulation that implicitly assumes a uniform prior on the domain Ω ⊂ R N x × M x of valid reconstructions, i . e ., R ( x ) = -log( p U ( x )) = -log( const · 1 Ω ( x )) , where 1 Ω ( x ) is 1 for x in the support Ω and 0 otherwise. This choice of R ( x ) is equivalent to considering an unbiased prior over the support of the data distribution and results in the following formulation

<!-- formula-not-decoded -->

To ensure that x ∈ Ω , we parameterize x via the initial noise of a pre-trained diffusion model. In the next section, we present our approach to infer the initial noise such that the generated image minimizes Eq. (12).

## 4.2 AN EFFICIENT DIFFUSION INVERSION

In this work, we adopt DDIM as our pre-trained generative process and we adopt a fully deterministic diffusion process ( σ t = 0 ). We propose a new iterative inversion process, which we summarize in Algorithm 1. We adopt an optimization-based iterative procedure to solve Eq. (12). In contrast to Kawar et al. (2022); Wang et al. (2023); Lugmayr et al. (2022), we explicitly use the correspondence between image samples x 0 and their associated latent vector x T . SHRED starts from an initial noise instance x 0 T ∼ N (0 , I ) and at the end of each iteration k , we update x k T . At each iteration, we first predict ˆ x 0 | t from x t using the pre-trained DDIM model ϵ θ .

<!-- formula-not-decoded -->

Rather than denoising x T iteratively for all the steps used in the pre-training, we make larger steps by using intermediate estimates of ˆ x 0 | t . We define a hyper-parameter δt that controls the number of denoising steps and we can directly jump to estimate x t -δt from ˆ x 0 | t and x t using

<!-- formula-not-decoded -->

Table 1: LPIPS/FID metrics for different degradation tasks on the CelebA dataset. The best and second best methods are in bold and underlined respectively.

| Method          | Inpainting   | SR × 8        | SR × 16       | SR × 32        | Blind Deconv   | CS 1%          | CS 5%         |
|----------|----------|----------|----------|----------|----------|----------|----------|
| DIP Ulyanov et al. (2018)     | 0.373/125.56 | 0.423/195.74  | 0.507/228.14  | 0.581/267.27   | -          | -          | -          |
| SelfDeblur Ren et al. (2020)  | -          | -          | -          | -          | 0.667/343.40   | -          | -          |
| Repaint Lugmayr et al. (2022) | 0.028/7.82   | -          | -          | -          | -          | 0.583/181.57   | 0.389/142.61  |
| DDRM Kawar et al. (2022)      | 0.039/8.69   | 0.239/123.54  | 0.385/139.15  | 0.492/149.12   | -          | 0.567/162.22   | 0.462/149.79  |
| DDNMWang et al. (2023)        | 0.018 / 6.09 | 0.299 /113.76 | 0.376/122.95  | 0.447/143.86   | -          | 0.570/179.43   | 0.319 /105.06 |
| SHRED          | 0.021/6.51   | 0.304/ 78.84  | 0.325 / 85.82 | 0.416 / 110.75 | 0.362 / 90.28  | 0.558 / 139.88 | 0.344/ 85.42  |

Higher step size δt allows us to favor speed while a lower one will favor fidelity. This iterative procedure results in an estimate of x 0 that is differentiable in x T . We denote by f the mapping between x T and x t . The estimate x 0 = f ( x T ) is used to update x T such that it minimizes the image restoration loss. Our method supports both non-blind and blind image degradation.

Non-blind case In the non-blind case, H is supposed to be known, the restoration loss becomes

<!-- formula-not-decoded -->

Given an initial noise x 0 T , the optimization iteration is simply derived as the following

<!-- formula-not-decoded -->

where α is the learning rate.

Blind case In the blind case, H is unknown and can be approximated by a parametric function H θ , the restoration loss becomes

<!-- formula-not-decoded -->

Given an initial noise x 0 T and H θ 0 , the optimization iteration is simply derived as the following

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where α is the learning rate.

One might wonder if the proposed fast inversion procedure suffers from local minima, such that different initial noise samples x T might result in different estimates of x 0 . We show in Fig. 2 that our inversion method is stable and is insensitive to the initialization. We use images from both ImageNet and CelebA and perform denoising without noise ( i . e ., H is the identity function and η = 0 ). As can be observed, the proposed inversion method is stable.

## 5 EXPERIMENTS

## 5.1 IMAGE RESTORATION TASKS

In this work, we showcase our proposed method for different image restoration tasks. Specifically, we evaluate our method on 4 typical noise-free IR tasks, including image inpainting, × 4, × 8, × 16 and × 32 image super-resolution (SR), blind deconvolution, and compressed sensing (CS) with a 1% and 5% sampling rates. For our super-resolution experiments, we adopt a uniform downsampling. For the case of blind deconvolution, we use an anisotropic Gaussian kernel and we follow Ren et al. (2020) and use the same feed-forward network for the kernel estimation. Our compressed sensing experiments use a uniform pixel sampling strategy. Lastly, for our inpainting task, we consider a square centered mask. For each image restoration task, we use the same degradation operator for all methods. We evaluate our method both on ImageNet (Deng et al., 2009) and CelebA (Liu et al., 2018) at 256 × 256 resolution. We follow Chung et al. (2022a;b) and report the LPIPS Zhang et al. (2018) and FID Heusel et al. (2017) values obtained for each experiment.

× 16

Masked

Bicubic

DDNM

Repaint

DDRM

SHRED

00

Table 2: LPIPS/FID metrics for different degradation tasks on the ImageNet dataset. The best and second best methods are highlighted in bold and underlined respectively.

| Method          | Inpainting   | SR × 4       | SR × 8         | SR × 16        | Blind Deconv   | CS 1%         | CS 5%         |
|----------|----------|----------|----------|----------|----------|----------|----------|
| DIP Ulyanov et al. (2018)     | 0.175/75.12  | 0.369/114.83 | 0.544/267.31   | 0.738/376.35   | -          | -          | -          |
| SelfDeblur Ren et al. (2020)  | -          | -          | -          | -          | 0.719/368.94   | -          | -          |
| Repaint Lugmayr et al. (2022) | 0.068/7.85   | -          | -          | -          | -          | 0.769/338.31  | 0.623/292.17  |
| DDRM Kawar et al. (2022)      | 0.042 / 5.32 | 0.302/75.34  | 0.478/195.82   | 0.578/277.86   | -          | 0.784/345.39  | 0.607/295.52  |
| DDNMWang et al. (2023)        | 0.056/7.13   | 0.284 /70.88 | 0.450/183.64   | 0.582/281.15   | -          | 0.700 /315.86 | 0.532 /276.63 |
| SHRED          | 0.063/6.45   | 0.329/ 65.78 | 0.394 / 152.74 | 0.470 / 183.11 | 0.467 / 185.37 | 0.736/ 295.12 | 0.642/ 262.17 |

## 5.2 EXPERIMENTAL RESULTS

Our experimental results are summarized in Tables 1 and 2 for CelebA and ImageNet, respectively. We report the performance of SHRED as well as DIP (Ulyanov et al., 2018), SelfDeblur (Ren et al., 2020), RePaint (Lugmayr et al., 2022), DDRM (Kawar et al., 2022) and DDNM (Wang et al., 2023). We showcase the versatile nature of SHRED and its ability to restore degraded images with a higher level of details thanks to its explicit use of the correspondence property of the underlying pre-trained DDIM Song et al. (2020) model. We report the performance of super-resolution methods in Tables 1 and 2 for the × 8, × 16 and × 32 upscaling factors on CelebA and × 4, × 8 and × 16 on ImageNet respectively.

Figure 3: Visual results of inpaintings on CelebA.

<!-- image -->

For both datasets, we observe that our method is ranked first in both LPIPS and FID. We observe a similar pattern as in our super-resolution experiments for compressed sensing where SHRED generates more detailed images and achieves a better FID while being on par or slightly worse in terms of LPIPS.

Figure 4: Visual results of super-resolution with different upsampling factors.

<!-- image -->

DDNM

Bicubic

SHRED

DDNM

GT

SHRED

Blurry

SelfDeblur

SHRED

GT

For the case of blind deconvolution, SHRED largely outperforms SelfDeblur (Ren et al., 2020) both in terms of FID and LPIPS which highlights its ability to also tackle blind restoration tasks through joint optimization. We illustrate this in Fig. 5 on two blurred images using a large blur kernel. Due to the significant blur, SelfDeblur is unable to restore the degraded image. SHRED on the other hand successfully deblurs the image especially around the main object in the image. Lastly, in the case of image inpainting SHRED is ranked second and third on CelebA and ImageNet respectively. On CelebA, SHRED achieves a comparable FID and LPIPS to DDNM while significantly outperforming other inpainting methods. We showcase the different considered methods in Fig. 3, where we observe that SHRED is capable of generating highly detailed and consistent features in the masked regions.

Figure 5: Visual results of blind deconvolution using a random Gaussian kernel with size 41 × 41 . Figure 6: Robustness of SHRED to the initial noise x T .

<!-- image -->

Figure 7: Restoring real-world photos using SHRED. Degraded images are collected from the internet

<!-- image -->

## 6 ABLATIONS STUDY

## 6.1 EFFECT OF DIFFERENT HYPER-PARAMETERS

We study the influence of hyper-parameters: the learning rate α , the number of iterations N and the step size δt , on SHRED performance. Table 4 shows the effect of varying the step size δt on both FID and the computation time. δt = 100 gives the best trade-off between the speed and the visual quality. In Table 5, we report the FID values achieved by SHRED for different learning rates. SHRED acheives competitive FID with the different learning rates and this shows that it is robust enough to the choice of learning rate. Table 6 shows SHRED performance for different values of N (the total number of iterations). SHRED needs around 100 iterations to achieve the best FID indicating that a limited number of iterations is sufficient to produce a decent reconstruction. We note that N can be set automatically using the criteria L IR &lt; ϵ . We empirically observe that tuning on a small set generalizes well in the zero-shot setting. For all IR tasks, δt = 100 and α = 0 . 003 produces competitive performance.

Table 3: Runtime (in seconds) comparison of different methods on CelebA per image on superresolution. All experiments are conducted on a Geforce GTX 1080 Ti.

| Method          |   Time[s] |
|----------|----------|
| DIP Ulyanov et al. (2018) |       520 |
| DDRM Kawar et al. (2022)  |        13 |
| DDNMWang et al. (2023)    |        12 |
| Naive Inversion          |      5750 |
| SHRED          |        55 |

## 6.2 SHRED ROBUSTNESS

## Robustness to the initial noise

We fix the degraded image and we test our method by varying the initial noise x 0 T . Figure6 shows that SHRED is robust to the input noise initialization and that for each time it can output a different plausible solution.

## Robustness to real-world degradations

We apply our method to the task of old image enhancement where the images are real and collected from internet depicting complex and unknown degradation (noise, jpeg artifacts, blur..). The problem is framed as the mixture of image colorization and super-resolution. Figure 7 shows SHRED is able to enhance the image quality and that is robust to the different degradation.

## 6.3 COMPUTATIONAL COST

In Table 3, we report the runtime per image of the different compared methods. We denote by Naive Inversion a similar approach as SHRED that works by inverting the fully unrolled diffusion model. SHRED is × 100 time faster than the Naive Inversion . Despite being an iterative method, SHRED has a reasonable runtime making it a practical zero-shot IR method. We note that the reported speed is using δt = 100 and a further speed boost is possible by opting for a higher value of δt at a small reduction of the visual quality.

| α   |   5 × 10 - 4 |   10 - 3 |   2 . 5 × 10 - 3 |   5 × 10 - 3 |
|-----|----------|----------|----------|----------|
| FID |        87.85 |    72.36 |          74.52 |        77.85 |

Table 5: Effect of the learning rate α on the visual quality of SR on CelebA.

## 7 CONCLUSION

Wehave introduced SHRED, a zero-shot framework for solving IR tasks by using pre-trained diffusion generative models as learned priors. Our method exploits the deterministic correspondence between noise and images in DDIM by casting the inverse restoration problem as a latent estimation problem. We leverage the capabilty of DDIM to skip ahead in the forward diffusion process and provide an efficient diffusion inversion in the context of IR. SHRED is comprehensively utilized on various tasks such as super-resolution, inpainting, blind-deconvolution, and compressed sensing , demonstrating the capabilities of SHRED on unified image restoration.

## REFERENCES

Liang Chen, Faming Fang, Shen Lei, Fang Li, and Guixu Zhang. Enhanced sparse model for blind deblurring. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XXV , pp. 631-646. Springer, 2020a.

N

FID

50

78.03

75

75.63

100

74.31

125

73.85

150

74.42

Table 6: Effect of the number of iterations N on the visual quality of SR on CelebA.

| Step size ( δt )    |      50 |     100 |    200 |
|----------|---------|---------|--------|
| FID          |    6.38 |    6.51 |   8.32 |
| Time/Iteration (ms) | 2330    | 1150    | 576    |

Table 4: Effect of the step size δt on the visual quality and the runtime of image inpainting on CelebA.

- Liang Chen, Faming Fang, Jiawei Zhang, Jun Liu, and Guixu Zhang. Oid: Outlier identifying and discarding in blind image deblurring. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XXV 16 , pp. 598-613. Springer, 2020b.
- Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models. arXiv preprint arXiv:2108.02938 , 2021.
- Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687 , 2022a.
- Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems , 2022b. URL https://openreview. net/forum?id=nJJjv0JDJju .
- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition , pp. 248-255. Ieee, 2009.
- Ben Fei, Zhaoyang Lyu, Liang Pan, Junzhe Zhang, Weidong Yang, Tianyue Luo, Bo Zhang, and Bo Dai. Generative diffusion prior for unified image restoration and enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 9935-9946, 2023.
- Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems , 30, 2017.
- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems , 33:6840-6851, 2020.
- Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. In Proc. NeurIPS , 2021.
- Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. arXiv preprint arXiv:2201.11793 , 2022.
- Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Large-scale celebfaces attributes (celeba) dataset. Retrieved August , 15(2018):11, 2018.
- Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 11461-11471, 2022.
- Xingang Pan, Xiaohang Zhan, Bo Dai, Dahua Lin, Chen Change Loy, and Ping Luo. Exploiting deep generative prior for versatile image restoration and manipulation. IEEE Transactions on Pattern Analysis and Machine Intelligence , 44(11):7474-7489, 2021.
- Dongwei Ren, Kai Zhang, Qilong Wang, Qinghua Hu, and Wangmeng Zuo. Neural blind deconvolution using deep priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 3341-3350, 2020.
- Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 , 2020.
- Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems , 32, 2019.
- Yang Song, Liyue Shen, Lei Xing, and Stefano Ermon. Solving inverse problems in medical imaging with score-based generative models. arXiv preprint arXiv:2111.08005 , 2021.
- Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Deep image prior. In Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 9446-9454, 2018.
- Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. The Eleventh International Conference on Learning Representations , 2023.
- Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang. Gan prior embedded network for blind face restoration in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 672-681, 2021.

- Yongsheng Yu, Libo Zhang, Heng Fan, and Tiejian Luo. High-fidelity image inpainting with gan inversion. In European Conference on Computer Vision , pp. 242-258. Springer, 2022.
- Bowen Zhang, Shuyang Gu, Bo Zhang, Jianmin Bao, Dong Chen, Fang Wen, Yong Wang, and Baining Guo. Styleswin: Transformer-based gan for high-resolution image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pp. 11304-11314, 2022.
- K. Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, and Radu Timofte. Plug-and-play image restoration with deep denoiser prior. IEEE Transactions on Pattern Analysis and Machine Intelligence , 44: 6360-6376, 2020.
- Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 586-595, 2018.

## A APPENDIX

## A.1 IMPLEMENTATION DETAILS

For evaluation, We choose a set of 100 random images from ImageNet 1K and CelebA 1K datasets with image size 256 × 256 for validation. For a fair comparison, we use the same pre-trained denoising models used in Wang et al. (2023) and apply them to all the diffusion-based methods. We use the Adam optimizer all IR tasks. The step size δt is set to 100 . All experiments are conducted on a Geforce GTX 1080 Ti.

Figure 8: An application of SHRED on a non-linear inverse problem (non-uniform deblurring). The top row shows degraded images with non-uniform blur. The second row depicts the results of our method. The bottom row shows the ground truth.

<!-- image -->

## A.2 SUPER-RESOLUTION

For superresolution, we use × 4 , × 8 , and × 16 uniform downsampling factors for ImageNet and × 8 , × 16 , and × 32 uniform downsampling factors for CelebA. During training, we use Adam as our optimizer with a learning rate of 0 . 003 and 50 iterations. We also add comparisons with both OID Chen et al. (2020b) and Chen et al. Chen et al. (2020a) . Additional visual results are depicted in Fig. 11, Fig. 12 and Fig. 13. While other methods tend to generate over-smoothed images especially at higher downsampling factors, our method (SHRED) outputs sharper and more realistic images. For instance, SHRED is able to generate detailed grass and hair textures in Fig. 11 and Fig. 12 especially at × 16 upscaling where other methods fail.

## A.3 BLIND DECONVOLUTION

For Blind Deconvolution, we apply a random anisotropic Gaussian kernel of size 41x41. During training, we use Adam as our optimizer with a learning rate of 0 . 003 and 150 iterations. Visual results are shown in Fig. 16 and Fig. 17. Despite the heavy blur, SHRED is able to better deblur the main object present in the considered image compared to other methods.

## A.4 COMPRESSED SENSING (CS)

For Compressed Sensing, we randomly subsample 1% and 5% of the total available pixels. During training, we use Adam as our optimizer with a learning rate of 0 . 01 and 150 iterations. More visual results are depicted in Fig. 14 and Fig. 15. SHRED performs better than the other methods at preserving the identity of the original image especially at the higher subsampling rates. Indeed, we see in the second and forth rows of Fig. 14 that other method hallucinate new faces instead of reconstructing the correct one.

## A.5 INPAINTING

For Inpainting, we apply a square mask for CelebA and a text mask for ImageNet. During training, we use Adam as our optimizer with a learning rate of 0 . 01 and 200 iterations. We show additional results with different masks in Fig. 9 and Fig. 10.

Masked brem ipsum

onsectetur ac elus od temp

et dolere mag!

nim veniam orem ipsum d

consectetur ac eiusmodtemp

et dolore magr min m veniam

Repaint

DDRM

DDNM

SHRED

GT

Figure 9: Visual results of inpaintings on CelebA.

<!-- image -->

Bicubic

Lorem Ipsum c

DDNM

DDNM

SHRED

SHRED

consectetu

Ismo temp

core magr ninına

venian um o

urac mp

nam

Bicubic x4

Masked x8

× 16

Repaint

Masked

DDRM

× 16

×32

Repaint

DDRM

DDNM

SHRED

Figure 10: Visual results of inpaintings on ImageNet.

<!-- image -->

Figure 11: Visual results of super-resolution with different upsampling factors.

DDNM

SHRED

GT

GT

Bicubic

Bicubic

Bicubic

Bicubic

DDNM

DDNM

DDNM

DDNM

SHRED

SHRED

SHRED

SHRED

x4

x4

X8

птак

x8

× 16

×32

× 16

× 16

×32

x8

× 16

Figure 12: Visual results of super-resolution with different upsampling factors.

<!-- image -->

Figure 13: Visual results of super-resolution with different upsampling factors.

<!-- image -->

5 %

1 %

5 %

1%

Masked

Repaint

DDRM

DDNM

SHRED

GT

Figure 14: Visual results of CS on CelebA.

<!-- image -->

1%

5 %

1%

Masked

Blurry

SelfDeblur

Repaint

DDRM

DDNM

SHRED

GT

OID

Chen et al.

SHRED

GT

Figure 15: Visual results of CS on ImageNet.

<!-- image -->

Figure 16: Visual results of blind deconvolution using a random Gaussian kernel with size 41 × 41 on CelebA.

<!-- image -->

Blurry

SelfDeblur

OID

Chen et al.

SHRED

GT

Figure 17: Visual results of blind deconvolution using a random Gaussian kernel with size 41 × 41 on ImageNet.

<!-- image -->