## Fast Sampling via Discrete Non-Markov Diffusion Models with Predetermined Transition Time

## Zixiang Chen Huizhuo Yuan Yongqian Li Yiwen Kou Junkai Zhang Quanquan Gu

Department of Computer Science University of California, Los Angeles Los Angeles, CA 90095

{chenzx19,hzyuan,yongqianl,evankou,jkzhang,qgu}@cs.ucla.edu

## Abstract

Discrete diffusion models have emerged as powerful tools for high-quality data generation. Despite their success in discrete spaces, such as text generation tasks, the acceleration of discrete diffusion models remains under-explored. In this paper, we propose discrete non-Markov diffusion models (DNDM), which naturally induce the predetermined transition time set. This enables a training-free sampling algorithm that significantly reduces the number of function evaluations (i.e., calls to the neural network), making the sampling process much faster. Furthermore, we study the transition from finite to infinite step sampling, offering new insights into bridging the gap between discrete and continuous-time processes for discrete diffusion models. Extensive experiments on natural language generation and machine translation tasks demonstrate the superior performance of our method in terms of both generation speed and sample quality compared to existing methods for discrete diffusion models. Codes are available at https://github.com/ uclaml/DNDM .

## 1 Introduction

Diffusion-based generative models, as first introduced by Sohl-Dickstein et al. (2015), have shown remarkable capabilities in generating high-quality samples across various domains, including images (Ho et al., 2020; Song and Ermon, 2020), audio (Chen et al., 2020; Kong et al., 2020), and videos (Ho et al., 2022). The diffusion model utilizes an innovative approach comprising a forward process that gradually transforms training data into pure noise and a reverse process that reconstructs clean data from the noise. Throughout the training phase, the model optimizes a neural network by minimizing an objective derived from maximum likelihood estimation. Once trained, the model can generate samples using various decoding strategies, including implicit dynamics (Song et al., 2020a), analytical processes (Bao et al., 2022), or differential equation solvers (Song et al., 2020b; Liu et al., 2022; Lu et al., 2022). In particular, Song et al. (2020a) introduced the denoising diffusion implicit model (DDIM), providing a non-Markov and de-randomized version of the Denoising Diffusion Probabilistic Model (DDPM) (Sohl-Dickstein et al., 2015; Ho et al., 2020), which enables faster generation of high-quality samples.

Although diffusion models were initially introduced for both discrete and continuous-state spaces (Sohl-Dickstein et al., 2015), these studies have largely focused on Gaussian diffusion processes in continuous-state spaces. Recently, Discrete Denoising Diffusion Probabilistic Models (D3PMs) (Austin et al., 2021) working in discrete-state spaces have gained increasing interest due to their applications in diverse areas such as text generation (Hoogeboom et al., 2021b), medical record generation (Ceritli et al., 2023), and protein design (Gruver et al., 2024). These models, which are distinct from their Gaussian counterparts, employ discrete noises, such as the multinomial distribution, for diffusion processes. Very recently, Zheng et al. (2023) introduced a reparameterized diffusion

model (RDM) that can improve sampling speed and sample quality in text generation tasks. However, their proposed algorithm is a training-based approach. Compared with diffusion models using Gaussian noise, discrete diffusion models remain under-studied, especially regarding training-free sampling acceleration.

In this work, we introduce a training-free approach aiming at enhancing the sampling speed of discrete diffusion models. This approach stems from a unique characteristic of discrete diffusion models: unlike continuous diffusion models, which typically employ Gaussian noise for data corruption (Ho et al., 2020; Song and Ermon, 2020; Song et al., 2020b,a), discrete diffusion models often use categorical white noises (Hoogeboom et al., 2021b; Austin et al., 2021; Zheng et al., 2023).

By delving into this special property, we develop a discrete non-Markov diffusion model, together with a design of accelerated algorithm. Notably, this new sampling technique does not require any modifications to the training objec-

Table 1: Cross Comparison of Diffusion Models.

|          | Continuous          | Discrete          |
|----------|----------|----------|
| Markov     | DDPM (Sohl-Dickstein et al., 2015) | D3PM Austin et al. (2021) |
| Non-Markov | DDIM (Song et al., 2020a)          | DNDM (Ours)          |

tive of diffusion models and is, therefore, training-free. Our contributions are summarized as follows:

- We propose discrete non-Markov diffusion models (DNDM), which naturally induces a set of latent variables T , termed as the transition time set . This key feature enables us to develop a training-free sampling algorithm that can accelerate a large family of discrete diffusion models. Importantly, DNDM preserves the essential properties of the original discrete diffusion model: for any diffusion trajectory { x t } starting from real data x 0 , it provably maintains both the marginal distribution q ( x t ) and the conditional distribution q ( x 0 | x t ) . Our method can accelerate the two most widely used discrete diffusion models: multinomial diffusion (Hoogeboom et al., 2021b) and absorbing diffusions (Austin et al., 2021). Similar to how DDIM introduces a de-randomized, faster sampling algorithm compared to DDPM in continuous space, DNDM achieves acceleration through a predetermined transition time set in discrete space (See Table 1).
- Based on the predetermined transition time set T in DNDM, we design an accelerated sampling algorithm that reduces the required number of neural network function evaluations. In a standard T time-step discrete diffusion process, while D3PM, including Multinomial (Ho et al., 2020) and absorbing state discrete sampling (Austin et al., 2021), requires evaluating the neural network function T times, our approach only requires |T | function evaluations, where |T | is the cardinality of the transition set T . Moreover, |T | is provably less than T and approaches O (1) as T goes to infinity. We provide both theoretical analysis and empirical experiments showing that the improvement in the number of function evaluations (NFE) is significant. Notably, our algorithm is about 3 × faster than baselines for T = 50 and about 30 × faster for T = 1000 while preserving the sample quality.
- To further illustrate the effectiveness of DNDM, we explore the limit as T →∞ and introduce an infinite-step sampling algorithm. With a pretrained neural network, we can generate an initial noise x T and a transition time set T ⊆ [0 , 1] with infinitesimal spacing, such that |T | = O (1) . This enables the generation of the real data distribution with only |T | neural network evaluations. This study offers new insights into bridging the gap between discrete and continuous-time processes for discrete diffusion models.

Notation. We use |T | to denote the cardinality of the set T (excluding repeated elements). We use lowercase letters to denote scalars, boldface lowercase letters to denote vectors, and boldface uppercase letters to denote matrices. The notation 1 : N indicates the sequence from 1 through N . The symbol q designates the real distribution in a diffusion process, while p represents the distribution during sampling. With its success probability inside the parentheses, the Bernoulli distribution is denoted by Bernoulli( · ) . We further use Cat( x ; p ) to denote a categorical distribution over a one-hot row vector x with probabilities given by the row vector p .

## 2 Background

In this section, we provide the background of discrete diffusion models. We begin by introducing the discrete Markov diffusion model, designed for handling categorical random variables. Specifically,

consider a diffusion model trying to generate distributions over a discrete random variable x ∈ R K that is one-hot encoded with K categories, i.e., x can be chosen as one of K categories, and for any k ∈ [ K ] , x is categorized as k if x aligns with the standard basis vector e k . The sequence { x t } T t =0 represents how this random variable changes over time 0 ≤ t ≤ T , starting from an x 0 ∈ R K drawn from the real distribution q data . In this paper, we focus on the two most widely used D3PMs: multinomial diffusion (Hoogeboom et al., 2021b) and absorbing diffusions (Austin et al., 2021).

Forward Process. During the forward process, the real distribution q data is gradually transformed into a noise distribution named q noise . The transformation occurs through T steps, with T intermediate latent variables x 1 , . . . x T and update rules given by:

<!-- formula-not-decoded -->

Here b t is randomly drawn from a Bernoulli distribution with parameter β t , denoted by b t ∼ Bernoulli( β t ) , and w t is randomly drawn from the noise distribution q noise , while for different t the samples are independent. In this work, we focus on cases where the noise q noise can be either a uniform distribution over the vocabulary { 1 , 2 , . . . , K } (Hoogeboom et al., 2021b), or a point mass with all of the probability mass lying on an absorbing state (Austin et al., 2021). Following this notation, the process in (1) defines a Markov process characterized by the transition kernel

<!-- formula-not-decoded -->

Moreover, the Markov chain property allows us to get samples x 0: t from x 0 by multiplying the transition probabilities at each step as p ( x 1: t | x 0 ) = ∏ t i =1 q ( x t | x t -1 ) . It further leads to the following marginal distribution.

<!-- formula-not-decoded -->

where α t := Π t s =1 β s is determined by the sequence of β t of our choice and decreases from 1 to 0 .

Reverse Process. Given the forward Markov process, the reverse process can be derived by Bayes' rule (Hoogeboom et al., 2021b; Austin et al., 2021; Zheng et al., 2023). The conditional probability q ( x t -1 | x 0 , x t ) can be determined by q ( x t -1 | x 0 , x t ) = q ( x t | x t -1 ) q ( x t -1 | x 0 ) /q ( x t | x 0 ) . The reverse process can be used for synthetic data generation by sampling from the noise distribution q noise and repeatedly applying a learned predictor (neural network) p θ ( ·| x t ) parameterized by θ :

<!-- formula-not-decoded -->

We note that the reverse process q ( x t -1 | x t , ̂ x 0 ) is stochastic and thus requires function evaluation at every step.

Training the Neural Network. The neural network p θ ( ·| x t ) that predicts ̂ x 0 is trained by maximizing the evidence lower bound (ELBO) (Sohl-Dickstein et al., 2015),

<!-- formula-not-decoded -->

Here KL denotes Kullback-Liebler divergence and the last term E q ( x T | x 0 ) KL( q ( x T | x 0 ) ∥ q noise ( x T )) equals zero. Building on this foundation, Austin et al. (2021) introduced an auxiliary denoising objective, which refines the data predictions x 0 at each time step. Since this paper primarily focuses on reverse sampling, we leave detailed discussions of these losses to Appendix B.

## 3 Discrete Non-Markov Diffusion Models (DNDM)

## 3.1 Forward and Reverse Process

In this section, we introduce a non-Markov process such that the joint distribution of ( x 0 , x t ) remains the same as the one defined with Markov process in Section 2. The new process aims to gradually

transform input data q data to the noise distribution q noise through T intermediate latent variables x 1 , . . . x T with the following process:

<!-- formula-not-decoded -->

where b t is independently drawn from the Bernoulli distribution Bernoulli( β t ) and w is drawn from the noise distribution q noise . The only difference between (6) and (1) is that we replace w t in (1) by w , which is time-invariant during the diffusion. Therefore, the process in (6) becomes non-Markov since q ( x t | x t -1 , . . . , x 0 ) doesn't necessarily equals q ( x t | x t -1 ) . The following theorem shows that the conditional distribution q ( x t | x 0 ) remains unchanged.

Theorem 3.1. For the non-Markov process in (6) , we have

<!-- formula-not-decoded -->

where α t := Π s i =1 β s is specified to decrease from 1 to 0 .

Using the Bayes' rule, we have q ( x 0 | x t ) ∝ q ( x t | x 0 ) q ( x 0 ) . Consequently, the condtional distribution q ( x 0 | x t ) remains consistent with the one induced by the process process in (1). Therefore, neural network p θ ( ·| x t ) trained by the Markov process in (1), remains applicable to our non-Markov process (6) (see Appendix B for detail).

Based on the discrete non-Markov diffusion model, we can give a simple characterization of the reverse process by introducing the transition time.

Definition 3.2. Transition time τ is the time that the token x t transition from x 0 to noise, i.e., τ := min t { t | b t = 0 } .

Remark 3.3 . The concept of transition time has also been introduced in Hoogeboom et al. (2021a). However, Hoogeboom et al. (2021a) restricts the transition time to be the first time of entering the absorbing state, which is only applicable to absorbing diffusion. Our definition is more general and applicable to discrete diffusion with various noise including multinomial diffusion.

Given the transition time τ , the forward process reduces to:

<!-- formula-not-decoded -->

which shows that the token will be a real token x 0 before the time τ and will be the noise w after the transition time. Since token only get changed at the transition time τ , we can derive a reverse process based on (7),

̸

<!-- formula-not-decoded -->

Therefore, the process in (8) is de-randomized given transition time τ . Specifically, after independently sampled transition times τ , x t -1 becomes deterministically known and fixed if we observe x 0 and x t . It is also worth noting that given x 0 and τ , the exact reverse process (8) is Markovian, since x t -1 solely depends on x 0 , τ, x t . Plugging (8) into (4) gives the generation process. We can prove the ELBO of the DNDM is equivalent to the ELBO of the original process (5) up to some constant, which further supports the neural network p θ ( ·| x t ) trained by the Markov process in (1), remains applicable to DNDM. (See Appendix B.3 for details).

Remark 3.4 . (7) and (8) suggest that even though there are T distinct time steps, not every time in the range 1 : T is crucial for capturing the process. Therefore, our primary focus should be on the most significant time step, i.e., the transition time τ , enabling faster reverse sampling. We further note that although transition happens only at time τ , the transition time is random, differs across runs, and covers the full range from 1 to T on average.

Remark 3.5 . While Song et al. (2020a) proposed a non-Markov multinomial diffusion model in Appendix A, DDIM and DNDM are fundamentally different models when specialized to multinomial diffusion. DDIM's discrete process remains stochastic at every step, even with deterministic noise scheduling. In contrast, DNDM achieves full de-randomization by pre-determined transition time τ (Equation 8 in our paper). By sampling these transition times upfront, DNDM establishes a predetermined transition time set that guides the sampling process, enabling deterministic evolution and faster sampling speed even under the same number of sampling steps, which is not reported under DDIM framework. For detailed technical comparison, see Appendix B.1.

## 3.2 Accelerated Reverse Sampling

In this section, we demonstrate that sampling from DNDM can lead to accelerated reverse sampling. Although our algorithm is quite general, we focus on text generation in the presentation.

In Section 3.1, we only consider the case of a single token x ∈ R K being one hot encoding of K categories. In real applications, we are interested in generating a sentence with multiple tokens. So, we extend the terminology in Section 3.1, and we denote the sequence of tokens at t -th time step to be x t, 1: N = [ x t, 1 , . . . , x t,N ] where x t,n is the n -th token and N is the sequence length. The noise will be added to each token in a sequence independently. Therefore, each token will have its own transition time defined in Definition 3.2. We denote the transition time for each token x n to be τ n and further denote the transition time set T := { τ n } N n =1 . Given the transition times τ n ∈ T , our DNDM can now be extended to the sequence with multiple tokens

̸

<!-- formula-not-decoded -->

Learning the Reverse Process. We first generate the transition times τ n for n ∈ [ N ] , then we follow (9) to generate the learned reverse process. Since x 0 ,n is unknown in the process, we use the neural network evaluation p θ ( ·| x t ) obtained in Section 3.1 to predict x 0 ,n . In detail, the noisy sequence x t, 1: N is fed into p θ ( ·| x t, 1: N ) and the prediction tokens ̂ x 0 , 1: N ∼ p θ ( ·| x t, 1: N ) are collected.

Transition time. Transition time, denoted by τ , is crucial in our reverse process. This is because the reverse sampling becomes deterministic upon using (9). Each instance of transition time τ is a random variable within the set { 1 , 2 , . . . , T } . Let's assume it follows the distribution D τ . Given the schedule { α t } T t =0 , we can derive the distribution for D τ .

Theorem 3.6. Each specific transition time τ n in Definition 3.2 is independent. Furthermore, they collectively adhere to the distribution D τ , which obeys the rule P ( τ n = t ) = α t -1 -α t .

From Theorem 3.6, we discern that the nature of the diffusion model scheduler, α t , clarifies the distribution of τ . Take the linear schedule as an example, as given by Austin et al. (2021), the relationship is α t = 1 -t/T . This translates to P ( τ n = t ) = 1 /T for every t in the range 1 to T . As a result, transition time distributes uniformly across each moment in the set { 1 , . . . , T } . Generally, if we express α t as g ( t/T ) , then we can simplify to P ( τ n = t ) = g (( t -1) /T ) -g ( t/T ) , which further refines to (1 /T ) | g ′ ( t/T ) | + o (1 /T ) . This indicates that transitions are more likely where | g ′ | is large.

In practice, we observed that the shape of the transition time does not need to exactly match the theoretically predicted schedule D τ in Theorem 3.6. Algorithm 1 works even if D τ is unknown. In particular, we can approximate the schedule with a Beta distribution by first sampling a time t ∈ [0 , 1] from a Beta distribution, then adjusting these samples to fit by multiplying by T and rounding the result to obtain an integer.

Accelerated Sampling. According to (9), a token x t -1 ,n is updated only if step t is the transition time for the n -th token. If step t is not the transition time for any token, the sentence from the previous step can be directly copied: x t -1 , 1: N = x t, 1: N . As a result, there is no need to do a function evaluation for the current step. Our attention, therefore, can be solely centered

## Algorithm 1 Sampling From DNDM

```
Require: Trained prediction function p θ , q noise , D τ 1: for n = 1 . . . N do 2: Initiate each token x T,n ∼ q noise 3: Initiate the transition time τ n ∼ D τ 4: end for 5: Collect transition time set T = { τ n } N n =1 6: for t = T . . . 1 do 7: if t ∈ T then 8: Generate ˜ x 0 , 1: N from p θ ( ·| x t, 1: N ) 9: for n = 1 . . . N do 10: Update x t -1 ,n based on condition of τ n 11: end for 12: else 13: Update x t -1 , 1: N = x t, 1: N 14: end if 15: end for 16: Return x 0 , 1: N
```

on the transition set T , necessitating function evaluations only for t within T . For our method, when N is fixed while T → ∞ , the total NFE |T | will reach N . On the other hand, when T is fixed and N → ∞ , the NFE T will reach T (See Theorem D.1 for detail). It is worth noting that the auto-regressive diffusion model (ARDM) (Hoogeboom et al., 2021a) can also achieve at most N NFE when T = ∞ . However, ARDM only focuses on infinite time steps, while our method here is

able to accelerate sampling for finite time steps. More detailed discussion and theoretical analysis can be found in Section D, where additional experiments also demonstrate that our DNDM achieves an NFE that is less than half of the original Markov sampling method for discrete diffusion.

By incorporating the forward process with different noises, we can develop DNDM-Multi and DNDMAbsorb, which accelerate the Multinomial and Absorbing sampling methods respectively. Recent works have demonstrated that the quality of samples can be enhanced by utilizing supplementary information derived from the neural network, (Ghazvininejad et al., 2019; Savinov et al., 2021; Chang et al., 2022; He et al., 2022; Zheng et al., 2023). Our DNDM can also be improved using this idea. We call it a discrete non-Markov Diffusion Model with Top-k Transition Time (DNDMk ). Due to the limit of the pages, we leave the detailed Algorithm and discussion to Appendix E.

## 3.3 Continous-time (Infinite Step) Reverse Sampling

In the context of continuous state spaces, continuous-time processes have been proposed to accommodate algorithms that offer faster sampling speeds and enhanced sample quality (Jolicoeur-Martineau et al., 2021; Zhang and Chen, 2022; Salimans and Ho, 2022; Chung et al., 2022; Song et al., 2020b; Dockhorn et al., 2021). However, the application of continuous-time schemes to discrete-state spaces remains largely unexplored. Campbell et al. (2022) first developed a continuous framework for discrete-time diffusion for the Markovian process and randomized sampling, but not in our nonMarkovian setting. In this section, we investigate the transition from finite to infinite step sampling, providing new insights into bridging the gap between discrete and continuous-time processes for discrete diffusion models.

Continuous-time Forward and Backward process. Recall that the forward process described in (6) can be sampled from x 0 ,n through the following process:

<!-- formula-not-decoded -->

In the previous section, we are constrained to discrete time steps, where we must define a maximum step, denoted by T . The values of x t are computed only for t = 1 , . . . , T . As a result, during the training process, it is only possible to predict x 0 at these predetermined time steps. This constraint confines the computation of our reverse process exclusively to these fixed time stamps. To derive the continuous limit of (10), for each T we rescale (10) to a diffusion process on [0 , 1] , e.g., x T,n = ̂ x 1 ,n , x 0 ,n = ̂ x 0 ,n , and x t,n = ̂ x t/T,n . Therefore, when T →∞ , ̂ x t,n represents the continuous process that has values at arbitrary t ∈ [0 , 1] . If the choice of α t for

Algorithm 2 Sampling from DNDM-C

Require: Trained prediction function p θ , q noise , D τ

- 1: for n = 1 . . . N do
- 2: Initiate each token x T,n ∼ q noise
- 3: Initiate the transition time τ n ∼ D τ and order them as τ n 1 &lt; . . . &lt; τ n N
- 4: end for
- 5: for k = N . . . 1 do
- 6: Generate ˜ x 0 , 1: N from p θ ( ·| x τ n k , 1: N , τ n k )
- 7: for n = 1 . . . N do
- 8: Update x τ n k -1 ,n based on condition of τ n
- 9: end for
- 10: end for
- 11: Return x 0 , 1: N

each T is scale-invariant, we can define a continuous function α ( t ) as the continuous α schedule of the discrete counterpart 1 . More specifically, we obtain

<!-- formula-not-decoded -->

For the reverse-time process, we define the transition time set T := { τ n } N n =1 consistent with Theorem 3.6 and sample it from P ( τ n = t ) = -α ′ ( t ) (we always use decreasing α ( t ) ). With T defined, the updates to x t,n only occur at { τ n } . Consequently, we arrange τ n to obtain an ordered sequence τ n k , where τ n 1 &lt; τ n 2 &lt; . . . &lt; τ n N . When omitting the infinitely many time steps between τ n k and τ n k -1 , the resulting reverse process is then given by:

̸

<!-- formula-not-decoded -->

for all n ∈ [ N ] . The detailed algorithm named DNDM-C is shown in Algorithm 2.

1 If we represent α t with maximum step T as α t ( T ) , the scale-invariant property states that α ct ( cT ) = α t ( T ) . The simplest example of such an α t schedule is α t ( T ) = 1 -t/T , under which α ( t ) = 1 -t .

Remark 3.7 . Autoregressive Diffusion Model (ARDM) (Hoogeboom et al., 2021a) is a discrete diffusion model built upon the autoregressive nature of data. ARDM is shown to be equivalent to a continuous-time absorbing diffusion model and thus provides a unique perspective for discrete diffusion. For continuous-time ( T = ∞ ) reverse sampling, both ARDM and our method achieve N NFEs. Unlike ARDM which is limited to absorbing-state transitions, our method provides a unified framework including both absorbing and multinomial diffusions, applicable to both finite time and continuous time diffusions. For infinite timesteps, Hoogeboom et al. (2021a) also proposed an advanced parallelizing technique that can reduce NFE according to the log-likelihood, which we have not considered in DNDM-C.

## 4 Experiments

In this section, we evaluate DNDM and demonstrate its superior performance on two types of tasks: conditional sequence-to-sequence text generation (i.e., machine translation) and unconditional text generation. For the fairness of comparison, all the experiments are conducted using a single NVIDIA RTX A6000 GPU with 48 GB memory. Additional experiment details are provided in Appendix F.

## 4.1 Conditional Text Generation

We evaluate DNDM's effectiveness on conditional text generation through machine translation tasks. Following Zheng et al. (2023), we use Byte Pair Encoding (BPE) (Sennrich et al., 2016) to create a shared vocabulary of words and subwords from both source and target languages. We implement our experiments using FairSeq (Ott et al., 2019), which employs an encoder-decoder architecture. The model uses bi-directional self-attention blocks without causal masking, allowing tokens to attend to both past and future positions during training and inference. The encoder processes the source text, while the decoder generates the target translation.

Datasets. We use the following three datasets to compare with the baselines for machine translation tasks: (1) IWSLT14 DE -EN (Cettolo et al., 2014), a dataset with German as the source language and English as the target language. It consists of 174272 examples (sentence pairs), and each of the validation set and the testing set accounts for 7283 and 6750 of the dataset; (2) WMT14 EN -DE (Bojar et al., 2014), which is an English-to-German translation dataset consisting of 3967182 examples. Each of the validation set and the testing set accounts for 3000 and 3003 of the dataset; and (3) WMT16 EN -RO (Bojar et al., 2016), which is an English-to-Russian translation dataset consisting of 612317 examples. Each of the validation sets and the testing set accounts for 1999 and 1999 of the dataset. The train-validation-test split is fixed across all experiments for all machine translation datasets to ensure fair comparison.

Performance Metrics. We use the BLEU score (Papineni et al., 2002) to evaluate the machine translation quality, where the BLEU score is calculated based on the similarity between the actual target sequence and the predicted target sequence. The sampling speed is measured by wall-clock time (in second).

Baselines. The main baselines we are comparing with are RDM and RDMk from Zheng et al. (2023). Here, we use RDMk and RDM to denote the sampling method proposed in their paper with and without the usage of topk selection for the token generation technique (see Appendix E for more details), respectively. RDM and RDMk are applied to two previously proposed state-ofthe-art discrete diffusion models: Multinomial Diffusion (Hoogeboom et al., 2021b) and Absorbing Diffusion (Austin et al., 2021).

Results and Discussion. Tables 2 and 3 present the performance evaluations of our algorithms in machine translation tasks. Table 2 presents results for multinomial diffusion, while Table 3 displays results for absorbing diffusion. Our reported time and BLEU scores are averaged over 5 repeated experiments, except for the baseline RDM experiment 2 .

From Tables 2 and 3, we observe that methods based on DNDM significantly accelerate the sampling process compared to baseline diffusion models. This acceleration allows for greater flexibility in increasing the number of steps (up to infinity) without imposing a significant computational burden.

2 Due to computational intensity, we did not repeat the 1000-step sampling for the RDM baseline. However, reproducing it was deemed unnecessary as the sampling time is largely stable across repeated experiments, and the precise averaged timing is not critical for demonstrating the speed improvement of DNDM.

In particular, more sampling steps lead to better generation quality (BLEU) at the expense of longer sampling time, as indicated in each column of Tables 2 and 3. For RDM-based methods, generation time increases linearly with the number of sampling steps. On the contrary, for our DNDM-based method, generation time only increases marginally (See Figure 4 in Section G). As a result of the difference in the growing speed of sampling time with respect to sampling steps, the more sampling steps, the more speedup DNDM can obtain.

Continuous-time results, as the ultimate limit of increasing sampling steps, are presented in the last row of each dataset with the tag ∞ . Given that the results with 1000 steps consistently outperform those with 50 steps, we compare ∞ with 1000 steps in Table 2 and 3. For IWSLT14 and WMT16 , where the generation BLEU score is relatively high, we observe a consistent performance improvement of up to 0 . 3 in BLEU score when utilizing the DNDM-C algorithm, with the exception of a single case in the absorbing diffusion setting for WMT16 without the use of topk selection. The performance gain of the continuous-time method on WMT14 is less significant, with both drops and gains. However, WMT14 itself has not reached a high level of performance, with a BLEU score significantly lower than other datasets. In general, training WMT14 poses challenges across all diffusion models, including multinomial diffusion (Hoogeboom et al., 2021b), absorbing diffusion (Austin et al., 2021), and RDM diffusion (Zheng et al., 2023), etc. We defer a more detailed discussion on WMT14 to Appendix F.1. Finally, when compared with the results obtained with 50 steps, the performance of DNDM-C demonstrates improvement consistently. Furthermore, we note that regardless of the dataset or the method (i.e., RDM or DNDM) employed, topk token generation consistently outperforms vanilla methods. This approach enhances the BLEU score by approximately 1 -2 points without introducing significant increases in sampling time.

Table 2: BLEU score comparison of multinomial diffusion on machine translation benchmarks IWSLT14 DE -EN , WMT14 EN -DE , and WMT16 EN -RO . Below the dataset, we present the amount of data used to run the evaluation (sentences). The blue background highlights our algorithms, and the bold number indicates the best performance within each row and each setting (i.e., with or without top-k).

| Dataset         | Steps        | RDM-Multi          | RDM-Multi          | DNDM-Multi          | DNDM-Multi          | RDM- k -Multi       | RDM- k -Multi        | DNDM- k -Multi          | DNDM- k -Multi         |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Dataset         | Steps        | BLEU          | Time (s)          | BLEU          | Time (s)          | BLEU          | Time(s)          | BLEU          | Time (s)          |
| IWSLT14 (6.75k) | 25 50 1000 ∞ | 31.26 31.50 31.69 - | 166.9 328.6 6308.9 - | 30.95 31.45 31.82 31.89 | 52.9 83.9 191.3 225.2  | 32.82 32.82 32.64 - | 161.9 321.2 6321.3 - | 32.30 32.80 33.15 33.44 | 52.6 93.2 191.5 228.1  |
| WMT14 (3k)      | 25 50 1000 ∞ | 25.25 25.75 25.66 - | 237.3 466.1 8996.7 - | 25.01 25.33 25.71 24.79 | 90.7 138.4 265.4 307.5 | 26.03 26.14 25.82 - | 230.9 500.2 8991.7 - | 25.98 26.37 26.88 26.39 | 90.5 138.3 265.5 307.3 |
| WMT16 (2k)      | 25 50 1000 ∞ | 32.29 32.53 32.63 - | 145.2 286.1 5588.9 - | 31.97 32.50 32.86 32.91 | 36.4 63.2 171.4 196.4  | 33.12 33.41 33.67 - | 143.5 312.4 5601.0 - | 32.94 33.26 33.79 33.86 | 36.4 62.7 171.2 196.3  |

Scaling Law in Sampling Speed. For illustrative purposes, we use the example of IWSLT14 to visualize how the sample quality scales regarding sampling speed for different methods. In Figure 1, we observe the trend of the BLEU score in relation to computational time. Each line in the legend represents a different sampling algorithm, and a steeper slope indicates a larger marginal gain when sampling for longer periods. Figure 1 demonstrates that our algorithm displays nearly linear growth in BLEU score over the log of time, which is remarkable in contrast with the flat curve of the baseline. Particularly, for multinomial diffusion, the BLEU score increases by 1 in less than 60 seconds of additional sampling time. For absorbing diffusion, DNDM outperforms RDM before RDM samples 50 steps. In Tables 7 and 8 in Appendix D, we further use the average number of function evaluations (NFE) to measure the improved speed within the specified number of sampling steps. Additionally, in Figure 2, we visualize how the BLEU score and the generated text change throughout the sampling process.

Table 3: BLEU score comparison of absorbing diffusion on machine translation benchmarks IWSLT14 DE -EN , WMT14 EN -DE , and WMT16 EN -RO . Below the dataset, we present the amount of data used to run the evaluation (sentences). The blue background highlights our algorithms, and the bold number indicates the best performance within each row and each setting (i.e., with or without top-k).

| Dataset         | Steps        | RDM-Absorb          | RDM-Absorb          | DNDM-Absorb          | DNDM-Absorb          | RDM- k -Absorb      | RDM- k -Absorb       | DNDM- k -Absorb         | DNDM- k -Absorb        |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Dataset         | Steps        | BLEU          | Time (s)          | BLEU          | Time (s)          | BLEU          | Time(s)          | BLEU          | Time (s)          |
| IWSLT14 (6.75k) | 25 50 1000 ∞ | 31.58 31.80 31.91 - | 116.3 227.2 4197.4 - | 32.43 32.63 32.93 33.03 | 67.2 95.9 161.1 174.6  | 34.50 34.58 34.60 - | 108.9 213.9 4205.9 - | 34.14 34.34 34.56 34.65 | 67.3 96.2 162.3 180.7  |
| WMT14 (3k)      | 25 50 1000 ∞ | 24.97 24.95 25.22 - | 116.4 231.1 4169.4 - | 25.79 26.10 26.43 26.50 | 68.1 102.0 178.3 180.1 | 27.50 27.73 27.75 - | 107.5 255.2 4167.4 - | 27.18 27.66 27.82 27.50 | 68.0 102.5 179.1 181.2 |
| WMT16 (2k)      | 25 50 1000 ∞ | 32.86 32.93 33.25 - | 75.5 148.4 2951.7 -  | 33.20 33.30 33.60 33.42 | 41.2 62.5 121.3 121.8  | 33.92 34.10 34.44 - | 69.9 166.1 2718.7 -  | 33.96 34.20 34.38 34.41 | 41.4 62.7 122.7 121.9  |

Figure 1: Generation quality to generation time comparison on IWSLT14 . x -axis: computational time in seconds; y -axis: BLEU score.

<!-- image -->

## 4.2 Unconditional Text Generation

For unconditional text generation, we evaluate our approach on language modeling tasks, where the model learns to generate text that matches the statistical patterns of the training data. Unlike conditional generation, this task involves directly learning q ( x 0 | x t ) without conditioning on any input text. We conduct experiments on the text8 and enwik8 datasets using a decoder-only architecture similar to GPT models. Since unconditional generation does not require encoding input sequences, we employ a 12-layer Transformer decoder without an encoder component.

Datasets. The natural language generation task is evaluated on two language datasets following Hoogeboom et al. (2021b): text8 and enwik8 . Both datasets are from Wikipedia, but their contents are highly distinct. In text8 , the plain text consists of English words (all the letters are in lower case) and spaces, and it is tokenized into 26 characters and one blank space, resulting in 27 categories. In contrast to the cleanness of text8 , enwik8 preserves the original XML dump contents, and there exist various special symbols in its raw text, so its text is tokenized into 1 Byte, resulting in 256 categories. We utilize text8 dataset with sequence length 256 and enwik8 dataset with sequence length 320. The train/val/test splits are 9 e 7 / 5 e 6 / 5 e 5 for both text8 and enwik8 .

Performance Metrics. Our evaluation of text generation quality relies on the perplexity score. When generating text8 data, we calculate perplexity scores using the GPT2 model, while for enwik8 data generation, we employ the GPT2-large model. The sampling speed is measured in seconds.

<!-- image -->

- (a) The BLEU Score in the Generation Process
- t = 100 [noise] [noise] [noise] [noise] · · ·
- t = 75 [noise] · · · [noise] and we [noise] · · · [noise] govern [noise] [noise] year [noise]

t = 67 we [noise] [noise] fello [noise] [noise] [noise] and we let them [noise] [noise] city govern [noise] every year.

- t = 39 we choose some fellows every year and we let them work with city governance every year.
- t = 0 we choose some fellows every year and we let them work with city governance every year.

(b) Text in the Generation Process

Figure 2: We demonstrate the 100-step generation process of DNDMk -Multi as an example, where the left is the change of the BLEU score along the generation process, and the right is the text at different time steps. As the time goes from 100 to 0, noise is gradually removed until the corresponding English text emerges. Since the transition time follows a Beta distribution as described in Section 3.2, the majority of transitions occur near the starting time.

Baselines. We compare our proposed DNDM on unconditional text generation task with the vanilla Multinomial Diffusion (Hoogeboom et al., 2021b).

Results and Discussion. Table 4 displays the performance of our algorithms in text generation tasks. We run the multinomial diffusion model on the text8 dataset for 1000 diffusion steps and on the enwik8 dataset for 4000 diffusion steps. Our DNDM-based algorithms outperform the vanilla sampling algorithm used in Hoogeboom et al. (2021b) in terms of both sampling time and perplexity score. Specifically, for the text8 dataset, DNDMbased algorithms are 5 times faster than the vanilla algorithm. For the enwik8 dataset, DNDM-based algorithms are 14 times faster than the vanilla algorithm.

## 5 Conclusion and Future Work

This paper presents a novel discrete non-Markov diffusion model (DNDM) accompanied by an accelerated sampling algorithm designed to boost sampling speed in a discrete-state space. Our discrete diffusion model incorporates "transition time set" latent variables, establishing itself as an efficacious diffusion and data generation method. Thanks to our acceleration technique, we significantly decrease the number of neural network function evaluations without sacrificing sample quality. We also introduce an infinite-step sampling algorithm, DNDM-C, which provides new insights into bridging the gap between discrete and continuous-time processes for discrete diffusion models. While this study focuses on text generation using non-autoregressive models, a promising direction for future exploration is applying our method to other tasks, such as audio and image generation.

## Acknowledgement

We thank the anonymous reviewers and area chair for their helpful comments. ZC, HY, YL, YK, JZ, and QG are supported in part by the National Science Foundation CAREER Award 1906169, IIS-2008981, and the Sloan Research Fellowship. The views and conclusions contained in this paper are those of the authors and should not be interpreted as representing any funding agencies.

Table 4: Comparison of different sampling methods for unconditional text generation (multinomial diffusion) on text8 and enwik8 benchmarks. Sampling time is computed by generating a single text sample of length 256 for text8 and length 320 for enwik8 , averaged over 10 runs. The blue background represents our algorithms, and the bold number indicates the optimal value.

|        |          | Vanilla        | DNDM        |
|--------|----------|----------|----------|
| text8  | Perplexity Time (s) | 1,465.75 135.9 | 600.02 31.1 |
| enwik8 | Perplexity Time (s) | 801.78 602.8   | 556.78 47.4 |

## References

- ALAIN, G., BENGIO, Y., YAO, L., YOSINSKI, J., THIBODEAU-LAUFER, E., ZHANG, S. and VINCENT, P. (2016). Gsns: generative stochastic networks. Information and Inference: A Journal of the IMA 5 210-249.
- ALIAS PARTH GOYAL, A. G., KE, N. R., GANGULI, S. and BENGIO, Y. (2017). Variational walkback: Learning a transition operator as a stochastic recurrent net. Advances in Neural Information Processing Systems 30 .
- AUSTIN, J., JOHNSON, D. D., HO, J., TARLOW, D. and VAN DEN BERG, R. (2021). Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems 34 17981-17993.
- BAO, F., LI, C., ZHU, J. and ZHANG, B. (2022). Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. arXiv preprint arXiv:2201.06503 .
- BENGIO, Y., LAUFER, E., ALAIN, G. and YOSINSKI, J. (2014). Deep generative stochastic networks trainable by backprop. In International Conference on Machine Learning . PMLR.
- BOJAR, O., BUCK, C., FEDERMANN, C., HADDOW, B., KOEHN, P., LEVELING, J., MONZ, C., PECINA, P., POST, M., SAINT-AMAND, H., SORICUT, R., SPECIA, L. and TAMCHYNA, A. (2014). Findings of the 2014 workshop on statistical machine translation. In Proceedings of the Ninth Workshop on Statistical Machine Translation . Association for Computational Linguistics, Baltimore, Maryland, USA.
- BOJAR, O., CHATTERJEE, R., FEDERMANN, C., GRAHAM, Y., HADDOW, B., HUCK, M., JIMENO YEPES, A., KOEHN, P., LOGACHEVA, V., MONZ, C., NEGRI, M., NÉVÉOL, A., NEVES, M., POPEL, M., POST, M., RUBINO, R., SCARTON, C., SPECIA, L., TURCHI, M., VERSPOOR, K. and ZAMPIERI, M. (2016). Findings of the 2016 conference on machine translation. In Proceedings of the First Conference on Machine Translation: Volume 2, Shared Task Papers . Association for Computational Linguistics, Berlin, Germany.
- BORDES, F., HONARI, S. and VINCENT, P. (2017). Learning to generate samples from noise through infusion training. arXiv preprint arXiv:1703.06975 .
- CAMPBELL, A., BENTON, J., DE BORTOLI, V., RAINFORTH, T., DELIGIANNIDIS, G. and DOUCET, A. (2022). A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems 35 28266-28279.
- CERITLI, T., GHOSHEH, G. O., CHAUHAN, V. K., ZHU, T., CREAGH, A. P. and CLIFTON, D. A. (2023). Synthesizing mixed-type electronic health records using diffusion models. arXiv preprint arXiv:2302.14679 .
- CETTOLO, M., NIEHUES, J., STÜKER, S., BENTIVOGLI, L. and FEDERICO, M. (2014). Report on the 11th IWSLT evaluation campaign. In Proceedings of the 11th International Workshop on Spoken Language Translation: Evaluation Campaign . Lake Tahoe, California.
- CHANG, H., ZHANG, H., JIANG, L., LIU, C. and FREEMAN, W. T. (2022). Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition .
- CHEN, N., ZHANG, Y., ZEN, H., WEISS, R. J., NOROUZI, M. and CHAN, W. (2020). Wavegrad: Estimating gradients for waveform generation. arXiv preprint arXiv:2009.00713 .
- CHUNG, H., SIM, B. and YE, J. C. (2022). Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition .
- DOCKHORN, T., VAHDAT, A. and KREIS, K. (2021). Score-based generative modeling with critically-damped langevin diffusion. arXiv preprint arXiv:2112.07068 .
- DOCKHORN, T., VAHDAT, A. and KREIS, K. (2022). Genie: Higher-order denoising diffusion solvers. Advances in Neural Information Processing Systems 35 30150-30166.

- GHAZVININEJAD, M., LEVY, O., LIU, Y. and ZETTLEMOYER, L. (2019). Mask-predict: Parallel decoding of conditional masked language models. arXiv preprint arXiv:1904.09324 .
- GRUVER, N., STANTON, S., FREY, N., RUDNER, T. G., HOTZEL, I., LAFRANCE-VANASSE, J., RAJPAL, A., CHO, K. and WILSON, A. G. (2024). Protein design with guided discrete diffusion. Advances in Neural Information Processing Systems 36 .
- HE, Z., SUN, T., WANG, K., HUANG, X. and QIU, X. (2022). Diffusionbert: Improving generative masked language models with diffusion models. arXiv preprint arXiv:2211.15029 .
- HO, J., CHAN, W., SAHARIA, C., WHANG, J., GAO, R., GRITSENKO, A., KINGMA, D. P., POOLE, B., NOROUZI, M., FLEET, D. J. ET AL. (2022). Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 .
- HO, J., JAIN, A. and ABBEEL, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems 33 6840-6851.
- HOOGEBOOM, E., GRITSENKO, A. A., BASTINGS, J., POOLE, B., BERG, R. V. D. and SALIMANS, T. (2021a). Autoregressive diffusion models. arXiv preprint arXiv:2110.02037 .
- HOOGEBOOM, E., NIELSEN, D., JAINI, P., FORRÉ, P. and WELLING, M. (2021b). Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in Neural Information Processing Systems 34 12454-12465.
- JOLICOEUR-MARTINEAU, A., LI, K., PICHÉ-TAILLEFER, R., KACHMAN, T. and MITLIAGKAS, I. (2021). Gotta go fast when generating data with score-based models. arXiv preprint arXiv:2105.14080 .
- KARRAS, T., AITTALA, M., AILA, T. and LAINE, S. (2022). Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems 35 26565-26577.
- KONG, Z. and PING, W. (2021). On fast sampling of diffusion probabilistic models. arXiv preprint arXiv:2106.00132 .
- KONG, Z., PING, W., HUANG, J., ZHAO, K. and CATANZARO, B. (2020). Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761 .
- LIU, L., REN, Y., LIN, Z. and ZHAO, Z. (2022). Pseudo numerical methods for diffusion models on manifolds. arXiv preprint arXiv:2202.09778 .
- LU, C., ZHOU, Y., BAO, F., CHEN, J., LI, C. and ZHU, J. (2022). Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems 35 5775-5787.
- LYU, S. (2012). Interpretation and generalization of score matching. arXiv preprint arXiv:1205.2629 .
- MOVELLAN, J. R. (2008). Contrastive divergence in gaussian diffusions. Neural Computation 20 2238-2252.
- NACHMANI, E., ROMAN, R. S. and WOLF, L. (2021). Non gaussian denoising diffusion models. arXiv preprint arXiv:2106.07582 .
- NICHOL, A. Q. and DHARIWAL, P. (2021). Improved denoising diffusion probabilistic models. In International Conference on Machine Learning . PMLR.
- OTT, M., EDUNOV, S., BAEVSKI, A., FAN, A., GROSS, S., NG, N., GRANGIER, D. and AULI, M. (2019). fairseq: A fast, extensible toolkit for sequence modeling. arXiv preprint arXiv:1904.01038 .
- PAPINENI, K., ROUKOS, S., WARD, T. and ZHU, W.-J. (2002). Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics .

- REID, M., HELLENDOORN, V. J. and NEUBIG, G. (2022). Diffuser: Discrete diffusion via edit-based reconstruction. arXiv preprint arXiv:2210.16886 .
- SALIMANS, T. and HO, J. (2022). Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512 .
- SAN-ROMAN, R., NACHMANI, E. and WOLF, L. (2021). Noise estimation for generative diffusion models. arXiv preprint arXiv:2104.02600 .
- SAVINOV, N., CHUNG, J., BINKOWSKI, M., ELSEN, E. and OORD, A. V. D. (2021). Step-unrolled denoising autoencoders for text generation. arXiv preprint arXiv:2112.06749 .
- SENNRICH, R., HADDOW, B. and BIRCH, A. (2016). Neural machine translation of rare words with subword units.
- SOHL-DICKSTEIN, J., BATTAGLINO, P. and DEWEESE, M. R. (2009). Minimum probability flow learning. arXiv preprint arXiv:0906.4779 .
- SOHL-DICKSTEIN, J., WEISS, E., MAHESWARANATHAN, N. and GANGULI, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning . PMLR.
- SONG, J., MENG, C. and ERMON, S. (2020a). Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 .
- SONG, Y., DHARIWAL, P., CHEN, M. and SUTSKEVER, I. (2023). Consistency models. arXiv preprint arXiv:2303.01469 .
- SONG, Y. and ERMON, S. (2019). Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems 32 .
- SONG, Y. and ERMON, S. (2020). Improved techniques for training score-based generative models. Advances in neural information processing systems 33 12438-12448.
- SONG, Y., SOHL-DICKSTEIN, J., KINGMA, D. P., KUMAR, A., ERMON, S. and POOLE, B. (2020b). Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 .
- SUN, H., YU, L., DAI, B., SCHUURMANS, D. and DAI, H. (2022). Score-based continuous-time discrete diffusion models. arXiv preprint arXiv:2211.16750 .
- VAHDAT, A., KREIS, K. and KAUTZ, J. (2021). Score-based generative modeling in latent space. Advances in Neural Information Processing Systems 34 11287-11302.
- WATSON, D., HO, J., NOROUZI, M. and CHAN, W. (2021). Learning to efficiently sample from diffusion probabilistic models. arXiv preprint arXiv:2106.03802 .
- YE, J., ZHENG, Z., BAO, Y., QIAN, L. and GU, Q. (2023). Diffusion language models can perform many tasks with scaling and instruction-finetuning. arXiv preprint arXiv:2308.12219 .
- ZHANG, Q. and CHEN, Y. (2022). Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902 .
- ZHENG, L., YUAN, J., YU, L. and KONG, L. (2023). A reparameterized discrete diffusion model for text generation. arXiv preprint arXiv:2302.05737 .

## Broader Impact

This paper presents work that aims to advance the field of diffusion models. We believe this work may enable future applications of synthetic data generation, which may lead to positive impacts. Our experiments demonstrate that the proposed method achieves state-of-the-art performance in the acceleration of the generative model. However, proper controls may be needed whenever applying our method to tasks that involve sensitive data data. There may be other potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## Limitations

- The scope of the empirical claims is limited to the text domain with non-auto regressive setting. The applicability and performance of DNDM for other tasks like audio and image generation, as well as with other architectures like auto-regressive GPT models, are not explored and left as future work.
- While DNDM-C, the infinite-step sampling algorithm, offers new insights into bridging the gap between discrete and continuous-time processes for discrete diffusion models, the sample quality is not guaranteed to be superior to the accelerated algorithm with 1000 steps. Some intuitions here: the assumption that the neural network can be optimally trained is an ideal case and is often not realized in practice. There is an inherent estimation error associated with the training process. As the number of steps increases, these estimation errors can accumulate, potentially leading to a degradation in performance. This cumulative estimation error might explain why using an infinite number of steps does not necessarily yield better results than a finite number of steps like 1000 in the conditional generation experiments. How to further improve sample quality of infinite steps is interesting but beyond the scope of this paper.
- This paper focuses on the comparison with discrete Markov diffusion models since it aims to propose an accelerated algorithm for discrete diffusion with DNDM. Other text generation models, such as continuous diffusion models or auto-regressive models, are not considered in this paper.
- This paper focuses on acceleration while maintaining good sample quality. The hyper parameter regions with poor sample qualities are not explored in this paper.

By highlighting these limitations, this paper aims to clearly scope its contributions and spark future work on addressing these important challenges with discrete diffusion models for generative modeling.

## A Related Work

Continous Diffusion Models. Generative modeling via continuous-time stochastic process has been investigated thoroughly in a series of work (Movellan, 2008; Lyu, 2012; Sohl-Dickstein et al., 2009; Bengio et al., 2014; Alain et al., 2016; ALIAS PARTH GOYAL et al., 2017; Bordes et al., 2017). The two lines of probabilistic modeling, denoising diffusion probabilistic model (Sohl-Dickstein et al., 2015; Ho et al., 2020) and score matching with Langevin dynamics (Song and Ermon, 2019) are unified by Song et al. (2020b) through introducing the SDE framework for SGM. Based on it, subsequent works (Dockhorn et al., 2021; Nachmani et al., 2021; Vahdat et al., 2021) introduced a more complex diffusion process to improve the generation speed and quality. On the other hand, the score-based sampling process is time-consuming and has attracted much attention for improvements in speed (San-Roman et al., 2021; Watson et al., 2021; Kong and Ping, 2021; Karras et al., 2022; Song et al., 2023). 'Gotta go fast' (GGF), an SDE solver with adaptive step size tailored to SGM, is proposed in Jolicoeur-Martineau et al. (2021). Song et al. (2020a) introduced a non-Markov diffusion process that corresponds to a deterministic sampling process, enabling the generation of high-quality samples more rapidly. Dockhorn et al. (2022); Liu et al. (2022) proposed a high-order SDE/ODE solver to achieve lower discretization error. Lu et al. (2022); Zhang and Chen (2022) leveraged the semi-linear structure of reverse ODE to reduce the discretization error and achieve state-of-the-art sampling speed.

Discrete Diffusion Models. Research on discrete diffusion models was initiated by Sohl-Dickstein et al. (2015), who investigated diffusion processes over binary random variables. The methodology was expanded upon by Ho et al. (2020), integrating categorical random variables through transition matrices with uniform probabilities. Though Song et al. (2020a) suggested a similar extension in

their supplementary content, they abstained from experimenting with this model type. Later on, Austin et al. (2021) unveiled a more intricate framework for diffusion concerning categorical random variables, enhancing the discrete diffusion models by merging them with Masked language models (MLMs). Contemporary research has furthered this domain by introducing features like editingbased operations (Jolicoeur-Martineau et al., 2021; Reid et al., 2022), auto-regressive diffusion models (Hoogeboom et al., 2021a; Ye et al., 2023), the evolution of a continuous-time structure (Campbell et al., 2022), and the exploration of neural network analogs for learning (Sun et al., 2022). Additionally, Zheng et al. (2023) introduced a re-parameterized loss and an associated sampling technique, attaining commendable outcomes in fewer iterations. Our contributions run parallel to these aforementioned studies.

## B Additional details of Discrete Diffusion

In our paper, we treat all the x , q noise as a row vector and treat 1 as a column vector with all elements equal 1 .

## B.1 Comparison between D3PM and DNDM

In Section 3.1, we introduced two different diffusion processes, the Markov process in (1) and the non-Markov process in (6). In this section, we explain why they are different but result in the same joint distribution of ( x 0 , x t ) for every time step t . Since q ( x 0 ) keeps the same, we only need to prove that the conditional distribution q ( x t | x 0 ) is the same for the two processes.

Markov Process. 1 is a Markov process since w n is independent with x t -1 , . . . , x 0 , so x t is independent of all the past states given the present state. This can also be inferred from the following distribution, which does not depend on x 0 , . . . , x t -2 ,

<!-- formula-not-decoded -->

Denote Q t := β t I +(1 -β t ) 1 q noise , then we have that

<!-- formula-not-decoded -->

where the last equality holds due to the fact that x t -1 is a one hot vector and thus x t -1 1 = 1 . Therefore, we can rewrite (13) as q ( x t | x t -1 ) = Cat ( x t ; p = x t -1 Q t ) . Then, it is a Markov process with transition kernel Q t . So q ( x t | x 0 ) = Cat ( x t ; p = x 0 Q 0 . . . Q t ) (Austin et al., 2021). We can then have that

<!-- formula-not-decoded -->

where the last equality holds since identity matrix I multiplying any vector equals the vector itself and 1 q noise 1 q noise = 1 ( q noise 1 ) q noise = 1 q noise . Therefore, we have that

<!-- formula-not-decoded -->

where the last equality holds due to the definition α t = Π t s =0 β s . This gives rise to why the Markov process (1) results in conditional distribution q ( x t | x 0 ) = Cat ( x t ; p = α t x 0 +(1 -α t ) q noise ) .

Non-Markov Process. Recall that our DNDM is defined by

<!-- formula-not-decoded -->

where w is fixed for any time t . Therefore, w is no longer independent with x 0 , . . . , x t -1 . Therefore, we can't define the transition kernel and compute q ( x t | x 0 ) by using the property of Markov. Therefore, we need to advance the technique to calculate the conditional distribution.

Proof of Theorem 3.1. By (6), we can derive the following explicit expression for a recursive sequence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where second equality is by cancellation of terms, the last inequality holds by defining a t = b 1 . . . b t . Since a t either equals to 1 or 0 . Besides, a t equals 1 if and only if b 1 = b 2 = . . . = b t = 1 , so we have that a t follows Bernoulli distribution Bernoulli( β 1 . . . β t ) = Bernoulli( α t ) where α t = Π t i =1 β s . Therefore, we can conclude that q ( x t | x 0 ) = Cat ( x t ; p = α t x 0 +(1 -α t ) q noise ) , which completes the proof.

Comparison between D3PM-Absorb and DNDM. Recall the forward processes of D3PM and DNDMas follows:

$$D3PM : x t = b t x t - 1 +(1 - b t ) w t , ∀ t = 1 . . . T, DNDM : x t = b t x t - 1 +(1 - b t ) w , ∀ t = 1 . . . T.$$

̸

For absorbing diffusion where w = [Mask] , DNDM's forward process becomes equivalent to D3PM since w t = w = [Mask] in this special case. However, for multinomial diffusion or other diffusion processes where w t = w , these two processes exhibit different behaviors. In addition, even for absorbing diffusion, our proposed reverse sampling algorithm for DNDM is still different from that for D3PM.

To elucidate the key differences between the sampling algorithm in DNDM and that in D3PM for absorbing diffusion, let's directly compare the algorithms:

̸

- For the D3PM-Absorb algorithm: We begin with an all [Mask] sequence. At each time step t , we sample x 0 ∼ p θ ( x 0 | x t ) . If x t = [Mask] , x t -1 transitions to [Mask] with probability (1 -α t -1 ) / (1 -α t ) and to x 0 with probability ( α t -1 -α t ) / (1 -α t ) . If x t = [Mask] , it remains unchanged.
- For the DNDM-Absorb algorithm: We also start with an all [Mask] sequence, but crucially, we first determine the transition time set. During sampling, if x t = [Mask] , the transition probabilities for x t -1 are identical to D3PM. However, we only sample x 0 ∼ p θ ( x 0 | x t ) when at least one token needs to change, as determined by our pre-computed transition set. This selective sampling is the key to our algorithm's efficiency.

Therefore, you can see that DNDM will skip many steps during the sampling process to avoid function evaluation and save computational cost. Even though the forward process of DNDM is the same as that of D3PM for absorbing diffusion, our DNDM approach introduces an algorithm design in the sampling process by pre-computing the transition time set and selectively applying function evaluations. This distinguishes DNDM from D3PM algorithm, offering a more computationally efficient approach to inference in discrete diffusion.

Comparison between DDIM and DNDM for Multinomial Diffusion. While there are similarities between DNDM and DDIM (Appendix A), they are fundamentally different models, and DNDM is not a special case of DDIM. DNDM introduces a novel framework specifically designed for discrete spaces, while DDIM was originally developed for continuous diffusion models. The key differences for multinomial diffusion are as follows.

- DDIM: Following Song et al. (2020a) (eq. 19 in Appendix A), q ( x t -1 | x t , x 0 ) = Cat ( σ t x t + ( α t -1 -σ t α t ) x 0 +((1 -α t -1 ) -(1 -α t ) σ t ) 1 K ) . Even with σ t = 1 -α t -1 1 -α t , the process remains stochastic: q ( x t -1 | x t , x 0 ) = Cat ( σ t x t +(1 -σ t ) x 0 ) . This means at every step, there's a probability of choosing x 0 , regardless of whether it has transitioned to x 0 or not. Unlike Absorbing discrete diffusion, no [Mask] exists in multinomial diffusion. Therefore, DDIM cannot distinguish whether x t already equals x 0 or not. In particular, although the sampling process becomes less stochastic in the DDIM setting, it will still be predicted x 0 with high probability 1 -σ t = α t -1 -α t 1 -α t .
- DNDM: Achieves full de-randomization using transition time τ , where:

̸

<!-- formula-not-decoded -->

This crucial difference allows DNDM to achieve full de-randomization once τ is sampled, leading to a deterministic evolution that DDIM cannot achieve.

While DNDM and DDIM are both non-Markov models for multinomial diffusion, their fundamental approaches to and achievements in de-randomization differ significantly in discrete spaces.

## B.2 Training Objective

Hoogeboom et al. (2021b) utilized L t derived from the negative variational bound. In detail,

<!-- formula-not-decoded -->

where ̂ x 0 ∼ p θ ( ·| x t ) , θ post = ( β t x t + (1 -β t ) /K 1 ⊤ ) ⊙ ( α t -1 x 0 + (1 -α t -1 ) /K 1 ⊤ ) and θ post = ( β t x t +(1 -β t ) /K 1 ⊤ ) ⊙ ( α t -1 ̂ x 0 +(1 -α t -1 ) /K 1 ⊤ ) . This loss evolves KL divergence between two categorical distributions.

Building on this foundation, Austin et al. (2021) introduced an auxiliary denoising objective to strengthen the data predictions x 0 at each time step. In detail, the auxiliary objective is as follows,

<!-- formula-not-decoded -->

where the auxiliary loss term is minimized exactly when p θ ( ·| x t ) has all its mass on the data point x 0 .

Furthering the advancements, Zheng et al. (2023) put forth a reparametrized loss L t that incorporates a re-weighted parameter λ t . The detailed loss is

<!-- formula-not-decoded -->

This loss can be related to the standard multi-class cross-entropy loss function, which is also simple and powerful. That's why we consider Zheng et al. (2023) as the baseline model.

In Section 3.3, we consider the continuous-time forward and backward process. Based on that, we were motivated to analyze the infinite limit of the average loss lim t →∞ 1 T ∑ T t =1 L t . We find that the new loss can provide a better checkpoint than the loss averaged on the finite step on some tasks.

## B.3 Calculation of the Evidence Lower Bound

## B.3.1 Finite Time DNDM

In this section, we derive the evidence lower bound (ELBO) for our model. The derivatives are inspired by the reasoning in DDIM (Song et al., 2020a). Specifically, We denote the generative process as p θ ( x 0: T | τ ) = p ( T ) θ ( x T | τ ) ∏ T t =1 p ( t ) θ ( x t -1 | x t , τ ) . Here, p ( T ) θ is the pure noise and p ( t ) θ ( x t -1 | x t , τ ) = q ( x t -1 | x t , ̂ x 0 , τ ) , where ̂ x 0 is given by a neural network p θ , i.e., ̂ x 0 = p θ ( x t , t ) . Notice that by Jensen's inequality,

<!-- formula-not-decoded -->

The evidence lower bound inequality gives

<!-- formula-not-decoded -->

Plugging (17) into (16) gives the following ELBO,

<!-- formula-not-decoded -->

We factorize the p θ and q by

<!-- formula-not-decoded -->

Here q admits such a decomposition due to our definition of the diffusion process in (6), which introduce the following reverse process:

̸

<!-- formula-not-decoded -->

Therefore, x 1: T is Markovian when conditioned on x 0 and τ . Based on the factorization, we have

<!-- formula-not-decoded -->

By a slight abuse of notations we use q ( x t -1 | x t , x 0 ) , p ( t ) θ ( x 0 | x 1 ) to indicate the distribution of the diffusion process defined in Zheng et al. (2023), that is, the standard Markov discrete diffusion process. In particular, we have

̸

<!-- formula-not-decoded -->

Thus, we can obtain that

<!-- formula-not-decoded -->

Here L t matches the loss terms in Zheng et al. (2023). In the practical training process, Zheng et al. (2023) samples t from Unif { 1 , · · · , T } in each iteration and optimizes λ t · L t , where λ t 's are weights. Thus, when we sample τ and optimize L τ , our ELBO indeed leads to the same training objective as Zheng et al. (2023) up to reweighting. Since Zheng et al. (2023) is a parametrization of existing works (Austin et al., 2021; Hoogeboom et al., 2021b), our training objective indeed aligns with previous discrete diffusion models.

## B.3.2 Continous Time DNDM

In Section B.3, we derived an ELBO for DNDM and its accelerated algorithm defined in Section 3.1 and 3.2. While for finite sampling steps, we can decompose the diffusion process via the sampling steps 1 , . . . , T in (17), it becomes intractable for continuous Time DNDM (Infinite steps T →∞ ). Therefore, we can formulate the ELBO of continuous time DNDM by decomposing the transition times. The idea of decomposition of transition times follows Hoogeboom et al. (2021a), but their

̸

proof is only applicable to absorbing discrete diffusion, while ours can deal with discrete diffusion with various noise q noise including multinomial diffusion.

In Section B.3, we only consider the case of a single token x ∈ R K for simplicity as we decompose with the sampling steps T . In this section, we decompose over the transition time τ . Therefore, we need to consider a sentence with multiple tokens x t, 1: N = [ x t, 1 , . . . , x t,N ] where x t,n is the n -th token and N is the sequence length. Recall that we defined the transition time set T = { τ n } N n =1 in Section 3.2. We arrange τ n to obtain an ordered sequence τ n k , where 0 = τ n 0 &lt; τ n 1 &lt; τ n 2 &lt; . . . &lt; τ n N = T . Then conditioning on the transition time set T = { τ 1 , . . . , τ N } , we have that

<!-- formula-not-decoded -->

where we omit the time superscript of p for simplicity. Then, the evidence lower bound inequality gives

<!-- formula-not-decoded -->

By Jensen's inequality, we have

<!-- formula-not-decoded -->

Plugging (18) into (19) gives the following ELBO,

<!-- formula-not-decoded -->

We factorize the p θ and q by

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

Remark B.1 . (20) represents the ELBO utilized by the DNDM-C architecture. As our transition times τ n are independently and identically drawn from the distribution D τ , we are unable to further decompose (20) into a loss function related to the position information 1 : N , as was accomplished by Hoogeboom et al. (2021a).

## C Choice of the Transition Time

Transition time τ in Definition 3.2 plays an important role in DNDM. In this section, we provide a deeper discussion of the transition time. We first give a proof of the Theorem 3.6.

Proof of Theorem 3.6. By the definition of τ , we know that τ n = t is equivalent to b 0 ,n = 1 , . . . , b t -1 ,n = 1 and b t,n = 0 . Since { b t,n } T t =0 is independent for different n by definition, each τ n is also independent. Therefore, we drop the subscript n for simplicity. On the other hand if b 0 = 1 , . . . , b t -1 = 1 and b t = 0 we can also conclude that τ = t . Therefore, we have that

<!-- formula-not-decoded -->

where the second equality is due to b s , s = 1 , 2 , . . . , t are independent random variable following Bernoulli( β s ) distribution and the last equality is by the definition of α t = Π t s =1 β s .

Notice that α t is a decreasing sequence in the 0 to 1 range. Therefore, P ( τ = t ) ∈ [0 , 1] for any t ∈ { 1 , . . . , T } . Besides ∑ P ( τ = t ) = ∑ T t =1 ( α t -1 -α t ) = α 0 -α T = 1 . Therefore, the derived distribution is valid as long as the α t is decreasing from 1 to 0 .

Figure 3: Different distribution of transition time for T = 50 . a ) , b ) , c ) The transition time sampled 1K times under the different α t schedule. d) The approximated transition time for t = 1 , . . . , T using different hypter-parameters.

<!-- image -->

From Theorem 3.6, we discern that the nature of the diffusion model scheduler, α t , clarifies the distribution of τ .

Linear α schedule. This is a schedule studied in Austin et al. (2021), where α t = 1 -t/T . This will result in P ( τ n = t ) = 1 /T for every t in the range 1 to T . As a result, transition time distributes uniformly across each moment in the set { 1 , . . . , T } . This can be verified in a) of Figure 3.

Cosine α schedule. This is a schedule studied in Hoogeboom et al. (2021b), where α t = cos( π ∗ t/ 2 T ) . For numerical consideration of the noise, a small offset s is added, i.e., α t = f ( t ) /f (0)

where f ( t ) = cos(( s + t/T ) / (1 + s ) ∗ π/ 2) . As shown in b) of Figure 3, the transition time will concentrate more on the large T .

Cosine square α schedule. This is a schedule studied in Zheng et al. (2023), where α t = cos 2 ( π ∗ t/ 2 T ) , which motivated by Nichol and Dhariwal (2021). Again, for numerical consideration of the noise, a small offset s is added, i.e., α t = f ( t ) /f (0) where f ( t ) = cos ( ( s + t/T ) / (1 + s ) ∗ π/ 2) . As shown in c) of Figure 3, the transition time will concentrate more on the middle of the range.

Generally, if we express α t as g ( t/T ) , then we can simplify to P ( τ = t ) = g (( t -1) /T ) -g ( t/T ) , which further refines to (1 /T ) | g ′ ( t/T ) | + o (1 /T ) . This indicates that transitions are more likely where | g ′ | is large. Such a mathematical finding can match our observation in Figure 3.

In practice, we find that the shape of the transition time doesn't need to match the theoretical prediction schedule exactly. As we can see from d) in Figure 3. A reshaped Beta distribution can approximate all the transition time distributions in a fixed range. We first extract a time t ∈ [0 , 1] from a Beta distribution, then adjust these samples to fit by multiplying T and round them to acquire the integer. Our experiment finds that a properly chosen Beta distribution (tuned on the validation set) makes DNDM perform better on the translation tasks. Specifically, the chosen Beta distributions and the searching method are reported in Appendix F. The performance of the four transition time schedules mentioned above, including the reported Beta distributions for comparison, are listed in Table 5, where we find the other three schedules affect the performance, and most of their scores are lower than the scores of Beta distribution, but their scores are at least still close to the reported Beta distributions, especially for DNDM-k-absorb and DNDM-absorb. The efficiencies (measured by NFE) are also similar to one another.

Additionally, the ablation study on a reasonable range of different Beta distributions with 50 and 1000 sampling steps are shown in Tables 10 and 9, where the BLEU scores and NFE values on the test set of one of the three machine translation datasets, WMT16 , are shown for demonstration. The range of Beta distributions covers our chosen Beta schedules based on validation sets and a variety of basic Beta distribution shapes. These results show that the different Beta distributions influence the performance, but most of these choices of parameters still achieve results close to the optimal. Since the Beta distributions of the reported results in Tables 2 and 3 are selected using the validation set, they do not always have the highest scores on the test set, but their scores still at least belong to the top tiers according to these tables.

Another view of the transition time. In Algorithm 1, we only need to call the neural network when t ∈ T , which can significantly speed up the sampling since we reduce the function call. Notice that after we get the x 0 prediction, we only update the x t for those tokens at the transition time. However, (7) implies that x t = x 0 as long as τ &gt; t . Therefore, instead of only updating the x t for those tokens at the transition time, i.e., τ = t , we can also update those tokens with transition time τ &gt; = t . This motivates us to consider a variation presented as Algorithm 3, which keeps almost the same sampling time but will update the tokens several times rather than just once. Since the tokens now get the chance to be corrected over time. The new Algorithm 3 will be more robust than Algorithm 1.

Table 5: The BLEU scores and average number of function evaluations (NFE) values of different distributions of transition time for 1000 sampling steps with batch size 100. The parameters of the Beta distributions in this table are the same as in Tables 2 and 3 and are reported in Appendix F.

| Datasets   | Schedules          | DNDM-multi        | DNDM-multi        | DNDM-absorb         | DNDM-absorb       | DNDM-k-multi      | DNDM-k-multi      | DNDM-k-absorb     | DNDM-k-absorb     |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Datasets   | Schedules          | BLEU          | Avg NFE          | BLEU          | Avg NFE          | BLEU          | Avg NFE          | BLEU          | Avg NFE          |
| IWSLT14    | Cosine Cosine 2 Linear α | 31.72 31.78 31.77 | 31.71 31.74 31.82 | 32.71 32 . 93 32.65 | 31.21 31.21 31.33 | 32.91 32.78 32.83 | 31.71 31.74 31.82 | 34.50 34.53 34.53 | 31.21 31.21 31.33 |
|          | Beta (reported)          | 31 . 82          | 30 . 33          | 32 . 93          | 31 . 08          | 33 . 15          | 30 . 33          | 34 . 56          | 31 . 08          |
| WMT14      | Cosine          | 25 . 80          | 39.61          | 26 . 54          | 39.18          | 26.63          | 39.61          | 27.81          | 39.18          |
|          | Cosine 2          | 25.52          | 39.48          | 26.53          | 39.18          | 25.01          | 39.48          | 27 . 95          | 39.18          |
|          | Linear α          | 25.58          | 39.97          | 26.33          | 39.82          | 25.47          | 39.97          | 27.63          | 39.82          |
|          | Beta (reported)          | 25.71          | 38 . 94          | 26.43          | 38 . 76          | 26 . 88          | 38 . 94          | 27.82          | 38 . 76          |
| WMT16      | Cosine          | 32.71          | 40.50          | 33.56          | 40.45          | 33.46          | 40.50          | 34.37          | 40.45          |
| WMT16      | Cosine 2          | 32.73          | 40.50          | 33.51          | 40.45          | 33.44          | 40.50          | 34.24          | 40.45          |
| WMT16      | Linear α          | 32.85          | 40.36          | 33.46          | 40.36          | 33.47          | 40.36          | 33.88          | 40.36          |
| WMT16      | Beta (reported)          | 32 . 86          | 38 . 46          | 33 . 60          | 38 . 27          | 33 . 79          | 38 . 45          | 34 . 38          | 38 . 27          |

Table 6: Comparison of left-to-right and right-to-left transition approaches across different datasets and step counts.

|   Steps | Direction          | IWSLT14     | WMT14       | WMT16       |
|---------|----------|----------|----------|----------|
|      25 | Left-to-right Right-to-left | 31.08 30.54 | 24.41 23.33 | 31.67 31.33 |
|      50 | Left-to-right Right-to-left | 32.87 32.47 | 26.46 25.18 | 33.37 32.78 |
|    1000 | Left-to-right Right-to-left | 34.45 34.04 | 27.93 27.02 | 34.43 34.15 |

Impact of Transition Order. We further evaluate the impact of transition order. Building upon the results in Table 3, we investigate how the model performance will change if the transition time is influenced by the position of the tokens: from left to right and from right to left. In the left-to-right approach, tokens positioned on the left are transitioned to x 0 earlier, and vice versa for the right-toleft approach. Our experiments show that the left-to-right approach consistently outperforms the right-to-left approach across all datasets and step counts, as demonstrated in Table 6.

This result suggests that the order of token transitions significantly influences the model's performance, with earlier transitions of left-side tokens leading to better generation quality.

## D Discussion on the Number of Function Evaluations (NFE).

In this section, we discuss the number of function evaluations (NFE) in DNDM. According to (9), the update of a token x t -1 ,n occurs solely at its designated transition time. Meanwhile, if step t does not coincide with a transition time for any token, we maintain the sentence from the preceding step unchanged: x t, 1: N = x t -1 , 1: N . Therefore, our algorithm removes the need of function evaluation for steps outside the set of transition times. Given this structure, our analytical emphasis is on the transition set T since function evaluations are required only at times t that are members of T . Consequently, the NFE is precisely the cardinality of the transition set, denoted by |T | . In our main paper, we propose a naive upper bound for |T | as min { N,T } , which effectively demonstrates the speed of our method when T &gt; N . Next, we demonstrate that DNDM also reduces the NFE when T &lt; N , by providing a precise estimation of |T | .

Theorem D.1. Suppose transition time follows distribution D τ , and consider a sequence of length N . Then, the cardinality of the transition set T := { τ 1 , . . . , τ N } satisfies:

- 1 ≤ |T | ≤ min { N,T } ,
- E [ |T | ] = [1 -C T,N, D τ ] · T , where C T,N, D τ is a constant in the range (0 , 1) . Furthermore,

<!-- formula-not-decoded -->

where p i = P ( τ = i ) for τ ∼ D τ , and the equality holds if and only if D τ is a uniform distribution.

Proof. The first statement is straightforward. For completeness, the proof is provided. Since there are only N transition times (possibly repeated): τ 1 , . . . , τ N , the distinct transition times must satisfy |T | ≤ N . Additionally, since T ⊆ { 1 , . . . , T } , we also have |T | ≤ T .

To prove the second statement, we decompose T and use the property of expectation. Note that |T | = ∑ T i =1 1 { i ∈ T } . Thus,

<!-- formula-not-decoded -->

Assuming P D τ ( τ = i ) = p i , and that τ n are i.i.d. draws from D τ , we have

<!-- formula-not-decoded -->

Substituting (22) into (21) yields

<!-- formula-not-decoded -->

where C T,N, D τ = ( ∑ T i =1 (1 -p i ) N ) /T . An upper bound for C T,N, D τ is given as

<!-- formula-not-decoded -->

where the inequality holds if and only if p i = 1 /T for all i ∈ [ T ] , i.e., D τ is a uniform distribution.

Remark D.2 . Theorem D.1 suggests that even when T ≤ N , our method still provides a significant improvement. Specifically, for T = N ≥ 4 , we have C T,N, D τ = (1 -1 /N ) N ≥ 0 . 3 . This implies that our model requires at most 0 . 7 T even in the worst case. Moreover, if we consider a special scenario where the number of p i satisfying p i &lt; ϵ is more than M , then we have C T,N, D τ &gt; M (1 -ϵ ) N /T , indicating that with M sufficiently large and ϵ sufficiently small, C T,N, D τ can be pretty close to 1 .

Remark D.3 . In practical applications of our model, we employ a beta distribution for D τ , which typically exhibits a right-heavy tail. Therefore C T,N, D τ tends to be larger than that in the worst-case scenario. In Tables 7 and 8, we list the average NFE for each experiment we run in §4. These results demonstrate a significant reduction in NFE compared to the original counts: for T = 25 , the NFE is only about half of the original count; for T = 50 , it is approximately one-third; and for T = 1000 , it reduces to less than one-twentieth of the original count.

Remark D.4 . By Bernoulli's inequality, (1 -p ) N &gt; 1 -N · p for 1 &gt; p &gt; 0 . Therefore, C T,N, D τ &gt; 1 -N/T , implying that E [ |T | ] &lt; N . As T → ∞ , assuming the transition time does not concentrate at a single point, the probability that two transitions occur simultaneously is zero. Consequently, the generation process will sequentially go through each token. Thus, the expected number of function evaluations (NFE), E [ |T | ] , will be N . In contrast, when T is finite, there is a non-zero probability that multiple transitions happen at the same time. Hence, in this case, the NFE, |T | , is strictly less than N

Table 7: BLEU score and the average number of function evaluations (NFE) comparison of multinomial diffusion on machine translation benchmarks IWSLT14 DE -EN , WMT14 EN -DE , and WMT16 EN -RO . The blue background highlights our algorithms. The average NFE values are calculated by dividing the number of times calling the denoising function (neural network) during generation by the number of batches, where the batch sizes of all experiments are 100.

| Dataset   | Steps        | RDM-Multi          | RDM-Multi    | DNDM-Multi          | DNDM-Multi          | RDM- k -Multi       | RDM- k -Multi   | DNDM- k -Multi          | DNDM- k -Multi          |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Dataset   | Steps        | BLEU          | Avg NFE      | BLEU          | Avg NFE          | BLEU          | Avg NFE         | BLEU          | Avg NFE          |
| IWSLT14   | 25 50 1000 ∞ | 31.26 31.50 31.69 - | 25 50 1000 - | 30.95 31.45 31.82 31.89 | 9.03 14.07 30.33 32.73  | 32.82 32.82 32.64 - | 25 50 1000 -    | 32.30 32.80 33.15 33.44 | 9.03 14.07 30.33 32.73  |
| WMT14     | 25 50 1000 ∞ | 25.25 25.75 25.66 - | 25 50 1000 - | 25.01 25.33 25.71 24.79 | 13.52 20.58 38.94 40.67 | 26.03 26.14 25.82 - | 25 50 1000 -    | 25.98 26.37 26.88 26.39 | 13.52 20.58 38.94 40.67 |
| WMT16     | 25 50 1000 ∞ | 32.29 32.53 32.63 - | 25 50 1000 - | 31.97 32.50 32.86 32.91 | 8.5 14.73 38.45 41.64   | 33.12 33.41 33.67 - | 25 50 1000 -    | 32.94 33.26 33.79 33.86 | 8.5 14.73 38.45 41.64   |

## E Discrete Non-Markov Diffusion Model with Top-k Transition Time (DNDM-K).

Table 8: BLEU score and the average number of function evaluations (NFE) comparison of absorbing diffusion on machine translation benchmarks IWSLT14 DE -EN , WMT14 EN -DE , and WMT16 EN -RO . The blue background highlights our algorithms. The average NFE values are calculated by dividing the number of times calling the denoising function (neural network) during generation by the number of batches, where the batch sizes of all experiments are 100.

| Dataset   | Steps        | RDM-Absorb          | RDM-Absorb   | DNDM-Absorb          | DNDM-Absorb          | RDM- k -Absorb      | RDM- k -Absorb   | DNDM- k -Absorb         | DNDM- k -Absorb         |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Dataset   | Steps        | BLEU          | Avg NFE      | BLEU          | Avg NFE          | BLEU          | Avg NFE          | BLEU          | Avg NFE          |
| IWSLT14   | 25 50 1000 ∞ | 31.58 31.80 31.91 - | 25 50 1000 - | 32.43 32.63 32.93 33.03 | 13.81 19.24 31.08 32.07 | 34.50 34.58 34.60 - | 25 50 1000 -     | 34.14 34.34 34.56 34.65 | 13.81 19.24 31.08 32.07 |
| WMT14     | 25 50 1000 ∞ | 24.97 24.95 25.22 - | 25 50 1000 - | 25.79 26.10 26.43 26.50 | 15.09 22.45 38.76 40.39 | 27.50 27.73 27.75 - | 25 50 1000 -     | 27.18 27.66 27.82 27.50 | 15.09 22.45 38.76 40.39 |
| WMT16     | 25 50 1000 ∞ | 32.86 32.93 33.25 - | 25 50 1000 - | 33.20 33.30 33.60 33.42 | 13.91 20.95 38.27 41.59 | 33.92 34.10 34.44 - | 25 50 1000 -     | 33.96 34.20 34.38 34.41 | 13.91 20.95 38.27 41.59 |

## Algorithm 3 Sampling From DNDM (Version 2)

Require: Trained prediction function p θ , q noise ,

```
D τ 1: for n = 1 . . . N do 2: Initiate each token x T,n ∼ q noise 3: Initiate the transition time τ n ∼ D τ 4: end for 5: Collect transition time set T = { τ n } N n =1 6: for t = T . . . 1 do 7: if t ∈ T then 8: Generate ˜ x 0 , 1: N from p θ ( ·| x t, 1: N ) 9: for n = 1 . . . N do 10: Update x t -1 ,n if τ n ≥ t 11: end for 12: else 13: Update x t -1 , 1: N = x t, 1: N 14: end if 15: end for 16: Return x 0 , 1: N
```

## Algorithm 4 Sampling From DNDM-K

```
Input: Trained prediction function p θ , q noise and D τ for n = 1 . . . N do Initiate each token x T,n ∼ q noise Initiate the top K number { K t } Initiate an empty set U = {} , which includes the index of the tokens that have been updated. end for for t = T . . . 1 do if K t -1 > K t then Calculate the P = argtop K t { s t,n } N n =1 ; Generate ˜ x 0 , 1: N from p θ ( ·| x t, 1: N ) Update x t -1 ,n = ˜ x 0 ,n for all n in the set P but not in the set U (top score but not updated yet) Update the set U by appending the index of the updated tokens else Update x t -1 , 1: N = x t, 1: N ; end if end for Return x 0 , 1: N .
```

Recent works have demonstrated that the quality of samples can be enhanced by utilizing supplementary information derived from the neural network (Ghazvininejad et al., 2019; Savinov et al., 2021; Chang et al., 2022; He et al., 2022). Very recently, Zheng et al. (2023) applied this idea in their RDM framework and can achieve significant performance improvement. Specifically, after decoding ̂ x 0 , 1: N from transformer p θ ( ·| x t, 1: N ) , the score corresponding to this decoded token from the transformer's last layer, is also recorded and denote as s t,n . Tokens with high scores are more likely to be selected for updates.

Inspired by Zheng et al. (2023), we introduce the discrete non-Markov discrete diffusion Model with top-K transition time (DNDM-K). Instead of directly determining which token gets updated at step t by first drawing transition time τ ∼ D τ , we employ a two-step process.

1. We first compute K t = ∑ N n =1 1 ( τ n ≥ t ) . k t represents how many tokens should be decoded at the current step.
2. Compare K t -1 and K t , if K t -1 = K t . There is no transition time at time t , we just update x t -1 , 1: N = x t, 1: N . If K t -1 &gt; K t , Then there exist transition time at time t , we calculate and select the indexes with topK t -1 scores. Then we update those tokens if it hasn't been updated yet.

Subsequently, we will only update those tokens with the highest K t score that hasn't been changed yet. Since the function evaluation occurs only when K t changes, DNDM-K can give an accelerated sampling algorithm. The details are presented in Algorithm 4.

## F Experiment details

## F.1 Conditional Text Generation

Parameter choices. In all experiments, the batch size is chosen to be 100 . For RDM and RDMk , our hyperparameter settings follow the original paper (Zheng et al., 2023) except for the batch size. Before the sampling, we used the saved checkpoint of trained models provided by the authors for discrete sampling experiments, and we trained the corresponding models for continuous sampling experiments.

For finite-step DNDM, the transition times are determined by the schedule, and we approximate the schedule with a Beta distribution Beta ( α, β ) (please refer to Section 3.2 for detailed explanation). The α and β values are selected by applying grid search on the validation sets. Based on the BLEU scores on the validation sets, we have selected Beta (15 , 7) for Multinormial Diffusion on IWSLT14 , Beta (3 , 3) for Absorbing Diffusion on both IWSLT14 and WMT14 , Beta (5 , 3) for Multinormial Diffusion on WMT14 and Absorbing Diffusion on WMT16 , and Beta (20 , 7) for Multinormial Diffusion on WMT16 .

For infinite-steps (continuous-step) diffusion (DNDM-C), the transition timestamps are sampled from Beta ( α, β ) , where the choice of ( α, β ) are chosen from (100 . 0 , 4 . 0) or (17 . 0 , 4 . 0) , based on the performance comparison on the validation set. In the end we choose Beta (17 , 4) for IWSLT14 and Beta (100 , 4) for WMT14 and WMT16 .

We conduct a performance comparison based on varying configurations of the Beta and Alpha distributions. The results of these comparisons are presented in Tables 10 and 9. Furthermore, to evaluate the efficacy of discrete versus continuous step schemes, we also conduct an ablation study under the same set of parameters (100 , 4) in Table 11.

Table 9: BLEU scores on dataset WMT16 from the ablation study of other different Beta ( α, β ) distributions of the transition time with 1000 sampling steps.

| Model   | Alpha   |   Beta |   Beta |   Beta |   Beta | Beta    | Beta    |   Beta | Beta    | Beta    | Beta    |
|---------|---------|--------|--------|--------|--------|---------|---------|--------|---------|---------|---------|
| Model   | Alpha   |   3    |   5    |   7    |   9    | 11      | 13      |  15    | 17      | 19      | 21      |
|         | 3       |  33.47 |  33.67 |  33.62 |  33.77 | 33 . 87 | 33.64   |  33.73 | 33.60   | 33.68   | 33.56   |
|         | 5       |  33.18 |  33.47 |  33.68 |  33.53 | 33.71   | 33.69   |  33.73 | 33.72   | 33.74   | 33.82   |
|         | 7       |  32.99 |  33.2  |  33.49 |  33.56 | 33.58   | 33.61   |  33.67 | 33.72   | 33.78   | 33.83   |
|         | 3       |  32.73 |  32.66 |  32.74 |  32.82 | 32.77   | 32 . 92 |  32.8  | 32.81   | 32.76   | 32.86   |
|         | 5       |  32.32 |  32.62 |  32.7  |  32.8  | 32.83   | 32.83   |  32.9  | 32.95   | 32.91   | 32.87   |
|         | 7       |  32.35 |  32.35 |  32.53 |  32.67 | 32.75   | 32.78   |  32.86 | 32.80   | 32.86   | 32.88   |
|         | 3       |  34.19 |  34.38 |  34.34 |  34.22 | 34.21   | 34.24   |  34.07 | 34.31   | 34 . 42 | 34.36   |
|         | 5       |  32.15 |  33.99 |  34.29 |  34.3  | 34.29   | 34.40   |  34.4  | 34.24   | 34.30   | 34.22   |
|         | 7       |  27.67 |  32.87 |  33.94 |  34.28 | 34.27   | 34.38   |  34.31 | 34.29   | 34.38   | 34.40   |
|         | 3       |  33.53 |  33.6  |  33.67 |  33.71 | 33.71   | 33.70   |  33.58 | 33.63   | 33.53   | 33.54   |
|         | 5       |  32.7  |  33.33 |  33.52 |  33.6  | 33.66   | 33.73   |  33.7  | 33 . 74 | 33.72   | 33 . 74 |
|         | 7       |  30.56 |  32.65 |  33.28 |  33.37 | 33.51   | 33.52   |  33.61 | 33.67   | 33.63   | 33.67   |

Table 10: BLEU scores on dataset WMT16 from the ablation study of other different Beta ( α, β ) distributions of the transition time with 50 sampling steps.

| Model   | Alpha   |   Beta |   Beta |   Beta | Beta    | Beta    |   Beta |   Beta |   Beta |   Beta |   Beta |
|---------|---------|--------|--------|--------|---------|---------|--------|--------|--------|--------|--------|
| Model   | Alpha   |   3    |   5    |   7    | 9       | 11      |  13    |  15    |  17    |  19    |  21    |
|         | 3       |  33.31 |  33.47 |  33.39 | 33.48   | 33.29   |  33.23 |  33.25 |  33.27 |  33.11 |  33.17 |
|         | 5       |  32.93 |  33.28 |  33.29 | 33 . 58 | 33.45   |  33.21 |  33.4  |  33.49 |  33.16 |  33.19 |
|         | 7       |  32.61 |  32.98 |  33.31 | 33.20   | 33.27   |  33.41 |  33.39 |  33.53 |  33.35 |  33.08 |
|         | 3       |  32.63 |  32.46 |  32.44 | 32.56   | 32.59   |  32.55 |  32.37 |  32.33 |  32.22 |  32.23 |
|         | 5       |  32.31 |  32.43 |  32.66 | 32.64   | 32 . 68 |  32.55 |  32.55 |  32.44 |  32.35 |  32.3  |
|         | 7       |  31.95 |  32.11 |  32.22 | 32.26   | 32.54   |  32.52 |  32.5  |  32.58 |  32.48 |  32.41 |
|         | 3       |  34.05 |  34.2  |  34.31 | 34.37   | 34.15   |  34.05 |  34.06 |  33.77 |  33.81 |  33.84 |
|         | 5       |  32.3  |  34.08 |  34.3  | 34 . 38 | 34.26   |  34.23 |  34.09 |  34.06 |  34.02 |  34.13 |
|         | 7       |  27.39 |  32.64 |  33.71 | 34.18   | 34.02   |  34.33 |  34.31 |  34.17 |  34.12 |  34.19 |
|         | 3       |  33.26 |  33.3  |  33.29 | 33.24   | 33.23   |  32.97 |  33.06 |  32.85 |  32.89 |  32.63 |
|         | 5       |  32.47 |  33.08 |  33.31 | 33.22   | 33 . 41 |  33.25 |  33.15 |  33.27 |  33.04 |  32.98 |
|         | 7       |  30.34 |  32.27 |  33.27 | 33.03   | 33.16   |  33.14 |  33.27 |  33.11 |  33.11 |  33.07 |

Table 11: The BLEU scores on dataset WMT16 with Beta(100,4) as the transition time schedule for discrete sampling or the distribution to sample transition timestamps for continuous sampling.

| Steps   |   DNDM-k-multi |   DNDM-k-absorb |   DNDM-multi |   DNDM-absorb |
|---------|----------|----------|----------|----------|
| 50      |          31.6  |          31.74 |        30.39 |         29.69 |
| 1000    |          33.59 |          34.37 |        32.87 |         33.52 |
| ∞       |          33.86 |          34.41 |        32.91 |         33.42 |

Continuous time vs discrete time diffusions. To test our hypothesis that the continuous-time sampler will produce more accurate results in reverse sampling if our x 0 estimator consistently approximates the true x 0 over time, we conduct various sampling experiments using a shared pretrained neural network. For discrete-time sampling, we consider three cases: T = 25 , 50 , 1000 . In each case, we rescale the interval [0 , T ] to [0 , 50] and divide it into T fractions. In contrast, for continuous-time sampling, we directly sample from a continuous distribution over the interval [0 , 50] without any partitioning.

Training approach. In machine translation tasks, the neural network is designed to learn q ( x 0 | x t , z ) , where z represents the embedding of the source text obtained using transformer encoder layers. For a fair comparison, we employ the same neural network structure as our baseline, with detailed architecture specifications available in Section E.2 of Zheng et al. (2023). Furthermore, given that the primary focus of this paper is the speed and effectiveness of our sampling algorithm, we omit the training procedure and instead use a state-of-the-art diffusion-based pretrained checkpoint from Zheng et al. (2023). In the Appendix, we present additional results of continuous sampling based on a continuously trained checkpoint. In this setting, we rescale our network input to the interval [0 , 1] and uniformly sample from this interval. The rest of the architecture follows that of Zheng et al. (2023).

Performance on WMT14. Our work primarily focuses on the sampling process, and for the training, we utilized a pretrained checkpoint trained on 50 steps. In our sampling experiments we noticed that our method does not work ideally on WMT14 , this could be possibly attributed to the fact that the training performance on WMT14 was not ideal. Specifically, when we performed sampling using 1000 steps, the network was trained with exposure to only 50 time steps, specifically at intervals of 20 (0, 20, 40, ..., 980, 1000). As a result, when we apply our model to generation using 1000 steps, the checkpoint NN has only been explicitly trained on these intervals. While we generally assume that the network can still provide a good estimate for the untrained steps, this might not hold under some hard scenarios. Considering the longer training time and poorer performance of WMT14 , it is likely that the training performance is insufficient for us to rely on those unseen steps. In a word, the model's trained checkpoint may not be robust enough to effectively handle unseen steps, especially for timesteps 1000 or infinite timesteps.

## F.2 Unconditional Text Generation

Parameter choices. We recover the checkpoints of the multinomial diffusion model employing the provided code by Hoogeboom et al. (2021b). We train 12-layer Transformers for both text8 and enwik8 datasets for 500 epochs with the cosine schedule. For the text8 dataset, we utilize a training batch size of 256, while for the enwik8 dataset, we use a batch size of 128. During training, we employ a learning rate of 0.0001, a weight decay parameter of 0.99, and the Adam optimizer.

## G Additional Experiments

In this section, we present additional experimental results. We begin by plotting the relationship between computational time and the number of sampling steps, using the absorbing diffusion in IWSLT14 as an example. Figure 4 displays the growth of computational time for absorbing diffusion (yellow and orange lines), RDM-absorbing diffusion, and our model DNDM-Absorb and DNDM-TAbsorb (green and blue lines). We see from Figure 4 that previous algorithms, including absorbing

Figure 4: The growth of computational time with the increase of the sampling steps

<!-- image -->

diffusion and RDM-absorbing diffusion all suffer from linear growth of computational time.

## G.1 Continuous Training

In Section 4.1, we introduce the DNDM-C algorithm, designed for continuous-time, over discretetime algorithms. However, this algorithm assumes that we have learned a sufficiently accurate neural network at any timestamp t ∈ [0 , 1] . Using the checkpoint trained with 50 discrete time partitions might not suffice for the purpose of continuous sampling. In this section, we investigate the performance of continuous sampling when training is also done continuously.

Table 12: Continuous Training + Continuous Sampling

| Dataset   | Step scheme   | C-DNDM-Multi   | C-DNDM-Multi   | C-DNDM-Absorb   | C-DNDM-Absorb   |
|----------|----------|----------|----------|----------|----------|
|          |          | Default        | Top-k          | Default         | Top-k          |
| IWSLT14   | Continuous    | 32.07          | 33.57          | 32.80          | 34.52          |
| WMT16     | Continuous    | 33.48          | 33.71          | 33.50          | 34.36          |

In Table 12, we summarize the performance of DNDM-C based on a neural network estimated continuously during training time. This involves sampling time uniformly from [0 , 1] during training, and the forward process follows (11) in Section 3.3. The training objective remains the same as in discrete-time training. In Table 12 we list the result of IWSLT14 and WMT16 with continuous training followed by continuous sampling. In addition, we compare the value with the corresponding value during discrete training and continuous sampling in Section 4.1 and mark every item that improves in bold. As demonstrated in Table 12, there is room for enhancement in the overall sampling scores by training the neural network in a complete space of timestamps.

## G.2 Comparison with more generative models

In our study, a key aspect of evaluating our fast discrete generative model involves comparisons with prior work known for speed in sampling with minimal steps. Specifically, we draw a direct comparison with the Mask-Predict (Ghazvininejad et al., 2019), which is notable for its ability to generate high-quality results within just 10 iterations. The results are shown in Table 13. All experiments were conducted on the same GPU and within the same machine setup.

Table 13: The performance comparison on WMT16 of DNDM with Mask-Predict (Ghazvininejad et al., 2019). We align the number of sampling steps used in Mask-Predict with a similar number of function evaluations (NFE) in our DNDM algorithm. We see that our Algorithm runs faster, with better BLEU score.

| Mask-Predict   | Mask-Predict   | Mask-Predict   | DNDM-Absorb   | DNDM-Absorb   | DNDM-Absorb   | DNDM-Absorb   | DNDM-k-Absorb   | DNDM-k-Absorb   | DNDM-k-Absorb   | DNDM-k-Absorb   |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Steps          | BLEU          | Time          | Steps         | BLEU          | Time          | NFE          | Steps          | BLEU          | Time          | NFE          |
| 10          | 33.08          | 49.25          | 25          | 33.20         | 41.2          | 13.91         | 25          | 33.96          | 41.4          | 13.91          |
| 15          | 33.06          | 67.94          | 50          | 33.30         | 62.5          | 20.95         | 50          | 34.20          | 62.7          | 20.95          |
| 25          | 33.16          | 111.89         | 1000          | 33.60         | 121.3         | 38.27         | 1000          | 34.38          | 122.7          | 38.27          |
| 40          | 33.10          | 169.95         | ∞          | 33.42         | 121.8         | 41.59         | ∞          | 34.41          | 121.9          | 41.59          |

## G.3 Samples from the multinomial text models

Conditional Generation. For DNDM-Multi trained on IWSLT14 , we provide a full generation process with 100 steps in Figure 5. A token ending with @@ indicates it is an incomplete word; it will be concatenated with the following token to form a complete word. For example, ' fel@@ lo@@ ws ′′ means ' fellows ′′ . We can see that after t = 39 , the generate sentence converges.

##