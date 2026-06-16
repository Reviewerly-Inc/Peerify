## Graph Convolutions Enrich the Self-Attention in Transformers!

## Jeongwhan Choi ∗

Yonsei University

jeongwhan.choi@yonsei.ac.kr

## Jayoung Kim KAIST

jayoung.kim@kaist.ac.kr

## Hyowon Wi ∗ KAIST

hyowon.wi@kaist.ac.kr

## Yehjin Shin

KAIST

yehjin.shin@kaist.ac.kr

## Kookjin Lee

Arizona State University kookjin.lee@asu.edu

## Nathaniel Trask

## Noseong Park †

University of Pennsylvania ntrask@seas.upenn.edu

## Abstract

Transformers, renowned for their self-attention mechanism, have achieved state-ofthe-art performance across various tasks in natural language processing, computer vision, time-series modeling, etc. However, one of the challenges with deep Transformer models is the oversmoothing problem, where representations across layers converge to indistinguishable values, leading to significant performance degradation. We interpret the original self-attention as a simple graph filter and redesign it from a graph signal processing (GSP) perspective. We propose a graph-filter-based self-attention (GFSA) 1 to learn a general yet effective one, whose complexity, however, is slightly larger than that of the original self-attention mechanism. We demonstrate that GFSA improves the performance of Transformers in various fields, including computer vision, natural language processing, graph-level tasks, speech recognition, and code classification.

## 1 Introduction

Transformers are arguably one of the best feats in the field of deep learning. They are now showing state-of-the-art performance in various fields, ranging from computer vision to natural language processing, prediction tasks on graphs, speech recognition, and so forth [77, 16, 60, 61, 19, 74, 101, 45, 27, 41, 63, 51, 59, 38, 95, 72]. Recently, there have been several studies conducted on better understanding them [25, 3, 80]; there exists a common agreement among researchers that the self-attention is one of the keys leading to the success.

∗ Equal contribution.

† Corresponding author.

Figure 1: Performance improvements (%) of our GFSA when integrated with different Transformer backbones in various domains. We achieve these results with only tens to hundreds of additional parameters to Transformers.

<!-- image -->

1 The source code of GFSA is available at: https://github.com/jeongwhanchoi/GFSA .

KAIST

noseong@kaist.ac.kr

Figure 2: Filter frequency response, cosine similarity, and singular values on ImageNet-1k for DeiT-S and DeiT-S + GFSA. Details and more visualizations are in Appendices C and D.

<!-- image -->

However, there also exist several studies pointing out potential limitations of the self-attention [101, 18, 25, 28]. For instance, Shi et al. [71] revealed an analogy between the self-attention and the residual graph convolutional network (GCN), showing that BERT also suffers from a notorious problem of GCNs, called oversmoothing , i.e., tokens' latent representations become similar to each other at the deeper layers. In every self-attention layer, value vectors are aggregated in a weighted average manner since each row-wise sum of the attention matrix is always 1. Although each self-attention layer has its own attention matrix, this aggregation method causes the oversmoothing problem, not only in Transformers but also in graph neural networks [54, 7, 79, 66, 37, 101, 25, 92, 53, 71, 80, 3, 89, 90]. However, we confine our discussion to the oversmoothing of Transformers (see Section 2).

Being inspired by them, we redesign the self-attention from the perspective of graph signal processing (GSP) - in particular, we resort to GSP on directed graphs since the attention matrix is asymmetric. However, performing graph convolutions in the self-attention layer may incur non-trivial computational overheads. Therefore, our key design point is to learn a general but effective graph filter with minimal overhead. In general, a graph filter on a graph G is represented by a polynomial expression based on its adjacency or Laplacian matrix - in this regard, the existing self-attention mechanism can be understood as the simplest graph filter with ¯ A only, where ¯ A ∈ [0 , 1] n × n means a learned attention matrix and n is the number of input tokens.

Our proposed graph filter consists of an identity term and two matrix polynomial terms, ¯ A and ¯ A K . One can design better filters with more polynomial terms, but we avoid it since Transformers already require very large computation. The K -th power, ¯ A K , may also require a high computation when the number of tokens is large. To avoid this, we further approximate ¯ A K using the element-wise first-order Taylor approximation. Therefore, one can consider that our proposed graph filter is the very next complicated filter after the one used by the original self-attention mechanism. However, its efficacy is tremendous in various fields (cf. Fig. 1).

Our proposed filter enriches the self-attention with more diverse frequency information (see Fig. 2(a)) -low (resp. high) frequency signals on G mean neighboring nodes have similar (resp. different) values. Therefore, our method is able to not only effectively address the oversmoothing problem but also learn better latent representations for downstream tasks.

There exist a couple of prior works to enrich the self-attention mechanism with high frequency information [80, 4]. In comparison with them, our proposed graph filter is distinctive in the following aspects: i) our proposed filter is more effective and shows better performance with comparable computational overheads, ii) our proposed filter is well-aligned with recent advancements in the GCN community - in other words, some graph filters used by recent advanced GCN methods are special cases of our proposed graph filter, which is not the case for prior works, and iii) other methods were typically studied for certain domains only whereas we test our method in 6 domains - for instance, DiversePatch [25] works only for Vision Transformers (ViTs).

We replace the self-attention layer of selected Transformers in various fields with our proposed graph filter-based layer without changing other parts. Therefore, the accuracy increases in them are solely by our proposed graph filter-based self-attention. These enriched Transformers increase the model performance by 1.63% for image classification, 6.25% for natural language understanding, 0.31% for

causal language modeling, 4.03% for graph regression, 4.76% for speech recognition, and 2.40% for code classification (see Fig. 1). Our core contributions are as follows:

- We provide a novel perspective on self-attention as a graph filter. This perspective allows us to design more effective self-attention that can address the oversmoothing problem.
- We propose a graph filter-based self-attention (GFSA) mechanism, integrating an identity term and two polynomial terms for general yet effective than the simple self-attention mechanism (Section 3).
- We demonstrate that GFSA improves the performance of Transformers on a variety of tasks. GFSA achieves improved results on natural language processing, computer vision, speech recognition, graph-level tasks, and code classification (Sections 5.1 to 5.6).
- We devise a strategy to selectively apply GFSA to even-numbered layers, effectively mitigating the computational overhead while preserving GFSA's performance (Section 6).

## 2 Background &amp; Related Work

## 2.1 Self-Attention in Transformers

The core building block of the Transformer architecture is the self-attention mechanism, which enables the model to learn attention patterns over its input tokens [77]. The self-attention mechanism, denoted as SA : R n × d → R n × d , can be expressed as follows:

<!-- formula-not-decoded -->

where X ∈ R n × d is the input feature matrix, W key ∈ R d × d , W qry ∈ R d × d , and W val ∈ R d × d are the key, query, and value trainable parameters, respectively, and d is the dimension of each token. The self-attention mechanism allows the model to weigh the importance of each token in the input sequence relative to the others, enabling the model to capture long-range contextual information better. The Transformer architecture includes multiple layers, each with a multi-head self-attention layer followed by a position-wise feed-forward layer.

## 2.2 Self-Attention and Graph Convolutional Filter

The self-attention matrix used in Transformers has the form of symmetrically normalized adjacency matrix where each token become a node [71, 28] - the symmetrically normalized adjacency matrix is a special case of asymmetric (or directed) adjacency matrix where each row is normalized and is frequently used for the graph signal processing (GSP) on directed graphs [49]. A weighted graph G with adjacency matrix A can be constructed by using the input tokens as n nodes and the edge weights between node i and node j as exp(( XW qry ) ⊺ i ( XW key ) j ) . We can rewrite the self-attention matrix ¯ A ij as exp (( XW qry ) ⊺ i ( XW key ) j ) ∑ d k =1 exp ( XW qry ) ⊺ i ( XW key ) k . This allows ¯ A to be interpreted as the symmetrically normalized adjacency matrix. In other words, ¯ A = D -1 A , where D = diag ( d 1 , d 2 , . . . , d n ) and d i = ∑ j A i,j .

Our new attention method is designed on top of GSP which has a close connection to discrete signal processing (DSP) [67, 68]. In DSP, a discrete signal with a length of n can be represented by a vector x ∈ R n . Let g ∈ R n be a filter that we want to apply to x . The convolution x ∗ g can be written as follows:

<!-- formula-not-decoded -->

where the index, denoted as i , refers to the i -th element in each vector.

GSP can be understood as a generalized concept of DSP. Signals are defined on the nodes of a graph, and the graph's structure influences signal processing operations. In addition, the linear and shift-invariant graph convolution filter H with n nodes can be written with a shift operator S as

follows S can be from a directed graph [48]:

<!-- formula-not-decoded -->

where x ∈ R n is a 1-dimensional graph signal, K is the maximum order of polynomial, and w k ∈ [ -∞ , ∞ ] is a coefficient. S is an n × n matrix where ( i, j ) -th element is non-zero if and only if there is an edge from node i to j . Two representative samples of S are adjacency and Laplacian matrices. The graph filter H is the same as ∑ K k =0 w k S k with a large enough value of K , which is called matrix polynomial [48]. We note that this graph filtering operation can be extended to d -dimensional cases as in Eq. (1). Being inspired by Zou et al. [106] and Maskey et al. [49], we rely on the singular value domain analysis to understand the low/high-pass characteristics of filters on directed graphs (cf. Fig. 2). See more discussion in Appendices E and F.

In the context of the self-attention within Transformers, the core part of the self-attention in Eq. (1), i.e., ¯ AX , can be considered as a d -dimensional graph filter with ¯ A only, where H = ¯ A . Our goal in this paper is to design a simple (for computational purposes) yet effective form of H .

## 2.3 Oversmoothing in GCNs and Transformers

Oversmoothing is a phenomenon observed in deep learning models, especially in GCNs [39, 78]. As information is aggregated over multiple layers for multiple nodes (tokens), latent representations tend to become similar to each other, leading to a loss of distinctiveness in the representations [54, 104, 66].

Surprisingly, an oversmoothing-like phenomenon is also observed in Transformers [80, 71]. Unlike CNNs, Transformers can not benefit from simply deepening layers after a certain depth. Earlier studies hypothesize that this may be due to issues such as the attention or feature collapse or due to uniformity among patches or tokens [101, 25, 92]. Dong et al. [18] also point out that the output of a pure Transformer, i.e., an attention mechanism without skip connections or MLPs, tends to converge to a rank-1 matrix [18]. This analysis is followed by [53], which suggests that rank collapses incur vanishing gradients of attention queries and keys.

In this context, the self-attention acts as a low-pass filter, since the self-attention calculates the weighted average of the value vectors of tokens. Wang et al. [80, Theorem 1] also reveal that the self-attention is a low-pass filter, continuously reducing high-frequency information. This nature contributes to the oversmoothing phenomenon as unique high-frequency features are lost in deeper layers of the network, further worsening the uniformity of token representations. Therefore, we extend the term 'oversmoothing' to describe the degeneration challenge observed in Transformers.

There have been proposed many empirical countermeasures for ViT, such as patch diversification [102, 25], rank collapse alleviation [101, 99], and training stabilization [74, 98]. Similar alleviating methods have been also proposed in the field of NLP, such as unsupervised learning [9], and resolve the oversmoothing and the token uniformity (or information diffusion) problems [18, 92]. There are studies on utilizing high frequency information via frequency domain analyses [80, 4], but they are not designed on top of graph filtering perspectives. Dovonon et al. [20] find that Transformers are not inherently low-pass filters, but oversmoothing depends on the eigenspectrum of the self-attention layers. They propose a reparametrization of the Transformer weights, ensuring that oversmoothinng does not occur.

Our paper addresses the oversmoothing problem with graph filters since the self-attention mechanism is a basic graph filtering operation as seen in the previous subsection.

## 3 Graph Filter-based Self-Attention Layers

Let ¯ A ∈ [0 , 1] n × n , where n is the number of tokens in the input to the self-attention layer, be a self-attention matrix. Since Transformers use multi-head self-attentions, there are multiple such matrices. For simplicity, but without loss of generality, we discuss only one head in one layer.

From the GSP perspective, using ¯ A as the shift operator, a graph filter can be represented as a matrix polynomial filter, as mentioned in Section 2.2. We aim to design this matrix polynomial filter using the two lowest-order terms and one high-order term in Eq. (3). The following theorem shows that,

despite using the three terms, the filter can be either a low-pass filter or a high-pass filter, depending on the coefficient values.

Theorem 3.1 (Filter characteristics based on coefficient values) . Let ¯ A be a self-attention matrix interpreted as a graph with connected components. Consider the polynomial graph filter defined by ∑ K k =0 w k ¯ A k , where w 2 , w 3 , . . . , w K -1 = 0 and only w 0 , w 1 , and w K are non-zero. If the coefficients w k for k = 0 , 1 , K are positive and their sum is 1, then the polynomial filter acts as a low-pass filter, attenuating high-frequency components and promoting smoothness across the graph. Conversely, if w k = ( -α ) k for k = 0 , 1 , K and α ∈ (0 , 1) with sufficient large K , the polynomial filter exhibits high-pass filter behavior.

The proof of Theorem 3.1 is in Appendix G. Based on Theorem 3.1, we propose to use the following graph filter, H GFSA, where the two lowest-order terms and one high-order term of the matrix polynomial are used:

<!-- formula-not-decoded -->

where w 0 , w 1 , w K are coefficients and K is a hyper-parameter where K ≥ 2 . The coefficients can be learnable weights and we learn them with gradient descent algorithms.

Approximation of the high-order term. In Eq. (4), it is costly to calculate ¯ A K when K is large, so we need a way to approximate the high-order term ¯ A K in GFSA. We use the first-order Taylor approximation at point a = 1 for this purpose:

<!-- formula-not-decoded -->

thus, we approximate f ( K ) = ¯ A K as follows:

<!-- formula-not-decoded -->

Computing the derivative of ¯ A K directly at the evaluation point requires high computational costs. To overcome this problem, we adopt the forward finite difference method, which approximates derivatives with the difference term:

<!-- formula-not-decoded -->

where the approximation error is O ( h 2 ) . To balance the trade-off between computational efficiency 2 and accuracy, we set h = 1 . This method is inspired by the approach in Brouwer et al. [6], which uses the difference term between two consecutive hidden states in discrete time to approximate the derivatives. Therefore, we approximate ¯ A K as:

<!-- formula-not-decoded -->

The approximation for ¯ A K with ¯ A and ¯ A 2 provides a simpler computation that can significantly reduce the required computational resources and time.

GFSA: our graph filter-based self-attention. Our proposed graph filter-based self-attention (GFSA) is defined with the graph filter ˜ H GFSA as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last term is the approximated ¯ A K from Eq. (8). We replace the original self-attention layer in various Transformers with the proposed graph filter-based layer without changing other parts. Therefore, GFSA can be plugged into any Transformers that rely on the self-attention. For pseudocode, see Appendix I.

2 Calculating the power of a matrix for small h requires a high computational cost since it is calculated by diagonalizing the matrix or using Schur normal form [31].

## 4 Properties of GFSA

This section analyzes the theoretical error of the ¯ A K approximation used by GFSA and how GFSA can mitigate oversmoothing. We also explain the meaning of GFSA's high-order term in the context of Transformers and provide comparisons of GFSA in other models.

Theoretical characteristics of approximation error in GFSA. Weprovide a theorem that provides an upper bound on the error introduced by approximating the power of a matrix, specifically using the first-order Taylor expansion. The following theorem specifically analyzes the error of matrix ¯ A K when approximated using a first-order Taylor expansion.

Theorem 4.1 (Error bound for approximated high-order term in GFSA) . Define the error term, E K , as the difference between the exact value and approximated value of ¯ A K , which is given by E K = || ¯ A K -( ¯ A +( K -1)( ¯ A 2 -¯ A )) || F , where || · || F denotes the Frobenius norm. Then, the error bound can be shown that E K ≤ 2 √ nK.

The error bound provides an upper limit for the difference between the actual value of ¯ A K and its approximation. The proof of Theorem 4.1 is in Appendix H. It shows theoretical validity for using approximations to the high-order term in the filters of our GFSA. In terms of performance, we report a difference of approximately ¯ A K between the actual calculated values in Appendix J.

How to alleviate the oversmoothing problem? The key leading to the low/high pass filtering behavior of our proposed filter is the coefficients { w 0 , w 1 , w K } -note that in the self-attention of Transformers, w 0 = w K = 0 and w 1 = 1 . Since our method can learn any appropriate values for them for a downstream task, it can be reduced to low-pass-only, high-pass-only, or combined filters. According to Theorem 3.1, our graph polynomial filter can be said to be a low-pass filter when w 1 , w K are positive and a high-pass filter when they are negative. Therefore, our method can learn the appropriate coefficients { w 0 , w 1 , w K } for downstream tasks, so it can be reduced to a low-pass-only, high-pass-only, or combined filter, alleviating the oversmoothing problem of self-attention.

The meaning of the high-order term in GFSA in the context of Transformers. Existing selfattention only captures simple pairwise similarities between tokens and is limited in capturing high-order dependencies. For example, given the two sentences, 'Books are more expensive than pencils' and 'Books are cheaper than computers', to understand the relationship between 'computers' and 'pencils', we need to capture the high-order dependencies connected through the 'Book' token. However, it is difficult to capture these high-order dependencies with traditional self-attention [103]. Therefore, from a Transformer perspective, the approximated ¯ A K in GFSA can be interpreted as being able to capture these high-order dependencies.

Comparison to Transformers. In the field of computer vision, there has been recent research on adjusting the frequency response of ViT. HAT [4] creates adversarial examples by altering clean images with high-frequency perturbations and jointly trains the ViT on clean images and adversarial examples. Through this, they aim to solve the problem of the ViT being unable to capture highfrequency by allowing us to capture the high-frequency components of the images. However, HAT has the disadvantage of requiring more epochs than the existing ViT, as it must perform adversarial training in some initial epochs and train normally in the remaining epochs. Wang et al. [80] use the concept of DSP, which is a special case of GSP, to isolate the lowest frequency component in the Fourier domain and use a filter learned by rescaling the low and high-frequency components. On the other hand, our GFSA extends the concept to graph signal processing and redesigns self-attention as a graph filter. While GFSA seeks to design a better graph filter by interpreting self-attention as a graph filter, Shi et al. [71] are inspired by JKNet [91], and they solve the oversmoothing problem by fusing the hidden vectors of each layer. However, their method has a limitation with memory increasing, and they only applied it to BERT.

Comparison to GCNs. Comparisons to GCNs that can be interpreted as graph filters [39, 15, 24] are inevitable. GFSA without a high-order term is analogous to ChebNet [15] with K = 1 . In addition, GFSA reduces to the vanilla GCN [39] when K = 1 , w 0 = 0 , w 1 = 1 . GPR-GNN [12], which approximates graph convolutions using the monomial basis, is identical to GFSA if it only considers up to first order and additionally uses a K -order term and learns the coefficients. When we

use only a high-order term and w K is learned to a negative value, GFSA can become similar to the reaction-diffusion layer of GREAD [13], ¯ AX + β ( ¯ A -¯ A 2 ) , depending on the higher order terms.

## 5 Experiments

In this section, we demonstrate the effectiveness of GFSA through a series of experiments. These experiments encompass various tasks: i) language understanding and causal language modeling, ii) image classification, iii) graph-level task, and iv) code classification. We replace the self-attention of base Transformers in those fields with our GFSA. Our modification adds only tens to hundreds of parameters, which are negligible in comparison with the original size of base models.

## 5.1 Experiments on Natural Language Understanding

Setting. We integrate GFSA into 3 pre-trained large language models: BERT, ALBERT, and RoBERTa. We evaluate them on the GLUE benchmark, which includes 3 categories of natural language understanding tasks: i) single-sentence, ii) similarity and paraphrasing, and iii) natural language inference tasks. For each task, we select the best hyperparameters for GFSA, and the other hyperparameters are fixed. The detailed experimental settings are in Appendix K.1.

Results. The results are shown in Table 1. When GFSA was plugged into backbones, average performance scores improved across all models over pure backbones. This indicates that GFSA is effective in both large models like BERT and RoBERTa, as well as relatively smaller models like ALBERT. It is worth mentioning that in the case of RoBERTa finetuned on the CoLA dataset; there is a significant margin increase from 60.34% to 64.11%, which is a 3.77% improvement with only 144 additional parameters. When compared to ContraNorm, GFSA shows a greater performance improvement on average. Fig. 5 in Appendix C shows that these performance enhancements can be attributed to addressing the oversmoothing issue through the designed graph filter.

Table 1: Results comparison on GLUE benchmark. Avg denotes the average performance.

| Method          | #Params   |   CoLA |   SST-2 |   MRPC |   QQP |   STS-B | MNLI-m/mm     |   QNLI |   RTE |   Avg |
|----------|----------|--------|---------|--------|-------|---------|----------|--------|-------|-------|
| BERT BASE [16] + ContraNorm + GFSA    | 110M      |  56.79 |   93.81 |  88.7  | 88.32 |   88.16 | 84.96/84.15   |  91.63 | 66.06 | 82.51 |
| BERT BASE [16] + ContraNorm + GFSA    | 110M      |  59.89 |   93.92 |  89.88 | 88.51 |   88.36 | 85.11/84.50   |  91.84 | 69.31 | 83.48 |
| BERT BASE [16] + ContraNorm + GFSA    | 110M      |  59.56 |   94.15 |  90.6  | 88.46 |   88.33 | 85.12 / 85.06 |  91.95 | 68.95 | 83.58 |
| ALBERT BASE [40] + ContraNorm + GFSA  | 11M       |  57.86 |   92.32 |  91.8  | 85.3  |   90.37 | 85.37 /84.37  |  91.76 | 76.9  | 84.01 |
| ALBERT BASE [40] + ContraNorm + GFSA  | 11M       |  57.45 |   93    |  92.83 | 87.78 |   90.55 | 85.06/84.57   |  92.28 | 78.7  | 84.69 |
| ALBERT BASE [40] + ContraNorm + GFSA  | 11M       |  60.21 |   93.23 |  92.47 | 87.79 |   90.63 | 85.29/ 84.92  |  92.17 | 78.7  | 85.05 |
| RoBERTa BASE [44] + ContraNorm + GFSA | 125M      |  60.34 |   94.84 |  92.28 | 88.86 |   89.99 | 87.94/87.30   |  92.57 | 78.7  | 85.87 |
| RoBERTa BASE [44] + ContraNorm + GFSA | 125M      |  63.06 |   95.41 |  93.17 | 88.91 |   90.34 | 87.88/87.40   |  92.82 | 80.51 | 86.61 |
| RoBERTa BASE [44] + ContraNorm + GFSA | 125M      |  64.11 |   95.41 |  93.52 | 89.09 |   90.35 | 87.99 / 87.54 |  92.97 | 80.14 | 86.79 |

## 5.2 Experiments on Causal Language Modeling

Setting. We also validate the effectiveness of GFSA on causal language modeling problems. We finetune GPT2 [61] on the following 3 datasets: Penn Treebank (PTB) [47], WikiText-2, and WikiText103 [50]. Following the evaluation method in Yao et al. [93], we finetune models for 15 epochs with PTB, 4 epochs with WikiText-103, and 10 epochs with WikiText-2, and report the perplexity for sensitivity metric. The detailed experimental settings are in Appendix L.1.

Results. Table 2 shows the perplexity on PTB, WitiText-2, and WikiText-103. Across all datasets, GPT2 with GFSA consistently outperforms the vanilla GPT2. Our GFSA improves the average perplexity from 18.806 to 18.764. Note that performance improvements

Table 2: Results comparison on GPT-2 finetuned with GFSA. Avg denotes the average performance.

| Method   | #Params   |    PTB |   WikiText-2 WikiText-103 |   WikiText-2 WikiText-103 |    Avg |
|----------|----------|--------|----------|----------|--------|
| GPT2     | 117M      | 19.513 |          20.966 |          15.939 | 18.806 |
| + GFSA   | 117M      | 19.45  |          20.923 |          15.919 | 18.764 |

are made with only 144 additional learnable parameters for 12 layers with 12 heads.

## 5.3 Experiments on Vision Transformers

Setting. We aim to demonstrate the efficacy of our GFSA across a spectrum of ViT backbones. We choose DeiT [74], CaiT [75], and Swin [45] as the backbone, and the models are trained from scratch. When training the 12-layer DeiT, we follow the same training recipe, hyperparameters, and data augmentation from Touvron et al. [74]. For detailed experimental settings, see Appendix M.1.

Results. The experimental evaluations are summarized in Table 3. We compare various models on the ImageNet-1k benchmark. The results show that the proposed GFSA successfully enhances DeiT, CaiT, and Swin across all depth settings and training methods. GFSA provides additional parameters less than 72 for 12-layer DeiT while improving top-1 accuracy by 1.63%. To sum up, we observed that both shallow and deep ViTs can achieve the following benefits from GFSA: i) The filter response shows GFSA can preserve higher-frequency representation (cf. Fig. 2 (a)) and ii) Fig. 2 (b) shows that GFSA mitigates the increase in the cosine similarity of representation as the layer gets deeper. We further compare with state-of-the-art models that use Fourier transforms rather than graph filters in Appendix M.5. We also show results under the same settings as ContraNorm [28] in Appendix M.6.

Table 3: Results comparison on ImageNet-1k. Our full results with other models are in Appendix M.4.

| Category    | Method          |   Input Size |   #Layers | #Params   |   Top-1 Acc |
|----------|----------|----------|----------|----------|----------|
| Transformer | DeiT-S [74]          |          224 |        12 | 22M       |        79.8 |
| Transformer | DeiT-S + AttnScale [80]  |          224 |        12 | 22M       |        80.7 |
| Transformer | DeiT-S + FeatScale [80]  |          224 |        12 | 22M       |        80.9 |
| Transformer | DeiT-S + ContraNorm [28] |          224 |        12 | 22M       |        80.4 |
| Transformer | Swin-S [45]          |          224 |        12 | 50M       |        82.9 |
| Transformer | DeiT-S [74]          |          224 |        24 | 43M       |        80.5 |
| Transformer | CaiT-S [75]          |          224 |        24 | 47M       |        82.6 |
| Transformer | DeiT-S + AttnScale [80]  |          224 |        24 | 44M       |        81.1 |
| Transformer | DeiT-S + FeatScale [80]  |          224 |        24 | 44M       |        81.3 |
| Transformer | DeiT-S + ContraNorm [28] |          224 |        24 | 43M       |        80.7 |
| GFSA        | DeiT-S + GFSA          |          224 |        12 | 22M       |        81.1 |
| GFSA        | DeiT-S + GFSA          |          224 |        24 | 43M       |        81.5 |
| GFSA        | CaiT-S + GFSA          |          224 |        24 | 47M       |        82.8 |
| GFSA        | Swin-S + GFSA          |          224 |        12 | 50M       |        83   |

## 5.4 Experiments on Graph-level Tasks

Setting. To evaluate the efficacy of GFSA on graph-level tasks, we conduct experiments on a broader range of datasets. We use datasets from Long-Range Graph Benchmark (LRGB) [21] (e.g., Peptide-func and Peptide-struct), Benchmarking GNNs [22] (e.g., ZINC, MNIST, CIFAR10), Open Graph Benchmark (OGB) dataset [32] (e.g., Molhiv and MolTox21), and OGB-LSC dataset (i.e., PCQM4M-LSC) [33]. We choose Graphormer [94], Graph-ViT [30], and GPS [63] as our backbone architectures, following their original experimental protocols for fair comparison. For GPS, we replace its self-attention module with our GFSA while maintaining its best configuration and other hyperparameters. For Graph-ViT, we apply GFSA to the Hadamard self-attention method, which He et al. [30] propose as optimal. For a detailed experimental setting, see Appendix O.1.

Results. Tables 4, 5, and 6 show consistent performance improvements when GFSA is integrated with backbone architectures. Graph-ViT + GFSA shows improvements on all datasets. On Peptidefunc, it achieves a 0.65% increase in AP. Notably, in PCQM4M, incorporating GFSA improves the validation MAE by 7.20%. Due to space constraints, the results with standard deviation are included in Appendix O.2.

## 5.5 Experiments on Automatic Speech Recognition

Setting. We conduct automatic speech recognition (ASR) experiments on the LibriSpeech 3 dataset [55], which consists of audio recordings paired with their transcriptions. We use Branch-

3 http://www.openslr.org/12

Table 5: Results on PCQM4M and PCQM4Mv2

| Table 4: Results on ZINC   | Table 4: Results on ZINC   | Table 4: Results on ZINC   |          |         | PCQM4M    | PCQM4M         | PCQM4Mv2    | PCQM4Mv2       |
|----------|----------|----------|----------|---------|----------|----------|----------|----------|
| Method          | #Params          | MAE ( ↓ )          | Method          | #Params | Train ( ↓ | Validate ( ↓ ) | Train ( ↓ ) | Validate ( ↓ ) |
| Graphormer + GFSA          | 500K 500K          | 0.1240 0.1189          | Graphormer + GFSA | 48.3M   | 0.0535    | 0.1286         | 0.0250      | 0.0862         |
|          |          |          |          | 48.3M   | 0.0312    | 0.1193         | 0.0249      | 0.0860         |

Table 6: Experimental evalutation of GFSA plugged into GPS and Graph-ViT. Results marked with † indicate settings where we conducted our own experiments due to unavailable Hadamard self-attention performance in He et al. [30]'s paper.

| Method          | Peptide-func          | Peptide-struct          | MNIST          | CIFAR10          | Molhiv          | MolTOX21          | ZINC          |
|----------|----------|----------|----------|----------|----------|----------|----------|
| Method          | AP ( ↑ )          | MAE ( ↓ )          | Accuracy ( ↑ )          | Accuracy ( ↑ )          | ROCAUC ( ↑ )          | ROCAUC ( ↑ )          | MAE ( ↓ )          |
| GPS          | 0.6535 ± 0.0041          | 0.2500 ± 0.0005 ± 0.0013        | 0.9805 ± 0.0013 0.7230 ± 0.0036 |          | -          | -          | 0.070 ± 0.004          |
| + GFSA          | 0.6593 ± 0.0094          | 0.2496          | 0.9814 ± 0.0014          | 0.7244 ± 0.0048          | -          | -          | 0.069 ± 0.002          |
| Graph-ViT + GFSA | 0.6919 ± 0.0085 † 0.6964 ± 0.0025 | 0.2474 ± 0.0016 0.2461 ± 0.0024 | 0.9820 ± 0.0005 0.9826 ± 0.0004 | 0.6967 ± 0.0040 0.6987 ± 0.0028 | 0.7792 ± 0.0149 0.7830 ± 0.0109 | 0.7851 ± 0.0077 0.7895 ± 0.0069 | 0.0849 ± 0.0047 0.0845 ± 0.0032 |

former [59] and a pure Transformer. For implementation, we follow the recipes of SpeechBrain [65] and the detailed settings are in Appendix N.1.

Results. Table 7 compares word error rates (WERs) on LibriSpeech 100h and 960h. For 100h, Transformer+GFSA achieves 10.30/25.30 on the test clean/other set, which is a 6.53% improvement over the Transformer for the WER of the test clean. For 960h, Transformer+GFSA shows a WER result of 2.31 in test clean, a 4.55% improvement over Transformer and Branchformer+GFSA achieves 2.31/5.49 with an LM on the test clean/other sets. Fig. 8 in Appendix N.2 depicts the learning curves of train loss and valid loss when using GFSA, showing the effectiveness of our proposed filter.

Table 7: Results for ASR training on LibriSpeech 100h and 960h with GFSA

| Method          | #Params   | LibriSpeech 100h   | LibriSpeech 100h   | LibriSpeech 960h   | LibriSpeech 960h   |
|----------|----------|----------|----------|----------|----------|
| Method          | #Params   | test-clean WER     | test-other WER     | test-clean WER     | test-other WER     |
| Transformer + GFSA  | 71.5M     | 11.02          | 25.42          | 2.42          | 5.50          |
| Transformer + GFSA  | 71.5M     | 10.30          | 24.30          | 2.31          | 5.49          |
| Branchformer + GFSA | 109.8M    | 9.63          | 22.43          | 2.13          | 5.00          |
| Branchformer + GFSA | 109.8M    | 9.60          | 22.25          | 2.11          | 4.94          |

## 5.6 Experiments on Code Classification

Setting. We conduct a code defect detection task based on Devign dataset provided by Zhou et al. [105]. We use RoBERTa [44], CodeBERT [23], PLBART [2], and CodeT5 [84] as our backbone models. The detailed settings are in Appendix P.1.

Results. Table 8 shows the accuracy of all models; GFSA results better than the base models. The biggest improvement is 2.40% for RoBERTa. In the case of CodeT5-base, using GFSA shows an accuracy of 64.75, an improvement of 1.95% from 63.51. CodeT5small+GFSA has only about 100 additional parameters compared to CodeT5-small with 60M parameters, and even more impressively, it surpasses the accuracy of CodeT5-base. The biggest improvement is 2.40% for RoBERTa. In Appendix P.2, we include case studies for this task. We also report the results of the code clone detection task in Appendix Q.

Table 8: Results on code classification. The number in ( ↑ ) indicates the improvement rate.

| Method          | Accuracy         |
|----------|----------|
| RoBERTa          | 62.88          |
| + GFSA + GFSA      | 64.39 ( ↑ 2.40%) |
| CodeBERT          | 64.31 64.49 ( ↑  |
|          | 0.12%)          |
| PLBART          | 62.63          |
| + GFSA          | 62.96 ( ↑ 0.52%) |
| CodeT5-small       | 63.25          |
| + GFSA CodeT5-base | 63.69 ( ↑ 0.70%) |
|          | 63.51          |
| + GFSA          | 64.75 ( ↑ 1.95%) |

## 6 Discussion on Runtime Overheads

Limitation. The introduction of our GFSA layer results in a slight increase in training and inference time. We report the runtimes when plugging GFSA in Appendices R and S. For GLUE benchmark, integrating GFSA into BERT enhances the average performance from 82.51% to 83.58% (see Table 1) with more overhead of less than 36 seconds per epoch based on average training time (see Table 23). Considering the improvements, the increases in training time are negligible.

## GFSA in selected layers: a strategy to mitigate the limi-

tation. As GFSA requires more calculation than the original self-attention, the runtime after using GFSA slightly increases. Our experiments initially applied GFSA across all Transformer layers (as discussed in Section 5); however, to reduce computational load, we propose a selective application strategy. For this purpose, GFSA is used only on even-numbered layers. In Tables 35 to 39 of Appendix T, the results show that this strategy effectively reduces runtime increases while preserving comparable performance to the full-layer GFSA integration. Notably, the selective application of GFSA cuts the per-epoch runtime increase by 26.90% relative to its full-layer application, with only a 7.39% increase in runtime per epoch compared to the backbone model in Table 39.

GFSA in linear Transformers. Although GFSA requires additional computation for calculating ¯ A 2 , we explore integrating GFSA with linear attention variants to maintatin efficiency and scalability. Recent approaches [36, 70] achieve linear complexity by reformulating softmax operations and reordering matrix multiplication in self-attention. We apply similar principles to compute second-order self-attention efficiently, enabling ˜ H GFSA calculation with linear complexity with respect to sequence length. Fig. 4 shows the performance, runtime and GPU usage changes when applying our GFSA to Transformers with linear complexity. GFSA still improves performance compared to the backbone model, while the increase in time and GPU usage is minimal. Notably, when GFSA is applied to Efficient Attention [70], the performance is improved and the runtime is 11.82 times faster

<!-- image -->

even

Figure 3: Effectiveness of our selective layer strategy on ImageNet-1k. This shows out strategy's ability to maintain accuracy benefits while mitigating runtime increases.

Figure 4: Performance ( x -axis), runtime ( y -axis), and GPU usage (circle sizes) of various Transformers and integrated GFSA on Long-Range benchmark

<!-- image -->

than when GFSA is applied to the vanilla self-attention. This shows that GFSA can be effectively implemented with linear complexity architectures while preserving its benefits and providing a solution for addressing computational concerns.

## 7 Conclusion

Our proposed GFSA achieves high performance with improvements on a variety of tasks. GFSA is a simple yet effective method that enriches self-attention in Transformers with more diverse frequency information. This enables GFSA to address the oversmoothing problem and learn better latent representations for downstream tasks. However, our GFSA does not bring significant overheads in those Transformers' empirical runtime complexity. One can use more complicated graph filters to enhance accuracy more, but our goal is to find a balance between accuracy enhancements and overheads in runtime complexity.

We believe that GFSA suggests a promising new direction for improving Transformers. GFSA can be implement with simple way and used in conjunction with other techniques to further improve the performance of Transformers. Considering the ongoing advancements in large language models, such as GPT-4 [1] and LLaMA [76], we hope that our approach may offer new insights for enhancing their performance and efficiency.

## Acknowledgements

N. Park was partly supported by the Korea Advanced Institute of Science and Technology (KAIST) grant funded by the Korea government (MSIT) (No. G04240001, Physics-inspired Deep Learning, 10%), Institute for Information &amp; Communications Technology Planning &amp; Evaluation (IITP) grants funded by the Korea government (MSIT) (No. RS-2020-II201361, Artificial Intelligence Graduate School Program (Yonsei University), 20%; No. RS-2024-00457882, AI Research Hub Project, 50%), and Samsung Electronics Co., Ltd. (No. G01240136, KAIST Semiconductor Research Fund (2nd), 10%). K. Lee acknowledges support from the U.S. National Science Foundation under grant IIS 2338909. Dr. Trask acknowledges funding under the Department of Energy under the Mathematical Multifaceted Integrated Capability Centers program.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 , 2023.
- [2] Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. Unified pre-training for program understanding and generation. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 2655-2668, 2021.
- [3] Ameen Ali, Tomer Galanti, and Lior Wolf. Centered self-attention layers. arXiv preprint arXiv: 2306.01610 , 2023.
- [4] Jiawang Bai, Li Yuan, Shu-Tao Xia, Shuicheng Yan, Zhifeng Li, and Wei Liu. Improving vision transformers by revisiting high-frequency components. In European Conference on Computer Vision , pages 1-18. Springer, 2022.
- [5] Luisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. The fifth pascal recognizing textual entailment challenge. TAC , 7:8, 2009.
- [6] Edward De Brouwer, Jaak Simm, Adam Arany, and Yves Moreau. GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series. In Advances in Neural Information Processing Systems (NeurIPS) , 2019.
- [7] Chen Cai and Yusu Wang. A note on over-smoothing for graph neural networks. arXiv preprint arXiv:2006.13318 , 2020.
- [8] Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation. arXiv preprint arXiv:1708.00055 , 2017.
- [9] Nuo Chen, Linjun Shou, Ming Gong, Jian Pei, Bowen Cao, Jianhui Chang, Daxin Jiang, and Jia Li. Alleviating over-smoothing for unsupervised sentence representation. arXiv preprint arXiv:2305.06154 , 2023.
- [10] Tianlong Chen, Zhenyu Zhang, Yu Cheng, Ahmed Awadallah, and Zhangyang Wang. The principle of diversity: Training stronger vision transformers calls for reducing all levels of redundancy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 12020-12030, 2022.
- [11] Zihan Chen, Hongbo Zhang, Xiaoji Zhang, and Leqi Zhao. Quora question pairs, 2018.
- [12] Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. Adaptive universal generalized PageRank graph neural network. In Proceedings of the International Conference on Learning Representations (ICLR) , 2021.
- [13] Jeongwhan Choi, Seoyoung Hong, Noseong Park, and Sung-Bae Cho. Gread: Graph neural reaction-diffusion networks. In International Conference on Machine Learning (ICML) , pages 5722-5747. PMLR, 2023.

- [14] George Dasoulas, Kevin Scaman, and Aladin Virmaux. Lipschitz normalization for selfattention layers with application to graph neural networks. In International Conference on Machine Learning (ICML) , pages 2456-2466. PMLR, 2021.
- [15] Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in Neural Information Processing Systems (NeurIPS) , 2016.
- [16] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) , pages 4171-4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423 .
- [17] Bill Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In Third International Workshop on Paraphrasing (IWP2005) , 2005.
- [18] Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In International Conference on Machine Learning (ICML) , pages 2793-2803. PMLR, 2021.
- [19] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In Proceedings of the International Conference on Learning Representations (ICLR) , 2021.
- [20] Gbètondji JS Dovonon, Michael M Bronstein, and Matt J Kusner. Setting the record straight on transformer oversmoothing. arXiv preprint arXiv:2401.04301 , 2024.
- [21] Vijay Prakash Dwivedi, Ladislav Rampášek, Mikhail Galkin, Ali Parviz, Guy Wolf, Anh Tuan Luu, and Dominique Beaini. Long range graph benchmark. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track , 2022. URL https://openreview.net/forum?id=in7XC5RcjEn .
- [22] Vijay Prakash Dwivedi, Chaitanya K Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. Journal of Machine Learning Research , 24(43):1-48, 2023.
- [23] Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, et al. CodeBERT: A pre-trained model for programming and natural languages. In Findings of the Association for Computational Linguistics: EMNLP 2020 , pages 1536-1547, 2020.
- [24] Johannes Gasteiger, Stefan Weißenberger, and Stephan Günnemann. Diffusion improves graph learning. In Advances in Neural Information Processing Systems (NeurIPS) , volume 32, 2019.
- [25] Chengyue Gong, Dilin Wang, Meng Li, Vikas Chandra, and Qiang Liu. Vision transformers with patch diversification. arXiv preprint arXiv:2104.12753 , 2021.
- [26] Alex Graves and Navdeep Jaitly. Towards end-to-end speech recognition with recurrent neural networks. In International Conference on Machine Learning (ICML) , pages 1764-1772. PMLR, 2014.
- [27] Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, et al. Conformer: Convolution-augmented transformer for speech recognition. arXiv preprint arXiv:2005.08100 , 2020.
- [28] Xiaojun Guo, Yifei Wang, Tianqi Du, and Yisen Wang. ContraNorm: A contrastive learning perspective on oversmoothing and beyond. In Proceedings of the International Conference on Learning Representations (ICLR) , 2023.

- [29] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- [30] Xiaoxin He, Bryan Hooi, Thomas Laurent, Adam Perold, Yann LeCun, and Xavier Bresson. A generalization of vit/mlp-mixer to graphs. In International conference on machine learning (ICML) , pages 12724-12745. PMLR, 2023.
- [31] Nicholas J Higham. Functions of matrices: theory and computation . SIAM, 2008.
- [32] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. Advances in neural information processing systems (NeurIPS) , 33:22118-22133, 2020.
- [33] Weihua Hu, Matthias Fey, Hongyu Ren, Maho Nakata, Yuxiao Dong, and Jure Leskovec. OGBLSC: A large-scale challenge for machine learning on graphs. arXiv preprint arXiv:2103.09430 , 2021.
- [34] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 4700-4708, 2017.
- [35] John J Irwin, Teague Sterling, Michael M Mysinger, Erin S Bolstad, and Ryan G Coleman. Zinc: a free tool to discover chemistry for biology. Journal of chemical information and modeling , 52(7):1757-1768, 2012.
- [36] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning , pages 5156-5165. PMLR, 2020.
- [37] Nicolas Keriven. Not too little, not too much: a theoretical analysis of graph (over) smoothing. In Advances in Neural Information Processing Systems (NeurIPS) , volume 35, pages 22682281, 2022.
- [38] Jayoung Kim, Yehjin Shin, Jeongwhan Choi, Hyowon Wi, and Noseong Park. Polynomialbased self-attention for table representation learning. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 24509-24526. PMLR, 21-27 Jul 2024. URL https://proceedings.mlr.press/v235/kim24ae.html .
- [39] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In Proceedings of the International Conference on Learning Representations (ICLR) , 2017.
- [40] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942 , 2019.
- [41] Siddique Latif, Aun Zaidi, Heriberto Cuayahuitl, Fahad Shamshad, Moazzam Shoukat, and Junaid Qadir. Transformers in speech processing: A survey. arXiv preprint arXiv:2303.11607 , 2023.
- [42] James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon. Fnet: Mixing tokens with fourier transforms. arXiv preprint arXiv:2105.03824 , 2021.
- [43] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pages 7871-7880, 2020.

- [44] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 , 2019.
- [45] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision , pages 10012-10022, 2021.
- [46] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 , 2017.
- [47] Mitchell Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of english: The penn treebank. 1993.
- [48] Antonio G. Marques, Santiago Segarra, and Gonzalo Mateos. Signal processing on directed graphs: The role of edge directionality when processing and learning from network data. IEEE Signal Processing Magazine , 37(6):99-116, 2020. doi: 10.1109/MSP.2020.3014597.
- [49] Sohir Maskey, Raffaele Paolino, Aras Bacho, and Gitta Kutyniok. A fractional graph laplacian approach to oversmoothing. In Advances in Neural Information Processing Systems (NeurIPS) , 2023.
- [50] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843 , 2016.
- [51] Luis Müller, Mikhail Galkin, Christopher Morris, and Ladislav Rampášek. Attending to graph transformers. Transactions on Machine Learning Research , 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=HhbqHBBrfZ .
- [52] Tam Nguyen, Tan Nguyen, and Richard Baraniuk. Mitigating over-smoothing in transformers via regularized nonlocal functionals. In Advances in Neural Information Processing Systems (NeurIPS) , volume 36, 2023.
- [53] Lorenzo Noci, Sotiris Anagnostidis, Luca Biggio, Antonio Orvieto, Sidak Pal Singh, and Aurelien Lucchi. Signal propagation in transformers: Theoretical perspectives and the role of rank collapse. In Advances in Neural Information Processing Systems (NeurIPS) , volume 35, pages 27198-27211, 2022.
- [54] Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node classification. In Proceedings of the International Conference on Learning Representations (ICLR) , 2020.
- [55] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP) , pages 5206-5210. IEEE, 2015.
- [56] Daniel S Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D Cubuk, and Quoc V Le. Specaugment: A simple data augmentation method for automatic speech recognition. Interspeech 2019 , 2019.
- [57] Badri Patro and Vijay Agneeswaran. Scattering vision transformer: Spectral mixing matters. In Advances in Neural Information Processing Systems (NeurIPS) , volume 36, 2023.
- [58] Badri N Patro, Vinay P Namboodiri, and Vijay Srinivas Agneeswaran. Spectformer: Frequency and attention is what you need in a vision transformer. arXiv preprint arXiv:2304.06446 , 2023.
- [59] Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji Watanabe. Branchformer: Parallel mlp-attention architectures to capture local and global context for speech recognition and understanding. In International Conference on Machine Learning (ICML) , pages 17627-17643. PMLR, 2022.
- [60] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.

- [61] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog , 1(8):9, 2019.
- [62] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research , 21(1):5485-5551, 2020.
- [63] Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. Recipe for a general, powerful, scalable graph transformer. In Advances in Neural Information Processing Systems (NeurIPS) , volume 35, pages 14501-14515, 2022.
- [64] Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou. Global filter networks for image classification. In Advances in Neural Information Processing Systems (NeurIPS) , volume 34, pages 980-993, 2021.
- [65] Mirco Ravanelli, Titouan Parcollet, Peter Plantinga, Aku Rouhe, Samuele Cornell, Loren Lugosch, Cem Subakan, Nauman Dawalatabad, Abdelwahab Heba, Jianyuan Zhong, Ju-Chieh Chou, Sung-Lin Yeh, Szu-Wei Fu, Chien-Feng Liao, Elena Rastorgueva, François Grondin, William Aris, Hwidong Na, Yan Gao, Renato De Mori, and Yoshua Bengio. SpeechBrain: A general-purpose speech toolkit, 2021. arXiv:2106.04624.
- [66] T. Konstantin Rusch, Michael M. Bronstein, and Siddhartha Mishra. A survey on oversmoothing in graph neural networks. arXiv preprint arXiv: Arxiv-2303.10993 , 2023.
- [67] Aliaksei Sandryhaila and José MF Moura. Discrete signal processing on graphs. IEEE transactions on signal processing , 61(7):1644-1656, 2013.
- [68] Aliaksei Sandryhaila and Jose MF Moura. Discrete signal processing on graphs: Frequency analysis. IEEE Transactions on Signal Processing , 62(12):3042-3054, 2014.
- [69] Roy Schwartz, Jesse Dodge, Noah A Smith, and Oren Etzioni. Green AI. Communications of the ACM , 63(12):54-63, 2020.
- [70] Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, and Hongsheng Li. Efficient attention: Attention with linear complexities. In Proceedings of the IEEE/CVF winter conference on applications of computer vision , pages 3531-3539, 2021.
- [71] Han Shi, Jiahui Gao, Hang Xu, Xiaodan Liang, Zhenguo Li, Lingpeng Kong, Stephen Lee, and James T Kwok. Revisiting over-smoothing in bert from the perspective of graph. In Proceedings of the International Conference on Learning Representations (ICLR) , 2022.
- [72] Yehjin Shin, Jeongwhan Choi, Hyowon Wi, and Noseong Park. An attentive inductive bias for sequential recommendation beyond the self-attention. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 8984-8992, 2024.
- [73] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing , pages 1631-1642, 2013.
- [74] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers &amp; distillation through attention. In International Conference on Machine Learning (ICML) , pages 10347-10357. PMLR, 2021.
- [75] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou. Going deeper with image transformers. In Proceedings of the IEEE/CVF international conference on computer vision , pages 32-42, 2021.
- [76] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 , 2023.
- [77] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems (NeurIPS) , volume 30, 2017.

- [78] Petar Veliˇ ckovi´ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph Attention Networks. In Proceedings of the International Conference on Learning Representations (ICLR) , 2018.
- [79] Guangtao Wang, Rex Ying, Jing Huang, and Jure Leskovec. Multi-hop attention graph neural network. In IJCAI , 2021.
- [80] Peihao Wang, Wenqing Zheng, Tianlong Chen, and Zhangyang Wang. Anti-oversmoothing in deep vision transformers via the fourier domain analysis: From theory to practice. In Proceedings of the International Conference on Learning Representations (ICLR) , 2022.
- [81] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768 , 2020.
- [82] Wei Wang, Ming Yan, and Chen Wu. Multi-granularity hierarchical attention fusion networks for reading comprehension and question answering. arXiv preprint arXiv:1811.11934 , 2018.
- [83] Wenhan Wang, Ge Li, Bo Ma, Xin Xia, and Zhi Jin. Detecting code clones with graph neural network and flow-augmented abstract syntax tree. In 2020 IEEE 27th International Conference on Software Analysis, Evolution and Reengineering (SANER) , pages 261-271, 2020. doi: 10.1109/SANER48275.2020.9054857.
- [84] Yue Wang, Weishi Wang, Shafiq Joty, and Steven CH Hoi. CodeT5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing , pages 86968708, 2021.
- [85] Alex Warstadt, Amanpreet Singh, and Samuel R Bowman. Neural network acceptability judgments. Transactions of the Association for Computational Linguistics , 7:625-641, 2019.
- [86] Ross Wightman. Pytorch image models. https://github.com/rwightman/ pytorch-image-models , 2019.
- [87] Adina Williams, Nikita Nangia, and Samuel R Bowman. A broad-coverage challenge corpus for sentence understanding through inference. arXiv preprint arXiv:1704.05426 , 2017.
- [88] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. CvT: Introducing convolutions to vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision , pages 22-31, 2021.
- [89] Xinyi Wu, Amir Ajorlou, Zihui Wu, and Ali Jadbabaie. Demystifying oversmoothing in attention-based graph neural networks. In Advances in Neural Information Processing Systems (NeurIPS) , 2023.
- [90] Xinyi Wu, Zhengdao Chen, William Wei Wang, and Ali Jadbabaie. A non-asymptotic analysis of oversmoothing in graph neural networks. In Proceedings of the International Conference on Learning Representations (ICLR) , 2023.
- [91] Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. In International Conference on Machine Learning (ICML) , pages 5453-5462, 2018.
- [92] Hanqi Yan, Lin Gui, Wenjie Li, and Yulan He. Addressing token uniformity in transformers via singular value transformation. In Uncertainty in Artificial Intelligence , pages 2181-2191. PMLR, 2022.
- [93] Zhewei Yao, Xiaoxia Wu, Conglong Li, Connor Holmes, Minjia Zhang, Cheng Li, and Yuxiong He. Random-ltd: Random and layerwise token dropping brings efficient training for large-scale transformers. arXiv preprint arXiv:2211.11586 , 2022.
- [94] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. Do transformers really perform badly for graph representation? In Advances in Neural Information Processing Systems (NeurIPS) , volume 34, pages 28877-28888, 2021.

- [95] Youn-Yeol Yu, Jeongwhan Choi, Woojin Cho, Kookjin Lee, Nayong Kim, Kiseok Chang, ChangSeung Woo, ILHO KIM, SeokWoo Lee, Joon Young Yang, SOOYOUNG YOON, and Noseong Park. Learning flexible body collision dynamics with hierarchical contact mesh transformer. In The Twelfth International Conference on Learning Representations , 2024. URL https://openreview.net/forum?id=90yw2uM6J5 .
- [96] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In Proceedings of the IEEE/CVF international conference on computer vision , pages 558-567, 2021.
- [97] Zhanpeng Zeng, Yunyang Xiong, Sathya Ravi, Shailesh Acharya, Glenn M Fung, and Vikas Singh. You only sample (almost) once: Linear cost self-attention via bernoulli sampling. In International conference on machine learning , pages 12321-12332. PMLR, 2021.
- [98] Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, and Joshua M. Susskind. Stabilizing transformer training by preventing attention entropy collapse. In Proceedings of the 40th International Conference on Machine Learning (ICML) , volume 202, pages 40770-40803. PMLR, 2023.
- [99] Aston Zhang, Alvin Chan, Yi Tay, Jie Fu, Shuohang Wang, Shuai Zhang, Huajie Shao, Shuochao Yao, and Roy Ka-Wei Lee. On orthogonality constraints for transformers. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers) , pages 375-382, 2021.
- [100] Ying Zhang, Mohammad Pezeshki, Philémon Brakel, Saizheng Zhang, César Laurent, Yoshua Bengio, and Aaron Courville. Towards end-to-end speech recognition with deep convolutional neural networks. Interspeech 2016 , 2016.
- [101] Daquan Zhou, Bingyi Kang, Xiaojie Jin, Linjie Yang, Xiaochen Lian, Zihang Jiang, Qibin Hou, and Jiashi Feng. DeepViT: Towards deeper vision transformer. arXiv preprint arXiv:2103.11886 , 2021.
- [102] Daquan Zhou, Yujun Shi, Bingyi Kang, Weihao Yu, Zihang Jiang, Yuan Li, Xiaojie Jin, Qibin Hou, and Jiashi Feng. Refiner: Refining self-attention for vision transformers. arXiv preprint arXiv:2106.03714 , 2021.
- [103] Haoyi Zhou, Siyang Xiao, Shanghang Zhang, Jieqi Peng, Shuai Zhang, and Jianxin Li. Jump self-attention: Capturing high-order statistics in transformers. In Advances in Neural Information Processing Systems (NeurIPS) , volume 35, pages 17899-17910, 2022.
- [104] Jie Zhou, Ganqu Cui, Shengding Hu, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. Graph neural networks: A review of methods and applications. AI open , 1:57-81, 2020.
- [105] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in neural information processing systems (NeurIPS) , volume 32, 2019.
- [106] Chunya Zou, Andi Han, Lequan Lin, and Junbin Gao. A simple yet effective svd-gcn for directed graphs. arXiv preprint arXiv:2205.09335 , 2022.

## Appendix

## A Reproducibility Statement

To ensure the reproducibility and completeness of this paper, we include the Appendix with 12 sections. Appendix I provides our PyTorch-style pseudo code for our GFSA method. The pseudo code helps to implement our GFSA to any Transformers used a pure self-attention. All experiments in the paper are reproducible with additional implementation details provided in Appendices K to Q.

## B Broader Impact

In terms of the broader impact of this research on society, we do not see the very negative impacts that might be expected. However, this paper may have implications for the carbon footprint and accessibility of learning algorithms. The computations required for machine learning research are rapidly growing, resulting in a larger carbon footprint [69]. Our study improves performance and increases runtime very slightly, but the runtime increase is not very significant. However, in future research, it will also be important to study and improve our GFSA by taking carbon footprints into account.

GFSA improves the performance of existing Transformer-based models, which can have many positive impacts on society through services that utilize natural language processing, computer vision, and speech recognition. However, it will also be important to improve GFSA by considering other dimensions of AI, such as robustness to adversarial examples, fairness, and explainability.

## C Oversmoothing and Additional Visualizations

In Fig. 2, we show the visualizations of oversmoothing characteristics in DeiT. We also provide visualizations in other domains. We show the filter response, cosine similarity, and singular value of BERT finetuned on STS-B dataset of GLUE tasks in Fig. 5 and Graphormer finetuned on ZINC dataset in Fig. 6.

To characterize self-attention, we first analyze the filter response of self-attention in the frequency domain. We follow the method used by Wang et al. [80] for spectral visualization of the self-attention matrix. As shown in Fig. 2 (a), DeiT has a near-zero magnitude for the high frequencies, which is characteristic of a low-frequency filter and is likely to result in oversmoothing when applied multiple times.

We follow the calculation method of Guo et al. [28] for cosine similarity. As shown in Fig. 2 (b), the higher similarity as the layers of the model get deeper is related to the oversmoothing problem. To further analyze this issue, we also consider the dimensionality collapse in Transformer-based models. We plot the singular value distribution of the feature in the last block. As shown in Fig. 2 (c), insignificant, near-zero values dominate the feature distribution. As layers get deeper, the similarity of features increases and dimensional collapse occurs. The oversmoothing problem is the same in BERT and Graphormer, as shown in Fig. 5 and Fig. 6.

Figure 5: Filter frequency response, cosine similarity, and singular values on STS-B for BERT and BERT+GFSA

<!-- image -->

Figure 6: Filter frequency response, cosine similarity, and singular values on ZINC for Graphormer and Graphormer+GFSA

<!-- image -->

## D Analysis of Frequency Responses with Visualizations

We analyze the frequency responses, which represent the impact of learned coefficients, for all 12 layers of BERTBASE with and without GFSA. From Fig. 7, our analysis reveals that GFSA learns various filter types between layers. In early layers, we observe a tendency towards low-pass filtering, with prominent peaks at low frequencies. This aligns with the need for broader feature extraction in initial layers. The middle layers show a mix of low-pass and high-pass characteristics, with more complex frequency responses. This suggests GFSA is learning to balance between feature extraction and refinement. In deep layers, there is a noticeable shift towards higher frequency responses, indicating a move towards high-pass filtering. This shift supports our claim that GFSA can mitigate oversmoothing in deeper layers. BERTBASE+GFSA shows a consistently higher magnitude response at higher frequencies, especially in deeper layers, compared to vanilla BERT. In other word, vanilla self-attention works primarily as a low-pass filter, while GFSA utilizes a wider range of frequencies.

Figure 7: Visualization of the frequency responses for all 12 layers of BERT trained on STS-B dataset. The top-left figure corresponds to the first layer, and the bottom-right figure corresponds to the last layer.

<!-- image -->

## E Frequency Analyses in the Singular Value Domain

Graph signal processing (GSP) [67, 68] can be understood as a generalized concept of DSP - in other words, DSP is a special case of GSP where a line graph with n nodes is used and therefore, the graph Fourier transform (GFT) of the line graph is identical to the discrete Fourier transform.

In the definition of GFT, we assume that the graph shift operator (GSO) S is diagonalizable. Considering the eigendecomposition of the GSO S = V ⊺ Λ V with eigenvector V , we can write the graph filter output as follows:

<!-- formula-not-decoded -->

where x ∈ R n is a 1-dimensional graph signal, Λ is a diagonal matrix with eigenvalues, and w k ∈ [ -∞ , ∞ ] is a coefficient.

However, one can use the singular value decomposition, when the GSO is not diagonalizable but symmetrically normalized, instead of the eigendecomposition [49]. Both the singular value decomposition and the eigendecomposition project the original signal onto a set of basis, but they use different basis sets. In the singular value decomposition, we sort the set of basis in ascending order of their eigenvalues, and perform frequency domain-like analyses [106, 49].

Since the self-attention matrix's row-wise sum is always 1, the following is the case: ¯ A = D -1 A = 1 n A , where n is the number of tokens. Maskey et al. [49] define the following symmetrically normalized adjacency matrix (SNA): D -1 / 2 in AD -1 / 2 out . Since the degree of every node is n in the self-attention matrix, the following is the case: D -1 / 2 in AD -1 / 2 out = D -1 / 2 AD -1 / 2 = 1 √ n A 1 √ n =

1 n A = ¯ A . Therefore, the self-attention matrix is a special case of SNAs.

## F Matrix Polynomial vs. Graph Fourier Transform

There are two paradigms of implementing graph filters: i) matrix polynomial, which does not require diagonalizability, and ii) graph Fourier transform, which uses the eigendecomposition for diagonalizable adjacency matrices or uses the Jordan decomposition or the singular value decomposition for non-diagonalizable adjacency matrices.

Those two paradigms have their own weaknesses: i) the matrix polynomial approach requires explicit matrix multiplications, and ii) the graph Fourier transform approach requires expansive spectral decompositions. The matrix polynomial is preferred when there are not many matrix multiplications. Otherwise, the graph Fourier transform approach may be better since the matrix multiplication can be simplified after the decomposition.

Among those two, we use the first matrix polynomial approach with only three non-zero coefficients { w 0 , w 1 , w K } since it does not require the complicated spectral decomposition. Since we do not rely on any explicit spectral decomposition but on the matrix polynomial, any adjacency matrix can be used.

## G Proof of Theorem 3.1

Theorem 3.1 (Filter characteristics based on coefficient values) . Let ¯ A be a self-attention matrix interpreted as a graph with connected components. Consider the polynomial graph filter defined by ∑ K k =0 w k ¯ A k , where w 2 , w 3 , . . . , w K -1 = 0 and only w 0 , w 1 , and w K are non-zero. If the coefficients w k for k = 0 , 1 , K are positive and their sum is 1, then the polynomial filter acts as a low-pass filter, attenuating high-frequency components and promoting smoothness across the graph. Conversely, if w k = ( -α ) k for k = 0 , 1 , K and α ∈ (0 , 1) with sufficient large K , the polynomial filter exhibits high-pass filter behavior.

Note that without filtering, the singular value ratio is ∣ ∣ σ 0 i ∣ ∣ / ∣ ∣ σ 0 1 ∣ ∣ = 1 . In the case where | g ( σ i ) /g ( σ 1 ) | &lt; 1 ∀ i ≥ 2 , it implies that after applying the graph filter g , the lowest frequency component further dominates, indicating that the graph filter acts as a low-pass filter. Conversely, in

the case where | g ( σ i ) /g ( σ 1 ) | &gt; 1 ∀ i ≥ 2 , it implies that after applying the graph filter g , the lowest frequency component σ i no longer dominates, indicating that the graph filter acts as a high-pass filter.

Proof. We prove the low-pass filter result. For the case where w 0 , w 1 , and w K are positive and their sum is 1 , we show that

<!-- formula-not-decoded -->

Hence, proving Theorem G is equivalent to show | g ( σ i ) | &lt; 1 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the high-pass filter result, when w k = ( -α ) k / ( k +1) where k = 0 , 1 , K and α ∈ (0 , 1) , then we show that when w 0 = 1 , w 1 = -α/ 2 ,

<!-- formula-not-decoded -->

since

<!-- formula-not-decoded -->

It shows that the graph filter with w k = ( -α ) k / ( k +1) for k = 1 , 2 , K emphasizes high-frequency components and acts as a high-pass filter.

This proof supports that the behavior of the polynomial filter as either a low-pass or high-pass filter directly depends on the sign and values of the coefficients, as specified in Theorem 3.1.

## H Proof of Theorem 4.1

Proof. The Frobeinus norm of the self-attention is directly related to how far the softmax probabilities are from being uniform. For any matrix M ∈ R m × n , we have

<!-- formula-not-decoded -->

where S i is the i -th row of softmax ( M ) , U n is the uniform distribution over n elements, and d χ 2 ( p, q ) = ∑ i q i ( p i /q i -1) 2 is the χ 2 -divergence between p and q [14]. The Frobenius norm is maximized when the whole mass of the probabilities is on one element, which is a case for d χ 2 ( S i , U n ) = n -1 and ∥ softmax ( M ) ∥ F = √ m . Therefore, we can calculate the upper bound of Frobenius norm for ¯ A as follows:

<!-- formula-not-decoded -->

Note that ¯ A ∈ R n × n is a right-stochastic matrix normalized with row-wise softmax: i) all the elements of ¯ A lie within [0, 1], and ii) the row-sum in ¯ A is equal to 1. Since the self-attention matrix

<!-- formula-not-decoded -->

is a right-stochastic matrix, the power of the self-attention is also a right-stochastic matrix. Therefore, Eq. (21) is also hold for ¯ A K as follows:

<!-- formula-not-decoded -->

Now, considering the error term E K as given by Theorem 4.1, and applying the triangle inequality for matrix norms:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## I Implementation of GFSA

The pseudo code of our GFSA is shown in Algorithm 1. For implementation, w 0 and w 1 can be set as hyperparameters optionally.

## Algorithm 1 PyTorch-style pseudocode for GFSA

```
w_0 = torch.zeros(h) w_1 = torch.ones(h) w_K = torch.zeros(h) I = torch.eyes(n)[None, None, ...] def GFSA (att, K) att: original self-attention att_K: high order term att_K = att + (K-1) * (torch.mm(att,att)-att) gf_att: GFSA attention gf_att = w_0[None, :, None, None] * I + w_1[None, :, None, None] * att + w_K[None, :, None, None] * att_K return gf_att
```

## J Comparison with Actual and Approximated High-order Terms

To compare the impact of the actual ¯ A K and the approximated ¯ A K in terms of accuracy, we experimented with BERT on GLUE and the results are summarized in Table 9. BERTBASE+ ¯ A K denotes using the exactly calculated ¯ A K instead of the approximated ¯ A K .

Table 9: Comparison of performance using the exactly calculated ¯ A K vs. the approximated ¯ A K for GLUE tasks

| Datasets          | #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI RTE   |   #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI RTE |   #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI RTE |   #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI RTE |   #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI RTE |   #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI RTE | #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI RTE   | #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI RTE   |   Avg |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|-------|
| BERT BASE [16]          | 110M          |          56.79 |          93.81 |          88.7 |          88.32 |          88.16 | 84.96/84.15          | 91.63 66.06          | 82.51 |
| + GFSA (approximated ¯ A K ) | 110M          |          59.56 |          94.15 |          90.6 |          88.46 |          88.33 | 85.12/85.06          | 91.95 68.95          | 83.58 |
| + GFSA (actual ¯ A K )       | 110M          |          59.85 |          94.27 |          89.8 |          88.43 |          88.32 | 84.95/84.89          | 91.76 68.23          | 83.39 |

## K Natural Language Understanding

## K.1 Detailed Experimental Settings

We integrate GFSA into 3 pre-trained large language models: BERT, ALBERT, and RoBERTa. We evaluate them on the GLUE benchmark, which includes 3 categories of natural language understanding tasks: i) single-sentence tasks CoLA and SST-2; ii) similarity and paraphrasing tasks MRPC, QQP, and STS-B; iii) natural language inference tasks MNLI, QNLI, and RTE. For MNLI task, we experiment on both the matched (MNLI-m) and mismatched (MNLI-mm) versions. Following Devlin et al. [16], we report Matthews correlation for CoLA, F1 scores for QQP and MRPC, Spearman correlations for STS-B, and accuracy scores for the other tasks. For each task, we select the best hyperparameters for GFSA, and the other hyperparameters are fixed. We compare our GFSA with ContraNorm [28], one of the related methods that address oversmoothing. We finetune ContraNorm with the recommended hyperparameters in Guo et al. [28]. We initialize with a pre-trained language model and finetune with added GFSA for 5 epochs.

Dataset. The benchmark datasets we used are listed below.

- CoLA. The Corpus of Linguistic Acceptability [85] consists of English acceptability judgments drawn from books and journal articles. The target task is a binary classification task, and each sentence is determined to be grammatically acceptable or not.
- SST-2. The Stanford Sentiment Treebank [73] is a dataset in which each sentence is sourced from movie reviews and accompanied by human annotations of their sentiment. The target task is to classify binary sentiments for a single sentence.
- MRPC. The Microsoft Research Paraphrase Corpus [17] is a corpus of sentence pairs, which are automatically extracted from online news sources and annotated by humans. The target is to determine whether the sentences in the pair are semantically equivalent.
- QQP. The Quora Question Pairs [11] dataset is a collection of question pairs from the community question-answering website Quora. The target is to determine whether the questions in the pair are semantically equivalent.
- STS-B. The Semantic Textual Similarity Benchmark [8] is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data with human annotation. The target is a regression task to predict a similarity score from 0 to 5.
- MNLI. The Multi-Genre Natural Language Inference Corpus [87] is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The standard test set consists of private labels from the authors and evaluates both the matched (in-domain) and mismatched (cross-domain) sections.
- QNLI. The Stanford Question Answering [82] dataset is a question-answering dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph contains the answer to the corresponding question written by an annotator. The task is to determine whether the context sentence contains the answer to the question.
- RTE. The Recognizing Textual Entailment [5] dataset comes from a series of annual textual entailment challenges. The target task is a binary entailment classification task.

BERT. BERT [16] consists with 12 layers, 12 heads, 768 hidden size, 512 maximum sequence length, and MLP dimension of 3072.

ALBERT. ALBERT [40] consists of 12 layers, 12 heads, 768 hidden dimensions, 512 maximum sequence length, 128 embedding dimensions, and MLP dimension of 3072.

RoBERTa. RoBERTa [44] consists of 12 layers, 12 heads, 768 hidden size, 514 maximum sequence length, and MLP dimension of 3072.

Training. For implementation, we adopt HuggingFace framework. We trained all models with 5 epochs with 32 batch size. The linear learning rate decay is used and initial learning rate is set to 2 × 10 -5 . We use AdamW [46] optimizer, and weight decay is set to 0. All models are trained on 1 GPU and of NVIDIA RTX A5000 24GB.

## K.2 Sensitivity to K

In this section, we explore the influence of the polynomial order, denoted as K , in our GFSA, conducting experiments on BERTBASE finetuned with GLUE tasks. We search for values of K from 2 to 10, and the results are presented in Table 10. For each dataset, there is an optimal K and the performance of models using GFSA is generally robust to changes in K .

Table 10: Sensitivity results on various K with BERTBASE finetuned on GLUE tasks

|   K |   CoLA |   SST2 |   MRPC |   QQP |   STSB |   MNLI-m |   MNLI-mm |   QNLI |   RTE |
|-----|--------|--------|--------|-------|--------|----------|----------|--------|-------|
|   2 |  57.83 |  94.15 |  90.6  | 88.41 |  88.27 |    84.96 |     84.9  |  91.74 | 68.23 |
|   3 |  58.56 |  93.46 |  89.77 | 88.41 |  88.33 |    85.08 |     84.75 |  91.78 | 68.59 |
|   4 |  59.56 |  93.46 |  89.77 | 88.45 |  88.29 |    85.06 |     85.06 |  91.76 | 68.59 |
|   5 |  58.1  |  93.58 |  90.07 | 88.39 |  88.29 |    84.87 |     84.99 |  91.85 | 68.95 |
|   6 |  59.4  |  93.58 |  90.4  | 88.29 |  88.27 |    84.93 |     84.97 |  91.69 | 68.23 |
|   7 |  59.12 |  94.04 |  90.48 | 88.43 |  88.26 |    85.12 |     84.94 |  91.82 | 68.94 |
|   8 |  58.58 |  93.69 |  90.12 | 88.46 |  88.24 |    84.92 |     84.81 |  91.95 | 68.95 |
|   9 |  58.88 |  93.46 |  89.54 | 88.41 |  88.26 |    85.06 |     91.67 |  67.87 | 85.04 |
|  10 |  59.31 |  93.35 |  89.98 | 88.41 |  88.3  |    84.84 |     91.73 |  68.59 | 84.93 |

## L Causal Language Modeling

## L.1 Detailed Experimental Settings

Dataset. The benchmark datasets we used are listed below.

- PTB. Penn Treebank [47] dataset is a collection of text documents that have been extensively annotated with linguistic information, primarily syntactic and grammatical structures.
- WikiText. WikiText [50] dataset is a collection of over 100 million tokens extracted from the set of verified good and featured articles on Wikipedia. Compared to the preprocessed version of Penn Treebank (PTB), WikiText-2 is over 2 times larger and WikiText-103 is over 110 times larger.

GPT2. GPT2 [61] is a Transformer pretrained on a very large corpus of English data in a selfsupervised fashion without any human labelling on dataset. It automatically generate inputs and labels from those texts, and trained to guess the next word in sentences. For implementation, we adopt HuggingFace Framework 4 . For all experiments, GPT2 has 12 layers with 12 attention heads, 768 hidden size and 1024 maximum sequence length, resulting in a total of 117 million parameters.

Training. We finetune GPT2 with 4 batch size, 5 × 10 -5 learning rate and linear learning weight decay using adamW [46] optimizer. We also apply dropout with probability 0.1. Following [93], we train models for 15 epochs with PTB, 4 epochs with WikiText-103 and 10 epochs with WikiText-2. We use sensitivity metric, i.e., perplexity, which is a commonly used metric to evaluate the performance of language models, particularly in language modeling and text generation tasks. perplexity measures how well a language modeling can predict a sequence of words in a given text or a test dataset. All the experiments are conducted on 1 GPU and of NVIDIA RTX 3090 24GB.

## L.2 Sensitivity to K

We conducted a sensitivity study on K of GPT-2 across all datasets, and the results are presented in Table 11. For PTB and WikiText-2, GFSA exhibits the best performance when K is high, typically

4 https://github.com/huggingface/transformers

around 8 or 9. However, for WikiText-103, GFSA achieves the best perplexity when K is small, specifically when K is 3 or 4.

Table 11: Results comparison on GPT-2 finetuned with GFSA

| Method          | #Params   |    PTB |   WikiText-2 |   WikiText-103 |
|----------|----------|--------|----------|----------|
| GPT2 [61]          | 117M      | 19.513 |       20.966 |         15.939 |
| GPT2 + GFSA( K = 2 ) | 117M      | 19.459 |       20.929 |         15.92  |
| GPT2 + GFSA( K = 3 ) | 117M      | 19.456 |       20.927 |         15.919 |
| GPT2 + GFSA( K = 4 ) | 117M      | 19.453 |       20.927 |         15.919 |
| GPT2 + GFSA( K = 5 ) | 117M      | 19.452 |       20.925 |         15.92  |
| GPT2 + GFSA( K = 6 ) | 117M      | 19.451 |       20.925 |         15.92  |
| GPT2 + GFSA( K = 7 ) | 117M      | 19.45  |       20.925 |         15.921 |
| GPT2 + GFSA( K = 8 ) | 117M      | 19.45  |       20.924 |         15.921 |
| GPT2 + GFSA( K = 9 ) | 117M      | 19.45  |       20.923 |         15.921 |

## M Image Classification

## M.1 Detailed Experimental Settings

Our code is implemented based on the timm library [86]. In the case of our training recipe, it is the same as experimental setting of Wang et al. [80] that follows the training recipes of Touvron et al. [74] and Touvron et al. [75]. To apply our GFSA to existing base models such as DeiT, Cait, and Swin, we consider a range of K between 2 and 5. For 12-layer DeiT, we follow the same hyperparameters from Wang et al. [80]. We set the dropout rate to 0 and 0.2 for 12-layer and 24-layer DeiT, respectively. For CaiT, we apply our GFSA on only to the patch embedding layer. All other hyper-parameters are kept consistent with the original papers of DeiT [74], CaiT [75] and, Swin [45]. All models are trained on NVIDIA RTX 3090 24GB.

## M.2 FLOPs &amp; Throughput

In Table 12, we report the number of FLOPs and throughput. With GFSA plugged in, the FLOP count is either the same or no different. For DeiT-S with 24 layers, which shows a slight FLOP increase with GFSA plugged in. However, for the rest of the settings, the models have the same number of Flops. For throughput, it tends to decrease because calculating the high-order term is an additional cost.

Table 12: Experimental evalutation of GFSA plugged into DeiT-S, CaiT-S, and Swin-S

| Backbone   | Method        |   Input Size |   #Layers | #Params   | #FLOPs   | #Throughput   | Top-1 Acc          |
|----------|----------|----------|----------|----------|----------|----------|----------|
| DeiT       | DeiT-S        |          224 |        12 | 22.0M     | 4.57G    | 856.07 614.54 | 79.8 81.1 ( ↑ 1.3) |
| DeiT       | DeiT-S + GFSA |          224 |        12 | 22.0M     | 4.57G    |          |          |
| DeiT       | DeiT-S        |          224 |        24 | 43.3M     | 9.09G    | 423.68        | 80.5          |
| DeiT       | DeiT-S + GFSA |          224 |        24 | 43.3M     | 9.10G    | 314.75        | 81.5 ( ↑ 1.0)      |
| CaiT       | CaiT-S        |          224 |        24 | 46.9M     | 9.34G    | 574.66        | 82.6          |
| CaiT       | CaiT-S + GFSA |          224 |        24 | 47.0M     | 9.34G    | 406.96        | 82.8 ( ↑ 0.2)      |
| Swin       | Swin-S        |          224 |        24 | 49.6M     | 8.75G    | 912.38        | 82.9          |
| Swin       | Swin-S + GFSA |          224 |        24 | 49.6M     | 8.75G    | 714.60        | 83.0 ( ↑ 0.1)      |

## M.3 Sensitivity to K

We also perform the sensitivity analysis for K . Tables 13 and 14 show the results of sensitivity analysis for DeiT-S and CaiT-S with GFSA plugged in. For 12-layer DeiT-S, GFSA performance of 81.12 is highest when K = 3 . When the GFSA has a K of 2, the performance is worse than the

original DeiT-S, but when the K is 3 or higher, the performance is better than the original DeiT-S, and most surprisingly, the performance is better than the 24-layer DeiT-S.

CaiT-S shows the highest performance of 82.84 when K = 4 . For CaiT-S, the accuracy is slightly lower than that of the original CaiT-S when K = 2 , but it starts to exceed the accuracy of CaiT-S when K is 3 or higher.

Table 13: Sensitivity to K for 12-layer DeiT-S + GFSA Table 14: Sensitivity to K for 24-layer CaiT-S + GFSA

| K 2 3 4 5          | K 2 3 4          |
|----------|----------|
| Top-1 Acc (%) 79.27 81.12 80.86 81.07 | Top-1 Acc (%) 82.54 82.65 82.84 |

## M.4 Full Experimental Results

In Table 15, we consider all three classes, CNN only, CNN + Transformer, and pure Transformer, to compare more different models than in Table 3. In particular, in the Transformer category, we only test with lightweight models with similar number of parameters, such as ViT-S and DeiTS. Compared to existing techniques, the improvements by GFSA already surpasses LayerScale (0.7%) [75], LateInsertion (0.6%) [75], and HAT [4] (1.38%).

Table 15: Compared with state-of-the-art models on ImageNet-1k dataset. The number in ( ↑ ) indicates the performance improvement over the base model.

| Category    | Method          |   Input Size |   #Layers | #Params   | Top-1 Acc     |
|----------|----------|----------|----------|----------|----------|
| CNN         | ResNet-152 [29]          |          224 |       152 | 230M      | 78.1          |
|          | DenseNet-201 [34]          |          224 |       201 | 77M       | 77.6          |
| CNN +       | CVT-21 [88]          |          224 |        21 | 32M       | 82.5          |
| Transformer | Refiner [102]          |          224 |        16 | 86M       | 81.2          |
|          | ViT-S/16 [19]          |          224 |        12 | 49M       | 78.1          |
|          | ViT-B/16 [19]          |          224 |        12 | 86M       | 79.8          |
|          | DeiT-S [74]          |          224 |        12 | 22M       | 79.8          |
|          | DeiT-S + LayerScale [75]     |          224 |        12 | 22M       | 80.5          |
|          | DeiT-S + LateInsertion [75]  |          224 |        12 | 22M       | 80.5          |
|          | DeiT-S + ClassAttention [75] |          224 |        12 | 22M       | 80.6          |
|          | DeiT-S + AttnScale [80]      |          224 |        12 | 22M       | 80.7          |
|          | DeiT-S + FeatScale [80]      |          224 |        12 | 22M       | 80.9          |
|          | DeiT-S + HAT [4]          |          224 |        12 | 22M       | 80.9          |
| Transformer | DeiT-S + Diverse [10]        |          224 |        12 | 22M       | 80.6          |
|          | DeiT-S + ContraNorm [28]     |          224 |        12 | 22M       | 80.4          |
|          | Swin-S [45]          |          224 |        12 | 50M       | 82.9          |
|          | T2T-ViT-24 [96]          |          224 |        24 | 64M       | 82.3          |
|          | DeepViT-24B [101]          |          224 |        24 | 36M       | 80.1          |
|          | DeiT-S [74]          |          224 |        24 | 43M       | 80.5          |
|          | CaiT-S [75]          |          224 |        24 | 47M       | 82.6          |
|          | DeiT-S + DiversePatch [25]   |          224 |        24 | 44M       | 82.2          |
|          | DeiT-S + LayerScale [75]     |          224 |        24 | 44M       | 82.4          |
|          | DeiT-S + AttnScale [80]      |          224 |        24 | 44M       | 81.1          |
|          | DeiT-S + FeatScale [80]      |          224 |        24 | 44M       | 81.3          |
|          | DeiT-S + ContraNorm [28]     |          224 |        24 | 43M       | 80.7          |
| GFSA        | DeiT-S + GFSA          |          224 |        12 | 22M       | 81.1 ( ↑ 1.3) |
|          | DeiT-S + GFSA          |          224 |        24 | 43M       | 81.5 ( ↑ 1.0) |
|          | CaiT-S + GFSA          |          224 |        24 | 47M       | 82.8 ( ↑ 0.2) |
|          | Swin-S + GFSA          |          224 |        12 | 50M       | 83.0 ( ↑ 0.1) |

## M.5 Additional Comparison with SOTA Models

Our main experiment aims to determine whether introducing the GFSA layer would help improve performance in a base model, such as DeiT. We also compare our method with the recent models: SpectFormer [58], SVT [57], NeuTRENO [52], FNet [42], and GFNet [64]. We use a 12-layer setup to ensure a fair comparison in Table 16.

GFNet [64] can reduce the number of parameters, but there is a performance penalty. However, the performance improvement of DeiT-S+GFSA is relatively greater than DeiT-S compared to other models. SpectFormer [58] and SVT [57] have advantages in calculation amount and model complexity, and performance is improved over DeiT-S, but Top-1 and Top-5 accuracies are lower than those using GFSA. Additionally, NeuTRENO [52] also improves as much as GFSA compared to DeiT-S, but GFSA still has higher Top-1 accuracy.

Table 16: Compared with state-of-the-art models on ImageNet-1k

| Method          |   Input Size |   #Layers | #Params   |   Top-1 Acc | Top-5 Acc   |
|----------|----------|----------|----------|----------|----------|
| DeiT-S          |          224 |        12 | 22M       |        79.8 | 95.0        |
| Fnet-XS [42]          |          224 |        12 | 20M       |        71.2 | -          |
| GFNet-XS [64]          |          224 |        12 | 16M       |        78.6 | 94.2        |
| SpectFormer-XS [58]    |          224 |        12 | 20M       |        80.2 | 94.7        |
| SVT-XS [57]          |          224 |        12 | 20M       |        79.9 | 94.5        |
| DeiT-S + NeuTRENO [52] |          224 |        12 | 20M       |        80.7 | 95.4        |
| DeiT-S + GFSA          |          224 |        12 | 22M       |        81.1 | 95.4        |

## M.6 Additional Experiments with Guo et al. [28]'s setting

To make a fair comparison with ContraNorm [28], one of the related studies that mitigates oversmoothing, we run additional experiments to match their experimental setup.

Setting. We follow the training recipe used by Guo et al. [28], which is a slightly modified version of Touvron et al. [74]'s recipe. Guo et al. [28] use AdamW optimizer with cosine learning rate decay. We select the DeiT-T and DeiT-S for ImageNet-1k. 'T' and 'S' denote tiny and small model sizes, respectively. For all experiments, the image size is set to be 224x224. We train each model for 300 epochs and the batch size is set to 1024. For ContraNorm, we train with their recommended hyperparameters. All models are trained on 4 GPUs and of NVIDIA RTX A6000 48GB.

Results. In Table 17, DeiT-T and DeiT-S with GFSA outperform vanilla DeiT-T and DeiT-S in all layer settings. GFSA improves the performance of DeiT-T with 12 layers by 1.52%. The largest gain is a 4.88% improvement on 16-layer DeiT-T. This shows that the effect of GFSA is larger than the effect of ContraNorm. For DeiT-S with 16 layers, surprisingly, GFSA is able to increase the performance by 80.83%, meaning that GFSA brings benefits with a 3.23% improvement.

Table 17: Experiment results on ImageNet-1k

| Method          |   #Layers=12 |   #Layers=16 |   #Layers=24 |
|----------|----------|----------|----------|
| DeiT-T          |        76.52 |        75.34 |        76.76 |
| DeiT-T + ContraNorm |        77.03 |        78.72 |        78.12 |
| DeiT-T + GFSA       |        77.68 |        79.02 |        78.64 |
| DeiT-S          |        77.32 |        78.25 |        77.69 |
| DeiT-S + ContraNorm |        77.8  |        79.04 |        78.67 |
| DeiT-S + GFSA       |        79.86 |        80.83 |        79.15 |

Sensitivity to K . In Table 18, we experiment with a sensitivity analysis for K . For DeiT-T, the performance of GFSA generally improves when K is 4 or 5. On the other hand, GFSA performs better at lower K for settings that are layers 16 and 24 for DeiT-S.

Table 18: Varying K for DeiT-T and DeiT-S

| Method        |   K |   #Layers=12 |   #Layers=16 |   #Layers=24 |
|----------|-----|----------|----------|----------|
| DeiT-T + GFSA |   2 |        76.92 |        78.14 |        78.4  |
| DeiT-T + GFSA |   3 |        77.41 |        77.76 |        78.09 |
| DeiT-T + GFSA |   4 |        77.01 |        79.02 |        78.64 |
| DeiT-T + GFSA |   5 |        77.68 |        78.14 |        78.64 |
| DeiT-S + GFSA |   2 |        79.84 |        80.83 |        79.15 |
| DeiT-S + GFSA |   3 |        79.85 |        79.39 |        79.07 |
| DeiT-S + GFSA |   4 |        79.86 |        79.44 |        79.1  |

## N Automatic Speech Recognition

## N.1 Detailed Experimental Settings

Dataset. We conduct experiments on the LibriSpeech 5 dataset [55], which consists of audio recordings paired with their transcriptions. The LibriSpeech dataset has approximately 1,000 hours of read English speech with a sampling rate of 16 kHz. We keep the original 16,000Hz sampling rate and compute 80-dim log-Mel filterbanks for a 25ms sliding window, strided by 10ms. The filterbank features are then normalized to zero mean and unit variance per input sequence. For implementation, we follow the recipes of SpeechBrain [65].

Evaluation Metric. Word error rate (WER (%)) is derived from the Levenshtein distance and compares a reference to a hypothesized word-level transcription. It is calculated by summing the number of word insertions, deletions, substitutions and dividing it by the total number of words in the reference transcription.

Vanilla Transformer. We use a vanilla Transformer to apply our GFSA. For implementation, we use a SpeechBrain [65] framework. The vanilla Transformer consists of i) 1D convolution to perform striding, ii) Transformer encoder with 12 layers, 4 heads, embedding dimension of 512, MLP dimension of 2048, and post-LayerNorm iii) decoder with 6 layers, 4 heads, embedding dimension of 512, MLP dimension of 2048, joint beamsearch, and iv) external Transformer language model with 12 layers, 12 heads, embedding dimension of 768, and MLP dimension of 3072.

Branchformer. We use one of the SOTA models, Branchformer [59] to plug-in our GFSA. Branchformer has two parallel branches, one for capturing global interactions using attention and the other for more localized context using convolutional gating MLP. The Branchformer architecture for speech recognition consists of i) 1D convolution to perform striding, ii) Branchformer encoder with 18 layers, 8 heads, embedding dimension of 512, and MLP dimension of 3072, iii) decoder with 6 layers, 8 heads, embedding dimension of 512, a convolutional spatial gating unit (CSGU) dimension of 3072, joint beamsearch, and iv) external Transformer language model with 12 layers, 12 heads, embedding dimension of 768, and MLP dimension of 3072.

Training. We follow a training recipe from SpeechBrain [65]. The standard LibriSpeech validation sets (dev-clean and dev-other) are used to tune all parameters and select the best models. Test sets (test-clean and test-other) are used only to report final WER performance. We train the pure Transformer for 100 epochs and the Branchformer for 120 epochs with a batch size of 16. We use a data augmentation method on all models using SpecAugment [56]. SpecAugment applies time and frequency masking as well as time warping to the input spectrum. For Branchformer, we use AdamW [46] optimizer with 0.9 and 0.98 coefficients for computing running averages of gradient and its square. The learning rate and weight decay in all models are 0.0008 and 0.01, respectively. We use a connectionist temporal classification (CTC) loss [26, 100]. We also apply dropout with probability 0.1 and label smoothing with weight 0.1 to mitigate overfitting. We fix the random seed as 74443 on all experiments. All models are trained on 1 GPU and of NVIDIA RTX A6000 48GB.

5 http://www.openslr.org/12

Hyperparameters. In Table 19, we describe main hyperparameters used in the automatic speech recognition task. For Transformer+GFSA and Branchformer+GFSA, we also report the best K hyperparameter.

Table 19: Main hyperparameters used in ASR

| Model          | Experimental Setting          |
|----------|----------|
| Transformer       | Encoder: Transformer (12 layers) Decoder: Transformer (6 layers) + (CTC/ATT joint) beamsearch + TransformerLM Augmentation: SpecAugment Features: 40 fbanks Pretraining: no Dropout: 0.1 Batchnorm: yes Number of epochs: 100 Batch size: 32 Learning rate: 0.0008 LR scheduler: Noam Optimizer: Adam Loss: CTC + KLdiv (Label Smoothing loss)          |
| Transformer+GFSA  | Encoder: Transformer (12 layers) Decoder: Transformer (6 layers) + (CTC/ATT joint) beamsearch + TransformerLM Augmentation: SpecAugment Features: 40 fbanks Pretraining: no Dropout: 0.1 Batchnorm: yes Number of epochs: 100 Batch size: 32 Learning rate: 0.0008 LR scheduler: Noam Optimizer: Adam Loss: CTC + KLdiv (Label Smoothing loss) CTC weight: 0.3 K : 2          |
| Branchformer      | Encoder: Branchformer Decoder: Transformer (6 layers) + (CTC/ATT joint) beamsearch + TransformerLM Augmentation: SpecAugment Features: 40 fbanks Pretraining: no Dropout: 0.1 Batchnorm: yes Number of epochs: 120 Batch size: 16 Learning rate: 0.0008 LR scheduler: Noam Optimizer: AdamW with coefficients 0.9 and 0.98 Loss: CTC + KLdiv (Label Smoothing loss) CTC weight: 0.3       |
| Branchformer+GFSA | Encoder: Branchformer Decoder: Transformer (6 layers) + (CTC/ATT joint) beamsearch + TransformerLM Augmentation: SpecAugment Features: 40 fbanks Pretraining: no Dropout: 0.1 Batchnorm: yes Number of epochs: 120 Batch size: 16 Learning rate: 0.0008 LR scheduler: Noam Optimizer: AdamW with coefficients 0.9 and 0.98 Loss: CTC + KLdiv (Label Smoothing loss) CTC weight: 0.3 K : 3 |

## N.2 Training Curve

We compare the training and validation curves for LibriSpeech 100h in Fig. 8. The training loss curve of GFSA is lower than the pure Transformer. GFSA stabilizes the loss curve of pure Transformer slightly earlier.

Figure 8: Training curve on LibriSpeech 100h

<!-- image -->

## O Graph-level Tasks

## O.1 Detailed Experimental Settings

Experimental settings for Graphormer. We describe benchmark datasets and Graphormer as the backbone model we used. ZINC [35] is the most popular real-world molecular dataset to predict graph property regression for constrained solubility, an important chemical property for designing generative GNNs for molecules. Uniform sampling is adopted for data splitting. We use a ZINCsubset of small-scale dataset. PCQM4M-LSC [33] is 2D molecular graphs, which is one of the most practically relevant quantum chemical properties of molecule science. The task is to predict density functional theory (DFT)-calculated HOMO-LUMO energy gap of molecules given their graphs. PCQM4M-LSC is unprecedentedly large in scale comparing to other labeled graph-level prediction datasets, which contain more than 3.8M graphs. We use PCQM4M and PCQM4Mv2 large-scale datasets.

Following Graphormer [94], we use Graphormer for PCQM4M and GraphormerSLIM for ZINC. Graphormer consists of 12 encoder layers, 80 encoder embedding dimension, and 768 MLP dimension. It employs 32 encoder heads and 24 hidden dimension for each head. GraphormerSLIM consists of 12 encoder layers, 80 encoder embedding dimension, and 80 MLP dimension. It employs 8 encoder heads and 10 hidden dimension for each head. We use adamW [46] optimizer with 0.9 and 0.999 coefficients for running averages of gradient and its square, and use Mean Absolute Error (MAE) as loss function. We use polynomial learning rate decay, with initial learning rate set to 2 × 10 -4 and end learning rate set to 1 × 10 -9 . For ZINC, we set batch size as 256, max epochs as 10k, and warm-up stage step as 40k. For PCQM4M and PCQM4Mv2, we set batch size as 1024, max epochs as 300, and warm-up stage step as 60k. All models are trained on 1 GPU and of NVIDIA RTX 3090 24GB. We conduct experiments with 4 different seeds.

Experimental settings for GPS and Graph-ViT. Weuse various benchmark datasets to experiment with our GFSA on GPS [63] and Graph-ViT [30]: Peptide-func and Peptide-struct from Long-Range Graph Benchmark (LRGB) [21], MNIST, CIFAR10, and ZINC from Benchmarking GNNs [22], and Moltox21 and Molhiv from OGB [32]. We fix all hyperparameters of GPS and Graph-ViT as recommended in their paper to ensure a fair comparison. To plug GFSA into GPS, we replace the self-attention module of GPS with GFSA. For Graph-ViT, we apply GFSA instead of the Hadamard self-attention mechanism and compare it with this self-attention method. We conduct experiments on GPS and Graph-ViT in their open code frameworks for a fair comparison:

- GPS: https://github.com/rampasek/GraphGPS
- Graph-ViT: https://github.com/XiaoxinHe/Graph-ViT-MLPMixer

## O.2 Experimental Results with Standard Deviation

We conduct experiments following the experimental environments of Graphormer [94] using 4 different seeds. Due to space constraints, only the mean values are reported in Tables 4 and 5. In Tables 20 and 21, we report the results with mean and standard deviations.

Table 20: Experimental results and number of parameters on ZINC

| Method          | #Params   | MAE          |
|----------|----------|----------|
| Graphormer        | 500K      | 0.1240 ± 0.006 |
| Graphormer + GFSA | 500K      | 0.1189 ± 0.002 |

Table 21: Experimental results and number of parameters on PCQM4M and PCQM4Mv2

| Method          | #Params   | PCQM4M         | PCQM4M         | PCQM4Mv2       | PCQM4Mv2       |
|----------|----------|----------|----------|----------|----------|
|          |          | Train          | Validate       | Train          | Validate       |
| Graphormer        | 48.3M     | 0.0535 ± 0.038 | 0.1286 ± 0.016 | 0.0250 ± 0.000 | 0.0862 ± 0.000 |
| Graphormer + GFSA | 48.3M     | 0.0312 ± 0.001 | 0.1193 ± 0.000 | 0.0249 ± 0.000 | 0.0860 ± 0.000 |

## P Code Defect Detection

## P.1 Detailed Experimental Settings

Dataset. We use Devign dataset provided by [105], which is a binary classification task to evaluate whether a C language code is vulnerable to software systems or not.

Implementation. We build our experiments on top of the open-sourced code 6 and recipes provided by Wang et al. [84].

RoBERTa. RoBERTa [44] is an encoder-only model trained with masked language modeling on code. All hyperparameters are consistent with the training method in the source code of Wang et al. [84].

PLBART. PLBART [2] is an encoder-decoder model based on BART [43] architecture. PLBART can support understanding and generation tasks. All hyperparameters are consistent with the training method in the source code of Wang et al. [84].

CodeBert. CodeBERT [23] is a model trained on masked language modeling and replaced token detection. CodeBERT is a bimodal pretrained model based on Transformer with 12 layers for programming language and natural language. All hyperparameters are consistent with the training method in the source code of Wang et al. [84].

CodeT5. CodeT5 is an encoder-decoder framework with the same architecture as T5 [62]. It aims to derive generic representations for programming language and natural language via pre-training on unlabeled source code. CodeT5-small has 6 encoder layers, 6 decoder layers, 8 attention heads, 512 dimensional hidden states, and 60M parameters. The other models have 12 encoder layers, 12 decoder layers, 12 attention heads, 768 dimensional hidden states, and 220M parameters. All hyperparameters are consistent with the training method in the source code of Wang et al. [84].

Training. The pre-trained models mentioned above are applied to this downstream task. We add GFSA directly on top of self-attention. We finetune baselines and GFSA models for 10 epochs with a batch size of 16. We use early stopping strategy with a patience of 2. Models generate binary labels from unigram sequences at the decoder for defect detection task. We employ accuracy for evaluating the code defect detection task. All models are trained on 1 GPU and of NVIDIA RTX A6000 48GB.

6 https://github.com/salesforce/CodeT5

## P.2 Case Study

In Listing 1, we show one of case for code snippets of defects in QEMU 7 that CodeT5-base does not predict correctly, but that only CodeT5-base+GFSA predicts. The commit message 8 for this case is as follow:

Needed for changing cpu\_has\_work() argument type to CPUState, used in h\_cede() .

h\_cede() is the hypercall that asks the hypervisor to shut down the CPU. Previously, this hypercall simply passed the CPUID, so the hypervisor did not know what state the CPU was in. This change allows the hypervisor to know whether the CPU is actually performing work. If the CPU is performing a task, the hypervisor waits for the CPU to complete the task.

In this context, accurately predicting defects like the one above is very important, and applying GFSA to CodeT5-base helps in terms of performance improvement.

```
1 @@ -204,7 +204,7 @@ static target_ulong put_tce_emu(sPAPRTCETable * tcet , target_ulong ioba , 2 -static target_ulong h_put_tce(CPUPPCState *env, sPAPREnvironment * spapr 3 + static target_ulong h_put_tce(PowerPCCPU *cpu, sPAPREnvironment * spapr 4 , target_ulong opcode , target_ulong * args) 5 { 6 target_ulong liobn = args[0]; 7 target_ulong ioba = args[1]; 8 target_ulong tce = args[2]; 9 VIOsPAPRDevice *dev = spapr_vio_find_by_reg(spapr->vio_bus , liobn); 10 VIOsPAPR_RTCE *rtce; 11 if (!dev) { 12 hcall_dprintf("LIOBN 0x" TARGET_FMT_lx " does not exist\n", liobn) ; 13 return H_PARAMETER; 14 } 15 ioba &= ~(SPAPR_VIO_TCE_PAGE_SIZE -1); 16 #ifdef DEBUG_TCE 17 fprintf(stderr , "spapr_vio_put_tce on %s ioba 0x" TARGET_FMT_lx " TCE 0x" TARGET_FMT_lx "\n", dev->qdev.id, ioba , tce); 18 #endif 19 if (ioba >= dev->rtce_window_size) { 20 hcall_dprintf("Out-of-bounds IOBA 0x" TARGET_FMT_lx "\n", ioba); 21 return H_PARAMETER; 22 } 23 rtce = dev->rtce_table + (ioba >> SPAPR_VIO_TCE_PAGE_SHIFT); 24 rtce ->tce = tce; 25 return H_SUCCESS; 26 }
```

Listing 1: An example commit history for defects in Devign dataset

## Q Code Clone Detection

## Q.1 Detailed Experimental Settings

Dataset. Code clone detection aims to measure the similarity between two code snippets and predict whether they have the same functionality. We experiment with the Java data provided by Wang et al. [83].

```
7 https://www.qemu.org 8
```

https://github.com/qemu/qemu/commit/b13ce26d3e8c6682044ae84920f2417b30ce356b

Implementation. We build our experiments on top of the open-sourced code 9 and recipes provided by Wang et al. [84].

Training. We finetune both CodeT5 and CodeT5+GFSA for one epoch with a batch size of 16. We also use early stopping with patience of 2. CodeT5 and CodeT5+GFSA encode source code and take the representation to calculate similarity of two code snippets. We employ F1 score for evaluating this task. All models are trained on 1 GPU and of NVIDIA RTX A6000 48GB.

## Q.2 Experiment Result

Table 22 shows results comparing CodeT5 and CodeT5 with GFSA. The result shows that by using our GFSA, CodeT5 models improve their performance. CodeT5-small+GFSA provides a 0.61% improvment over Code5T-small.

Table 22: Results on the code clone detection task

| Method          | Clone F1         |
|----------|----------|
| CodeT5-small [84]   | 94.36          |
| CodeT5-small + GFSA | 94.94 ( ↑ 0.61%) |
| CodeT5-base [84]    | 94.31          |
| CodeT5-base + GFSA  | 94.92 ( ↑ 0.64%) |

## R Time Complexity and Empirical Runtime Analysis

Time Complexity. The time complexity of original self-attention is O ( n 2 d ) . But our GFSA has a high order term. Therefore, the time complexity of GFSA has O ( n 2 d + n 3 ) . If n is smaller than d , the time complexity approaches O ( n 2 d ) , which is the complexity of original self-attention.

Empirical Runtime Analysis. We report the training time of various methods with GFSA in Tables 23 to 28. In general, the training time of methods with GFSA is slightly longer than that of existing methods. For example, the Transformer for the automatic speech recognition task increases from 190.5 seconds to 191.6 seconds on Librispeech 100h dataset, as increases of only 1 second. Instead of computing higher-order polynomial terms, our GFSA approximates them, with only a small increase in runtime, which is not very significant.

Table 23: Training time (seconds per epoch) on GLUE tasks. s denotes the abbreviation for second. Avg denotes the average training time across all tasks.

| Datasets          | #Params CoLA   | #Params CoLA   | SST-2   | MRPC   | QQP    | STS-B   | MNLI-m/mm   | QNLI   | RTE   | Avg   |
|----------|----------|----------|---------|--------|--------|---------|----------|--------|-------|-------|
| BERT BASE [16]      | 110M          | 17 s          | 182 s   | 17 s   | 1483 s | 24 s    | 2004 s      | 580 s  | 18 s  | 541 s |
| BERT BASE + GFSA    | 110M          | 19 s          | 192 s   | 19 s   | 1571 s | 25 s    | 2147 s      | 621 s  | 20 s  | 577 s |
| ALBERT BASE [40]    | 11M          | 15 s          | 188 s   | 20 s   | 1506 s | 25 s    | 2072 s      | 612 s  | 19 s  | 557 s |
| ALBERT BASE + GFSA  | 11M          | 16 s          | 197 s   | 21 s   | 1604 s | 26 s    | 2219 s      | 659 s  | 21 s  | 595 s |
| RoBERTa BASE [44]   | 125M          | 17 s          | 190 s   | 18 s   | 1492 s | 25 s    | 2012 s      | 593 s  | 18 s  | 546 s |
| RoBERTa BASE + GFSA | 125M          | 19 s          | 200 s   | 19 s   | 1580 s | 26 s    | 2151 s      | 634 s  | 20 s  | 581 s |

Table 24: Training time (seconds per epoch) on causal language modeling tasks.

| Method      | #Params   | PTB     | WikiText-2   | WikiText-103   | Avg      |
|----------|----------|---------|----------|----------|----------|
| GPT2 [61]   | 117M      | 89.1 s  | 195.7 s      | 9638.4 s       | 3307.8 s |
| GPT2 + GFSA | 117M      | 160.3 s | 354.2 s      | 17424.6 s      | 5979.7 s |

Table 25: Training time (seconds per epoch) on ImageNet-1k

| Backbone   | Method        | #Layers   | #Params     | #FLOPs          | #Throughput          | Runtime          |
|----------|----------|----------|----------|----------|----------|----------|
| DeiT       | DeiT-S        | 12 12 24  | 22.0M 22.0M | 4.57G 4.57G 9.09G | 856.07 614.54 423.68 | 551 s 814 s 1508 |
| DeiT       | DeiT-S + GFSA |          |          |          |          |          |
| DeiT       | DeiT-S        |          | 43.3M       |          |          | s          |
| DeiT       | DeiT-S + GFSA | 24        | 43.3M       | 9.10G          | 314.75          | 1798 s          |
| CaiT       | CaiT-S        | 24        | 46.9M       | 9.34G          | 574.66          | 1530 s          |
| CaiT       | CaiT-S + GFSA | 24        | 47.0M       | 9.34G          | 406.96          | 1624 s          |
| Swin       | Swin-S        | 24        | 49.6M       | 8.75G          | 912.38          | 1897 s          |
| Swin       | Swin-S + GFSA | 24        | 49.6M       | 8.75G          | 714.60          | 1970 s          |

Table 26: Training time (seconds per epoch) on graph-level tasks

| Method          | ZINC   | PCQM4M   | PCQM4Mv2   |
|----------|--------|----------|----------|
| Graphormer [94]   | 9 s    | 740 s    | 817 s      |
| Graphormer + GFSA | 9 s    | 896 s    | 955 s      |

Table 27: Training time (seconds per epoch) on LibriSpeech datasets

| Method          | LibriSpeech 100h   | LibriSpeech 960h   |
|----------|----------|----------|
| Transformer         | 190.5 s          | 3049.3 s          |
| Transformer + GFSA  | 191.6 s          | 3398.4 s          |
| Branchformer [59]   | 248.5 s          | 4990.1 s          |
| Branchformer + GFSA | 254.3 s          | 4999.3 s          |

Table 28: Training time (seconds per epoch) on the code defect prediction task

| Method          | Runtime   |
|----------|----------|
| RoBERTa [44]        | 543.96 s  |
| RoBERTa + GFSA      | 537.79 s  |
| CodeBERT [23]       | 555.28 s  |
| CodeBERT + GFSA     | 561.43 s  |
| PLBART [2]          | 467.80 s  |
| PLBART + GFSA       | 470.19 s  |
| CodeT5-small [84]   | 301.11 s  |
| CodeT5-small + GFSA | 309.04 s  |
| CodeT5-base [84]    | 362.28 s  |
| CodeT5-base + GFSA  | 373.22 s  |

## S Inference Time Analysis

We report the inference time of various methods with GFSA in Tables 29 to 34.

Table 29: Inference time on GLUE tasks. s denotes the abbreviation for second. Avg denotes the average training time across all tasks.

| Datasets          | #Params CoLA   | #Params CoLA   | SST-2   | MRPC   | QQP    | STS-B   | MNLI-m/mm   | QNLI   | RTE   | Avg    |
|----------|----------|----------|---------|--------|--------|---------|----------|--------|-------|--------|
| BERT BASE [16]      | 110M          | 1.0 s          | 1.4 s   | 1.2 s  | 48.7 s | 1.9 s   | 15.5 s      | 10.1 s | 1.2 s | 10.0 s |
| BERT BASE + GFSA    | 110M          | 1.1 s          | 1.4 s   | 1.2 s  | 52.3 s | 2.0 s   | 16.8 s      | 11.0 s | 1.3 s | 11.0 s |
| ALBERT BASE [40]    | 11M          | 1.1 s          | 1.6 s   | 1.4 s  | 58.4 s | 2.2 s   | 18.4 s      | 12.1 s | 1.3 s | 12.0 s |
| ALBERT BASE + GFSA  | 11M          | 1.2 s          | 1.7 s   | 1.4 s  | 62.1 s | 2.3 s   | 19.7 s      | 13.1 s | 1.4 s | 13.0 s |
| RoBERTa BASE [44]   | 125M          | 1.0 s          | 1.4 s   | 1.1 s  | 47.0 s | 1.9 s   | 15.0 s      | 9.9 s  | 1.2 s | 10.0 s |
| RoBERTa BASE + GFSA | 125M          | 1.1 s          | 1.4 s   | 1.2 s  | 50.4 s | 2.0 s   | 16.3 s      | 10.8 s | 1.2 s | 11.0 s |

Table 30: Inference time on causal language modeling tasks.

| Method      | #Params   | PTB   | WikiText-2   | WikiText-103   | Avg    |
|----------|----------|-------|----------|----------|--------|
| GPT2 [61]   | 117M      | 3.2 s | 7.4 s        | 7.4 s          | 6.0 s  |
| GPT2 + GFSA | 117M      | 5.5 s | 12.9 s       | 12.9 s         | 10.4 s |

Table 31: Inference time on ImageNet-1k

| Backbone   | Method          | #Layers   | Inference Time   |
|----------|----------|----------|----------|
| DeiT       | DeiT-S DeiT-S + GFSA DeiT-S | 12 12 24  | 52 s 53 s 68 s   |
| DeiT       | DeiT-S + GFSA          | 24        | 69 s          |
| CaiT       | CaiT-S          | 24        | 105 s          |
| CaiT       | CaiT-S + GFSA          | 24        | 107 s          |
| Swin       | Swin-S          | 24        | 17 s          |
| Swin       | Swin-S + GFSA          | 24        | 17 s          |

Table 32: Inference time on graph-level tasks

| Method          | ZINC   | PCQM4M   | PCQM4Mv2   |
|----------|--------|----------|----------|
| Graphormer [94]   | 8 s    | 99 s     | 31 s       |
| Graphormer + GFSA | 8 s    | 117 s    | 29 s       |

Table 33: Inference time on LibriSpeech datasets

| Method          | LibriSpeech 100h   | LibriSpeech 960h   |
|----------|----------|----------|
| Transformer         | 328.1 s          | 323.7 s          |
| Transformer + GFSA  | 329.5 s          | 343.3 s          |
| Branchformer [59]   | 299.4 s          | 328.7 s          |
| Branchformer + GFSA | 305.5 s          | 354.1 s          |

Table 34: Inference time on the code defect prediction task

| Method          | Inference Time   |
|----------|----------|
| RoBERTa [44]        | 22.4 s          |
| RoBERTa + GFSA      | 23.9 s          |
| CodeBERT [23]       | 23.8 s          |
| CodeBERT + GFSA     | 24.1 s          |
| PLBART [2]          | 37.7 s          |
| PLBART + GFSA       | 39.3 s          |
| CodeT5-small [84]   | 78.2 s          |
| CodeT5-small + GFSA | 82.5 s          |
| CodeT5-base [84]    | 83.2 s          |
| CodeT5-base + GFSA  | 88.5 s          |

## T Results of the Strategy for Efficiency

In Tables 35 to 39, the results show that this strategy can reduce the increase in runtime and maintain performance compared to using GFSA for all layers.

Table 35: Comparison of performance using GFSA on all layers vs. GFSAeven on even layers for GLUE tasks

| Datasets          | #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI   |   #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI |   #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI |   #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI |   #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI |   #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI | #Params CoLA SST-2 MRPC QQP STS-B MNLI-m/mm QNLI   | RTE         |   Avg |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|-------|
| BERT BASE [16]        | 110M          |          56.79 |          93.81 |          88.7 |          88.32 |          88.16 | 84.96/84.15          | 91.63 66.06 | 82.51 |
| BERT BASE + GFSA      | 110M          |          59.56 |          94.15 |          90.6 |          88.46 |          88.33 | 85.12/85.06          | 91.95 68.95 | 83.58 |
| BERT BASE + GFSA even | 110M          |          58.8  |          93.69 |          90.5 |          88.47 |          88.27 | 85.13/85.02          | 91.65 70.76 | 83.59 |

Table 36: Comparison of training time (seconds per epoch) using GFSA on all layers vs. GFSAeven on even layers for GLUE tasks

| Datasets          | #Params CoLA   | #Params CoLA   | SST-2   | MRPC   | QQP    | STS-B   | MNLI-m/mm   | QNLI   | RTE   | Avg   |
|----------|----------|----------|---------|--------|--------|---------|----------|--------|-------|-------|
| BERT BASE [16]        | 110M          | 17 s          | 182 s   | 17 s   | 1483 s | 24 s    | 2004 s      | 580 s  | 18 s  | 541 s |
| BERT BASE + GFSA      | 110M          | 19 s          | 192 s   | 19 s   | 1571 s | 25 s    | 2147 s      | 621 s  | 20 s  | 577 s |
| BERT BASE + GFSA even | 110M          | 17 s          | 185 s   | 18 s   | 1506 s | 24 s    | 2061 s      | 595 s  | 18 s  | 553 s |

Table 37: Comparison of performance using GFSA on all layers vs. GFSAeven on even layers for causal language modeling tasks

| Method          | #Params   |    PTB |   WikiText-2 |   WikiText-103 |    Avg |
|----------|----------|--------|----------|----------|--------|
| GPT2 [61]        | 117M      | 19.513 |       20.966 |         15.939 | 18.806 |
| GPT2 + GFSA      | 117M      | 19.45  |       20.923 |         15.919 | 18.764 |
| GPT2 + GFSA even | 117M      | 19.453 |       20.926 |         15.923 | 18.767 |

Table 38: Comparison of training time (seconds per epoch) using GFSA on all layers vs. GFSAeven on even layers for causal language modeling tasks

| Method          | #Params   | PTB     | WikiText-2   | WikiText-103   | Avg      |
|----------|----------|---------|----------|----------|----------|
| GPT2 [61]        | 117M      | 89.1 s  | 195.7 s      | 9638.4 s       | 3307.8 s |
| GPT2 + GFSA      | 117M      | 160.3 s | 354.2 s      | 17424.6 s      | 5979.7 s |
| GPT2 + GFSA even | 117M      | 127.4 s | 279.1 s      | 13761.4 s      | 4722.6 s |

Table 39: Comparison of using GFSA on all layers vs. GFSAeven on even layers for ImageNet-1k

| Method          |   Input Size |   #Layers | #Params   |   Top-1 Acc | Runtime   |
|----------|----------|----------|----------|----------|----------|
| DeiT-S          |          224 |        12 | 43M       |        79.8 | 551 s     |
| DeiT-S + GFSA      |          224 |        12 | 43M       |        81.1 | 814 s     |
| DeiT-S + GFSA even |          224 |        12 | 43M       |        81   | 595 s     |

## U GFSA in Linear Transformers

Table 40 shows the accuracy, runtime, and GPU usage results on Long Range Arena benchmark using ListOps and Image datasets.

Table 40: Comparison of accuracy (%), runtime ( s per 1,000 steps) and GPU usage (GB) on Long Range Arena benchmark

| Method          | ListOps (2K)   | ListOps (2K)   | ListOps (2K)   | Image (4K)   | Image (4K)   | Image (4K)   |
|----------|----------|----------|----------|----------|----------|----------|
|          | Accuracy       | Runtime        | GPU usage      | Accuracy     | Runtime      | GPU usage    |
| Transformer [77]          | 37.1          | 198.3          | 5.50          | 38.2         | 345.1        | 5.88         |
| Transformer+GFSA          | 37.6          | 635.8          | 10.87          | 40.2         | 737.2        | 11.20        |
| Linformer [81]          | 37.3          | 63.4          | 1.73          | 37.8         | 158.5        | 3.45         |
| YOSO-E [97]          | 37.3          | 85.7          | 0.37          | 39.8         | 114.2        | 1.42         |
| Efficient Attention [70]   | 36.9          | 49.2          | 0.57          | 40.2         | 121.1        | 1.14         |
| Efficient Attention + GFSA | 37.9          | 53.8          | 0.67          | 40.4         | 135.8        | 1.33         |

##