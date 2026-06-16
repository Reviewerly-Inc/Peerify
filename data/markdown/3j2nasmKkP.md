## Cluster-wise Graph Transformer with Dual-granularity Kernelized Attention

Siyuan Huang 1 , 2 Yunchong Song 1 Jiayue Zhou 2 Zhouhan Lin 1 ∗

1 LUMIA Lab, Shanghai Jiao Tong University

2 Paris Elite Institute of Technology, Shanghai Jiao Tong University siyuan\_huang\_sjtu@outlook.com ycsong@sjtu.edu.cn

## Abstract

In the realm of graph learning, there is a category of methods that conceptualize graphs as hierarchical structures, utilizing node clustering to capture broader structural information. While generally effective, these methods often rely on a fixed graph coarsening routine, leading to overly homogeneous cluster representations and loss of node-level information. In this paper, we envision the graph as a network of interconnected node sets without compressing each cluster into a single embedding. To enable effective information transfer among these node sets, we propose the Node-to-Cluster Attention (N2C-Attn) mechanism. N2C-Attn incorporates techniques from Multiple Kernel Learning into the kernelized attention framework, effectively capturing information at both node and cluster levels. We then devise an efficient form for N2C-Attn using the cluster-wise message-passing framework, achieving linear time complexity. We further analyze how N2C-Attn combines bi-level feature maps of queries and keys, demonstrating its capability to merge dual-granularity information. The resulting architecture, Cluster-wise Graph Transformer (Cluster-GT), which uses node clusters as tokens and employs our proposed N2C-Attn module, shows superior performance on various graph-level tasks. Code is available at https: //github.com/LUMIA-Group/Cluster-wise-Graph-Transformer .

## 1 Introduction

Graph learning represents a rapidly evolving field. Techniques like Graph Neural Networks (GNNs) and Graph Transformers (GT) demonstrate impressive performance across a range of tasks [27, 36, 51], such as social networks [39, 37], time series [22, 30], traffic flow [54, 6] and drug discovery [15, 45]. These methods enhance performance by promoting message propagation at the node level and calculating attention between node pairs, thereby concentrating on node-level interactions.

Recent advancements have extended beyond node-level message propagation, adopting approaches that treat the graph as a hierarchical structure [53, 4, 20], capturing information at multiple levels of the graph [55]. For instance, node clustering pooling segments the graph into multiple clusters [17, 50]. Each cluster is then independently pooled, preserving the structural information of the hierarchical graph. Drawing inspiration from Vision Transformers [7], GraphViT [18] treats subgraphs as tokens and computes attention among them, which enables the model to capture long-distance dependencies and reduces the overall computational complexity compared to node-level Graph Transformers.

However, existing methods based on node clustering rely on a fixed graph coarsening routine [32]. This routine involves partitioning the graph into several clusters and subsequently pooling each cluster into a single node to generate a coarsened version of the original graph. While generally

∗ Zhouhan Lin is the corresponding author

lin.zhouhan@gmail.com

effective, research has shown that compressing each cluster into a single embedding can lead to overly uniform cluster representations, which may not accurately reflect the diversity within each cluster [34]. Furthermore, these methods typically simplify the interactions between clusters to basic vertex-level interactions on the coarsened graph. This oversimplification overlooks the rich node-level information contained within each cluster, thereby limiting the potential for richer cluster-wise interactions.

In this work, we propose a different strategy for enhancing cluster-wise interaction. Instead of reducing each cluster to a single node through coarsening, we envision the graph as a network of interconnected node sets. To enable message propagation among these node sets, we develop a method termed Node-to-Cluster Attention (N2C-Attn). N2C-Attn incorporates techniques from Multiple Kernel Learning (MKL) [16] into the kernelized attention framework [46]. By combining kernels at two different granularity levels, N2C-Attn effectively captures hierarchical graph structural information at the cluster level while also preserving node-level details within each cluster.

We propose treating the graph as interconnected node clusters without coarsening, which inherently increases computational complexity. To mitigate this issue, we employ the technique of kernelized softmax [24] to reduce the computational complexity to linear. Consequently, the computation process of N2C-Attn can be viewed as a cluster-wise message propagation: each cluster gathers internal keys and values, then propagates them along weighted edges to the queries of other clusters.

We present a further analysis of how N2C-Attn synthesizes new queries and keys by merging inputs from both node and cluster levels. We consider two scenarios: 1) using the product of kernels and 2) using the convex sum of kernels. The former implicitly conducts a tensor product of the feature maps from both the node-level and cluster-level queries (and keys), adopting this product as the new query (or key) for N2C-Attn. The latter concatenates node and cluster-level feature maps with learnable weights, maintaining their independence and allowing the model to adjust their relative significance. We also demonstrate that cluster-level attention can be regarded as a special case of N2C-Attn.

Our resulting architecture, Cluster-wise Graph Transformer (Cluster-GT), leverages our proposed N2C-Attn module in conjunction with a simple graph partitioning algorithm, Metis [23]. We conduct extensive evaluations of Cluster-GT across eight graph-level datasets, varying in size and domain. Cluster-GT outperforms existing Graph Transformers and graph pooling methods that employ more intricate graph partitioning algorithms, which highlights the effectiveness of enhancing inter-cluster interactions and preserving information at both granular levels. We further analyze the relative weights of the combined kernel, finding that Cluster-GT pays more attention to cluster-level information when handling graphs in the social network domain compared to graphs in the biological domain.

## 2 Background

Consider a graph G represented by the multi-tuple ( N , E , X , A ) . N denotes the set of n nodes, E denotes the set of m edges. X ∈ R n × d is the feature matrix and A ∈ R n × n is the adjacency matrix. We use the superscript P to indicate the cluster-level (coarsened) graph: ( N P , E P , X P , A P ) , where N P represents clusters of nodes, and E P denotes the edges connecting these clusters.

Node Clustering Pooling and Cluster Assignment Matrix Node clustering pooling captures hierarchical structural information by partitioning and iteratively coarsening the graph to a smaller size [32, 1, 33]. This process involves two main steps. Initially, a Cluster Assignment Matrix (CAM) C ∈ R n × m is generated using a carefully designed strategy, where n represents the number of original nodes, and m indicates the number of clusters. Once the Cluster Assignment Matrix is obtained, it is used to perform graph coarsening, i.e., pooling each cluster into a single node:

<!-- formula-not-decoded -->

where X P ∈ R m × d and A P ∈ R m × m are the new node features and adjacency matrix, defining the post-coarsening graph structure. C sj thus represents the weight of the s -th node in the j -th cluster.

Beyond node clustering pooling, methods exist that leverage node clusters to enhance graph attention [18, 3]. GraphViT [18] utilizes Metis [23] to partition the graph into multiple subgraphs. It then applies mean pooling to each subgraph, treating the pooled clusters as tokens for further attention computation. Despite promising results, GraphViT still adheres to the graph coarsening pipeline, which leads to overly similar cluster representations [34] and the loss of node-level information.

Generalized Self-attention and Kernelized Softmax Numerous studies suggest reevaluating the attention mechanism through the lens of kernel methods [46, 24]. The generalized formulation of self-attention utilizes a non-negative kernel function κ ( · , · ) : R d k × R d k → R + , which can be represented with a corresponding feature map ϕ . The self-attention mechanism can be expressed as:

<!-- formula-not-decoded -->

where k j , q i , and v j are the corresponding keys, queries, and values. By expressing κ with feature map κ ( q i , k j ) = ϕ ( q i ) T ϕ ( k j ) , the computation simplifies to:

<!-- formula-not-decoded -->

where the sums ∑ N j =1 ϕ ( k j ) T v j and ∑ N j =1 ϕ ( k j ) T are shared across all nodes and need to be computed only once, thus reducing computational complexity to O ( N ) [21]. Various choices of feature maps are shown effective, such as the RBF kernel [46] and Positive Random Features [5].

Multiple Kernel Learning The selection of an optimal kernel function κ ( · , · ) is critical for enhancing the performance of kernel-based learning methods. Multiple Kernel Learning (MKL) methods [56, 44] leverage a combination of kernel functions to integrate various features from different perspectives. The resultant kernel, κ η , is mathematically defined as:

<!-- formula-not-decoded -->

where f η : R M → R can be either a linear or nonlinear function. Each κ m : R D m × R D m → R is a valid kernel for vectors x m , y m ∈ R D m , with D m representing the dimensionality of each feature. There are various strategies for combining kernels, which represent a dynamic area of research. [16]

In this work, we concentrate on the pairwise scenario, where M = 2 . We note the two input spaces as X and X ′ . For constructing pairwise kernels when elements of each pair belong to different input spaces, we select two fundamental strategies: the tensor product of kernels and the convex linear combination of kernels, which are commonly used on the product space X × X ′ .

Given two kernels κ 1 : X ×X → R and κ 2 : X ′ ×X ′ → R , the tensor product method is defined as:

<!-- formula-not-decoded -->

where ( x, x ′ ) , ( y, y ′ ) are pairs of objects from X × X ′ . While the convex linear combination method is defined as:

<!-- formula-not-decoded -->

where α, β ≥ 0 and α + β = 1 . α and β are coefficients that balance the contribution of each kernel.

## 3 Node-to-Cluster Attention

In this section, we present the Node-to-Cluster Attention (N2C-Attn) mechanism. We begin in Section 3.1 by defining the concept of N2C-Attn. We then proceed to Section 3.2, where we devise an efficient form of N2C-Attn with the message-passing framework. In Section 3.3, we re-examine N2C-Attn, focusing on the integration of feature maps of queries and keys across two granularities.

## 3.1 Definition of Node-to-Cluster Attention

Node-to-Cluster Attention marks a departure from the graph coarsening pipeline that typically coarsens each cluster into a single embedding. Instead, as shown in Figure 1, we maintain the clusters uncompressed and use N2C-Attn to propagate messages among the inter-connected node clusters. The definition of N2C-Attn is based on the Cluster Assignment Matrix C , which can be obtained through various graph partitioning methods [32]. N2C-Attn focuses on the "post-clustering" phase.

4. Maintain inear computational complexity (subsection 3.&lt;)

tion nessage prop

work to facilit e-level and clu

Bi-level Q&amp;K

-level inform

Node-wise interaction:

Queries of clusters

GNN, Graph Transformer..

Keys of nodes

1 node posses de clustering,

he collective 1

ure of its clus

00-0

O

Cluster-wise interation:

Node-to-Cluster Attention

Node-level Xv

91,92,93

k1, kz, k3, ka, ks, k6, kir, kis, kg

Cluster-level Xc

Q1,02,23

K1, K2, Kз

{91,91}

O

(K1,k1) (K1,k2) (K1,kg)

{Q2,92}

{K2,k4) (K2,k5)

{Q3,93}

(K3, kig)

, (К3, k7)

(Кз, kg) (Кз, kg)

Step 1: Step 2: Step 3: Step 4: Figure 1: Definition of Node-to-Cluster Attention (N2C-Attn). N2C-Attn perceives the graph as interconnected node sets instead of coarsening each cluster into a single node. It integrates multiple kernel learning methods into the kernelized attention framework to facilitate message propagation among node clusters, simultaneously capturing both the node-level and cluster-level information.

<!-- image -->

Sum the key and value within every cluster according to the Cluster Assignment Matrix Calculate the cluster-wise similarity as the Edge Gate for message propagation among clusters Cluster-wise propagation with Message in Step 1 and Edge Gate in Step 2 Finally each cluster unpacking the aggregated keys and values with its own query. Bi-level Query and Key A key observation is that after node clustering, each node possesses two tiers of information: 1) its individual node feature and 2) the collective feature of its cluster. An effective attention mechanism needs to accommodate these two distinct levels of information. Thus, the t -th node in the j -th cluster is characterized by a bi-level pair of keys: { K j , k t } ∈ X C ×X N :

<!-- formula-not-decoded -->

where h t is the feature of the t -th node. k t ∈ X N is the node-level key, which is solely derived from the embedding of t -th node, and K j ∈ X C represents the cluster-level key, which depends on all nodes within the j -th cluster. W k and W ′ k are two different projections to X N and X C , respectively.

Since we are considering the attention between clusters and nodes, each cluster needs to provide a corresponding bi-level query. Thus, the i -th cluster is characterized by a bi-level pair of queries:

<!-- formula-not-decoded -->

where Q i denotes the cluster-level query, q i denotes the node-level query. W v and W ′ v are two different projections to X N and X C , respectively. The bi-level query is thus { Q i , q i } ∈ X C ×X N .

Note that we use uppercase letters to represent cluster-level queries and keys, e.g., { Q i , K j } , and lowercase letters to represent node-level queries and keys, e.g., { q i , k t } .

General Definition of Node-to-Cluster Attention Having obtained the bi-level queries and keys, we consider how to use kernels to measure their similarity. We denote a valid kernel in the clusterlevel space X C as κ C , and a valid kernel in the node-level space X N as κ N . We now consider how to construct a kernel κ B on the tensor product space X C ×X N . κ B stands for B i-level kernel.

Given { Q i , q i } , the bi-level query for the i -th node cluster, and { K j , k t } , the bi-level key for the t -th node in the j -th node cluster, the general Node-to-Cluster Attention for the i -th cluster is defined as:

<!-- formula-not-decoded -->

Equation 9 depicts the process of the i -th cluster gathering information from nodes of all connected clusters. The attention score between the i -th cluster and the t -th node in the j -th cluster is A P i,j C tj κ B ( { Q i ,q i } , { K j ,k t } ) ∑ j A P i,j ∑ t C tj κ B ( { Q i ,q i } , { K j ,k t } ) . κ B plays a pivotal role in integrating information across cluster and node levels. As described in Section 2, we mainly consider two options for κ B: the tensor product method and the linear combination method. Next, we introduce these two different implementations.

N2C

whe

Queries of clusters

Keys of nodes where

is a bi-level kernel on

Figure 2: An efficient implementation of N2C-Attn-T with the message-passing framework. |N P | denotes the number of clusters and |E P | denotes the number of edges between clusters. The computation can be decomposed into 4 steps: 1) aggregation of node-level keys and values within each cluster, 2) computation of gate on each edge with the cluster-level kernel, 3) message propagation among clusters, 4) dot product of aggregated value with the node-level query of each cluster.

<!-- image -->

Node-to-Cluster Attention with Tensor Product of Kernels (N2C-Attn-T) With the help of Equation 5, we can define the bi-level kernel κ B as:

<!-- formula-not-decoded -->

We can thus rewrite the Node-to-Cluster Attention defined in Equation 9 as:

<!-- formula-not-decoded -->

N2C-AttnT stands for Node-to-Cluster Attention with T ensor Product of Kernels. By performing the product between κ C and κ N , this construction enables interaction across all dimensions of the feature vectors at different granular levels, thereby capturing the dependencies within the combined feature space. We offer a more detailed explanation in subsection 3.3.

Node-to-Cluster Attention with Convex Linear Combination of Kernels (N2C-Attn-L) With the help of Equation 6, we can also define the bi-level kernel κ B as:

<!-- formula-not-decoded -->

where α, β ≥ 0 are learnable parameters and α + β = 1 . We can thus rewrite Equation 9 as:

<!-- formula-not-decoded -->

N2C-AttnL stands for Node-to-Cluster Attention with Convex L inear Combination of Kernels. By combining the kernels κ C and κ N with coefficients α and β , this construction allows for flexible integration of the similarities measured in X C and X N , letting the combined kernel adaptively scale the influence of the cluster-level and node-level information on the overall similarity measure.

## 3.2 Efficient Implementation of Node-to-Cluster Attention

N2C-Attn requires the computation of similarity between queries and keys at two different levels of granularity. Normally, this necessitates a computational complexity of O ( |N||N P | ) , where |N| denotes the number of nodes and |N P | denotes the number of clusters. To speed up this process, we devise a linear algorithm using the feature map and the message-passing framework. In this subsection, we focus on the efficient implementation of N2C-Attn-T. Following a similar method, we can also develop an efficient implementation for N2C-Attn-L, which is detailed in Appendix A.

To accelerate Equation 11, a key observation is that κ C is correlated to the edges between clusters, serving as the gates on the edges. While κ N involves queries of clusters and keys of nodes. Therefore, we propose separating the node-level and cluster-level computation of κ N , and then turning the computation of N2C-Attn-T into a cluster-wise message propagation. We represent κ N using the corresponding feature map: κ N ( q i , k t ) = ψ ( q i ) T ψ ( k t ) . Thus, Equation 11 can be rewritten as:

<!-- formula-not-decoded -->

With Equation 14, we observe that the computation of N2C-Attn-T can be encompassed within the message-passing framework. Figure 2 shows the complete process, which contains four steps: 1) aggregating node-level keys and values within each cluster, 2) calculating the gate on each edge using the cluster-level kernel, 3) propagating messages among clusters, and 4) computing the dot product of the aggregated value with the node-level query for each cluster. N2C-Attn-T can thus be seen as a form of cluster-wise message propagation. Each cluster acts as a sender, propagating the packaged keys and values of its internal nodes; it also acts as a receiver, using its own query to interpret the information from the received keys and values. The overall time complexity is thus reduced to linear.

## 3.3 Equivalent Feature Maps of Bi-level Kernels

In this subsection, we delve into how the Node-to-Cluster Attention mechanism integrates information across cluster and node levels through the lens of feature maps. We note Φ B as the feature map of κ B : κ B ( { Q i , q i } , { K j , k t } ) = ⟨ Φ B ( { Q i , q i } ) , Φ B ( { K j , k t } ) ⟩ where ⟨· , ·⟩ represents the inner product in the corresponding feature space. Equation 9 can thus be expressed as:

<!-- formula-not-decoded -->

Φ B ( Q i , q i ) represents the feature vector of the newly formulated bi-level query, while Φ B ( K j , k t ) represents the feature vector of the newly formulated bi-level key. We are interested in their relationship with the original queries and keys { Q i , q i , K j , k t } . We establish the following relationships:

Proposition 1 If κ C ( Q i , K j ) = ⟨ ϕ ( Q i ) , ϕ ( K j ) ⟩ and κ N ( q i , k t ) = ⟨ ψ ( q i ) , ψ ( k t ) ⟩ , where ϕ and ψ are feature maps for the respective kernels, then the Node-to-Cluster Attention with the tensor product kernel implies the following equivalent feature map:

<!-- formula-not-decoded -->

where ⊗ represents the outer product of the node-level and cluster-level feature maps. Conversely, the Node-to-Cluster Attention with the convex sum implies the following equivalent feature map:

<!-- formula-not-decoded -->

where ⊕ represents the concatenation of the weighted node-level and cluster-level feature maps.

This proposition provides an intuitive understanding of N2C-Attn: by integrating queries and keys from both node-level and cluster-level, N2C-Attn synthesizes new queries and keys enriched with bi-level information. Specifically, using the product of kernels, as detailed in Equation 16, N2C-AttnT implicitly performs a tensor product between the feature maps of the node-level query (key) and the cluster-level query (key), and finally using the product as the new query (key). This resulting equivalent feature map thus extends into a higher-dimensional space, offering a feature fusion of bi-level information. It's worth noting that we do not need to actually compute the tensor product between the cluster-level and node-level queries or keys, which requires high spatial complexity.

While employing the convex sum of kernels, as detailed in Equation 17, can be regarded as a concatenation of the feature maps of the original node-level and cluster-level queries (keys), appending learnable weights. This approach preserves the independence of queries (keys) at different levels, empowering the model to adjust their relative significance. Besides, we can leverage this point to design an efficient implementation method for N2C-Attn-L. We introduce it in detail in Appendix A.

Weoffer a further analysis by comparing the assigned attention scores between N2C-Attn and previous cluster-level attention methods. We prove that the attention mechanism used in GraphViT [18], which is based on the graph coarsening pipeline and serves as a cluster-level attention mechanism, can be seen as a special case of our proposed N2C-Attn. More details can be found in Appendix B.

## 4 Cluster-wise Graph Transformer

In this section, we introduce a simple yet performant architecture named Cluster-wise Graph Transformer (Cluster-GT) which takes node clusters as tokens and utilizes N2C-Attn defined in Section 3 to propagate information among clusters. Cluster-GT can be divided into three main modules: 1) a node-level convolution module, 2) a graph partition module, and 3) a cluster-wise interaction module.

Figure 3: Architecture of Cluster-wise Graph Transformer (Cluster-GT), which can be decomposed into three main modules: 1) a node-wise convolution module with GNN, 2) a graph partition module with Metis, and 3) a cluster-wise interaction module with N2C-Attn.

<!-- image -->

Figure 3 presents the overall architecture of our proposed Cluster-GT. We begin with a node-level convolution module to capture the local structural information. We try two common options, GCN [27] and GIN [49], during our implementation. We also utilize two graph positional encoding strategies, random-walk structural encoding (RWSE) [9] and Laplacian eigenvector encodings [8], to enhance the perception of the graph structure. More details can be found in Appendix D. For the graph partition module, we use a relatively simple graph partition algorithm, Metis [23], to assign nodes to different clusters. After node clustering assignment, we introduce our proposed N2C-Attn as the cluster-wise interaction module, which propagates information among clusters. This process is divided into two steps: we first calculate the corresponding bi-level keys and queries, and then execute the efficient algorithm of N2C-Attn introduced in subsection 3.2, which outputs a single embedding for each cluster. We finally perform average pooling to obtain the graph-level embedding.

The choice of kernel and feature map is not the main focus of our work. In our implementation, we use the common exp-dot-product exp ( Q T K √ d k ) as κ C . For the feature map of κ N , we try two basic options: ψ ( x ) = Elu( x ) + 1 [40] and ψ ( x ) = Relu( x ) [24], which we set as a hyperparameter.

Cluster-GT, in conjunction with N2C-Attn, is designed to enhance information exchange between node clusters after the graph partitioning. This process can be viewed as a "post-partitioning" phase, which is a key distinction from many other node-clustering-based methods that primarily focus on optimizing the graph partition itself. In our implementation, we utilize a non-learnable and rigid graph partitioning algorithm, Metis. Notably, the Graph Partition module in Cluster-GT can be replaced with other learnable or flexible graph partitioning strategies, allowing for potential enhancements.

## 5 Evaluation

To evaluate the performance of Cluster-GT, we compare it against two categories of methods: Graph Pooling and Graph Transformers. We conduct experiments on eight graph classification datasets from different domains, including social networks and biology. We further visualize the weight coefficients of the cluster-level and node-level kernels in N2C-Attn-L to observe how the model focuses on different information granularities across different datasets. Additionally, we perform an ablation study, restricting the attention mechanism to different granularities, to demonstrate the benefits of integrating both levels of information. We finally carry out an efficiency study of Cluster-GT. All experiments are conducted on NVIDIA RTX 3090s with 24GB of RAM. Detailed dataset information is available in Appendix E, and more details of the implementation are provided in Appendix F.

## 5.1 Comparison with Graph Pooling Methods

Given the close relationship between Cluster-GT and node clustering methods, we compare ClusterGT with mainstream Graph Pooling methods:two well-known GNN baselines: GCN [27], GIN [49], six hierarchical pooling approaches: DiffPool [53], SAGPool(H) [29], TopKPool [13], ASAP [42], MinCutPool [4], SEP [50] and five global pooling techniques: Set2Set [48], SortPool [57], SAGPool(G) [29], StructPool [55], GMT [2] We test Cluster-GT on six TU datasets [38]: IMDB-BINARY, IMDB-MULTI, COLLAB, MUTAG, PROTEINS, and D &amp; D. The first three datasets are in the field

Table 1: Comparison with Graph Pooling Methods on six TU datasets. The shown accuracies (%) are mean and standard deviation over 10 different runs. We highlight the best results.

| Model          | IMDB-BINARY          | IMDB-MULTI          | COLLAB          | MUTAG          | PROTEINS          | D&D          |
|----------|----------|----------|----------|----------|----------|----------|
| GCN GIN Set2Set SortPool SAGPool(G) StructPool GMT | 73 . 26 ± 0 . 46 72 . 78 ± 0 . 86 72 . 90 ± 0 . 75 72 . 12 ± 1 . 12 72 . 16 ± 0 . 88 72 . 06 ± 0 . 64 73 . 48 ± 0 . 76 | 50 . 39 ± 0 . 41 48 . 13 ± 1 . 36 50 . 19 ± 0 . 39 48 . 18 ± 0 . 83 49 . 47 ± 0 . 56 50 . 23 ± 0 . 53 50 . 66 ± 0 . 82 | 80 . 59 ± 0 . 27 78 . 19 ± 0 . 63 79 . 55 ± 0 . 39 77 . 87 ± 0 . 47 78 . 85 ± 0 . 56 77 . 27 ± 0 . 51 80 . 74 ± 0 . 54 | 69 . 50 ± 1 . 78 81 . 39 ± 1 . 53 69 . 89 ± 1 . 94 71 . 94 ± 3 . 55 76 . 78 ± 2 . 12 79 . 50 ± 0 . 75 83 . 44 ± 1 . 33 | 73 . 24 ± 0 . 73 71 . 46 ± 1 . 66 73 . 27 ± 0 . 85 73 . 17 ± 0 . 88 72 . 02 ± 1 . 01 75 . 16 ± 0 . 86 75 . 09 ± 0 . 59 | 72 . 05 ± 0 . 55 70 . 79 ± 1 . 17 71 . 94 ± 0 . 56 75 . 58 ± 0 . 72 71 . 54 ± 0 . 91 78 . 45 ± 0 . 40 78 . 72 ± 0 . 59 |
| SEP-G Cluster-GT          | 74 . 12 ± 0 . 56 75 . 10 ± 0 . 84          | 51 . 53 ± 0 . 65 52 . 13 ± 0 . 78          | 81 . 28 ± 0 . 15 80 . 43 ± 0 . 52          | 85 . 56 ± 1 . 09 87 . 11 ± 1 . 37          | 76 . 42 ± 0 . 39 76 . 48 ± 0 . 86          | 77 . 98 ± 0 . 57 79 . 15 ± 0 . 63          |

Table 2: Comparison with Graph Transformers on ZINC and MolHIV over 4 different runs of 4 different seeds. We highlight the best results. Missing values from literature are indicated as '-'.

Figure 4: Visualization of α (weight of the clusterlevel kernel) during the training process. N2C-Attn learns to pay more attention to cluster-level information in social networks than in bioinformatics.

| Model          | ZINC (MAE ↓ )          | MolHIV (ROCAUC ↑ )          |
|----------|----------|----------|
| GT GraphiT Graphormer GPS | 0 . 226 ± 0 . 014 0 . 202 ± 0 . 011 0 . 122 ± 0 . 006 . 070 ± 0 . 004 | - - - 0 . 7880 ± 0 . 0101 0 . 7775 ± 0 . 0061 |
|          | 0          |          |
| SAN+LapPE          | 0 . 139 ± 0 . 006          |          |
| Graph MLP-Mixer Graph ViT | 0 . 073 ± 0 . 001 0 . 085 ± 0 . 005          | 0 . 7997 ± 0 . 0102 0 . 7792 ± 0 . 0149       |
| Cluster-GT          | 0 . 071 ± 0 .          | . ± 0 .          |
|          | 004          | 0 8093 0136          |

<!-- image -->

of social networks, while the latter three are in the field of biology. For a fair comparison, we strictly follow the experimental setup of [50]. Table 1 shows the results, indicating that Cluster-GT outperforms all baselines on most datasets, even though it employs a relatively simple graph partitioning algorithm compared to other node clustering pooling methods. This result highlights the effectiveness of the N2C-Attn module and shows the importance of the interaction between clusters in the "post-partitioning" phase, which is often oversimplified by other node clustering pooling methods.

## 5.2 Comparison with Graph Transformers

To assess the effectiveness of Cluster-GT within the context of Graph Transformers, we compare Cluster-GT with a range of existing Graph Transformers, including GT [8], GraphiT [35], Graphormer [52], GPS [41], SAN+LapPE [28], SAN+RWSE [28], Graph MLP-Mixer [18] and Graph ViT [18]. We conduct the experiment on two datasets: ZINC from Benchmarking GNNs [8] and Mol-HIV from OGB [19]. For a fair comparison, we strictly follow the experimental setup of [18]. The result shown in Figure 2 demonstrates that Cluster-GT surpasses most existing Graph Transformers, underscoring the importance of integrating information at both the cluster and node levels and showcasing the potential of using node clusters as tokens in attention mechanisms.

## 5.3 Visualization of α in N2C-Attn-L

In this subsection, we present the dynamic changes in the weight coefficient α during the training process of N2C-Attn-L. α quantifies the contribution of cluster-level information in the combined kernel, whereas β = 1 -α quantifies the node-level information. By enabling the model to

Figure 5: Comparison of different attention strategies. We restrict the attention module in Cluster-GT to focus on different granularities. N2C-Attn-T and N2C-Attn-L represent schemes that integrate information at both the node and cluster granularities. Cluster-Level-Attn focuses solely on clusterlevel information, i.e., α = 1 , while Node-Level-Attn focuses solely on node-level information, i.e., α = 0 . We provide a detailed description of the methods compared here in subsection F.3.

<!-- image -->

autonomously learn these coefficients, it dynamically adjusts to the varying importance of information at different granularities. We plot the evolution of α across training steps for six diverse datasets, as shown in Figure 4. We observe that the model automatically adjusts the weights assigned to the two levels of granularity. Notably, for social network datasets, Cluster-GT shows a preference for cluster-level information, whereas, for biology datasets, Cluster-GT balances its attention more equally between both granularities. This result indicates that N2C-Attn has a stronger inclination towards cluster-level information in the social networks domain compared to the biology domain.

## 5.4 Necessity of Combining Cluster-level and Node-level Information

In this subsection, we explore the necessity of fusing kernels of dual granularities within the N2C-Attn module. We analyze four variants: the first two are N2C-Attn-T and N2C-Attn-L, which are the attention schemes utilized in Cluster-GT. N2C-Attn-T deeply integrates cluster-level and node-level information, whereas N2C-Attn-L autonomously adjusts the balance between these two granularities. Then, we create two additional variants that specifically focus on the node level or the cluster level by setting α in N2C-Attn-L to 0 (exclusively focusing on the node-level kernel) and 1 (exclusively focusing on the cluster-level kernel). We provide a detailed description of the methods compared here in subsection F.3. Figure 5 shows the experimental results. We find that the variants that combine attention from both levels significantly surpass those that do not, with N2C-Attn-T leading marginally. This highlights the effectiveness of N2C-Attn's multiple kernel learning approach in integrating diverse levels of information. We reference the performance of GCN from Table 1 as a baseline.

## 6 Other Methods Involving Graph Coarsening

In this section, we will briefly introduce some existing research on GNNs with graph coarsening to capture broader structural information, aside from the node clustering pooling introduced in section 2.

[12] utilizes a dual-graph structure, employing a hierarchical message passing strategy between a molecular graph and its junction tree to facilitate a bidirectional flow of information. This concept of interaction between the coarsened graph (clusters) and the original graph (nodes) is similar to our N2C-Attn. However, the difference lies in [12]'s approach to propagating messages between clusters and nodes, whereas N2C-Attn integrates cluster and node information directly in the attention calculation using a multiple-kernel method. [58] introduces a novel node sampling strategy as an adversarial bandit problem and implements a hierarchical attention mechanism with graph coarsening to efficiently address long-range dependencies. [31] uses graph pooling to coarsen nodes into fewer representatives, focusing attention on these pooled nodes to manage scalability and computational efficiency. [25] introduces the Subgraph-To-Node (S2N) translation method, coarsening subgraphs into nodes to improve subgraph representation learning. [14] introduces HIGH-PPI, a double-viewed hierarchical graph learning model that uses a hierarchical graph combining protein-protein interaction networks and chemically described protein graphs to accurately predict PPIs and interpret their molecular mechanisms. Despite achieving good results in their respective downstream tasks, these methods still follow the graph coarsening pipeline, whereas our work attempts to break this limitation and has demonstrated effectiveness on various graph-level tasks.

## 7 Conclusion

Our Node-to-Cluster Attention mechanism leverages the strengths of both node-level and cluster-level information processing without succumbing to the limitations of the graph coarsening pipeline. By conceptualizing the graph as interconnected node sets and integrating kernelized attention with multiple kernel learning, we effectively bridge the gap between cluster-level and node-level spaces, capturing the hierarchical structure of graphs as well as the node-level information. We develop an efficient form of N2C-Attn using the message-passing framework and techniques of kernelized softmax. Our Cluster-wise Graph Transformer, empowered by a straightforward partitioning strategy and the N2C-Attn module, demonstrates robust performance across diverse graph datasets. Extensive experiments have demonstrated the effectiveness of our Cluster-GT and N2C-Attn modules. We offer a further discussion on the current limitation and potential impact in Appendix H and Appendix I.

## Acknowledgement

This work was sponsored by the National Key Research and Development Program of China (No. 2023ZD0121402) and National Natural Science Foundation of China (NSFC) grant (No.62106143).

## References

- [1] D. Bacciu and L. D. Sotto. A non-negative factorization approach to node pooling in graph convolutional neural networks. In M. Alviano, G. Greco, and F. Scarcello, editors, AI*IA 2019 - Advances in Artificial Intelligence - XVIIIth International Conference of the Italian Association for Artificial Intelligence, Rende, Italy, November 19-22, 2019, Proceedings , volume 11946 of Lecture Notes in Computer Science , pages 294-306. Springer, 2019.
- [2] J. Baek, M. Kang, and S. J. Hwang. Accurate learning of graph representations with graph multiset pooling. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021 . OpenReview.net, 2021.
- [3] S. Bandyopadhyay, M. Aggarwal, and M. N. Murty. Robust hierarchical graph classification with subgraph attention. CoRR , abs/2007.10908, 2020.
- [4] F. M. Bianchi, D. Grattarola, and C. Alippi. Spectral clustering with graph neural networks for graph pooling. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event , volume 119 of Proceedings of Machine Learning Research , pages 874-883. PMLR, 2020.
- [5] K. M. Choromanski, V. Likhosherstov, D. Dohan, X. Song, A. Gane, T. Sarlós, P. Hawkins, J. Q. Davis, A. Mohiuddin, L. Kaiser, D. B. Belanger, L. J. Colwell, and A. Weller. Rethinking attention with performers. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021 . OpenReview.net, 2021.
- [6] A. Derrow-Pinion, J. She, D. Wong, O. Lange, T. Hester, L. Perez, M. Nunkesser, S. Lee, X. Guo, P. W. Battaglia, V. Gupta, A. Li, Z. Xu, A. Sanchez-Gonzalez, Y. Li, and P. Veliˇ ckovi´ c. Traffic Prediction with Graph Neural Networks in Google Maps. 2021.
- [7] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021 . OpenReview.net, 2021.
- [8] V. P. Dwivedi, C. K. Joshi, A. T. Luu, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. J. Mach. Learn. Res. , 24:43:1-43:48, 2023.
- [9] V. P. Dwivedi, A. T. Luu, T. Laurent, Y . Bengio, and X. Bresson. Graph neural networks with learnable structural and positional representations. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022 . OpenReview.net, 2022.
- [10] F. Errica, M. Podda, D. Bacciu, and A. Micheli. A fair comparison of graph neural networks for graph classification. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net, 2020.
- [11] M. Fey and J. E. Lenssen. Fast graph representation learning with pytorch geometric. CoRR , abs/1903.02428, 2019.
- [12] M. Fey, J. Yuen, and F. Weichert. Hierarchical inter-message passing for learning on molecular graphs. CoRR , abs/2006.12179, 2020.

- [13] H. Gao and S. Ji. Graph u-nets. In K. Chaudhuri and R. Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA , volume 97 of Proceedings of Machine Learning Research , pages 2083-2092. PMLR, 2019.
- [14] Z. Gao, C. Jiang, J. Zhang, X. Jiang, L. Li, P. Zhao, H. Yang, Y. Huang, and J. Li. Hierarchical graph learning for protein-protein interaction. 14(1):1093.
- [15] T. Gaudelet, B. Day, A. R. Jamasb, J. Soman, C. Regep, G. Liu, J. B. Hayter, R. Vickers, C. Roberts, J. Tang, et al. Utilising graph machine learning within drug discovery and development. arXiv preprint arXiv:2012.05716 , 2020.
- [16] M. Gönen and E. Alpaydin. Multiple kernel learning algorithms. J. Mach. Learn. Res. , 12:2211-2268, 2011.
- [17] D. Grattarola, D. Zambon, F. M. Bianchi, and C. Alippi. Understanding pooling in graph neural networks. IEEE Trans. Neural Networks Learn. Syst. , 35(2):2708-2718, 2024.
- [18] X. He, B. Hooi, T. Laurent, A. Perold, Y. LeCun, and X. Bresson. A generalization of vit/mlp-mixer to graphs. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , volume 202 of Proceedings of Machine Learning Research , pages 12724-12745. PMLR, 2023.
- [19] W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta, and J. Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , 2020.
- [20] W. Hu, X. Zhan, and M. Tong. Parsing netlists of integrated circuits from images via graph attention network. Sensors , 24(1):227, 2024.
- [21] S. Huang, Y. Song, J. Zhou, and Z. Lin. Tailoring self-attention for graph via rooted subtrees. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , 2023.
- [22] M. Jin, H. Y. Koh, Q. Wen, D. Zambon, C. Alippi, G. I. Webb, I. King, and S. Pan. A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection. CoRR , abs/2307.03759, 2023.
- [23] G. Karypis and V. Kumar. A fast and high quality multilevel scheme for partitioning irregular graphs. SIAM J. Sci. Comput. , 20(1):359-392, 1998.
- [24] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event , volume 119 of Proceedings of Machine Learning Research , pages 5156-5165. PMLR, 2020.
- [25] D. Kim and A. Oh. Translating subgraphs to nodes makes simple gnns strong and efficient for subgraph representation learning. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 . OpenReview.net, 2024.
- [26] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In Y. Bengio and Y. LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings , 2015.
- [27] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings . OpenReview.net, 2017.
- [28] D. Kreuzer, D. Beaini, W. L. Hamilton, V. Létourneau, and P. Tossou. Rethinking graph transformers with spectral attention. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual , pages 21618-21629, 2021.
- [29] J. Lee, I. Lee, and J. Kang. Self-attention graph pooling. In K. Chaudhuri and R. Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA , volume 97 of Proceedings of Machine Learning Research , pages 3734-3743. PMLR, 2019.
- [30] Y. Li, R. Yu, C. Shahabi, and Y. Liu. Diffusion convolutional recurrent neural network: Data-driven traffic forecasting. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings . OpenReview.net, 2018.
- [31] C. Liu, Y. Zhan, X. Ma, L. Ding, D. Tao, J. Wu, and W. Hu. Gapformer: Graph transformer with graph pooling for node classification. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China , pages 2196-2205. ijcai.org, 2023.

- [32] C. Liu, Y. Zhan, J. Wu, C. Li, B. Du, W. Hu, T. Liu, and D. Tao. Graph pooling for graph neural networks: Progress, challenges, and opportunities. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China , pages 6712-6722. ijcai.org, 2023.
- [33] N. Liu, S. Jian, D. Li, Y . Zhang, Z. Lai, and H. Xu. Hierarchical adaptive pooling by capturing high-order dependency for graph representation learning. IEEE Trans. Knowl. Data Eng. , 35(4):3952-3965, 2023.
- [34] D. P. P. Mesquita, A. H. S. Jr., and S. Kaski. Rethinking pooling in graph neural networks. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , 2020.
- [35] G. Mialon, D. Chen, M. Selosse, and J. Mairal. Graphit: Encoding graph structure in transformers. CoRR , abs/2106.05667, 2021.
- [36] E. Min, R. Chen, Y. Bian, T. Xu, K. Zhao, W. Huang, P. Zhao, J. Huang, S. Ananiadou, and Y. Rong. Transformer for graphs: An overview from architecture perspective. CoRR , abs/2202.08455, 2022.
- [37] F. Monti, F. Frasca, D. Eynard, D. Mannion, and M. M. Bronstein. Fake news detection on social media using geometric deep learning. arXiv preprint arXiv:1902.06673 , 2019.
- [38] C. Morris, N. M. Kriege, F. Bause, K. Kersting, P. Mutzel, and M. Neumann. Tudataset: A collection of benchmark datasets for learning with graphs. CoRR , abs/2007.08663, 2020.
- [39] A. Pal, C. Eksombatchai, Y. Zhou, B. Zhao, C. Rosenberg, and J. Leskovec. Pinnersage: Multi-modal user embedding framework for recommendations at pinterest. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining , pages 2311-2320, 2020.
- [40] Z. Qin, W. Sun, H. Deng, D. Li, Y. Wei, B. Lv, J. Yan, L. Kong, and Y. Zhong. cosformer: Rethinking softmax in attention. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022 . OpenReview.net, 2022.
- [41] L. Rampásek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, and D. Beaini. Recipe for a general, powerful, scalable graph transformer. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 , 2022.
- [42] E. Ranjan, S. Sanyal, and P. P. Talukdar. ASAP: adaptive structure aware pooling for learning hierarchical graph representations. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020 , pages 5470-5477. AAAI Press, 2020.
- [43] H. Shirzad, A. Velingker, B. Venkatachalam, D. J. Sutherland, and A. K. Sinop. Exphormer: Sparse transformers for graphs. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , volume 202 of Proceedings of Machine Learning Research , pages 31613-31632. PMLR, 2023.
- [44] S. Sonnenburg, G. Rätsch, C. Schäfer, and B. Schölkopf. Large scale multiple kernel learning. J. Mach. Learn. Res. , 7:1531-1565, 2006.
- [45] J. M. Stokes, K. Yang, K. Swanson, W. Jin, A. Cubillos-Ruiz, N. M. Donghia, C. R. MacNair, S. French, L. A. Carfrae, Z. Bloom-Ackermann, et al. A deep learning approach to antibiotic discovery. Cell , 180(4):688-702, 2020.
- [46] Y. H. Tsai, S. Bai, M. Yamada, L. Morency, and R. Salakhutdinov. Transformer dissection: An unified understanding for transformer's attention via the lens of kernel. In K. Inui, J. Jiang, V . Ng, and X. Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019 , pages 4343-4352. Association for Computational Linguistics, 2019.
- [47] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA , pages 5998-6008, 2017.
- [48] O. Vinyals, S. Bengio, and M. Kudlur. Order matters: Sequence to sequence for sets. In Y. Bengio and Y. LeCun, editors, 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings , 2016.
- [49] X. Wang and M. Zhang. How powerful are spectral graph neural networks. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA , volume 162 of Proceedings of Machine Learning Research , pages 23341-23362. PMLR, 2022.

- [50] J. Wu, X. Chen, K. Xu, and S. Li. Structural entropy guided graph hierarchical pooling. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA , volume 162 of Proceedings of Machine Learning Research , pages 24017-24030. PMLR, 2022.
- [51] Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and P. S. Yu. A comprehensive survey on graph neural networks. IEEE Trans. Neural Networks Learn. Syst. , 32(1):4-24, 2021.
- [52] C. Ying, T. Cai, S. Luo, S. Zheng, G. Ke, D. He, Y. Shen, and T. Liu. Do transformers really perform badly for graph representation? In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual , pages 28877-28888, 2021.
- [53] Z. Ying, J. You, C. Morris, X. Ren, W. L. Hamilton, and J. Leskovec. Hierarchical graph representation learning with differentiable pooling. In S. Bengio, H. M. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada , pages 4805-4815, 2018.
- [54] B. Yu, H. Yin, and Z. Zhu. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. In J. Lang, editor, Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI 2018, July 13-19, 2018, Stockholm, Sweden , pages 3634-3640. ijcai.org, 2018.
- [55] H. Yuan and S. Ji. Structpool: Structured graph pooling via conditional random fields. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net, 2020.
- [56] H. Zhang and M. Xu. Graph neural networks with multiple kernel ensemble attention. Knowl. Based Syst. , 229:107299, 2021.
- [57] M. Zhang, Z. Cui, M. Neumann, and Y. Chen. An end-to-end deep learning architecture for graph classification. In S. A. McIlraith and K. Q. Weinberger, editors, Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018 , pages 4438-4445. AAAI Press, 2018.
- [58] Z. Zhang, Q. Liu, Q. Hu, and C. Lee. Hierarchical graph transformer with adaptive node sampling. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 , 2022.

## A Efficient implementation of Node-to-Cluster Attention with Convex Linear Combination of Kernels

In this section, we will devise an efficient form for the Node-to-Cluster Attention with Convex Linear Combination of Kernels:

<!-- formula-not-decoded -->

We introduce the corresponding feature map: κ C ( Q i , K j ) = ⟨ ϕ ( Q i ) , ϕ ( K j ) ⟩ and κ N ( q i , k t ) = ⟨ ψ ( q i ) , ψ ( k t ) ⟩ . According to Prop.1 , we have Φ B ( { Q i , q i } ) = √ αϕ ( Q i ) ⊕ √ βψ ( q i ); Φ B ( { K j , k t } ) = √ αϕ ( K j ) ⊕ √ βψ ( k t ) . Thus we can rewrite Equation 18 as:

<!-- formula-not-decoded -->

where [ √ αϕ ( Q i ) ⊕ √ βψ ( q i ) ] is the weighted concatenation of the feature map of bi-level queries, while [ √ αϕ ( K j ) ⊕ √ βψ ( k t ) ] is the weighted concatenation of the feature map of bi-level keys.

Equation 19 allows us to implement N2C-Attn-L with message-passing framework, which is similar to the implementation that we have devised in subsection 3.2. We can first calculate the equivalent feature map of the bi-level query [ √ αϕ ( Q i ) ⊕ √ βψ ( q i ) ] for each cluster, and the equivalent feature map of the bi-level key [ √ αϕ ( K j ) ⊕ √ βψ ( k t ) ] for each node. Then we aggregate the keys of nodes within every cluster respectively to get ∑ t C tj [ √ αϕ ( K j ) ⊕ √ βψ ( k t ) ] v t and ∑ t C tj [ √ αϕ ( K j ) ⊕ √ βψ ( k t ) ] . After getting these two aggregated "messages", we perform a message passing according to the adjacency matrix of the coarsened graph A P . And finally, we unpack the aggregated information by calculating the dot product with the feature map of the bi-level query [ √ αϕ ( Q i ) ⊕ √ βψ ( q i ) ] T . In summary, by using the corresponding feature map and this cluster-level message propagation, we can achieve an implementation method for N2C-Attn-L with linear computational complexity.

## B Relationship between GraphViT and N2C-Attn mechanism

In this subsection, we present a detailed justification for why the attention mechanism used in GraphViT [18] can be regarded as a special case of our proposed N2C-Attn. Please note that in this section, we mainly focus on the similarities and differences in attention computation between GraphViT and N2C-Attn. While GraphViT also enhances its performance and expressive power through the use of various positional encodings, residual connections, and normalization techniques, these modules are not the primary focus of this section and therefore will not be discussed.

GraphViT first uses Metis to partition the graph (with overlapping nodes), and then performs average pooling within each partition. We denote the embedding of the clusters as X P , GraphViT then performs the Graph-based Hadamard Attention: G-Hadamard-Attn( X P ) to capture the dependencies between the clusters, where G-Hadamard-Attn is defined as ( A P ⊙ softmax ( QK T √ d )) V .

We denote the node set if the p -th cluster as V p , then the average pooling process can be written as: x p = 1 |V p | ∑ i ∈V p x i,p , where x i,p is the embedding of the i -th node within the p -th cluster, and x p is the embedding of the p -th cluster. And we denote the connected clusters (cluster-wise neighbors) of

the i -th cluster as N i . Then the Hadamard Attention used in GraphViT can be written as:

<!-- formula-not-decoded -->

where κ ( Q i , K j ) = exp ( Q T i K j √ d ) . Q and K can be seen as cluster-level queries and keys.

Now, we check the corresponding form of N2C-Attn in this case. Since we use the Metis graph partitioning algorithm, which divides the graph into several separate subgraphs and produces a hard cluster Assignment Matrix:

<!-- formula-not-decoded -->

With the help of Equation 21, we can rewrite Equation 11 as:

<!-- formula-not-decoded -->

Comparing Equation 20 and Equation 22, if we set κ C ( Q i , K j ) = κ ( Q i , K j ) = exp ( Q T i K j √ d ) , then the only difference between these two formulas lies in the coefficient before v t . In fact, Equation 20 can be seen as a special case of Equation 22 where κ N ( q i , k t ) = 1 |V j | .

From our analysis above, the difference between the cluster-level attention used in GraphViT and N2C-Attn is as follows: the former assigns the same weight to all nodes within each cluster = 1 |V j | , while the latter allows different attention weights for the nodes within each cluster and uses a node-level kernel κ N to learn these weights.

For simplicity, we only prove that the cluster-level attention used in GraphViT can be considered a special case of Node-to-Cluster Attention with Tensor Product of Kernels (N2C-Attn-T) in this section. In fact, we can similarly argue that the cluster-level attention used in GraphViT is a special case of Node-to-Cluster Attention with Convex Linear Combination of Kernels (N2C-Attn-L). Their most important difference is that GraphViT still follows the graph coarsening pipeline and only uses cluster-level kernels for attention calculation, whereas N2C-Attn integrates both cluster-level and node-level kernels to perform the attention computation.

## C Proof of Proposition 1

## C.1 Proof of Equation 16

In this subsection, we offer a detailed proof for Equation 16.

If Q i , K j are d C -dimensional vectors from the cluster-level space X C and q i , k t are d N -dimensional vectors from the node-level space X N . Consider two kernel functions κ C , κ N from the clusterlevel space X C and the node-level space X N respectively, with the corresponding feature map: κ C ( Q i , K j ) = ⟨ ϕ ( Q i ) , ϕ ( K j ) ⟩ and κ N ( q i , k t ) = ⟨ ψ ( q i ) , ψ ( k t ) ⟩ .

Now we consider the case of Node-to-Cluster Attention with Tensor Product of Kernels, where we use the product of the kernels κ C , κ N to construct the bi-level kernel: κ B ( { Q i , q i } , { K j , k t } ) = κ C ( Q i , K j ) κ N ( q i , k t ) where κ B is the bi-level kernel from the tensor product of two original spaces

<!-- formula-not-decoded -->

Thus, we can construct the following feature map:

<!-- formula-not-decoded -->

where Φ B is a d C d N × 1 feature map. For each pair ( i, j ) , Φ B , ( i,j ) ( u, v ) = ϕ i ( u ) ψ j ( v ) , where 1 ≤ i ≤ d C and 1 ≤ j ≤ d N . This composite feature map Φ B corresponds to the kernel κ B .

With the composite feature map Φ B , we can rewrite Equation 23 as:

<!-- formula-not-decoded -->

which proves Equation 16.

## C.2 Proof of Equation 17

In this subsection, we offer a detailed proof for the Equation 17.

Again, we use κ C , κ N to denote the two kernel functions from the cluster-level space X C and the nodelevel space X N respectively, with the corresponding feature map: κ C ( Q i , K j ) = ⟨ ϕ ( Q i ) , ϕ ( K j ) ⟩ and κ N ( q i , k t ) = ⟨ ψ ( q i ) , ψ ( k t ) ⟩ .

Now we consider the case of Node-to-Cluster Attention with Convex Linear Combination of Kernels, where we use the convex linear combination of kernels κ C , κ N to construct the bi-level kernel: κ B ( { Q i , q i } , { K j , k t } ) = ακ C ( Q i , K j ) + βκ N ( q i , k t ) where α, β ≥ 0 and α + β = 1 . α and β are coefficients that balance the contribution of each kernel. Then, for all ( Q i , K j ) ∈ X 2 C and ( q i , k t ) ∈ X 2 N , we have:

<!-- formula-not-decoded -->

Thus, we can construct the following feature map:

<!-- formula-not-decoded -->

where Φ B is a weighted concatenation of the feature maps ϕ and ψ . In other words, if the feature maps ϕ and ψ have d C and d N coordinates respectively, then Φ B has d C + d N coordinates; for any pair ( u, v ) ∈ X C × X N , the first d C coordinates of Φ B ( u, v ) are √ αϕ 1 ( u ) , √ αϕ 2 ( u ) , . . . , √ αϕ d C ( u ) and the remaining d N coordinates of Φ B ( u, v ) are √ βψ 1 ( v ) , √ βψ 2 ( v ) , . . . , √ βψ d N ( v ) .

With the composite feature map Φ B , we can rewrite Equation 26 as:

<!-- formula-not-decoded -->

which proves Equation 17.

Table 3: Summary statistics of datasets

| Dataset     | #Graphs   |   Avg. #Nodes |   Avg. #Edges | Task          | Metric   |
|----------|----------|----------|----------|----------|----------|
| IMDB-BINARY | 1000      |         19.8  |          96.5 | binary classif.  | Accuracy |
| IMDB-MULTI  | 1500      |         13    |          66   | 3-class classif. | Accuracy |
| COLLAB      | 5000      |         74.49 |        2457.8 | 3-class classif. | Accuracy |
| MUTAG       | 188       |         17.93 |          19.8 | binary classif.  | Accuracy |
| PROTEINS    | 1113      |         39.1  |          72.8 | binary classif.  | Accuracy |
| D&D         | 1178      |        284.3  |         715.7 | binary classif.  | Accuracy |
| ZINC        | 12,000    |         23.2  |          24.9 | regression       | MAE      |
| MolHIV      | 41,127    |         25.5  |          54.9 | binary classif.  | ROCAUC   |

## D More Details of Cluster-GT

Positinal Encoding Positional encoding in graphs plays a crucial role in providing spatial context to nodes. Following [18], we adopt two different strategies: random-walk structural encoding (RWSE) [9] and Laplacian eigenvector encodings [8]. We concatenate the positional encoding with node features as the model input. Additionally, we have tried the patch-wise positional encoding proposed by [18] and set it as a hyperparameter for the Cluster-GT architecture.

Node-wise Convolution In our Cluster-GT framework, we have experimented with incorporating GCN [27] and GIN [49] as the node-wise convolution modules, setting the choice between them as a hyperparameter to optimize performance. GIN is particularly notable for its ability to improve model expressiveness, which is crucial in distinguishing different graph structures. Moreover, the design of our Cluster-GT framework is modular, allowing the node-wise convolution module to be freely replaced by any other method.

Bi-level Queries and Keys After the node clustering assignment, we obtain various node clusters. When generating cluster-level queries or keys, we have tried two options: 1) using DeepSets, 2) aggregating the queries and keys within a cluster. We set these two options as a hyperparameter for the model architecture. In our experiments, we find that these two options have similar performance. Additionally, we find that having the same cluster-level and node-level queries does not affect the model's performance, as long as the bi-level keys remain different. Therefore, we treat whether the queries at the two levels are identical as an optional hyperparameter.

Other details Just like other Transformer structures [47], we incorporate residual connections between the attention layers and MLP, along with layer normalization to enhance training stability. In N2C-Attn, after outputting representations at each cluster level, we ultimately obtain a graph-level representation through average pooling. For N2C-Attn-T and N2C-Attn-L, we use the efficient algorithms introduced in subsection 3.2 and Appendix A in our implementation, respectively.

## E Dataset Information

Our experiments employ a variety of benchmark datasets commonly used in graph learning research. These datasets are chosen for their distinct characteristics and relevance in testing graph classification algorithms. The summary statistics of datasets are shown in Table 3.

## E.1 Datasets used in subsection 5.1

We organize the datasets into categories based on their application domains. · Social Networks : IMDB-BINARY and IMDB-MULTI are derived from the Internet Movie Database (IMDB) and include graphs representing the ego-networks of different movie genres. In IMDB-BINARY, each graph is labeled as either Action or Romance. IMDB-MULTI includes three genres: Comedy, Romance, or Sci-Fi. Nodes represent actors, and edges are placed between nodes if the actors have co-starred in a movie. These datasets are used to evaluate the capability of graph classification models in social network analysis. · Scientific Collaboration Networks : COLLAB represents the ego-collaboration networks of researchers from three fields: High Energy Physics, Condensed Matter

Physics, or Astrophysics. Nodes represent scientists, and edges are drawn between scientists who have co-authored papers. COLLAB tests the model's ability to recognize different collaborative patterns in scientific domains. · Biochemical Molecules : For the biochemical domain, we have chosen three datasets: MUTAG, PROTEINS and D&amp;D. MUTAG comprises 188 chemical compounds represented as graphs, where nodes symbolize atoms and edges denote chemical bonds. Each graph is labeled based on its mutagenic effect on bacteria, serving as a benchmark for bioinformatics applications in predicting chemical properties. PROTEINS consists of protein structures, where each graph corresponds to a protein, nodes to secondary structure elements (SSEs), and edges connect nodes if they are adjacent either in the amino acid sequence or in 3D space. Proteins are classified into enzymes or non-enzymes, providing a basis for studying complex biological structures. D&amp;D includes protein structures with nodes representing amino acids and edges based on spatial closeness. Graphs are labeled according to whether the protein is associated with a disease, challenging the algorithms to decode intricate biological interactions. These datasets collectively provide a comprehensive suite for evaluating across different real-world scenarios.

## E.2 Datasets used in subsection 5.2

Here, we select two datasets used in biochemical molecule and drug discovery research: · Biochemical Molecules : The ZINC dataset is a collection of chemical compounds that are representative of real-world molecular data. This dataset is utilized predominantly for regression tasks such as predicting the scalar measure of molecular solubility. Each compound is represented as a graph where nodes are atoms and edges are chemical bonds, making it crucial for testing the accuracy of models in predicting molecular attributes. · Drug Discovery : The MolHIV dataset is part of the MoleculeNet suite, specifically designed for binary classification tasks related to HIV drug activity. Graphs in this dataset represent molecular structures where nodes are atoms and edges correspond to bonds. The task is to predict whether a molecule inhibits the HIV virus, which is vital for speeding up the discovery of potential therapeutic agents.

## F Implementation Details

## F.1 Introduction of baselines

Baselines for subsection 5.1 Initially, we utilize two well-known GNN architectures for comparison: GCN [27] and GIN [49]. Subsequently, we incorporate six hierarchical pooling approaches as baselines: DiffPool [53], SAGPool(H) [29], TopKPool [13], ASAP [42], MinCutPool [4] and SEP [50]. In addition to these hierarchical pooling methods, considerable attention has been given to global pooling strategies for graph classification. Therefore, we also evaluate five global pooling techniques: Set2Se [48], SortPool [57], SAGPool(G) [29], StructPool [55], and GMT [2] for comparative analysis.

Baselines for subsection 5.2 Next, we compare Cluster-GT against popular Graph Transformers with SOTA results, including GT [8], GraphiT [35], Graphormer [52], GPS [41], SAN+LapPE [28], SAN+RWSE [28]. These models represent cutting-edge advancements in graph neural network technology, each introducing unique methods to handle graph-structured data effectively.

## F.2 Experimental Details

The model is implemented using PyTorch and PyG [11]. Experiments are conducted on NVIDIA RTX 3090 GPUs. For optimization, the Adam [26] optimizer is utilized, adhering to the default settings of β 1 = 0 . 9 , β 2 = 0 . 999 , and ε = 1 e -8 .

Experimental Details of subsection 5.1 The model's performance is assessed using a 10-fold cross-validation approach, with dataset splits adhering to the standard established training/test partitions [50]. Moreover, 10 percent of the training data is allocated as validation data to ensure a fair comparison, as per [10]. The initial feature inputs are aligned with the fair comparison setting [10]. An early stopping criterion is implemented, halting training if there is no improvement in validation loss over 50 epochs. The training process is capped at a maximum of 500 epochs. The average performance on the test sets is reported after conducting the experiments 10 times.

Experimental Details of subsection 5.2 Each experiment is run with four different seeds, and the averaged results are reported from the epoch that achieved the best validation metric. We use a batch size of 64. We utilize a standard train/validation/test dataset split following [18].

## F.3 Attention strtegies compared in subsection 5.4

In this section, we provide a detailed introduction to the four different node-to-clustering attention strategies: 'N2C-Attn-T','N2C-Attn-L','Cluster-Level-Attn','Node-Level-Attn', compared in the ablation study. Note that for these four variants, we simply replaced the attention mechanism in Cluster-GT without modifying any other parts of the model.

- N2C-Attn-T :
- N2C-Attn-L :

<!-- formula-not-decoded -->

- Cluster-Level-Attn :

<!-- formula-not-decoded -->

- Node-Level-Attn :

<!-- formula-not-decoded -->

It is worth noting that Cluster-Level-Attn is a special case of N2C-Attn-L when the cluster-level coefficient α = 1 , while Node-Level-Attn is also a special case of N2C-Attn-L when the node-level coefficient β = 1 -α = 1 .

## G On the Naming Issue Between RWSE and RWPE

We use the term "Random Walk Positional Encoding (RWPE)" in this work. In this section, we will briefly discuss the naming issue between RWSE and RWPE. In the original paper that introduced RWPE [9], the term "Random Walk Positional Encoding (RWPE)" was proposed. This paper utilized the self-landing probability of nodes in a random walk to capture neighborhood structural information.

Subsequently, an influential work in the graph transformer domain [41] made a clear distinction between two types of encodings for structure and position, naming them Positional Encoding (PE) and Structural Encoding (SE). Positional encodings are intended to provide an understanding of a node's position within the graph, while structural encodings aim to embed the structure of graphs or subgraphs, enhancing the expressivity and generalizability of GNNs.

Interestingly, [41] argues that the Random Walk Positional Encoding (RWPE) proposed in [9] actually serves as a Structural Encoding (SE). Based on our investigation, it is likely that [41] began using the term RWSE instead of RWPE. Many subsequent studies, likely influenced by [41], such as [43, 18], have also adopted RWSE over RWPE. In our work, we also use RWSE, the widely accepted term.

In conclusion, both RWSE and RWPE are widely recognized and used interchangeably in the academic community to refer to the same PE method (diagonal of the k -steps random-walk matrix).

## H Limitations

We employ Metis in conjunction with the N2C-Attn module for Cluster-GT. However, Metis is a non-learnable graph partitioning algorithm that provides hard assignments for node clustering.

<!-- formula-not-decoded -->

This poses a limitation as it restricts the flexibility of node groupings, potentially impacting the adaptability of the model in dynamic or complex network scenarios. For future enhancements, exploring combinations of N2C-Attn with other learnable graph partitioning algorithms capable of generating soft assignments may be an interesting direction. Additionally, aspects such as robustness and explainability also warrant further investigation to ensure reliability in real-world settings.

## I Potential Impacts

The proposed Node-to-Cluster Attention mechanism introduces a novel approach to information exchange between node clusters. Our research underscores the importance of incorporating diverse strategies for interactions at both the node and cluster levels. This perspective can be integrated with many existing node clustering-based graph learning methods, enhancing their efficacy and adaptability. Moreover, our experimental validations reveal that the method of interaction between clusters significantly impacts model performance. While current research primarily focuses on how to partition clusters within graphs, our findings suggest a promising direction for future work: optimizing the methods of interaction between clusters. Such advancements could unlock new possibilities for cluster-level graph learning, potentially leading to more robust and sophisticated models that better capture the complexities of large-scale networks.

##