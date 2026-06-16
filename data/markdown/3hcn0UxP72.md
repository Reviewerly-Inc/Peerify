## Topological obstruction to the training of shallow ReLU neural networks

## Marco Nurisso

Politecnico di Torino &amp; CENTAI Institute

Torino, 10100 - ITALY

marco.nurisso@polito.it

## Pierrick Leroy

Politecnico di Torino Torino, 10100 - ITALY pierrick.leroy@polito.it

## Francesco Vaccarino

Politecnico di Torino Torino, 10100 - ITALY

francesco.vaccarino@polito.it

## Abstract

Studying the interplay between the geometry of the loss landscape and the optimization trajectories of simple neural networks is a fundamental step for understanding their behavior in more complex settings. This paper reveals the presence of topological obstruction in the loss landscape of shallow ReLU neural networks trained using gradient flow. We discuss how the homogeneous nature of the ReLU activation function constrains the training trajectories to lie on a product of quadric hypersurfaces whose shape depends on the particular initialization of the network's parameters. When the neural network's output is a single scalar, we prove that these quadrics can have multiple connected components, limiting the set of reachable parameters during training. We analytically compute the number of these components and discuss the possibility of mapping one to the other through neuron rescaling and permutation. In this simple setting, we find that the non-connectedness results in a topological obstruction, which, depending on the initialization, can make the global optimum unreachable. We validate this result with numerical experiments.

## 1 Introduction

Training a neural network consists of navigating the complex geometry of the loss landscape to reach one of its deepest valleys. Gradient descent and its variants are, by far, the most commonly used algorithms to perform this task. While technically correct, the standard picture of the parameter space as Euclidean space with the trajectory rolling down the loss's surface in the steepest direction towards a minimum is slightly misleading because different choices of parameters can be observationally equivalent i.e. encode the same function [10]. The observational equivalence of parameters shape the loss landscape by imposing specific geometric structures on the parameter space. Minima are not isolated points but high-dimensional manifolds with complex geometry [17, 9, 42] and the loss function's gradients and Hessian are constrained to obey some specific laws [46, 27]. Gradient-based optimization methods, where the parameters are updated by performing discrete steps in the gradient's direction, are thus very much dependent on the symmetry-induced geometry [12, 29].

In this work, we provide a topological perspective on the constraints induced by some groups of network symmetries on the optimization trajectories. Topology is a field of mathematics that studies the properties of a space that are preserved under continuous deformations. Our main goal is to find and quantify in topological terms the impossibility of the training trajectories to freely explore the parameter space and get from any initialization to an optimal parameter. This idea is formalized in

the topological notion of connectedness and, in particular, with the 0-th Betti number , which counts the number of connected components the space is composed of. The presence, or the absence, of topological obstructions in the parameter space does not depend on the particular loss function or the training data but is intrinsic to the interplay between the geometry and the topology of the parameter space under the action of groups of symmetries inducing observationally equivalent networks.

Main contributions. Our main contributions are the following.

1. We find that, for two-layer neural networks, the gradient flow trajectories lie on an invariant set, which can be factored as the product of quadric hypersurfaces.
2. We analytically compute its Betti numbers, i.e., the number of connected components, holes, and higher-dimensional cavities.
3. We find that the invariant set can be disconnected when the network's output dimension is 1, leading to a clear topological obstruction.
4. We find that the obstruction is caused by 'pathological' neurons that cannot change the sign of their output weights when trained with gradient flow.
5. We discuss the relation between the invariant set and the network's symmetries, finding that if we consider permutations, the number of effective connected components scales linearly in the number of pathological neurons.
6. We perform numerical validations on controlled toy scenarios, displaying the effect of obstruction in practice.

## 2 Related work

A large body of work studies gradient flow and gradient descent optimization of one hidden layer networks with homogeneous activations. Convergence properties have been found for wide networks [37, 43] with bounded density at initialization [31]. The implicit regularization provided is studied under various assumptions on: orthogonal input data [4], initialization scale [30, 4], wide (overparameterized regime) and infinitely wide [6], linearly separable data [30, 45]. Deeper linear networks [24] have also been studied.

These works focus on proving convergence and understanding which (optimal) solution is found, whereas our work investigates the shape of the optimization space and focuses on cases where the optimum might not be reachable from a given initialization.

Closer to our work, Safran et al. [39] studies two-layer ReLU binary classifiers with single input and output, counting the number of their piecewise-linear components after training. Eberle et al. [13] focuses on the differential challenge posed by the ReLU activation function and studies properties like the uniqueness of the solution of a gradient flow differential equation for a given initialization.

ReLU activation is a nonnegative homogeneous function, meaning that particular weight rescalings do not change the neural network's function. This is at the heart of the counterargument to flatness measures made by Dinh et al. [10], which shows that the Hessian eigenvalues can be made arbitrarily large in this way. Neyshabur et al. [34] explores the effect such rescalings can have on the gradient, proposing a rescaling-invariant regularization, and Pittorino et al. [36] employs them to define invariant flatness measures. Generally speaking, neural networks possess symmetries [20], and symmetries influence the geometry of training. Du et al. [12] studies how symmetry leads ReLU networks to automatically balance the neurons' weights. Kunin et al. [27], Zhao et al. [51] study how it constrains the gradient and Hessian matrix, leading to conservation laws w.r.t. gradient flow and Tanaka et al. [46] leverages it to propose a network pruning scheme. Ziyin [52] studies general mirror-reflect symmetries of the loss function and their effect on the weights of the trained network. Other conserved quantities stem from batch normalization's scale invariance [23, 47]. The transition from gradient flow to finite step size gradient descent breaks the conservation laws, resulting in altered trajectories [14, 2, 27, 44].

Numerous works have explored the geometry and topology of the loss landscape to obtain insight into a neural network's training behavior. Motivated by the striking experimental observation that low loss points can be connected by simple curves [11, 18] or line segments [40, 16, 15], a large body of literature tries to understand this phenomenon of mode connectivity under the topological lens of the connectedness of the loss function's sublevel sets [17, 35, 26], especially for overparameterized neural networks [9, 8, 42]. Another line of work approaches the connectivity of minima from another point of view, studying the presence [50, 38, 48] or absence [28] of spurious minima, i.e. minima

which are not global. Bucarelli et al. [5] analytically derives bounds on the sum of the Betti numbers of the loss landscape's sublevel set. Topological data analysis methods have also been exploited to numerically study the shape of the loss landscape [1, 22].

## 3 Setup and preliminaries

## 3.1 One-hidden layer neural network

Unless otherwise stated, all vectors are column vectors, that is, x = ( x 1 , . . . , x d ) ⊺ ∈ R d ≅ R d × 1 . Let us consider a two-layer neural network f (⋅ , θ ) ∶ R d → R e specified by the function

<!-- formula-not-decoded -->

where x ∈ R d is the input, θ = ( W ( 1 ) , W ( 2 ) ) with W ( 1 ) ∈ R l × d and W ( 2 ) ∈ R e × l are the parameters, σ ∶ R → R is the component-wise activation function and l is the number of neurons in the hidden layer. Notice that we consider a network with no biases, as it allows us a discussion with lighter notation. The case with biases is discussed in Appendix E.

In this work, following [12], we focus on the case where σ is homogeneous , namely σ ( x ) = σ ′ ( x ) ⋅ x for every x and for every element of the sub-differential σ ′ ( x ) if σ is non-differentiable at x . The commonly used ReLU ( σ ( z ) = max { z, 0 } ) and Leaky ReLU ( σ ( z ) = max { z, γ } with 0 ≤ γ ≤ 1 ) activation functions satisfy this property.

We call parameter space the vector space Θ = { θ = ( W ( 1 ) , W ( 2 ) ) ∣ W ( 1 ) ∈ R l × d , W ( 2 ) ∈ R e × l } .

It will also be convenient to examine the single hidden neurons and their associated parameters for the following discussions.

Proposition 1. For the two-layer neural network defined in Equation (1) . Let k = 1 , . . . , l , let ( e 11 , e 12 , . . . , e ll ) be the canonical basis of R l × l and Θ k = { θ k = ( e kk W ( 1 ) , W ( 2 ) e kk ) ∣ ( W ( 1 ) , W ( 2 ) ) ∈ Θ } ⊂ Θ , then Θ = Θ 1 ⊕/uni22EF⊕ Θ l .

Details of the proof are provided in Appendix A. Fixing k ∈ { 1 , . . . , l } , we can consider Θ k as the parameter space of the k -th hidden neuron, which consists of the inputs and output weights of neuron k , namely the rows and columns of W ( 1 ) and W ( 2 ) , respectively. For simplicity, when we work in Θ k , we write W ( 1 ) k ∶= e kk W ( 1 ) and W ( 2 ) k ∶= W ( 2 ) e kk . Interestingly, the decomposition of Proposition 1 only holds for two-layer neural networks and will be crucial to the formulations of this paper's results.

## 3.2 Symmetries and observationally equivalent networks

It is well known that the properties of the activation function heavily influence the geometry of the parameter space Θ . The activation function's commutativity with some classes of transformations can result in the latter having no effect on the function implemented by the neural network. This means that, in general, the mapping from the parameter space to the hypothesis class of functions is not injective. Following the terminology in Dinh et al. [10], we say that two parameters θ 1 , θ 2 ∈ Θ are observationally equivalent , if they encode the same function f (⋅ ; θ 1 ) = f (⋅ , θ 2 ) and write θ 1 ∼ θ 2 .

In the case of homogeneous activations (ReLU or Leaky ReLU), we describe two kinds of transformations that send a parameter θ into an observationally equivalent one.

Neuron rescaling. The input weights of a hidden neuron can be rescaled by a positive scalar α &gt; 0 provided that its output weights are rescaled by the inverse α -1 (top panel of Figure 1a). We formalize this as the action of the group R + of positive real numbers on Θ k :

<!-- formula-not-decoded -->

This action can be naturally extended to the space of all parameters by considering the possibility of rescaling all hidden neurons simultaneously by different factors. If α = ( α 1 , . . . , α l ) ∈ R l +

<!-- formula-not-decoded -->

Figure 1: a. Depiction of the two group actions acting on the space of the network's parameters: the neuron rescaling of Equation (2) (top) and the neuron permutation of Equation (4) (bottom). b. Depiction of the geometry of the parameter space induced by the rescaling invariance of ReLU networks. The dotted lines denote the orbits T ( θ ) while the solid lines represent the invariant sets H( c ) associated with θ and the one associated with its rescaled version θ ′ . Notice how the gradient of the loss g ( θ ) is tangent to H( c ) and orthogonal to T ( θ ) .

<!-- image -->

Given that σ ( az ) = aσ ( z ) when a ∈ R + , we see how θ ∼ T α ( θ ) .

We write T ( θ ) to denote the orbit of a parameter θ under the action of T , i.e. the set of all parameters obtained from θ by arbitrarily rescaling the neurons T ( θ ) = { T α ( θ ) ∶ α ∈ R l + } .

Permutations of the neurons. Besides rescaling, we can obtain an observationally equivalent network by permuting the hidden neurons in such a way as to preserve their input and output weights (bottom panel of Figure 1a).

Given the symmetric group on l elements S l of the permutations of { 1 , . . . , l } , we write the action

<!-- formula-not-decoded -->

where R π is the l × l row-permutation matrix associated to the permutation π .

Given that the activation function σ is applied component-wise, we have that it commutes with R π , namely

<!-- formula-not-decoded -->

Having defined these two actions, we say that θ and θ ′ are observationally equivalent by rescalings and permutations if θ ′ can be obtained from θ by a finite sequence of actions of T and P or, equivalently thanks to Lemma 3 in the Appendix, if there exists a rescaling α and a permutation π such that θ ′ = P π ○ T α ( θ ) . In this case, we write θ rp ∼ θ ′ .

## 3.3 Conserved quantities and the invariant hyperquadrics

The presence of symmetries in the neural network's parameter-function map results in a specific geometric structure in the loss landscape. Let indeed D = {( x i , y i ) ∈ R d × R e } N i = 1 be a training set of N input-output pairs and fix a loss function L ∶ Θ → R which depends on the parameters only through the output of the neural network (1), that is

<!-- formula-not-decoded -->

where ℓ ∶ R e × R e → R is differentiable. In this work, as empirical risk minimization, we consider the continuous time version of the gradient descent (GD) algorithm (with learning rate h &gt; 0 )

<!-- formula-not-decoded -->

named gradient flow (GF), and defined as

<!-- formula-not-decoded -->

where ∇ θ L ( θ ( t )) is the Clarke sub-differential [7] which takes into account the parameters θ where L ( θ ) is non-differentiable. Given that the loss function L depends on the parameters only through f , its value at θ must be constant over the orbit T ( θ ) . This, together with the fact that the gradient of a differentiable function at a point is orthogonal to the level set at that point, means that

<!-- formula-not-decoded -->

at any parameter θ where L ( θ ) is differentiable, as represented in Figure 1b. This orthogonality condition constrains the possible values of the gradient and, by extension, the possible gradient flow trajectories. In particular, as proven in Liang et al. [29], Tanaka et al. [46], Equation (8) is equivalent to

<!-- formula-not-decoded -->

For convenience of notation, we define, for k = 1 , . . . , l , the following bilinear forms on Θ , which help us describe the geometry induced by the rescaling symmetry. If θ = ( W ( 1 ) , W ( 2 ) ) and η = ( V ( 1 ) , V ( 2 ) ) , we define

<!-- formula-not-decoded -->

which, notice, only depends on the k -th row of W ( 1 ) and k -th column of W ( 2 ) meaning that we can equivalently see it as a bilinear form on Θ k . Θ k , together with ⟪⋅ , ⋅⟫ k is a pseudo-Euclidean space .

With the notation given by Equation (10), we see that Equation (9) can be simply rewritten as ⟪ θ, g ( θ )⟫ k = 0 for every neuron k . This condition, akin to orthogonality w.r.t. the bilinear form of Equation (10), implies that, under gradient flow optimization,

<!-- formula-not-decoded -->

This result, first obtained in Saxe et al. [41] for linear networks and discussed in Du et al. [12], Liang et al. [29], Kunin et al. [27], tells us that the rescaling symmetry results in the quantities ⟪ θ, θ ⟫ k being conserved. This means that the difference between the Euclidean norm of the inputs and the outputs is constant for each neuron throughout the GF training trajectory. Moreover, under the condition of homogeneity of the activation function, Du et al. [12] proves that Equation (11) holds even at non-differentiable points of L and in the case of multiple layers.

Invariant sets. Assume that at the initialization θ 0 we have ⟪ θ 0 , θ 0 ⟫ k = c k , for all k , then Equation (11) implies that the GF trajectory will lie on the set characterized by the system of equations ⟪ θ, θ ⟫ k = c k for k = 1 , . . . , l . This subset is mapped to itself under the GF dynamics by Equation (11) (see Figure 1b) and constitutes the main object of our study.

Definition 1 (Invariant set) . Given c = ( c 1 , . . . , c l ) , we call invariant set the subset H( c ) ⊆ Θ given by the equations ⟪ θ, θ ⟫ k = c k ∀ k = 1 , . . . , l .

If we look at each single equation (i.e. to each hidden neuron), we see that Equation (11) can be written as

<!-- formula-not-decoded -->

which corresponds to a hyperquadric (or quadric hypersurface) in Θ k . We denote with Q( c k ) ⊆ Θ k this hypersurface and call it the invariant hyperquadric associated to the k -th hidden neuron.

Here c k ∈ R takes the role of a label associated with the k -th hidden neuron, which, we see in the next section, plays a key role in specifying the shape of Q( c k ) . Figure 2a shows how, for d = 2 and e = 1 , Q( c k ) is an hyperboloid with 1 sheet (connected) if c k &gt; 0 and 2 sheets if c k &lt; 0 .

## 4 Topology of the invariant set

As we discussed above, Equation (11) tells us that gradient flow trajectories can't explore the whole space Θ but are constrained to lie on the invariant set H( c ) . The values of c , in turn, depend on the initialization and, we see from Equation (12), quantify the balance between the norms of input and output weights in every hidden neuron.

The goal of this section is to provide a topological characterization of H( c ) that can tell us something about the presence or absence of fundamental obstructions to the network's training process. With obstruction, we mean the impossibility of a GF trajectory to travel freely from one point to the other in H( c ) . We refer the reader to Appendix B for an essential overview of some of the topological concepts that we rely on in the next paragraphs.

Counting high-dimensional holes. Our topological characterization will be framed using Betti numbers . Betti numbers are well-known topological invariants given by a sequence of natural numbers that intuitively encode the number of higher-dimensional holes and cavities present in space. In particular, the 0-th Betti number of a space X , β 0 ( X ) corresponds to the number of connected components of X and thus will be fundamental for our goal of identifying obstructions.

The invariant set H( c ) is given as the set of solutions of l polynomial equations of degree 2 sharing no variables. Furthermore, in the setting of two-layer neural networks, we can leverage the fact that the parameter space can be decomposed into the parameter spaces of the hidden neurons. This, in turn, allows us to decompose the invariant set as the product of the neurons' invariant hyperquadrics, greatly simplifying our study.

Lemma 1. In a two-layer ReLU neural network, the invariant set H( c ) is homeomorphic to the Cartesian product of the hidden neurons' invariant hyperquadrics, that is

<!-- formula-not-decoded -->

Lemma 1 tells us that we can understand the topology of H( c ) by studying independently its factors. Moreover, the hyperquadrics we encounter here are well-studied objects for which the next proposition (proven in Appendix D.2) gives a topological characterization.

Proposition 2. If c k &gt; 0 , Q( c k ) is a topological manifold homeomorphic to R e × S d -1 . If c k &lt; 0 , Q( c k ) is a topological manifold homeomorphic to R d × S e -1 . If c k = 0 , Q( 0 ) is a contractible space.

Leveraging the decomposition of Lemma 1 and the characterization of the factors given by Proposition 2, we can explicitly compute all the Betti numbers of the invariant set. We give the next result in terms of the Poincaré polynomial of H( c ) , namely the polynomial whose coefficients are the Betti numbers (see Appendix B).

Theorem 1. Let l + , l -, l 0 be the number of positive, negative, and zero components of c , respectively. The Poincaré polynomial of H( c ) is given by

<!-- formula-not-decoded -->

This result, which is proven in Appendix D.3, contains a wealth of topological information as it gives us the exact number of holes and cavities of any order, depending on the network's hyperparameters ( d, e ) and initialization ( l + , l -). In the rest of this work, we focus only on the 0-th Betti number as the non-connectedness of H( c ) provides a clear obstruction to the GF trajectories.

Connectedness of the invariant set. With regard to the connectedness of H( c ) , we can leverage Theorem 1 to obtain the exact number of connected components.

Corollary 1. The 0 -th Betti number β 0 of H( c ) , corresponding to the number of its connected components, is given by

<!-- formula-not-decoded -->

Proof. This can be directly obtained from the coefficient of degree 0 of the Poincaré polynomial obtained through Theorem 1.

Wi1)*

0к°

CK &lt; 0

W (2)

1k 4

W (2)

Wk2)

(Wöl)°+ (WK2)3-(W12)º=ck

CK &gt; 0

Figure 2: a. The invariant hyperquadric Q( c k ) of a neuron with two inputs ( d = 2 ) and one output ( e = 1 ) in the cases where c k &lt; 0 (left) and c k &gt; 0 (right). b. Depiction of the invariant set H( c ) in the case where l -= 2 so that there are 2 l -= 4 connected components. C ±∓ denotes the connected component such that s = (± 1 , ∓ 1 ) . The blue lines separate the different effective components of H( c ) .

<!-- image -->

What we see in Equation (15) is that in most cases, the invariant set is connected, and gradient flow has no topological limitations in exploring the whole of H( c ) . Instead, when the hidden neurons have only one input or only one output, the space is fragmented into several components whose number scales exponentially in l + or l -, respectively.

Let us focus on the more interesting case where d &gt; 1 and e = 1 .

Corollary 2. If the output of a two-layer ReLU neural network is a single scalar e = 1 , its input has dimension d &gt; 1 , and the initial parameter θ 0 is such that ⟪ θ 0 , θ 0 ⟫ k &lt; 0 for l -&gt; 0 hidden neurons, then the set H( c ) is disconnected and has 2 l -connected components.

This means that neurons initialized with the norm of their outgoing weight strictly greater than their incoming weights' norm are responsible for disconnecting the space. We now precisely identify which connected component a parameter θ belongs to and clarify the meaning of the obstruction.

Proposition 3. Let e = 1 , d &gt; 1 , and θ ∈ H( c ) with c such that c k 1 , . . . , c k l -&lt; 0 while c k ≥ 0 for all other k . Let W ( 2 ) -∶= ( W ( 2 ) k 1 , . . . , W ( 2 ) k l -) ∈ R 1 × l -be the row vector whose components are the components of W ( 2 ) ∈ R 1 × l associated to c k &lt; 0 . Then the vector s ( θ ) = ( sign ( W ( 2 ) k 1 ) , . . . , sign ( W ( 2 ) k l -)) identifies uniquely the component θ belongs to, namely: θ and θ ′ belong to the same connected component of H( c ) if and only if s ( θ ) = s ( θ ′ ) .

Proposition 3, proven in Appendix D.4, implies that s ( θ ) does not change when we move in C on a continuous curve such as the one given by gradient flow. This gives us an interesting interpretation of the topological obstruction: gradient flow cannot change the signs of the outgoing weights of the hidden neurons k such that c k &lt; 0 (see Appendix G for an intuitive explanation of the phenomenon). This same observation is also mentioned in Boursier and Flammarion [3]. Proposition 3 extends one of the results of Boursier et al. [4] which proves that the same also holds when c k = 0 (balanced initialization).

By also considering Corollary 1, one obtains that a clever initialization of the parameters given by ⟪ θ 0 , θ 0 ⟫ k = c k &gt; 0 ∀ k = 1 , . . . , l can prevent the issue by ensuring the connectedness of the invariant set. We also find that under common initialization schemes such as Xavier [19] and Kaiming [21] the probability of having pathological neurons is negligible when the input dimension and number of hidden neurons is high (see Appendix F).

## 5 Taking symmetries into account

Corollary 2 states that neurons k such that c k &lt; 0 are 'pathological', in the sense that they are responsible for disconnecting the invariant set into several components, whose number scales exponentially in the number of those neurons. This result gives us a grim picture of the possibility of actually

2(Ck)

optimizing the neural network: if the initial parameter θ 0 is in a particular connected component and the global optimum θ ∗ lies in another, then any gradient flow trajectory will not be able to reach θ ∗ because it will be constrained in its connected component.

This result, however, provides us only with a partial picture of the parameter space's geometry. It is a priori possible that the training trajectory, moving in its connected component, reaches a parameter ζ , which itself is optimal as it is observationally equivalent to θ ∗ ( ζ ∼ θ ∗ ). In this case, the topological obstruction given by the non-connectedness would be only apparent.

To take this fact into account, we define the following notion.

Definition 2 (Effective component) . Let θ ∈ H( c ) and C ( θ ) be its connected component therein. We define its effective component Eff ( θ ) as the union of the connected component of all θ ′ such that θ ′ rp ∼ θ . So that Eff ( θ ) ∶= ⋃ θ ′ rp ∼ θ C ( θ ′ ) .

Figure 2b gives a picture which clarifies the definition, showing a space with 4 connected components that has only 3 effective components. If the optimum θ ∗ belongs to the same effective component as the initialization, then it is possible to reach a parameter that is observationally equivalent to it (through permutations and rescalings).

We present a useful result which tells us that the action of rescaling of Equation (3) can take any non-degenerate parameter θ ∈ H( c ) to any other invariant set H( c ′ ) for every c ′ ∈ R l . This means that any invariant set can realize all the neural network's functions.

Proposition 4. For every c k ∈ R and for every θ k ∈ Θ k such that W ( 1 ) k , W ( 2 ) k ≠ 0 , there exists a unique α k ∈ R + such that T α k ( θ k ) ∈ Q( c k ) . If W ( 1 ) k = 0 and W ( 2 ) k ≠ 0 , then the same holds for every c k &lt; 0 , while, if W ( 1 ) k ≠ 0 and W ( 2 ) k = 0 , it holds for every c k &gt; 0 .

The proof can be found in Appendix D.5 with the formula of the specific α which realizes the rescaling.

The following theorem leverages the power of Proposition 4 to give necessary and sufficient conditions for θ and θ ′ to belong to the same effective component.

Theorem 2. Let d &gt; 1 and e = 1 . Let c ∈ R l and l -be the number of neurons such that c k &lt; 0 . Assume that l -≥ 1 . Let C,C ′ ⊆ H( c ) be two distinct connected components of H( c ) such that s ( θ ) = s , ∀ θ ∈ C, and s ( θ ′ ) = s ′ , ∀ θ ′ ∈ C ′ . Then, the following statements are equivalent:

1. for every θ ∈ C there exists θ ′ ∈ C ′ such that θ rp ∼ θ ′ ;

<!-- formula-not-decoded -->

The theorem, proven in Appendix D.6, tells us that, while connected components are identified by s , the effective components are identified only by the values of ∑ i s i or, equivalently, by the distribution of ± 1 in s . Therefore, we find that the number of effective components scales much slower than the exponential growth of the number of connected components given by Corollary 1.

Corollary 3. The number of effective components of H( c ) is given by 1 + l -.

Proof. Theorem 2 tells us that two connected components C,C ′ belong to the same effective component if and only if their associated sign vectors s, s ′ ∈ {-1 , 1 } l -have the same sum. The number of effective components will thus equal the number of different values that the sum ∑ l -i = 1 s i can have. If s i = 1 ∀ i then ∑ l -i = 1 s = l -. Each switch of a component to -1 decreases the sum's value by 2 until it reaches the minimum -l -. Therefore, the total number of values of the sum will be 1 + l -.

## 6 Empirical Validation

Task, dataset, and model setup. We display here a toy example, showing how the initialization of the model can cause a topological obstruction, making the optimum unreachable.

We consider the function F ( x 1 , x 2 ) = -( x 1 + x 2 ) , which will be our ground-truth. Next, we generate a dataset of 8000 points ( x i , F ( x i )) by sampling x i ∼ U ([ 0 , 1 ] 2 ) . Our model, depicted in Figure 3a)

2.5-

2.0-

(0) T

1.5-

1.0-

0,5-

0.0-

Wit'

w22)

W(2)

W(2)

- Bad initialization

- Good initialization

100

200

2(c1)

c1 &lt; 0

° 0, (0)

0, (t.)

2(c2)

02 &lt; 0

0z(t=)° ° (O)

<!-- image -->

300

400

Optimization steps

Figure 3: Visualization of the experimental setup described in Section 6. a. The small 2-layer neural network architecture considered. b. The hidden neurons' parameter spaces, together with the invariant hyperquadrics associated with hidden neurons 1 (left) and 2 (right), for an initialization with topological obstruction (top) and without it (bottom). The colored curves represent the gradient descent trajectories from initialization θ k ( 0 ) up to t ∗ = 500 optimization steps. c. The loss curves for the bad (obstructed) and good initializations.

is a one hidden layer neural network with 2 hidden neurons, ReLU activations and no biases. All the weights are initialized by independently sampling from U ([- √ 2 , √ 2 ]) . From the task and the network's architecture, it is clear that at least one of the output weights has to be negative to approximate F correctly.

To standardize our results, we apply the rescaling of Proposition 4 and relocate the initial parameters to an observationally equivalent one in the invariant set H( c ) with c k ∈ {-0 . 1 , 0 . 1 } , controlling the sign of the weights on the last layer. We allow ourselves to do these two manipulations to control the experiments while only marginally modifying the network initialization, avoiding the introduction of massively unbalanced weights, which could change the dynamics, as shown in Neyshabur et al. [34]. Finally, we train the network using gradient descent on the MSE loss with a small learning rate of h = 0 . 01 . This limits the variations of c k values to less than one percent along training, giving us a good approximation of gradient flow.

Results. We initialize different models and collect all states and losses. First, when we initialize the model with an 'unlucky' configuration, namely c = (-0 . 1 , -0 . 1 ) (the space has 4 connected components) and s ( θ ) = (+ 1 , + 1 ) , we find that the trajectories are confined to the positive region of their invariant hyperquadric, resulting in a poor approximation of F , as we can see in Figure 3b (top) and in the loss of Figure 3c. Instead, with an initial configuration such that c = (-0 . 1 , + 0 . 1 ) (2 connected components) and s ( θ ) = (+ 1 , + 1 ) , the model can leverage the connectedness of Q( c 2 ) to learn F by flipping the sign of the second neuron's output weight (Figure 3b bottom right).

A more realistic experiment. We present here a further experiment to show how the topological obstruction can be a hindrance in a more realistic setting. We consider a simple binary classification task on the well-known breast cancer dataset [49], which we try to solve by fitting a one-layer ReLU neural network trained to minimize the BCE loss. We vary the number of hidden neurons l and, for each l , we change the number of non-pathological neurons l + (neurons with c k &gt; 0 ) from 0 to l . We repeat the experiment with 100 different random initializations and show how the model's average performance changes when the degree of disconnectedness of its invariant set is varied. The result, on the left panel of Figure 4, clearly shows the presence of a "gradient" in performance, where increasing the number of non-pathological neurons decreases the average value of the test loss after training. The right panel of Figure 4, moreover, shows how the impact of the obstruction depends on the number of non-pathological neurons and not on their fraction over the total number of hidden neurons.

b

W(2)

Figure 4: Left. Average test BCE loss of a two-layer ReLU neural network trained on the breast cancer dataset over 100 different initializations for each pair ( l , l + ), l = 2 , . . . , 9 and l + ≤ l , of numbers of hidden neurons and non-pathological neurons. Right. the y-axis displays the percentage of non-pathological neurons.

<!-- image -->

## 7 Conclusions

In this paper, we have given analytical results that clarify the nature of the constraints imposed by gradient flow on the parameter space of a two-layer neural network with homogeneous activations. In the case of a single scalar output, which appears in tasks such as binary classification and scalar regression, we identified initial conditions that lead to a topological obstruction in the form of the parameter space's fragmentation into multiple connected components. This is caused by pathological neurons whose output weights cannot change their sign during training. Moreover, if one also considers the network's symmetries under permutations of the hidden neurons, we find that most of the connected components are equivalent. The number of effective components of the resulting space scales linearly with the number of pathological neurons, contrasting with the exponential growth of the number of connected components obtained without considering the permutation symmetries.

As shown in the last numerical experiment, the lack of non-pathological neurons hinders learning, even when the network's width is scaled. Our probabilistic analysis outlined in Appendix F, however, shows that with common initialization schemes, the probability of creating a pathological neuron decreases rapidly with increased inner layer width. Therefore, the combination of specific initialization schemes and a large number of hidden neurons (beyond the minimum required to solve a task) appears to make this obstruction unlikely in practice. This work describes a simple safeguard to avoid obstructions, which can, for instance, discourage the usage of initialization schemes that result in the proliferation of pathological neurons.

## 8 Limitations

The main limitation of the work is the network's architecture, which is limited to only one hidden layer. Considering multiple layers, we can still define rescalings and permutations and find invariant hyperquadrics for each hidden neuron. The issue emerges in the fact that these hyperquadrics are not 'independent' anymore, and the invariant set cannot be factored into the product of the Q( c k ) . This intuitively results from the fact that in the multi-layer case, each weight in the hidden layers is shared by two neurons.

The second limitation is that our study focuses on gradient flow optimization. This idealized situation doesn't take into account the fact that moderate step size of gradient descent and stochastic gradient descent can break the conservation of ⟪ θ, θ ⟫ k and make the parameters drift away from the invariant set [2]. Moreover, popular optimizers like ADAM [25] update the parameters employing the gradients at previous iterations so their trajectories will not be constrained to lie on H( c ) as we defined it.

The inclusion of regularization terms in the loss function, such as ℓ p regularizations, also breaks the invariance to rescalings.

## Acknowledgements

M.N. acknowledges the project PNRR-NGEU, which has received funding from the MUR - DM 352/2022.

F.V. would like to thank the Isaac Newton Institute for Mathematical Sciences for the support and hospitality during the programme Hypergraphs: Theory and Applications when work on this paper was undertaken. This work was supported by: EPSRC Grant Number EP/V521929/1

This study was carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-GenerationEU (PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) - MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.3 - D.D. 1555 11/10/2022, PE00000013). This manuscript reflects only the authors' views and opinions; neither the European Union nor the European Commission can be considered responsible for them.

## References

- [1] Serguei Barannikov, Daria Voronkova, Ilya Trofimov, Alexander Korotin, Grigorii Sotnikov, and Evgeny Burnaev. Topological obstructions in neural networks learning. arXiv preprint arXiv:2012.15834 , 2020.
- [2] David GT Barrett and Benoit Dherin. Implicit gradient regularization. arXiv preprint arXiv:2009.11162 , 2020.
- [3] Etienne Boursier and Nicolas Flammarion. Early alignment in two-layer networks training is a two-edged sword. arXiv preprint arXiv:2401.10791 , 2024.
- [4] Etienne Boursier, Loucas Pillaud-Vivien, and Nicolas Flammarion. Gradient flow dynamics of shallow relu networks for square loss and orthogonal inputs. Advances in Neural Information Processing Systems , 35:20105-20118, 2022.
- [5] Maria Sofia Bucarelli, Giuseppe Alessio D'Inverno, Monica Bianchini, Franco Scarselli, and Fabrizio Silvestri. A topological description of loss surfaces based on betti numbers. arXiv preprint arXiv:2401.03824 , 2024.
- [6] Lenaic Chizat and Francis Bach. Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss. In Conference on learning theory , pages 1305-1338. PMLR, 2020.
- [7] Francis H Clarke, Yuri S Ledyaev, Ronald J Stern, and Peter R Wolenski. Nonsmooth analysis and control theory , volume 178. Springer Science &amp; Business Media, 2008.
- [8] Y Cooper. The critical locus of overparameterized neural networks. arXiv preprint arXiv:2005.04210 , 2020.
- [9] Yaim Cooper. The loss landscape of overparameterized neural networks. arXiv preprint arXiv:1804.10200 , 2018.
- [10] Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep nets. In International Conference on Machine Learning , pages 1019-1028. PMLR, 2017.
- [11] Felix Draxler, Kambis Veschgini, Manfred Salmhofer, and Fred Hamprecht. Essentially no barriers in neural network energy landscape. In International conference on machine learning , pages 1309-1318. PMLR, 2018.
- [12] Simon S Du, Wei Hu, and Jason D Lee. Algorithmic regularization in learning deep homogeneous models: Layers are automatically balanced. Advances in neural information processing systems , 31, 2018.
- [13] Simon Eberle, Arnulf Jentzen, Adrian Riekert, and Georg S. Weiss. Existence, uniqueness, and convergence rates for gradient flows in the training of artificial neural networks with relu activation. Electronic Research Archive , 2023.

- [14] Yuanyuan Feng, Tingran Gao, Lei Li, Jian-Guo Liu, and Yulong Lu. Uniform-in-time weak error analysis for stochastic gradient descent algorithms via diffusion approximation. arXiv preprint arXiv:1902.00635 , 2019.
- [15] Stanislav Fort, Gintare Karolina Dziugaite, Mansheej Paul, Sepideh Kharaghani, Daniel M Roy, and Surya Ganguli. Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the neural tangent kernel. Advances in Neural Information Processing Systems , 33:5850-5861, 2020.
- [16] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In International Conference on Machine Learning , pages 3259-3269. PMLR, 2020.
- [17] C Daniel Freeman and Joan Bruna. Topology and geometry of half-rectified network optimization. arXiv preprint arXiv:1611.01540 , 2016.
- [18] Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P Vetrov, and Andrew G Wilson. Loss surfaces, mode connectivity, and fast ensembling of dnns. Advances in neural information processing systems , 31, 2018.
- [19] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics , pages 249-256. JMLR Workshop and Conference Proceedings, 2010.
- [20] Grzegorz Głuch and Rüdiger Urbanke. Noether: The more things change, the more stay the same. arXiv preprint arXiv:2104.05508 , 2021.
- [21] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision , pages 1026-1034, 2015.
- [22] Stefan Horoi, Jessie Huang, Bastian Rieck, Guillaume Lajoie, Guy Wolf, and Smita Krishnaswamy. Exploring the geometry and topology of neural network loss landscapes. In International Symposium on Intelligent Data Analysis , pages 171-184. Springer, 2022.
- [23] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning , pages 448-456. pmlr, 2015.
- [24] Ziwei Ji and Matus Telgarsky. Gradient descent aligns the layers of deep linear networks. In International Conference on Learning Representations , 2019. URL https://openreview. net/forum?id=HJflg30qKX .
- [25] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR , abs/1412.6980, 2014. URL https://api.semanticscholar.org/CorpusID:6628106 .
- [26] Rohith Kuditipudi, Xiang Wang, Holden Lee, Yi Zhang, Zhiyuan Li, Wei Hu, Rong Ge, and Sanjeev Arora. Explaining landscape connectivity of low-cost solutions for multilayer nets. Advances in neural information processing systems , 32, 2019.
- [27] Daniel Kunin, Javier Sagastuy-Brena, Surya Ganguli, Daniel LK Yamins, and Hidenori Tanaka. Neural mechanics: Symmetry and broken conservation laws in deep learning dynamics. arXiv preprint arXiv:2012.04728 , 2020.
- [28] Shiyu Liang, Ruoyu Sun, Yixuan Li, and Rayadurgam Srikant. Understanding the loss surface of neural networks for binary classification. In International Conference on Machine Learning , pages 2835-2843. PMLR, 2018.
- [29] Tengyuan Liang, Tomaso Poggio, Alexander Rakhlin, and James Stokes. Fisher-rao metric, geometry, and complexity of neural networks. In The 22nd international conference on artificial intelligence and statistics , pages 888-896. PMLR, 2019.
- [30] Kaifeng Lyu, Zhiyuan Li, Runzhe Wang, and Sanjeev Arora. Gradient descent on two-layer nets: Margin maximization and simplicity bias. Advances in Neural Information Processing Systems , 34:12978-12991, 2021.

- [31] Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. Proceedings of the National Academy of Sciences , 115(33): E7665-E7671, 2018.
- [32] James R Munkres. Elements of algebraic topology . CRC press, 2018.
- [33] J.R. Munkres. Topology . Featured Titles for Topology. Prentice Hall, Incorporated, 2000. ISBN 9780131816299.
- [34] Behnam Neyshabur, Russ R Salakhutdinov, and Nati Srebro. Path-sgd: Path-normalized optimization in deep neural networks. Advances in neural information processing systems , 28, 2015.
- [35] Quynh Nguyen. On connected sublevel sets in deep learning. In International conference on machine learning , pages 4790-4799. PMLR, 2019.
- [36] Fabrizio Pittorino, Antonio Ferraro, Gabriele Perugini, Christoph Feinauer, Carlo Baldassi, and Riccardo Zecchina. Deep networks on toroids: removing symmetries reveals the structure of flat regions in the landscape geometry. In International Conference on Machine Learning , pages 17759-17781. PMLR, 2022.
- [37] Grant Rotskoff and Eric Vanden-Eijnden. Trainability and accuracy of artificial neural networks: An interacting particle system approach. Communications on Pure and Applied Mathematics , 75(9):1889-1935, 2022.
- [38] Itay Safran and Ohad Shamir. Spurious local minima are common in two-layer relu neural networks. In International conference on machine learning , pages 4433-4441. PMLR, 2018.
- [39] Itay Safran, Gal Vardi, and Jason D Lee. On the effective number of linear regions in shallow univariate relu networks: Convergence guarantees and implicit bias. Advances in Neural Information Processing Systems , 35:32667-32679, 2022.
- [40] Levent Sagun, Utku Evci, V Ugur Guney, Yann Dauphin, and Leon Bottou. Empirical analysis of the hessian of over-parametrized neural networks. arXiv preprint arXiv:1706.04454 , 2017.
- [41] Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120 , 2013.
- [42] Berfin Simsek, François Ged, Arthur Jacot, Francesco Spadaro, Clément Hongler, Wulfram Gerstner, and Johanni Brea. Geometry of the loss landscape in overparameterized neural networks: Symmetries and invariances. In International Conference on Machine Learning , pages 9722-9732. PMLR, 2021.
- [43] Justin Sirignano and Konstantinos Spiliopoulos. Mean field analysis of neural networks: A law of large numbers. SIAM Journal on Applied Mathematics , 80(2):725-752, 2020.
- [44] Samuel L Smith, Benoit Dherin, David GT Barrett, and Soham De. On the origin of implicit regularization in stochastic gradient descent. arXiv preprint arXiv:2101.12176 , 2021.
- [45] Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. Journal of Machine Learning Research , 19 (70):1-57, 2018.
- [46] Hidenori Tanaka, Daniel Kunin, Daniel L Yamins, and Surya Ganguli. Pruning neural networks without any data by iteratively conserving synaptic flow. Advances in neural information processing systems , 33:6377-6389, 2020.
- [47] Twan Van Laarhoven. L2 regularization versus batch and weight normalization. arXiv preprint arXiv:1706.05350 , 2017.
- [48] Luca Venturi, Afonso S Bandeira, and Joan Bruna. Spurious valleys in one-hidden-layer neural network optimization landscapes. Journal of Machine Learning Research , 20(133):1-34, 2019.
- [49] William Wolberg, Olvi Mangasarian, Nick Street, and W Street. Breast cancer wisconsin (diagnostic). UCI Machine Learning Repository , 10:C5DW2B, 1995.

- [50] Chulhee Yun, Suvrit Sra, and Ali Jadbabaie. Small nonlinearities in activation functions create bad local minima in neural networks. arXiv preprint arXiv:1802.03487 , 2018.
- [51] Bo Zhao, Iordan Ganev, Robin Walters, Rose Yu, and Nima Dehmamy. Symmetries, flat minima, and the conserved quantities of gradient flow. arXiv preprint arXiv:2210.17216 , 2022.
- [52] Liu Ziyin. Symmetry induces structure and constraint of learning. In Forty-first International Conference on Machine Learning , 2024.

## A Parameter spaces

Let ( e 11 , e 12 , . . . , e ll ) and ( e 1 , . . . , e l ) be the canonical bases of the spaces R l × l and R l , respectively and

<!-- formula-not-decoded -->

Θ k , notice, is the subspace of Θ consisting of the weight matrices W ( 1 ) with null rows except for the k -th one, and weight matrices W ( 2 ) with null columns except for the k -th one. We can check that Θ = Θ 1 ⊕/uni22EF⊕ Θ l , because, if I l is the l × l identity matrix, ∑ e kk = I l so that

<!-- formula-not-decoded -->

Moreover, Θ ≅ Θ 1 ×/uni22EF× Θ l via the linear isomorphism

<!-- formula-not-decoded -->

This, we see, is equivalent to decomposing the neural network of Equation (1) into the computations of the single hidden neurons. Indeed, let f ( x ; θ k ) ∶= f ( x, ( e kk W ( 1 ) , W ( 2 ) e kk )) , then, considering that e kk e kk = e kk and that σ ( e kk v ) = e kk σ ( v ) , it holds that

<!-- formula-not-decoded -->

Therefore ∑ k f ( x ; θ k ) = f ( x ; θ ) .

## B Primer on topology

Here, we recall some basic facts about the topology required to understand the paper's results. A self-consistent introduction is outside this work's scope, so we refer the interested reader to more complete expositions in Munkres [33, 32].

Topological manifold. An n -dimensional topological manifold is a topological space X which locally looks like the Euclidean space R n . More formally, for each p ∈ X , there exists a neighbourhood U of p and a homeomorphism mapping U to an open subset of R n .

Contractible space. A topological space X is contractible if it can continuously deform to a point p ∈ X . This means that there exists a continuous map

<!-- formula-not-decoded -->

such that F ( x, 0 ) = x and F ( x, 1 ) = p for every x ∈ X .

Betti numbers. Betti numbers formalize the notion of the hole in a topological space and extend it to describe higher-dimensional cavities. The general idea is that one can associate a sequence of Abelian groups named homology groups to any space X , which encodes rich information about the higher-dimensional cavities in X . For what we are concerned here, the rank of the k -th homology group is called the k -th Betti number β k ( X ) . β k ( X ) counts the number of k -dimensional holes in the space: β 0 ( X ) count the number of connected components, β 1 ( X ) the number of 'circular' holes and β 2 ( X ) the number of voids or cavities.

A contractible space X is connected and cannot have any holes, and thus its Betti numbers are β 0 ( X ) = 1 and β i ( X ) = 0 ∀ i &gt; 0 .

Betti numbers are topological invariants , meaning they are preserved when a space is transformed via a homeomorphism, namely a bijective, continuous map with continuous inverse.

Poincaré polynomials. The Poincaré polynomial of a topological space X is the polynomial whose k -th coefficient is given by the k -th Betti number

<!-- formula-not-decoded -->

Künneth formula. Künneth's theorem describes assembling the homology groups of a Cartesian product of spaces X × Y from the homology groups of the factors X,Y . One of its corollaries tells us that if we care only about Betti numbers, a simple relation holds between the Poincaré polynomials of X × Y and the ones of X and Y , namely,

<!-- formula-not-decoded -->

Types of connectedness. In topology, there are several kinds of connectedness. Two of them are particularly important for this work.

1.) A topological space X is connected if it cannot be divided into two disjoint non-empty open sets. If it is not connected, the connected component of a point x ∈ X is given by the union of all connected subsets of X which contain x .

A topological space equal to the Cartesian product of two spaces X = Y × Z is connected if and only if Y and Z are both connected.

2.) A topological space is path-connected if, for every pair of points x, y ∈ X , there exists a continuous curve γ ∶ [ 0 , 1 ] → X such that γ ( 0 ) = x, γ ( 1 ) = y . The path-component of x is the set of all y ∈ X such that a continuous curve exists connecting x to y .

This second notion is more relevant to our setting, where we care about the possible destinations of the optimization trajectories.

Path-connectedness implies connectedness, but not the opposite. There are situations, however, where these two notions are equivalent. For example, when X is a topological manifold, X is connected if and only if X is path connected.

Notice that the 0-th Betti number β 0 ( X ) counts the number of connected components but, in general, not the number of path components. With Lemma 2, we prove that these two notions are equivalent for our object of study.

## C Extra propositions and lemmas

Lemma 2. The invariant set H( c ) is connected if and only if it is path connected.

Proof. Lemma 1 tells us that H( c ) ≅ Q( c 1 ) × /uni22EF×Q( c l ) .

Let us focus on a particular Q( c k ) .

When c k ≠ 0 , Proposition 2 tells us that Q( c k ) is a topological manifold, and thus, it is connected if and only if it is path connected.

When c k = 0 , Q( 0 ) is not a topological manifold but contractible, implying that it is connected. Let us prove that it is also path-connected.

Let θ k , θ ′ k ∈ Q( 0 ) and define the curve γ ∶ [ 0 , 1 ] → Q( 0 )

<!-- formula-not-decoded -->

such that γ k ; θ ( 0 ) = θ, γ k ; θ ( 1 ) = 0 . γ k ; θ ( t ) ∈ Q( 0 ) for every t ∈ [ 0 , 1 ] because

<!-- formula-not-decoded -->

Therefore, the segment from θ k to 0 belongs to Q( 0 ) .

A continuous curve from θ k to θ ′ k can be thus obtained by

<!-- formula-not-decoded -->

which is continuous because γ k ; θ ( 1 ) = γ k ; θ ′ ( 1 ) = 0 . Therefore Q( 0 ) is path connected.

Finally, if H( c ) is connected, then all of its factors Q( c k ) are connected, which, in turn, is true if and only if they are path-connected. A product of path-connected space is again path-connected, and therefore H( c ) is path-connected. The other implication is true because path-connectedness implies connectedness, thus concluding the proof.

Lemma 3 (Interchange of rescalings and permutations) . Let θ ∈ Θ , α ∈ R l + and π ∈ S l , then, if ˜ α = R π -1 ( α )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we need to prove that diag ( α ) R π = R π diag ( ˜ α ) .

<!-- formula-not-decoded -->

Let us pick a generic ˜ α ∈ R l + .

<!-- formula-not-decoded -->

Let us consider the inverse permutation π -1 so that π -1 ( j ) = i if π ( i ) = j . Then, if ˜ α = R π -1 α ,

<!-- formula-not-decoded -->

and thus we get that diag ( α ) R π = R π diag ( ˜ α ) .

## D Proofs

## D.1 Proof of Lemma 1

Proof. Proposition 1 tells us that the invariant set can be decomposed as the direct sum of the single hidden neurons' parameter spaces. This means that, for every θ ∈ Θ , there exist unique θ 1 ∈ Θ 1 , . . . , θ l ∈ Θ l such that

<!-- formula-not-decoded -->

Therefore, we have a linear isomorphism φ ∶ Θ 1 ×/uni22EF× Θ k → Θ

<!-- formula-not-decoded -->

The invariant set is a subset of Θ , which is given as the set of solutions of l equations ⟪ θ, θ ⟫ k = c k k = 1 , . . . , l . Notice that each of these equations involves a set of variables that appear only in that particular equation. These variables are exactly the ones which belong to Θ k . In fact ⟪ θ, θ ⟫ k = ⟪ θ k , θ k ⟫ k .

Therefore, given θ 1 ∈ Q( c 1 ) , . . . , θ l ∈ Q( c l ) we have that

<!-- formula-not-decoded -->

On the opposite, given θ ∈ H( c ) we have that

<!-- formula-not-decoded -->

Therefore, H( c ) is in bijection with Q( c 1 )×/uni22EF × Q( c l ) through φ which, being a linear isomorphism, implies also that H( c ) and Q( c 1 ) × /uni22EF×Q( c l ) are homeomorphic.

## D.2 Proof of Proposition 2

Proof. Let us consider the three cases separately. If c k &gt; 0 , Q( c k ) is defined by the equation

<!-- formula-not-decoded -->

Proof. Given that

where ∥⋅∥ F is the Frobenius norm of a matrix, i.e., the square root of the sum of the squares of its elements. This can be rewritten as where √ c k +∥ W ( 2 ) k ∥ 2 F &gt; 0 because c k &gt; 0 . We define the map h ∶ Q( c k ) → S d -1 × R e as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where, notice, the first component belongs to the sphere S d -1 because of Equation (17) and W ( 2 ) k ∈ R e . This map is bijective, differentiable and has the following inverse h -1 ∶ S d -1 × R e →Q( k )

<!-- formula-not-decoded -->

which is differentiable. Therefore, h is a diffeomorphism from Q( c k ) to S d -1 × R e .

If c k &lt; 0 , we write the equation of Q( c k ) as

<!-- formula-not-decoded -->

where W ( 1 ) k and W ( 2 ) k have switched their role to guarantee the term on the right to be positive. The diffeomorphism is now built analogously to Equation (18) as a map h ∶ Q( c k ) → S e -1 × R d .

If c k = 0 , we prove that Q( 0 ) is a contractible space. To do that, we exhibit a homotopy equivalence between Q( 0 ) and the point 0 , i.e. a continuous map p ∶ [ 0 , 1 ]×Q( 0 ) → Q( 0 ) such that p ( 0 , θ k ) = θ k and p ( 1 , θ k ) = 0 ∀ θ k ∈ Q( 0 ) . The map is defined in the following way:

<!-- formula-not-decoded -->

This is continuous and well-defined because

<!-- formula-not-decoded -->

## D.3 Proof of Theorem 1

Proof. An implication of the Künneth formula is that the Poincaré polynomial of the Cartesian product of two spaces is equal to the product of their Poincaré polynomials:

<!-- formula-not-decoded -->

Starting from Proposition 2, we can apply this result to Q( c k ) .

<!-- formula-not-decoded -->

because a contractible space has 1 connected component and all of its other Betti numbers equal to zero.

Moreover, we know that R n is contractible for any n and its Poincaré polynomial is p R n ( x ) = 1 . The Poincaré polynomial of the sphere S n is given by p S n ( x ) = 1 + x n .

Equation (19) becomes

<!-- formula-not-decoded -->

Given that Lemma 1 tells us that H( c ) can be factored into the product of the Q( c k ) , we apply Künneth formula and find that

<!-- formula-not-decoded -->

## D.4 Proof of Proposition 3

Proof. In the following, we exploit Lemma 2 and use the terminology connected and path-connected interchangeably.

Let us first prove that s ( θ ) = s ( θ ′ ) means that θ and θ ′ belong to the same connected component. We do this by explicitly building a continuous curve δ ∶ [ 0 , 1 ] → H( c ) such that δ ( 0 ) = θ and δ ( 1 ) = θ ′ .

Let us proceed by leveraging the homeomorphism from H( c ) and Q( c 1 )×/uni22EF × Q( c l ) and consider the different components of δ in the invariant hyperquadrics associated to each neuron δ = ( δ 1 , . . . , δ l ) .

If c k ≥ 0 , we know that Q( c k ) is path-connected and therefore we fix δ k ( t ) to any continuous curve in Q( c k ) such that δ k ( 0 ) = θ k and δ k ( 1 ) = θ ′ k .

<!-- formula-not-decoded -->

for every i = 1 , . . . , l -.

γ i ; θ is a continuous curve which connects the point γ i ; θ ( 0 ) = θ k i with γ i ; θ ( 1 ) = ( 0 , s ( θ ) i √ -c k i ) .

If we define ¯ γ i ; θ ( t ) ∶= γ i ; θ ( 1 -t ) , which is the same curve as γ i ; θ but traversed in the opposite direction, we can define the curve

<!-- formula-not-decoded -->

i.e. the curve which travels on γ i ; θ for t ∈ [ 0 , 1 2 ] and on ¯ γ i ; θ ′ for t ∈ [ 1 2 , 1 ] , for i = 1 , . . . , l -.

<!-- formula-not-decoded -->

Notice now that δ k i ( 0 ) = θ k i and δ k i ( 1 ) = θ ′ k i . Moreover δ k i is continuous because under the hypothesis that s ( θ ) = s ( θ ′ ) .

Finally, we found a continuous curve δ = ( δ 1 , . . . , δ l ) such that δ ( t ) ∈ H( c ) ∀ t ∈ [ 0 , 1 ] and δ ( 0 ) = θ, δ ( 1 ) = θ ′ . Therefore, θ and θ ′ belong to the same connected component.

Let us now prove that if θ and θ ′ belong to the same connected component, then s ( θ ) = s ( θ ′ ) .

Let γ ∶ [ 0 , 1 ] → H( c ) be a continuous curve in H( c ) such that γ ( 0 ) = θ and γ ( 1 ) = θ ′ .

For each k ∈ K = { k 1 , . . . , k l -} such that c k &lt; 0 , we know that γ k ( t ) ∈ Q( c k ) means that

<!-- formula-not-decoded -->

for some function s k t 1 , 1 such that s k 0 s θ k and s k 1 s θ k

Assume, by contradiction, that s θ k s θ k . Assume also that s θ k 1 and s θ k 1 .

Notice that c k &lt; 0 implies that p ( t ) ∶= √ -c k +∥ γ ( 1 ) k ∥ 2 F &gt; 0 .

( ) ∈ {-} ( ) = ( ) ( ) = ( ′ ) . ( ) = -( ′ ) ( ) = + ( ′ ) = -

The function γ ( 2 ) k ( t ) = s k ( t ) p ( t ) , then, is a continuous function such that γ ( 2 ) k ( 0 ) &gt; 0 and γ ( 2 ) k ( 1 ) &lt; 0 and thus, by the intermediate value theorem, there exists t ∗ ∈ ( 0 , 1 ) such that γ ( 2 ) k ( t ∗ ) = 0 .

<!-- formula-not-decoded -->

Repeating the argument for s ( θ ) k = -1 we then prove by contradiction that s ( θ ) k = s ( θ ′ ) k for all k ∈ K , thus concluding the proof.

## D.5 Proof of Proposition 4

Proof. We have by Equation (10) and Equation (2):

<!-- formula-not-decoded -->

By renaming A = d ∑ i = 1 ( W ( 1 ) ki ) 2 = ∥ W ( 1 ) k ∥ 2 F , C = e ∑ j = 1 ( W ( 2 ) jk ) 2 = ∥ W ( 2 ) k ∥ 2 F and multiplying by α 2 k &gt; 0 we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Solving for α 2 k gives us:

Given that we want α k &gt; 0 , we discard the negative solution. The other is positive because ∆ &gt; c 2 k and thus √ ∆ &gt; ∣ c k ∣ .

<!-- formula-not-decoded -->

Of which we keep the positive solution only, with its full expression being:

<!-- formula-not-decoded -->

Hence, if α = ( α 1 , . . . , α l ) with α k given by Equation (23), we get that ⟪ T α ( θ ) , T α ( θ )⟫ k = c k ∀ k = 1 , . . . , l .

Let us consider now the pathological cases W ( 1 ) k = 0 or W ( 2 ) k = 0 .

If W ( 1 ) k = 0 , W ( 2 ) k ≠ 0 then A = 0 , C ≠ 0 . Therefore, we have that Equation (22) becomes

<!-- formula-not-decoded -->

which has solutions if and only if c k &lt; 0 . In that case α k = ∥ W ( 2 ) k ∥ F √ -c k . This means that a hidden neuron with zero input weights and nonzero output weights can be rescaled only to the invariant hyperquadrics with c k &lt; 0 .

If W ( 2 ) k = 0 , W ( 1 ) k ≠ 0 then C = 0 , A ≠ 0 . Therefore, we have that Equation (22) becomes

<!-- formula-not-decoded -->

which has solutions if and only if c k &gt; 0 . In that case α k = √ c k ∥ W ( 1 ) k ∥ F . This means that a hidden neuron with zero output weights and nonzero input weights can be rescaled only to the invariant hyperquadrics with c k &gt; 0 .

If W ( 1 ) k = 0 and W ( 2 ) k = 0 , then θ k = 0 ∈ Q( 0 ) and it cannot be rescaled to any other invariant hyperquadric.

## D.6 Proof of Theorem 2

Proof. Let us first prove that, if for every θ ∈ C there exists a θ ′ ∈ C ′ such that θ rp ∼ θ ′ then l -∑ i = 1 s ( θ ) i = l -∑ i = 1 s ( θ ′ ) i .

First, Lemma 3 tells us that we can interchange rescaling and permutation if we permute the rescaling factors accordingly. This means we can reduce any composite action of rescalings and permutations to the action of a single rescaling and a single permutation.

Let θ ∈ C and θ ′ ∈ C ′ such that θ rp ∼ θ ′ . Then there exist α ∈ R l + and π ∈ S l such that

<!-- formula-not-decoded -->

This means that T α P π ( θ ) and θ ′ belong to the same invariant set and, specifically, to the same connected component. Therefore,

Notice that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

because T α does not change the sign of W ( 2 ) as it acts by scaling it by positive factors. Let us focus on

If the neurons of θ such that c k &lt; 0 are indexed by k 1 , k 2 , . . . , k l -, we will have that the neurons of P π ( θ ) such that c k &lt; 0 are indexed by π ( k 1 ) , π ( k 2 ) , . . . , π ( k l -) . Therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some permutation π -∈ S l -. Therefore, Equation (24) implies that

<!-- formula-not-decoded -->

The action of rescaling and permutation can only reshuffle the label s of the connected component.

This means that

<!-- formula-not-decoded -->

Let us now prove the other implication. Let s, s ′ ∈ R l -such that l -∑ i = 1 s i = l -∑ i = 1 s ′ i .

Given that their sum is equal, s and s ′ have the same number of + 1 and -1 and thus there exists a permutation π -∈ S l -such that s ′ = sR ⊺ π -.

Let π ∈ S l be the permutation which permutes the neurons such that c k &lt; 0 according to π -and leaves the others fixed. In this way s ( P π ( θ )) = sR ⊺ π -= s ′ .

P π ( θ ) , however, doesn't belong to H( c ) but to another invariant set given by H( R π c ) .

Applying Proposition 4 we can find a rescaling α = α ( π ) ∈ R l + such that T α P π ( θ ) ∈ H( c ) .

Since neurons such that c k ≥ 0 , are left unchanged by the permutation, we can apply the proposition and we rescale them with α k = 1 . The permuted neurons are the ones such that c k &lt; 0 , namely the ones whose weights satisfy ∥ W ( 1 ) k ∥ 2 F -∥ W ( 2 ) k ∥ 2 F &lt; 0 , meaning that W ( 2 ) k ≠ 0 .

As noted above, the action of the rescaling doesn't change the sign vector, and thus

<!-- formula-not-decoded -->

If we name θ ′ ∶= T α P π ( θ ) this result means that we found a θ ′ rp ∼ θ such that θ ′ ∈ C ′ , thus concluding the proof.

## E Including biases

Let us consider the case where we include biases. The resulting two-layer neural network can be written as

<!-- formula-not-decoded -->

where b ( 1 ) ∈ R l and b ( 2 ) ∈ R e .

To work with this extended set of parameters, we re-define the space

<!-- formula-not-decoded -->

and the single hidden neuron spaces

<!-- formula-not-decoded -->

where the second bias term b ( 2 ) does not appear because it is not directly involved with the computations of the hidden neurons. This means that we can write

<!-- formula-not-decoded -->

where R e is included to describe the parameters in b ( 2 ) .

The neuron rescaling action now acts on the biases b ( 1 ) as well as the weights:

<!-- formula-not-decoded -->

and can be extended to the whole space of parameters

<!-- formula-not-decoded -->

Once again, we find that T α θ ∼ θ .

In this more general case, we can rewrite the bilinear form to include the biases. If θ = ( W ( 1 ) , b ( 1 ) , W ( 2 ) , b ( 2 ) ) and η = ( V ( 1 ) , p ( 1 ) , V ( 2 ) , p ( 2 ) ) , we define

<!-- formula-not-decoded -->

and see that, once gradient flow optimization, we have a conservation condition like the one of Equation (9)

<!-- formula-not-decoded -->

Once again, we call Q( c k ) the hypersurface of Θ k which satisfies the equation ⟪ θ, θ ⟫ k = c k and H( c k ) the set in Θ defined by ⟪ θ, θ ⟫ k = c k ∀ k = 1 , . . . , l .

With this in mind, it is not hard to extend the results of Proposition 2 and Theorem 1 which turn out to be slightly modified.

Proposition 5. If c k &gt; 0 , Q( c k ) is a topological manifold homeomorphic to R e × S d . If c k &lt; 0 , Q( c k ) is a topological manifold homeomorphic to R d × S e -1 . If c k = 0 , Q( 0 ) is contractible.

In this case, we can factor the space of parameters H( c ) as

<!-- formula-not-decoded -->

where the last factor is due to the freedom in choosing the values of b ( 2 ) .

Proposition 6. Let c k ≠ 0 ∀ k = 1 , . . . , l . Let l + , l -, l 0 be the number of positive, negative and zero elements of c , respectively. The Poincaré polynomial of H( c ) is given by

<!-- formula-not-decoded -->

Corollary 4. The 0 -th Betti number β 0 ( c ) of H( c ) , corresponding to the number of its connected components, is given by

<!-- formula-not-decoded -->

The result we obtain is similar to Corollary 1 although slightly modified by the fact that having a single input neuron does not cause H( c ) to become disconnected anymore. In the case of e = 1 , therefore, the picture presented in the main text is left unchanged.

Xavier normal initialization

Kaiming normal initialization

1

1

7-

7

3

.3

9

.9

5-

1

- 3

L 61

•9

1 0l

1 0

• 7

1

1

0

• O

0

0

Figure 5: Probability of the topological obstruction as a function of the number of input d and hidden l neurons, when the initial weights are sampled with Xavier normal (left) and Kaiming normal (right) initialization schemes.

<!-- image -->

## F Probability of obstruction

Let us consider the following question: what is the probability of having a disconnected invariant set given a realistic initialization?

Consider a one-layer ReLU neural network with e = 1 and assume that the weights are sampled independently of one another from a normal distribution W ( 1 ) ki ∼ N( 0 , σ 2 1 ) ∀ k, i , W ( 2 ) k ∼ N( 0 , σ 2 2 ) ∀ k, j . From Corollary 2 we know that the invariant set H( c ) will be disconnected if and only if there exists a hidden neuron satisfying ∑ d i = 1 ( W ( 1 ) ki ) 2 &lt; ( W ( 2 ) k ) 2 . Given independence of the initial weight sampling, this probability can be computed as

<!-- formula-not-decoded -->

where F is the cumulative distribution function of the Fisher-Snedecor distribution.

Having obtained this general expression, we can specify it to two common initialization schemes.

- We obtain Kaiming initialization [21] with σ 2 1 = 2 / d, σ 2 2 = 2 / l resulting in P [ obstruction ] = 1 -F 1 ,d ( l ) l .
- We obtain Xavier normal initialization [19] with σ 2 1 = 2 /( d + l ) , σ 2 2 = 2 /( 1 + l ) resulting in P [ obstruction ] = 1 -F 1 ,d ( d + ld d + l ) l .

We plot these two expressions in Figure 5. We can see how, for large values of d , the probability of obstruction quickly falls to 0 for any number of hidden neurons. Instead, we see an opposite trend for small values of d : the probability of disconnectedness grows with l . Moreover, it is interesting to notice that the region of high obstruction probability is much larger for Xavier initialization than for Kaiming initialization, further showing why the latter is preferred when working with ReLU networks.

## G Intuition on the occurrence of obstruction

We can give some intuition on why there is no obstruction for multiple outputs. First, we consider a single hidden neuron k , with d incoming weights and a single output e = 1 . If the neuron is pathological, we have that

<!-- formula-not-decoded -->

Since the weights W ( a ) ij ( t ) are continuous curves in time, for W ( 2 ) 1 k to change sign, its value needs to pass through 0 but, under the condition above, this cannot happen its square is always positive.

Consider now multiple outputs e &gt; 1 , resulting in the conservation condition being

<!-- formula-not-decoded -->

Now, any component W ( 2 ) jk can change sign by passing through 0 because the other components can compensate for it by increasing their magnitude to keep the condition satisfied.

##