## Unravelling in Collaborative Learning

Aymeric Capitaine 1

Antoine Scheid 1

Eric Moulines 1

Etienne Boursier 2

Michael I. Jordan 3

Alain Durmus 1

- 1 Centre de Mathématiques Appliquées - CNRS - École polytechnique - Palaiseau, 91120, France
- 2 INRIA Saclay, Université Paris Saclay, LMO - Orsay, 91400, France
- 3 Inria, Ecole Normale Supérieure, PSL Research University - Paris, 75, France

## Abstract

Collaborative learning offers a promising avenue for leveraging decentralized data. However, collaboration in groups of strategic learners is not a given. In this work, we consider strategic agents who wish to train a model together but have sampling distributions of different quality. The collaboration is organized by a benevolent aggregator who gathers samples so as to maximize total welfare, but is unaware of data quality. This setting allows us to shed light on the deleterious effect of adverse selection in collaborative learning. More precisely, we demonstrate that when data quality indices are private, the coalition may undergo a phenomenon known as unravelling , wherein it shrinks up to the point that it becomes empty or solely comprised of the worst agent. We show how this issue can be addressed without making use of external transfers, by proposing a novel method inspired by probabilistic verification. This approach makes the grand coalition a Nash equilibrium with high probability despite information asymmetry, thereby breaking unravelling.

## 1 Introduction

Collaborative learning is a framework in which multiple agents share their data and computational resources to address a common learning task [Blum et al., 2017, Kairouz et al., 2021]. A significant challenge arises when the quality of these distributions is unknown centrally and agents are strategic. Indeed, participants may be tempted to withhold or misrepresent the quality of their data to gain a competitive advantage. These strategic behaviors and their consequences have been studied extensively in the literature on information economics [Mas-Colell et al., 1995, Laffont and Martimort, 2001]. In particular, information asymmetry is known to result in adverse selection , whereby lowquality goods end up dominating the market. In the current paper we study collaborative learning from the perspective of information economics.

A vivid illustration of adverse selection is found in Akerlof's seminal work on the market for lemons [second-hand cars of low quality, Akerlof, 1970]. Because buyers cannot properly assess the quality of cars on the second-hand market, their inclination to pay decreases. As a consequence, sellers with high-quality cars withdraw from the market, since the proposed price falls below their reservation price. This in turn lowers the buyers' expectation regarding the average quality of cars on the market, so their willingness to pay decreases even more, which de facto crowds out additional cars. The

El-Mahdi El-Mhamdi 1

market may therefore enter a death spiral, up to the point where only low-quality cars are exchanged in any competitive Nash equilibrium. This phenomenon is known as unravelling . The insurance market serves as another poignant example of this effect [Rothschild and Stiglitz, 1976, Einav and Finkelstein, 2011, Hendren, 2013]. In insurance, information asymmetry arises due to the fact that insurees possess private knowledge about their individual risk profiles, which insurers lack. Individuals with higher risk are more inclined to purchase policy, while those with lower risk opt out. Consequently, insurers are left with a pool of policyholders skewed towards higher risk, leading to increased premiums to cover potential losses. This, in turn, prompts low-risk individuals to exit the market, exacerbating adverse selection further-a cycle reminiscent of the unravelling described by Akerlof.

We study the problem of whether collaborative learning could also fall victim to unravelling. We consider strategic agents who have access to sampling distributions of varying quality and wish to jointly train a model. They delegate the training process to a central authority who collects samples so as to maximize total welfare. We ask whether adverse selection can arise when data quality is private information. In particular, the presence of a low-quality data owner may harm the model, prompting high-quality owner to leave the collaboration and train a model entirely on their own. Their departure would decrease the average data quality even more, and create a vicious circle. In the worst case, the coalition of learners would reduce to the lowest data quality owner alone. This question is of prime importance from a practical point of view, because unravelling could jeopardize the long-run stability of collaborative models deployed at large scale.

Our contribution is threefold:

1. We provide a rigorous framework for analyzing collaborative learning with strategic agents having data distributions of varying quality. On the one hand, we leverage tools from domain adaptation to capture a notion of data quality formally. On the other hand, we model collaboration as a principal-agent problem, where the principal is an aggregator in charge of collecting samples so as to maximize social welfare. This setup allows us to derive the benchmark welfare-maximizing collaboration scheme when data quality is public information.
2. We show that when data quality is private, a naive aggregation strategy which consists in asking agents to declare their quality type and applying the optimal scheme results in a complete unravelling. More precisely, the set of agents willing to collaborate is either empty or made of the lowest-quality data owner alone at any pure Nash equilibrium.
3. We present solutions to unravelling. When transfers are allowed, the VCG mechanism suffices to re-establish optimality. When transfers are not possible, we leverage probabilistic verification techniques to design a mechanism which breaks unravelling. More precisely, we ensure that the optimal, grand coalition ranks with high probability among the Nash equilibria of the game induced by our mechanism. We demonstrate how to implement our mechanism practically in the setting of classification.

Related work. The issue of information asymmetry in machine learning has been an area of recent activity. Several learning settings have been considered, including bandits [Wei et al., 2024], linear regression [Donahue and Kleinberg, 2021a,b], classification [Blum et al., 2021] and empirical risk minimization [Dorner et al., 2023, Liu et al., 2023] in a federated context [Tu et al., 2022].

Most of these studies focus on the sub-problem of moral hazard , where agents take actions that are unobserved by others. This situation usually results in under-provision of effort and inefficiency at the collective scale [Laffont and Martimort, 2001]. This issue appears naturally in federated learning, because model updates are performed locally. For instance, Karimireddy et al. [2022] show that heterogeneity in sampling costs results in total free-riding without a proper incentive scheme. Huang et al. [2022] show that under-provision of data points in federated learning arises from privacy concerns. Yan et al. [2023] consider a federated classification game where agents can reduce the noise in their data distribution but incur a costly effort to do so. In the same vein, Huang et al. [2023] study the case of agents who are interested in different models, and may skew their sampling measure accordingly. Saig et al. [2023] and Ananthakrishnan et al. [2023] study hidden actions when a principal delegates a predictive task to another agent, and show that thresholds or linear contracts are able to approximate the optimal contracts.

Adverse selection is another type of information asymmetry, where preferences of agents are unobserved rather than actions. This issue also naturally arises in collaborative learning, because the data distributions from which agents sample or about which they care may not be public. While

data heterogeneity is a widely explored topic in federated learning (see for instance Gao et al. 2022 for a general survey, and Fu et al. 2023 for the specific problem of client selection ), the strategic aspect has been rarely considered. Most studies doing so focus on hidden sampling costs (see, e.g., Karimireddy et al. 2022, or Wei et al. 2024 in a bandit context), but few address the fundamental problem of distribution shift. Ananthakrishnan et al. [2023] mentions the issue of adverse selection when a principal delegates a predictive task to an agent, and provide qualitative insights about the optimal contract. However, they consider a single agent and leave aside the question of participation. Finally, Werner et al. [2024] and Tsoy and Konstantinov [2024] study the question of the stability of collaborative learning between competing agents. However, their analysis relies on ad-hoc market structures rather than information asymmetry per se. As such, our work is the first to demonstrate the effect of imperfect information on the sustainability of collaborative learning.

Organization. Section 2 presents our model and assumptions. Section 3 studies a full information benchmark, which allows us to derive the welfare-maximizing contribution scheme. In Section 4, we turn to the more realistic case where data quality is private information. We first show that in this case, a naive aggregation method leads to total unravelling. Second, we introduce a mechanism which breaks unravelling by inducing a game where the grand coalition is a pure strategy Nash equilibrium with high probability.

## 2 Model

Statistical framework. Let ( X , X ) and ( Y , Y ) be two measurable spaces and denote by P a family of probability measures on ( X × Y , X ⊗ Y ) . We consider [ J ] = { 1 , . . . , J } agents who aim to perform a prediction task associated with a hypothesis class G ⊂ { g : X → Y } , a loss function ℓ : Y × Y and a probability measure P 0 ∈ P . Each agent seeks to minimize g ∈ G ↦→ R P 0 ( g ) where for any probability distribution P ∈ P , R P ( g ) = ∫ ℓ ( g ( x ) , y )d P ( x, y ) is the risk associated to g ∈ G with respect to P .

We leverage tools and results from statistical learning theory. Denote by { ( X i , Y i ) } i ⩾ 1 the canonical process on X × Y and denote by P P and E P the canonical probability and expectation under which { ( X i , Y i ) } i ⩾ 1 are i.i.d. random variables with distribution P ∈ P . With this notation, we can introduce our assumptions on G and P .

H1. For δ ∈ (0 , 1) , there exist α δ &gt; 0 , β &gt; 0 and γ &gt; 0 such that for any distribution P ∈ P , hypothesis g ∈ G and n ⩾ 1

<!-- formula-not-decoded -->

This assumption covers a wide range of situations. For instance in the classification case where Y = {-1 , 1 } and ℓ : ( y, ˜ y ) ↦→ ✶ { y ˜ y ⩽ 0 } , α δ = √ ln(1 /δ ) / 2 , β = 2 RAD ( G ) and γ = 1 / 2 where RAD ( G ) is the empirical Rademacher complexity of G [Bousquet et al., 2003]. Similarly, the Bayesian PAC approach in the linear regression context with bounded loss leads to α δ = KL( ρ ∥ π ) + ln(1 /δ ) , β = ∥ ℓ ∥ 2 ∞ / 8 and γ = 1 where ρ and π are any distributions on G [Shalaeva et al., 2019, Corollary 4].

For ease of notation, we let R j ( g ) serve as a shorthand for R P j ( g ) . It is moreover assumed that agent j ∈ [ J ] cannot directly sample from P 0 , but has instead access to a distribution P j ∈ P which deviates from P 0 according to the G -divergence:

<!-- formula-not-decoded -->

Intuitively, for any j ∈ [ J ] , θ j ⩾ 0 models the bias incurred by having access to samples from P j instead of the target distribution P 0 . More precisely, the risk excess associated with empirical risk minimization (ERM) based on samples from P j is in the worst case at least θ j . A poor sampling distribution P j -which corresponds to a high θ j in the previous expression-might be the consequence of low-quality sensors or degraded experimental conditions resulting in noisier data points.

The class of discrepancies appearing in H 2 has been considered in the domain adaptation literature [see, e.g., Ben-David et al., 2010, Kifer et al., 2004, Konstantinov and Lampert, 2019]. It provides a natural

framework to analyze the behavior of models trained on diverse distributions, and is practically appealing since it can be easily estimated in the context of classification [Ben-David et al., 2010].

We make the following assumptions on the quality indexes ( θ , . . . , θ ) , hereafter referred to as types .

<!-- formula-not-decoded -->

1 J ¯ + 1 2 J ¯ .

H 3 The ordering assumption is just for ease of exposition but is neither used by the aggregator nor the agents. Our condition that types are strictly different is for convenience in simplifying the proofs.

Collaborative learning framework. We further suppose that agent j ∈ [ J ] can fit a model g ∈ G based on i.i.d. samples { ( X j 1 , Y j 1 ) , . . . , ( X j n j , Y j n j ) } of size n j ⩾ 0 from P j ∈ P in one of two ways: they can compute on their own, or collaborate. This is captured by the two following options:

Option 1) Agent j performs ERM on their own samples:

<!-- formula-not-decoded -->

This non-collaborative procedure is referred to as the outside option .

Option 2) Agent j can take part in a coalition orchestrated by a central data aggregator, encoded as B = ( B 1 , . . . , B J ) ∈ { 0 , 1 } J , where B j = 1 means that agent j is member of the coalition. We also write B = { j ∈ [ J ] : B j = 1 } . In exchange for their samples, agent j gains access to the collaborative model trained over the concatenation of samples:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Agent utilities. We assume that agents incur a unitary cost for sampling from their distribution and dislike statistical risk. Therefore, a baseline model for measuring the preferences associated with a model g ∈ G and a number of samples n is based on a linear map: ( g, n ) ↦→-a R 0 ( g ) -cn for a, c &gt; 0 . In practice, however, R 0 ( g ) is typically unknown so agents instead can use a PAC bound of the form P ( R 0 (ˆ g n ) ⩽ R ⋆ 0 + ε ) ⩾ 1 -δ to a assess a model ˆ g n ∈ G trained over their samples, where ε &gt; 0 , δ ∈ (0 , 1) and R ⋆ 0 = inf g ∈ G R 0 ( g ) . Our next result shows that H 1 and H 2 allow each agent to pin down such an ε &gt; 0 , under either Option 1) or Option 2). Assuming H 1, define the function ε for any ( θ, n ) ∈ Θ × R + by

<!-- formula-not-decoded -->

Lemma 1. Assume H 1 and H 2.

(i) Any agent j ∈ [ J ] picking the outside Option 1) obtains a model ̂ g j ∈ G achieving

<!-- formula-not-decoded -->

(ii) Any coalition B ∈ { 0 , 1 } J drawing n = ( n 1 , . . . , n J ) ∈ R J + , samples obtains a model ̂ g B ∈ G achieving

<!-- formula-not-decoded -->

where N is the total of samples and ϑ is the weighted average type within the coalition:

<!-- formula-not-decoded -->

When the context is clear, we write ε ( B , n ) = ε ( ϑ ( B , n ) , N ( B , n )) to lighten notation. Based on Lemma 1, we define the utility of agent j ∈ [ J ] as

<!-- formula-not-decoded -->

Note that for any j ∈ [ J ] , we take n j ⩾ 0 to be a real number for ease of presentation. In (5), a/c &gt; 0 captures the extent to which individuals are willing to trade off model quality against sampling cost.

For any j ∈ [ J ] and B -j = ( B 1 , . . . , B j -1 , B j +1 , . . . , B J ) ∈ { 0 , 1 } J -1 , we denote by a slight abuse of notation ( B, B -j ) = ( B 1 , . . . , B j -1 , B, B j +1 , . . . , B J ) for any B ∈ { 0 , 1 } . Similarly for n -j = ( n 1 , . . . , n j -1 , n j +1 , . . . , n j ) ∈ R J -1 + , we write ( n, n -j ) = ( n 1 , . . . , n j -1 , n, n j +1 , . . . , n j ) for any n ⩾ 0 . We can then characterize the optimal behavior of any agent picking the outside option as follows.

Proposition 1. Assume H 1 and H 2. For any j ∈ [ J ] , B -j ∈ { 0 , 1 } J -1 and n -j ∈ R J -1 + , the optimal number of samples to draw under Option 1) is argmax n ≥ 0 u j ((0 , B -j ) , ( n, n -j ) ; θ j ) = n where

<!-- formula-not-decoded -->

In what follows, we assume 2 a/c &gt; ( γα δ ) -1 to exclude the pathological case where no agent is willing to sample data points. From now on, we denote by

<!-- formula-not-decoded -->

the best achievable utility under the outside option. Note that n does not depend on θ j ∈ Θ (but u j does) so all agents outside of the coalition draw a same number of data points n &gt; 0 . This result, which may be surprising at first glance, comes from the fact that all agents have the same accuracy-to-sampling-cost ratio a/c in their utility.

Aggregator. We finally assume that the aggregator acts benevolently to set up a Pareto-optimal collaboration, by maximizing the total welfare under individual rationality. In other words, they solve:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the Social Choice literature, W is referred to as the utilitarian social welfare function. The participation constraint ensures that no agent within the coalition finds it beneficial to switch to their outside option. Note that n ∈ R J + is required to have non-negative entries, which prevents the aggregator from giving away data points to agents.

## 3 Full-Information Benchmark: First-Best Collaboration

In this section, we assume that the profile of types ( θ 1 , . . . , θ J ) ∈ Θ J is public, and study how the aggregator can implement an optimal collaboration among agents under this most-favorable scenario.

Exact solution. We are looking for a solution to the aggregator's problem (8). For any B ∈ { 0 , 1 } J and n ∈ R J + , denote by

<!-- formula-not-decoded -->

the maximum number of samples that agent j can be asked to provide within the coalition under its participation constraint, where n is defined as in (6), and u j as in (7). With this notation, problem (8) rewrites

<!-- formula-not-decoded -->

Theorem 1. Assume H 1, H 2, H 3. Problem (8) admits a unique solution ( B opt , n opt ( θ )) ∈ { 0 , 1 } J × R J + . Moreover,

<!-- formula-not-decoded -->

(ii) Denoting n opt ( θ ) = ( n opt ( θ ) , . . . , n opt ( θ )) , there exists L opt ∈ [ J ] such that for any j ∈ [ J ]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The couple ( B opt , n opt ) is referred to as the optimal contribution scheme . Although implicit, the condition (11) provides insights about the optimal scheme. The aggregator makes everyone enter the coalition, but only asks the L opt &gt; 0 first-best agents to contribute. This allows to obtain the best possible collaborative model while sparing any sampling cost to other agents. Moreover, the number of required samples n opt j ( θ ) slightly differs from n according to the relative performance of the collaborative model with respect to agent j 's one: if the agent gets a better accuracy by collaborating, the aggregator can ask them for more data; if on the other hand the agent gets a worse model by collaborating (i.e., they are a contributor with very high quality data), the aggregator can only ask less data because of the participation constraint.

Relaxed solution. Working with the optimal scheme ( B opt , n opt ) is difficult because n j ( 1 , n opt ( θ )) has no explicit expression. To make the analysis tractable, we slightly simply the optimal contribution scheme in Theorem 1 and consider the simplified optimal contribution scheme ( B ⋆ , n ⋆ ) where B ⋆ = B opt = (1 , . . . , 1) and for any j ∈ [ J ] ,

<!-- formula-not-decoded -->

Note that ( B ⋆ , n ⋆ ) only differs from ( B opt , n opt ) in two ways. First, in ( B ⋆ , n ⋆ ) all contributors' participation constraint bind, while in ( B opt , n opt ) the L opt -the one could be slack. Second, the total number of data points required from the coalition is fixed and equal to ¯ N = Θ( J 1 1+ γ ) . The quantity ¯ N comes from a natural relaxation of the original problem Equation (10) where we leave aside an intricate term of the objective function. This relaxation, which is formally described in Appendix A, provides a good approximation of the exact solution in reasonable settings. Indeed, the following result establishes that applying ( B ⋆ , n ⋆ ) instead of ( B opt , n opt ) comes at a negligible welfare cost when types are sufficiently evenly spaced.

Lemma 2. Assume H 1, H 2 and θ j -θ j -1 = O (1 /J ) for any j ∈ { 2 , . . . , J } . Then,

<!-- formula-not-decoded -->

Moreover, the following proposition shows that n ⋆ ( θ ) admits a workable expression.

Corollary 1. Assume H 1, H 2 and H 3. Then L ⋆ = Θ( J 1 1+ γ ) and for any j ∈ [ J ] ,

<!-- formula-not-decoded -->

Since ( B ⋆ , n ⋆ ) correctly approximates the optimal scheme while being more tractable, we work with it in the remainder to lighten proofs.

H4. The aggregator applies the simplified contribution scheme ( B ⋆ , n ⋆ ) .

## 4 Hidden information

The welfare-maximizing contribution scheme described in (12) depends explicitly on θ ∈ Θ J , so it is implementable only if types are public. This often unrealistic, either for legal or competitive reasons. We therefore turn to the problem of setting up a collaboration when types are private.

## 4.1 Naive aggregation and unravelling

A naive solution to coping with the private nature of θ ∈ Θ J is for the aggregator to ask agents to disclose their types, and apply the simplified optimal contribution scheme defined in (12). In this setting, however, agents may declare a type ˜ θ j different from their true type θ j .

This approach corresponds to a direct-revelation mechanism Γ : ( B , ˜ θ ) ↦→ n ⋆ ( ˜ θ ) which unfolds as follows.

1. Any agent j ∈ [ J ] declares a tuple ( B j , ˜ θ j ) ∈ { 0 , 1 } × Θ ∪ {†} . If B j = 1 , then agent j picks Option 2), and enters the coalition with type ˜ θ j . If B j = 0 , then agent j picks Option 1), their declared type ˜ θ j is † by convention.
2. Setting B = ( B 1 , . . . , B J ) and ˜ θ ∈ (Θ ∪{†} ) J , then the aggregator applies the contribution scheme defined in (12), so the vector of number of contributions within the coalition is n ⋆ ( ˜ θ ) .

Γ induces a direct revelation game ([ J ] , S J , ( v j ) j ∈ [ J ] ) where the action space is S = { (1 , ˜ θ ) : ˜ θ ∈ Θ } ∪ { (0 , † ) } and payoffs are for any j ∈ [ J ] and s ∈ S J ,

<!-- formula-not-decoded -->

This mechanism is obviously vulnerable to strategic manipulation, since it disregards incentive compatibility. This has severe consequences for the coalition, as shown by the following proposition.

Theorem 2 (Unravelling) . Assume H 1, H 2, H 3, and H 4. Let E ⊂ S J be the set of pure-strategy Nash equilibria of the game induced by Γ . We have

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 2 shows that under Γ , the coalition undergoes a full unravelling: it is either empty or comprised solely of the worst agent in any Nash equilibrium. Thus, collaborative learning is not immune to adverse selection, and may suffer from unravelling as any market characterized by information asymmetry.

̸

Sketch of proof. The profile of actions ((0 , † ) , . . . , (0 , † )) corresponding to B = (0 , . . . , 0) is a pure Nash equilibrium, since forming a lone coalition cannot bring more utility than picking the outside option. Conversely, consider a pure-strategy Nash equilibrium s ∈ E such that B = (0 , . . . , 0) . Denote by C = { j ∈ [ J ] : B j = 1 and n ⋆ j ( ˜ θ ) &gt; 0 } the set of contributors under this equilibrium. It can be shown that (i) for any ( j, k ) ∈ C 2 , θ j -˜ θ j = θ k -˜ θ k , and (ii) For any j ∈ C , (1 , ˜ θ j ) with ˜ θ j &gt; θ is strictly dominated by (1 , θ ) so ˜ θ j = θ at the equilibrium. As a consequence, θ j = θ k for any ( j, k ) ∈ C 2 , which implies by H 3 that |C| = 1 . From the definition of the contribution scheme (12), we can deduce that ∑ j ∈ [ J ] B j = |B| = 1 , because |B| &gt; 1 would entail |C| &gt; 1 . Finally, B J = 1 because v J ((1 , θ ) , s -J ) &gt; v J ((0 , † ) , s -J ) and s is a Nash equilibrium. This leads to B = (0 , . . . , 0 , 1) .

## 4.2 Breaking unravelling

The previous results motivate the design of a more sophisticated aggregation scheme that addresses adverse selection. In this section, we discuss how to design such a procedure.

Is VCG available in our framework? Unravelling occurs under Γ because agents do not find it beneficial to declare their true type and eventually opt for their outside option. This could be avoided by modifying Γ to make it

- (i) individually rational , that is v j ((1 , θ j ) , s -j ) ⩾ v j ((0 , † ) , s -j ) for any j ∈ [ J ] , s -j ∈ S J -1 ,
- (ii) and incentive compatible , that is v j ((1 , θ j ) , s -j ) ⩾ v j ((1 , ˜ θ j ) , s -j ) for any ˜ θ j ∈ Θ .

Under these conditions, the truthful, optimal profile of actions ((1 , θ 1 ) , . . . , (1 , θ J )) would emerge as a Nash equilibrium. Since the aggregator seeks to minimize the utilitarian function W , one option would be to rely on the VCG mechanism [Vickrey, 1961, Clarke, 1971, Groves, 1973], which is the direct-revelation mechanisms fulfilling these desiderata [Green and Laffont, 1977, Holmstrom, 1979]. Formally, the VCG mechanism writes Γ VCG : ˜ θ ↦→ ( n ⋆ ( ˜ θ ) , t ( ˜ θ )) where t ( ˜ θ ) = ( t 1 ( θ ) , . . . , t J ( θ )) ∈ R J is a set of transfers satisfying for any j ∈ [ J ] :

̸

<!-- formula-not-decoded -->

̸

It re-establishes truthfulness as a dominant strategy by aligning individual payoffs v VCG j ( ˜ θ ) = u j ( 1 , n ⋆ ( ˜ θ )) -t j ( ˜ θ ) with total social welfare. Unfortunately, the VCG approach is unavailable in our framework, because of the following observation.

<!-- formula-not-decoded -->

Lemma 3 shows that some agent would need to receive a strictly positive transfer. This is impossible without a monetary payment-utility can only be decreased by the aggregator, for instance through accuracy shaping [Karimireddy et al., 2022]-, which we exclude here.

A probabilistic verification-based mechanism. We now show how to design a mechanism that recovers the optimal collaboration as a Nash equilibrium in high probability without the need for transfers. Inspired by the probabilistic verification approach [Caragiannis et al., 2012, Ferraioli and Ventre, 2018, Ball and Kattwinkel, 2019], we assume that the aggregator can approximately estimate θ j with few samples from P j for any j ∈ [ J ] :

H5. There exists a decreasing function η δ : R ⋆ + → R ⋆ + , with δ ∈ (0 , 1) defined in H 1, such that for any j ∈ [ J ] and i.i.d samples ( X j 1 , Y j 1 ) , . . . , ( X j q , Y j q ) of size q &gt; 0 from P j , there exists a ( X j 1 , Y j 1 ) , . . . , ( X j q , Y j q ) -measurable estimator ̂ θ j satisfying

<!-- formula-not-decoded -->

In Section 4.2, we show how the aggregator can design such estimators. H 5 allows us to consider a new mechanism ̂ Γ : B ↦→ m ( B ) as follows:

1. any agent j ∈ [ J ] declares B j ∈ { 0 , 1 } . If B j = 1 , the principal asks for q ⩽ n -2( a/c )( ¯ θ -θ ) i.i.d samples from P j and estimates types as ̂ θ = ( ̂ θ j ) j ∈B following H 5.
2. Based on the estimated types ̂ θ = ( ̂ θ j ) j ∈B , the aggregator asks for [ n ⋆ j ( ̂ θ + η j ) -q ] + additional samples from P j , where n ⋆ ( · ) is defined as in (12) and

<!-- formula-not-decoded -->

Thus, the number of draws required from agent j is max[ q , n ⋆ j ( ̂ θ + η j )] .

3. The aggregator keeps m j ( ̂ θ ) = 1 { n ⋆ j ( ̂ θ + η j ) &gt; 0 } max[ q , n ⋆ j ( ̂ θ + η j )] samples from agent j , and trains a collaborative model with these pooled samples.

ˆ Γ induces a game ([ J ] , { 0 , 1 } J , (ˆ v j ) j ∈ [ J ] ) where any agent j ∈ [ J ] has a payoff function

<!-- formula-not-decoded -->

The rationale behind this mechanism is fairly intuitive: since ̂ θ is a correct estimate of θ , n ⋆ j ( ̂ θ + η j ) ≈ n ⋆ j ( ̂ θ ) correctly approximates the optimal contribution n ⋆ j ( θ ) for any contributor j ∈ B . Note that type estimates are purposely biased by η j when asking for samples. This is a safeguard against over-estimated types, which would lead to asking to many data points and could deter agents from participating in the coalition.

Critically, m ( ̂ θ ) does not depend on declared type, so agents are no longer able to strategically manipulate the mechanism. Moreover, ˆ Γ does not require agents to know their own types, which would be an unrealistic assumption. Finally, observe that the number of data points asked to estimate types q is low enough to never deter agents from participating in the coalition.

Theorem 3. Assume H 1, H 2, H 3, H 4 and H 5. B ⋆ = (1 , . . . , 1) T is a Nash equilibrium under ˆ Γ with probability 1 -δ .

Theorem 3 shows that the optimal coalition is a sustainable equilibrium under ˆ Γ , which effectively breaks unravelling: the set of (approximate) Nash equilibria is no more reduced to profiles of actions where the coalition is empty, or reduced to the worst agent.

Practical implementation. We now explain how to practically implement ̂ Γ in a collaborative learning setting. This requires defining a collection of estimators ( ̂ θ j ) j satisfying H 5. To this end, we assume that few samples from the target distribution are available.

H6. There are q ′ &gt; 0 i.i.d samples { ( X 0 1 , Y 0 1 ) , . . . , ( X 0 q ′ , Y 0 q ′ ) } from P 0 available to the aggregator and agents.

Under H 6, define ̂ R 0 ( g ) = q ′-1 ∑ q ′ i =1 ℓ ( g ( X 0 i ) , Y 0 i ) for g ∈ G . This allows us to devise suitable estimators ̂ θ j as follows.

Proposition 2. Assume H 1, H 2 and H 6. For any j ∈ [ J ] the estimator

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

satisfies H 5 with

Proposition 2 shows that the empirical version of the G -divergence defined in H 2 correctly estimate types. Note that the tighter the PAC bound in H 1, the better the approximation term in (13). The type estimator ̂ θ ERM 0 ,j defined in H 5 can easily be computed in the classification case, as shown with the following example.

H7 (Classification setting) . Y = {-1 , 1 } , ℓ = ℓ 0 , 1 : ( y, y ′ ) ∈ Y × Y ↦→ ✶ { yy ′ &lt; 0 } , and G is a symmetric ( g ∈ G if and only if -g ∈ G ) class of classifiers.

Example 4. Assume H 1, H 2, H 6 and H 7.

- (i) Denoting ̂ R j -( g ) = n -1 j ∑ n j i =1 ℓ 0 , 1 ( g ( X j i ) , -Y j i ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(ii) In H 1, assume α δ = ln(1 /δ ) 1 / 2 , β = 2 RAD ( G ) and γ = 1 [Bousquet et al., 2003]. With ̂ θ ERM 0 ,j defined in Proposition 2, we have

<!-- formula-not-decoded -->

Example 4 shows that in the classification case, it is sufficient to flip the labels of the data received from each contributor, merge these samples with those from P 0 , and perform an empirical risk minimization to compute ̂ θ ERM 0 ,j . The approximation error grows no more than logarithmically with the number of agents, while decreasing at rate γ with the number of samples used in the estimation.

## 5 Conclusion

In this work, we show that information asymmetry has deleterious consequences when strategic agents try to learn a collaborative model. More precisely, under a naive aggregation procedure, the ignorance of others' data quality leads the coalition of learners to be either empty or reduced to the lowest-quality agent. We introduce a transfer-free mechanism based on estimation of types. This effectively counteracts unravelling by letting the grand coalition ranks among the approximate Nash equilibria with high probability.

Several possible extensions can be considered. First, it would be interesting to relax the assumption that all agents have the same ratio a/c in their utility, and see how heterogeneity affects the results.

Second, the mechanism presented in Section 4 aims for individual rationality. A more desirable, yet difficult to achieve, property would be core stability, to ensure that no group of agents would benefit from a coordinated deviation, i.e., forming an alternative coalition. Finally, it would be interesting to check whether there exist mechanisms where the optimal collaboration not only emerges as a Nash equilibrium, but as a dominant equilibrium under imperfect information.

## Acknowledgements

Funded by the European Union (ERC, Ocean, 101071601). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References

- George A. Akerlof. The market for lemons: Quality uncertainty and the market mechanism. The Quarterly Journal of Economics , 84(3):488-500, 1970. URL https://ideas.repec.org/a/ oup/qjecon/v84y1970i3p488-500..html .
- Nivasini Ananthakrishnan, Stephen Bates, Michael Jordan, and Nika Haghtalab. Delegating data collection in decentralized machine learning, 2023.
- Ian Ball and Deniz Kattwinkel. Probabilistic verification in mechanism design. September 2019.
- Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Vaughan. A theory of learning from different domains. Machine Learning , 79:151-175, 05 2010. doi: 10.1007/s10994-009-5152-4.
- Avrim Blum, Nika Haghtalab, Ariel Procaccia, and Mingda Qiao. Collaborative PAC learning. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems . Curran Associates, 2017.
- Avrim Blum, Nika Haghtalab, Richard Lanas Phillips, and Han Shao. One for one, or all for all: Equilibria and optimality of collaboration in federated learning. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning , volume 139 of Proceedings of Machine Learning Research , pages 1005-1014. PMLR, 18-24 Jul 2021. URL https://proceedings.mlr.press/v139/blum21a.html .
- Olivier Bousquet, Stéphane Boucheron, Gábor Lugosi, Ulrike Luxburg, and Gunnar Rätsch. Introduction to statistical learning theory. Advanced Lectures on Machine Learning, 169-207 (2004) , 01 2003. doi: 10.1007/978-3-540-28650-9\_8.
- Ioannis Caragiannis, Edith Elkind, Mario Szegedy, and Lan Yu. Mechanism design: From partial to probabilistic verification. Proceedings of the ACM Conference on Electronic Commerce , 06 2012. doi: 10.1145/2229012.2229035.
- Edward Clarke. Multipart pricing of public goods. Public Choice , 11(1):17-33, September 1971. doi: 10.1007/BF01726210. URL https://ideas.repec.org/a/kap/pubcho/ v11y1971i1p17-33.html .
- Kate Donahue and Jon Kleinberg. Optimality and stability in federated learning: A game-theoretic approach, 06 2021a.
- Kate Donahue and Jon Kleinberg. Model-sharing games: Analyzing federated learning under voluntary participation. Proceedings of the AAAI Conference on Artificial Intelligence , 35:53035311, 05 2021b. doi: 10.1609/aaai.v35i6.16669.
- Florian E. Dorner, Nikola Konstantinov, Georgi Pashaliev, and Martin T. Vechev. Incentivizing honesty among competitors in collaborative learning and optimization. ArXiv , abs/2305.16272, 2023. URL https://api.semanticscholar.org/CorpusID:258887502 .

- Liran Einav and Amy Finkelstein. Selection in insurance markets: Theory and empirics in pictures. Journal of Economic Perspectives , 25(1):115-38, March 2011. doi: 10.1257/jep.25.1.115. URL https://www.aeaweb.org/articles?id=10.1257/jep.25.1.115 .
- Diodato Ferraioli and Carmine Ventre. Probabilistic verification for obviously strategyproof mechanisms, 2018.
- Lei Fu, Huanle Zhang, Ge Gao, Mi Zhang, and Xin Liu. Client selection in federated learning: Principles, challenges, and opportunities, 2023.
- Dashan Gao, Xin Yao, and Qiang Yang. A survey on heterogeneous federated learning. arXiv preprint arXiv:2210.04505 , 2022.
- Jerry Green and Jean-Jacques Laffont. Characterization of satisfactory mechanisms for the revelation of preferences for public goods. Econometrica , 45(2):427-38, 1977. URL https://EconPapers. repec.org/RePEc:ecm:emetrp:v:45:y:1977:i:2:p:427-38 .
- Theodore Groves. Incentives in teams. Econometrica , 41:617-631, 1973. URL https://api. semanticscholar.org/CorpusID:264740987 .
- Nathaniel Hendren. Private Information and Insurance Rejections. Econometrica , 81(5):17131762, September 2013. doi: ECTA10931. URL https://ideas.repec.org/a/ecm/emetrp/ v81y2013i5p1713-1762.html .
- Bengt Holmstrom. Groves' scheme on restricted domains. Econometrica , 47(5):1137-44, 1979. URL https://EconPapers.repec.org/RePEc:ecm:emetrp:v:47:y:1979:i:5:p:1137-44 .
- Baihe Huang, Sai Praneeth Karimireddy, and Michael I. Jordan. Evaluating and Incentivizing Diverse Data Contributions in Collaborative Learning. Papers 2306.05592, arXiv.org, June 2023. URL https://ideas.repec.org/p/arx/papers/2306.05592.html .
- Chao Huang, Shuqi Ke, Charles Kamhoua, Prasant Mohapatra, and Xin Liu. Incentivizing data contribution in cross-silo federated learning, 2022.
- Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, Rafael G. L. D'Oliveira, Hubert Eichner, Salim El Rouayheb, David Evans, Josh Gardner, Zachary Garrett, Adrià Gascón, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaid Harchaoui, Chaoyang He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara Javidi, Gauri Joshi, Mikhail Khodak, Jakub Konevcný, Aleksandra Korolova, Farinaz Koushanfar, Sanmi Koyejo, Tancrède Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock, Ayfer Özgür, Rasmus Pagh, Mariana Raykova, Hang Qi, Daniel Ramage, Ramesh Raskar, Dawn Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tramèr, Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu, and Sen Zhao. Advances and open problems in federated learning, 2021.
- Sai Praneeth Karimireddy, Wenshuo Guo, and Michael I. Jordan. Mechanisms that incentivize data sharing in federated learning, 2022.
- Daniel Kifer, Shai Ben-David, and Johannes Gehrke. Detecting change in data streams. pages 180-191, 04 2004.
- Nikola Konstantinov and Christoph Lampert. Robust learning from untrusted sources, 2019.
- Jean-Jacques Laffont and David Martimort. The Theory of Incentives: The Principal-Agent Model . Princeton University Press, Princeton, NJ, USA, 2001.
- Shutian Liu, Tao Li, and Quanyan Zhu. Game-theoretic distributed empirical risk minimization with strategic network design. IEEE Transactions on Signal and Information Processing over Networks , 9:542-556, 2023. doi: 10.1109/TSIPN.2023.3306106.
- Andreu Mas-Colell, Michael D. Whinston, and Jerry R. Green. Microeconomic Theory . Number 9780195102680 in OUP Catalogue. Oxford University Press, 1995. ISBN ARRAY(0x516030e0). URL https://ideas.repec.org/b/oxp/obooks/9780195102680.html .

- Michael Rothschild and Joseph Stiglitz. Equilibrium in competitive insurance markets: An essay on the economics of imperfect information. The Quarterly Journal of Economics , 90(4):629649, 1976. URL https://EconPapers.repec.org/RePEc:oup:qjecon:v:90:y:1976:i: 4:p:629-649.

Eden Saig, Inbal Talgam-Cohen, and Nir Rosenfeld. Delegated classification, 2023.

- Vera Shalaeva, Alireza Fakhrizadeh Esfahani, Pascal Germain, and Mihály Petreczky. Improved pac-bayesian bounds for linear regression. In AAAI Conference on Artificial Intelligence , 2019. URL https://api.semanticscholar.org/CorpusID:208857400 .
- Nikita Tsoy and Nikola Konstantinov. Strategic data sharing between competitors. Advances in Neural Information Processing Systems , 36, 2024.
- Xuezhen Tu, Kun Zhu, Nguyen Cong Luong, Dusit Niyato, Yang Zhang, and Juan Li. Incentive mechanisms for federated learning: From economic and game theoretic perspective. IEEE Transactions on Cognitive Communications and Networking , 8(3):1566-1593, 2022. doi: 10.1109/ TCCN.2022.3177522.
- William Vickrey. Counterspeculation, auctions, and competitive sealed tenders. Journal of Finance , 16 (1):8-37, 1961. URL https://EconPapers.repec.org/RePEc:bla:jfinan:v:16:y:1961: i:1:p:8-37 .
- Zhepei Wei, Chuanhao Li, Tianze Ren, Haifeng Xu, and Hongning Wang. Incentivized truthful communication for federated bandits, 2024.
- Mariel Werner, Sai Praneeth Karimireddy, and Michael I. Jordan. Defection-free collaboration between competitors in a learning system, 2024. URL https://arxiv.org/abs/2406.15898 .
- Yizhou Yan, Xinyu Tang, Chao Huang, and Ming Tang. Price of stability in quality-aware federated learning. In GLOBECOM 2023 - 2023 IEEE Global Communications Conference , pages 734-739, 2023. doi: 10.1109/GLOBECOM54140.2023.10437743.

## A Relaxation of the optimal aggregation problem.

Let B ∈ { 0 , 1 } J and the associated coalition B = { j ∈ [ J ] : B j = 1 } be fixed. We present in this appendix a natural relaxation of problem Equation (10) which motivates the choice ∑ j ∈B n ⋆ j ( θ ) = ¯ N = ( n +1) |B| 1 1+ γ -1 in the simplified contribution scheme ( B ⋆ , n ⋆ ) . The optimal aggregation problem with B fixed is

<!-- formula-not-decoded -->

Since agents outside of the coalition maximize their utility, the problem can equivalently be stated as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

We start by rewriting ( T B ) in a more convenient way. Instead of working with n ∈ R |B| + directly, we make appear (i) the total number of samples and (ii) the sharing out of samples between agents. Formally, for any n ∈ R |B| + consider N = n T B and λ = N -1 n ∈ ∆ |B| , where ∆ |B| is the simplex of dimension |B| . Observe that the average type in the coalition reads ϑ ( B , n ) = λ T θ . Moreover,

the welfare evaluated in ( B , n ) rewrites

<!-- formula-not-decoded -->

We denote by ˜ Ξ B = { ( λ , N ) ∈ ∆ |B| × R + : λ N ∈ Ξ B } and ˜ T B the problem:

<!-- formula-not-decoded -->

By definition, if ( λ , N ) is a solution to ˜ T B , then n = λ N is a solution to T B . We can further decompose ˜ T B by observing that

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

̸

Finally, we denote a slice of ˜ Ξ B along N ⩾ 0 as ˜ Ξ B ( N ) = { λ ∈ ∆ |B| : ( λ , N ) ∈ ˜ Ξ B } and N = { N ⩾ 0 : ˜ Ξ B ( N ) = ∅} . ˜ T B comes down to

<!-- formula-not-decoded -->

A strategy to solve Equation (16) is to (i) address the innermost problem min λ ∈ ˜ Ξ B ( N ) g ( λ ) with N ∈ N fixed, and denoting λ ( N ) ∈ ˜ Ξ B ( N ) its solution, (ii) solve the outermost problem:

<!-- formula-not-decoded -->

Point (i) is done in the proof of Theorem 1. However, the resulting problem in (ii) is hard to tackle because λ ( N ) has no simple form. Therefore, we leave aside the term g ( λ ( N ) ) (which can be easily controlled, as shown in Lemma 2) to determine N and only consider the problem

<!-- formula-not-decoded -->

Since f is differentiable and strictly convex, its minimizer is uniquely defined by f ′ ( ¯ N ) = 0 , which gives

<!-- formula-not-decoded -->

where n is defined in Proposition 1. As the solution of the relaxed problem (18), we take ∑ j ∈ [ J ] n ⋆ j ( θ ) = ¯ N is the simplified contribution scheme ( B ⋆ , n ⋆ ) . In many reasonable settings, this approximation is very satisfactory as shown by Lemma 2.

## B Proofs

Lemma 1. Assume H 1 and H 2.

(i) Any agent j ∈ [ J ] picking the outside Option 1) obtains a model ̂ g j ∈ G achieving

<!-- formula-not-decoded -->

(ii) Any coalition B ∈ { 0 , 1 } J drawing n = ( n 1 , . . . , n J ) ∈ R J + , samples obtains a model ̂ g B ∈ G achieving

<!-- formula-not-decoded -->

where N is the total of samples and ϑ is the weighted average type within the coalition:

<!-- formula-not-decoded -->

Proof. (i) Let { ( X j 1 , Y i 1 ) , . . . , ( X j n j , Y j n j ) } be n j &gt; 0 i.i.d samples from P j ∈ P and ̂ g j = inf g ∈ G ̂ R j ( g ) . First, observe that

<!-- formula-not-decoded -->

Let υ &gt; 0 and g υ j ∈ G such that R j ( g υ j ) ⩽ inf g ∈ G R j ( g ) + υ . Then with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where the last inequality holds taking the limit υ → 0 and using H 1. Finally, observing that inf g ∈ G R j ( g ) - R ⋆ 0 = inf g ∈ G R j ( g ) -inf g ∈ G R 0 ( g ) ⩽ sup g ∈ G |R j ( g ) -R 0 ( g ) | = θ j , and combining (20) as well as (21) yields

<!-- formula-not-decoded -->

with probability at least 1 -δ .

(ii) Let B ∈ { 0 , 1 } J and n = ( n 1 , . . . , n j ) ∈ R J + . For any j ∈ B , let { ( X j 1 , Y i 1 ) , . . . , ( X j n j , Y j n j ) } be a collection of n j &gt; 0 i.i.d samples from P j ∈ P . Denote by N = ∑ j ∈ [ J ] B j n j and consider λ = ( λ 1 , . . . , λ J ) such that λ j = B j n j /N . Note that λ belongs to the J -dimensional simplex. For any g ∈ G , the empirical risk over contributions is

<!-- formula-not-decoded -->

and the population risk is

<!-- formula-not-decoded -->

One the one hand, the collaborative model ̂ g B ∈ G satisfies:

<!-- formula-not-decoded -->

with ϑ ( B , n ) = N -1 ∑ j ∈ [ J ] B j n j θ j = ∑ j ∈ [ J ] B j λ j θ j . Now, let υ &gt; 0 and g υ ∈ G such that R B ( g υ ) ⩽ inf g ∈ G R B ( g ) + υ . We also have

<!-- formula-not-decoded -->

where the first inequality comes from ̂ R B ( ̂ g B ) = inf g ∈ G ̂ R B ( g ) ⩽ ̂ R B ( g υ ) , and the last is obtained by taking υ → 0 and applying assumption H 1. Now, observe that inf g ∈ G R B ( g ) - R ⋆ 0 = inf g ∈ G R B ( g ) -inf g ∈ G R 0 ( g ) ⩽ sup g ∈ G |R B ( g ) -R 0 ( g ) | ⩽ ∑ j B j λ j sup g ∈ G |R j ( g ) -R 0 ( g ) | = ϑ ( B , n ) . Combining this observation with Equation (25) and Equation (26) gives with probability 1 -δ

<!-- formula-not-decoded -->

Proposition 1. Assume H 1 and H 2. For any j ∈ [ J ] , B -j ∈ { 0 , 1 } J -1 and n -j ∈ R J -1 + , the optimal number of samples to draw under Option 1) is argmax n ≥ 0 u j ((0 , B -j ) , ( n, n -j ) ; θ j ) = n where

<!-- formula-not-decoded -->

Proof. For any j ∈ [ J ] and n ≥ 0 , Lemma 1 gives that

<!-- formula-not-decoded -->

Since f is strictly convex and differentiable, it admits a unique maximizer n ⩾ 0 determined by f ′ ( n ) = 0 . Simple algebra leads to n = (2 aγc -1 α δ ) 1 /γ +1 -1 .

Theorem 1. Assume H 1, H 2, H 3. Problem (8) admits a unique solution ( B opt , n opt ( θ )) ∈ { 0 , 1 } J × R J + . Moreover,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Recall that the optimal aggregation problem reads

<!-- formula-not-decoded -->

Define for any j ∈ [ J ] , B -j ∈ { 0 , 1 } J -1 , and n -j ∈ R J -1 + , the maximum number of samples agent j ∈ [ J ] having B j = 1 may be asked given their participation constraint:

<!-- formula-not-decoded -->

Given Equation (5) and Proposition 1, we obtain

<!-- formula-not-decoded -->

where N = ∑ j ∈ [ J ] B j n j = n T B . Problem (28) rewrites in canonical form

<!-- formula-not-decoded -->

where B = { j ∈ [ J ] : B j = 1 } . We first show that it admits a unique minimum.

Fix a B ∈ { 0 , 1 } J and consider problem (30) with respect to n ∈ R J + only. We call this subproblem T B . First, we show that it is enough to focus on ( n j ) j ∈B rather than ( n j ) j ∈ [ J ] in T B . With n B = (n j ) j ∈B and n B c = (n j ) j / ∈B , note that W ( B , n ) is decomposable in n B and n B c :

<!-- formula-not-decoded -->

where, by a slight abuse of notation W ( B , n B ) = ∑ j ∈ B u j ( B , n ) and W ( B , n B c ) = ∑ j ∈B c u j ( B , n ) . With

<!-- formula-not-decoded -->

problem T B is equivalent to

<!-- formula-not-decoded -->

by Proposition 1. Since ∑ j ∈ [ J ] -(1 -B j ) u j is constant, by Equation (32) it is enough to focus on the existence of min n B ∈ Ξ B -W ( B , n B ) . On the one hand, Ξ B is bounded. Indeed for any j ∈ B , by Equation (29) and Lemma 1

<!-- formula-not-decoded -->

Thus, Ξ B ⊂ [0 , M ] |B| . Moreover, Ξ B is closed and convex. For any j ∈ B , rewrite

<!-- formula-not-decoded -->

and define h j : n B ↦→ n B ,j -g j ( n B ) . Observe that Ξ B = { n B ∈ R |B| + : ∀ j ∈ B , n B ,j ⩾ 0 and h j ( n B ) ⩽ 0 } , that is Ξ B = A ∩ B with A = R |B| + and B = h -1 1 (( -∞ , 0]) ∩ . . . ∩ h -1 |B| (( -∞ , 0]) . For any j ∈ B , h -1 j (( -∞ , 0]) is convex, because so is h j . Additionally, h -1 j (( -∞ , 0]) is closed as the inverse image of a closed set by a continuous function. It follows that Ξ B is convex and closed as the intersection of convex and closed sets.

As a consequence Ξ B is compact and convex, and -W ( B , · ) is strictly convex so T B admits a unique minimizer n opt B ∈ Ξ B .

Since there are finitely many B ∈ { 0 , 1 } J , min {-W ( B , n opt B ) + ∑ j ∈B c u j , B ∈ { 0 , 1 } J } &gt; 0 exists and coincide with the minimum of problem (30). We show later in the proof that the optimal B opt ∈ { 0 , 1 } J is unique (see part 1 ), so the minimizer of Equation (30) is unique. This establishes the point (i) of the result.

The remainder of the proof proceeds as follows: we first characterize the optimal B opt ∈ { 0 , 1 } J , and then prove that the solution n opt B opt ∈ R |B| + of T B opt has the form presented in point (ii) of the result.

Part 1: Optimal B ∈ { 0 , 1 } [ J ]

We show that B opt = (1 , . . . , 1) T . By contradiction, assume there exists B ′ = (1 , . . . , 1) T and n ′ ∈ R J + such that for any ( B , n ) ∈ { 0 , 1 } J × R J +

̸

<!-- formula-not-decoded -->

Denote by j ∈ [ J ] an agent such that B ′ j = 0 , and first assume θ j ⩾ ϑ ( B ′ , n ′ ) . Consider the alternative allocation ( B ′′ , n ′′ ) = ((1 , B ′ -j ) , (0 , n ′ -j )) . Observe that B ′ j n ′ j = B ′′ j n ′′ j = 0 , so

<!-- formula-not-decoded -->

This in particular implies

<!-- formula-not-decoded -->

Thus by (36)

<!-- formula-not-decoded -->

̸

where the second inequality results from θ j ⩾ ϑ ( B ′ , n ′ ) and N ′ ⩾ n . Finally by definition of ( B ′′ , n ′′ ) , (35) and (36), for any k = j such that B ′′ k = 1 :

<!-- formula-not-decoded -->

Hence, (38) together with (37) give

̸

<!-- formula-not-decoded -->

̸

This contradicts (34). Now assume θ j &lt; ϑ ( B ′ , n ′ ) , and let R ∈ [ J ] be such that θ R = max { θ k , k ∈ B} . Consider ( B ′′ , n ′′ ) ∈ { 0 , 1 } J × R J + where

<!-- formula-not-decoded -->

which is feasible. Observe on the one hand that

<!-- formula-not-decoded -->

because θ j &lt; ϑ ( B ′ , n ′ ) ⩽ θ R . In particular

<!-- formula-not-decoded -->

̸

(40) in turn implies that for any k = j such that B ′′ k = 1 :

<!-- formula-not-decoded -->

because n ′ k = n ′′ k for k ∈ B \ { j, R } and n ′′ R &lt; n ′ R . On the other hand, we have

<!-- formula-not-decoded -->

by definition (29) of n j ( B ′′ , n ′′ ) . (41) and (42) together yield

̸

̸

<!-- formula-not-decoded -->

̸

which once again violates (34). Thus, we obtain B opt = (1 , . . . , 1) T , that is B opt = [ J ] .

Part 2: Characterization of n opt [ J ] ∈ R J +

We now focus on the problem T B opt . To lighten notations, we write n opt ∈ R J + instead of n opt B opt , and B instead of B opt = (1 , . . . , 1) . We first reformulate T B in a more convenient way. Recall that T B reads

<!-- formula-not-decoded -->

̸

where Ξ B is defined in (31). As explained in Appendix A, this problem can equivalently be stated as

<!-- formula-not-decoded -->

where with

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Writing ˜ Ξ B ( N ) = { λ ∈ ∆ J : ( λ , N ) ∈ ˜ Ξ B } for N ⩾ 0 and N = { N ⩾ 0 : ˜ Ξ B ( N ) = ∅} , ˜ T B comes down to

̸

<!-- formula-not-decoded -->

̸

In this proof, we address the innermost problem, which is enough to show that n opt satisfies the point (ii) of the result. Let N ⩾ 0 such that ˜ Ξ B ( N ) = ∅ , which exists because Ξ B = ∅ (for instance, ((1 , 0 , . . . , 0) , ( n, 0 , . . . , 0)) ∈ Ξ B ). By definition of ˜ Ξ B , it reads

̸

<!-- formula-not-decoded -->

The following lemma provides a necessary condition for any solution to problem (45).

Lemma 4. Let λ ( N ) = ( λ ( N ) 1 , . . . , λ ( N ) J ) ∈ ∆ J be a solution to (45) . If there exists r ∈ [ J ] such that λ ( N ) r &gt; 0 , then λ ( N ) k = N -1 n k ( B , N λ ( N ) ) for any k &lt; r .

Proof. We denote by ¯ n ( B , n ) = ( n 1 ( B , n ) , . . . , n J ( B , n )) T for any n ∈ R J + . The Lagrangian associated to problem (45), with µ , ρ and ν the associated dual variables, reads:

<!-- formula-not-decoded -->

Since the objective and the constraints are convex, L admits a saddle point ( λ ( N ) , µ , ρ , ν ) ∈ ∆ × R J × R J × R which is solution to problem (45) and verifies the following KKT conditions:

<!-- formula-not-decoded -->





<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assume there exists r ∈ B such that λ ( N ) r &gt; 0 . By complementary slackness (47), µ r = 0 , and (46) gives for any k &lt; r :

<!-- formula-not-decoded -->

That is

<!-- formula-not-decoded -->

By assumption θ r &gt; θ k , and since µ k ⩾ 0 as well as ρ r ⩾ 0 , we have ρ k &gt; 0 . It follows from (48) for agent k that λ opt k N = n ( B , N λ ( N ) ) .

Now, define

We show that

<!-- formula-not-decoded -->

To prove the first point, assume by contradiction that there exists ℓ &lt; L such that λ ( N ) ℓ &lt; n ℓ ( B , λ ( N ) N ) N -1 . By the contrapositive of Lemma 4, λ opt ℓ +1 = . . . = λ opt J = 0 . Thus

<!-- formula-not-decoded -->

by definition of L Equation (51) and ℓ &lt; L . This violates the third constraint of problem (45). For the second point, assume there exists ℓ &gt; L such that λ ( N ) ℓ &gt; 0 . By Lemma 4, λ ( N ) k = n k ( B , λ ( N ) N ) N -1 for any k ⩽ L &lt; ℓ , so

<!-- formula-not-decoded -->

which once again violates the constraints of the problem. Finally by (52),

<!-- formula-not-decoded -->

by (52). By (52) and (53) we obtain for any j ∈ [ J ] :

Now, consider the solution n opt ( θ ) ∈ R J to the initial problem (45), and define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

λ opt ( θ ) satisfies (54), and multiplying by N opt yields for any j ∈ [ J ] :

which establishes point (ii), and concludes the proof.

<!-- formula-not-decoded -->

Corollary 1. Assume H 1, H 2 and H 3. Then L ⋆ = Θ( J 1 1+ γ ) and for any j ∈ [ J ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By definition of the simplified scheme (12), n ⋆ j ( θ ) = ✶ { j ⩽ L ⋆ } n j ( 1 , n ⋆ ( θ )) = ✶ { j ⩽ L ⋆ } [ n -(2 a/c )( ε ( 1 , n ⋆ ( θ )) -ε ( θ j , n ))] for any j ⩽ L ⋆ . Since ∑ j ∈ [ J ] n ⋆ j ( θ ) = ¯ N , we get

<!-- formula-not-decoded -->

By Lemma 1 and denoting ¯ θ L ⋆ = L ⋆ -1 ∑ j ⩽ L ⋆ θ j . This yields

<!-- formula-not-decoded -->

Plugging back this value in n ⋆ j ( 1 , n ⋆ ( θ )) = n -(2 a/c )[ α δ ( (1 + ¯ N ) -γ -(1 + n ) -γ ) + ( ϑ ( 1 , n ⋆ ( θ )) -θ j )] gives for any j ∈ [ J ] :

<!-- formula-not-decoded -->

We now determine the order of magnitude of L ⋆ ∈ { 1 , . . . , J } . Recall on the one hand that by (33), there exists M &gt; 0 such that 0 ⩽ n ⋆ j ( θ ) ⩽ M for any j ∈ [ J ] , and on the other hand that ¯ θ L ⋆ -θ j ⩽ ¯ θ -θ = diam (Θ) &lt; ∞ . Therefore by (56),

<!-- formula-not-decoded -->

Since ¯ N = ( n +1) |B ⋆ | 1 1+ γ -1 and B ⋆ = [ J ] , (57) results in L = Θ( J 1 1+ γ ) .

Lemma 2. Assume H 1, H 2 and θ j -θ j -1 = O (1 /J ) for any j ∈ { 2 , . . . , J } . Then,

<!-- formula-not-decoded -->

Proof. Recall that for any admissible n ∈ Ξ B , we can define N = 1 T n and λ = N -1 n so that the social cost rewrites

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

By definition, ∑ j ∈ [ J ] n ⋆ j ( θ ) = ¯ N where ¯ N = ( n +1) J 1 1+ γ -1 = argmin N ⩾ 0 f ( N ) . Hence,

<!-- formula-not-decoded -->

Since θ j ⩽ θ L ⋆ for any j ∈ [ J ] such that λ ⋆ j &gt; 0 , and θ j ⩾ θ 1 for any j ∈ [ J ] such that λ opt j &gt; 0 :

<!-- formula-not-decoded -->

By assumption, for any j ∈ { 2 , . . . , J } θ j -θ j -1 = O (1 /J ) so there exists k j &gt; 0 and R j ⩾ 0 such that θ j -θ j -1 ⩽ k j J -1 for J ⩾ R j . Denoting R = max j ∈ [ J ] R j and k = max j ∈ [ J ] k j , for any J ⩾ R we have θ L ⋆ -θ 1 ⩽ kL ⋆ , so:

<!-- formula-not-decoded -->

so W ( B opt , n opt ( θ )) -W ( B ⋆ , n ⋆ ( θ )) = O ( L ⋆ ) = O ( J 1 1+ γ ) by Corollary 1.

Theorem 2 (Unravelling) . Assume H 1, H 2, H 3, and H 4. Let E ⊂ S J be the set of pure-strategy Nash equilibria of the game induced by Γ . We have

̸

- (i) E = ∅

<!-- formula-not-decoded -->

Proof. First, we show that the situation where the coalition is empty is a Nash equilibrium. Consider s = ((0 , † ) , . . . , (0 , † )) ∈ S N . For any j ∈ [ J ] and deviation (1 , ˜ θ j ) , B = { j } so ̂ g B = ̂ g j and v j ((0 , † ) , s ⋆ -j ) = u j ⩾ v j ((1 , ˜ θ j ) , s -j ) by Proposition 1. Thus, s ⋆ ∈ E .

Now, consider a pure-strategy Nash equilibrium s ⋆ ∈ E such that B = { j ∈ [ J ] : B j = 1 } ̸ = ∅ . Denote by C = { j ∈ B : n ⋆ j ( ˜ θ ) &gt; 0 } the set of contributors. We start with two technical lemmas:

Lemma 5. There exists ∆ ∈ R such that for any ( j, k ) ∈ C 2

<!-- formula-not-decoded -->

Proof. To lighten notation, we write in this proof

<!-- formula-not-decoded -->

By Equation (12) and H 4, for any j ∈ C , n ⋆ j ( ˜ θ ) = n j ( B , n ⋆ ( ˜ θ )) = n -a c ( ε ( ˜ ϑ ( ˜ θ ) , N ) -ε ( ˜ θ j , n )) , so their payoff reads

<!-- formula-not-decoded -->

Since s is a Nash equilibrium and u j = v j ((0 , † ) , s -j ) , we have in particular that

<!-- formula-not-decoded -->

that is θ j -˜ θ j = ∆ j ⩾ ∆ = ϑ ( ˜ θ ) -˜ ϑ ( ˜ θ ) for any j ∈ C . We now show that this holds with strict equality for any j ∈ C . By contradiction, assume there exists r ∈ C such that ∆ r = ∆ + χ with χ &gt; 0 . Then

<!-- formula-not-decoded -->

because λ ⋆ ( θ ) &gt; 0 as r ∈ C

r ˜ . This is a contradiction, and establishes the result.

Lemma 6. For any j ∈ C , ˜ θ j = θ .

Proof. Let j ∈ C , by (58):

where

<!-- formula-not-decoded -->

We prove that v j ((1 , · ) , s -j ) is strictly decreasing in ˜ θ j for j ∈ C , by showing that ∂ ˆ v j ((1 , ˜ θ j ) , s -i j ) /∂ ˜ θ j &lt; 0 For any j ∈ C . For v j to be differentiable, we need q j to be differentiable, that is λ ⋆ j ( ˜ θ ) = ( ∑ k ∈C n ⋆ k ( ˜ θ )) -1 n ⋆ j ( ˜ θ ) to be differentiable for any j ∈ C . We re-index C as { i 1 , . . . , i |C| } so that ˜ θ i 1 &lt; . . . &lt; ˜ θ i |C| . Observe that for 0 &lt; h &lt; δ for δ &gt; 0 sufficiently small, ˜ θ i j -1 &lt; ˜ θ j + h &lt; ˜ θ i j +1 because ˜ θ i j -1 &lt; ˜ θ j &lt; ˜ θ i j +1 by Lemma 5 and H 3. Hence, the bid ordering does not change for any infinitesimal variation d ˜ θ j &gt; 0 , nor does the indicator 1 { i j ⩽ i |C| } . Consequently by Corollary 1, n ⋆ j ( ˜ θ ) is differentiable in ˜ θ j and so is λ ⋆ ( ˜ θ ) . We have for any j ∈ C :

<!-- formula-not-decoded -->

where we have used Lemma 5 at the third line. This implies

<!-- formula-not-decoded -->

To see why, assume by contradiction that there exists j ∈ C such that s j = (1 , ˜ θ j ) with ˜ θ j &gt; θ . Then for any h ∈ (0 , ˜ θ j -θ ] , by (61):

<!-- formula-not-decoded -->

which contradicts s being a Nash equilibrium.

Combining Lemma 5 and Lemma 6 gives for any ( j, k ) ∈ C 2 :

<!-- formula-not-decoded -->

Recall that by H 3, θ m = θ n if and only if m = n , so (63) implies |C| = 1 . This in turn implies |B| = 1 . Indeed, assume by contradiction |C| = 1 and |B| ⩾ 2 . Denote r ∈ B the only contributing agent. They are asked n ⋆ r ( ˜ θ ) = n r ( ˜ θ ) = n . Moreover by (19) ¯ N = |B| 1 / 1+ γ ( n +1) -1 &gt; n , so by definition of the contribution scheme (12), there exists k ∈ B \ { r } such that n ⋆ k ( ˜ θ ) &gt; 0 . This contradicts |C| = 1 .

We now show that B = { J } . By contradiction, assume B = { j } with j &lt; J . In particular, s J = (0 , † ) . Consider a deviation s ′ J = (1 , ˜ θ J ) with ˜ θ J ∈ Θ , so B = { j, J } under ( s ′ J , s -J ) . By Lemma 6, ˜ θ j = ˜ θ J = θ , so (58) rewrites:

<!-- formula-not-decoded -->

for any k ∈ { j, J } . Since θ j &lt; ϑ ( B , n ⋆ ( ˜ θ )) &lt; θ J by H 3, we have

<!-- formula-not-decoded -->

which contradicts s begin a Nash equilibrium. Hence, B = { J } . This concludes the proof.

Lemma 3. There exists j ∈ [ J ] such that -t j ( ˜ θ ) &gt; 0 .

Proof. Let ( B , ˜ θ ) ∈ { 0 , 1 } J × Θ N be an equilibrium of the game induced by Γ VCG . Since the VCG mechanism is strategyproof in dominant strategy, ˜ θ = θ . For any j ∈ [ J ] , the VCG payment is

̸

̸

<!-- formula-not-decoded -->

Consider j ∈ { 1 , . . . , L ⋆ } . By the contribution scheme Corollary 1,

<!-- formula-not-decoded -->

where we used in the third inequality that n ⋆ j ( θ ) = n j ( B , n ⋆ ( θ )) . Thus, ( II ) = 0 . Moreover by Theorem 1, W ((0 , 1 -j ) , n ⋆ ( θ )) &lt; W ( 1 , n ⋆ ( θ )) so ( I ) &lt; 0 . Thus, by (65) we have t j ( ˜ θ ) &lt; 0 .

Theorem 3. Assume H 1, H 2, H 3, H 4 and H 5. B ⋆ = (1 , . . . , 1) T is a Nash equilibrium under ˆ Γ with probability 1 -δ .

Proof. Let B ⋆ = (1 , . . . , 1) T , so the aggregator has at her disposal ̂ θ 1 , . . . , ̂ θ J .

We first focus on the event under which ̂ θ j correctly approximates θ j for any j ⩽ J . Define E j ( δ ) = { ω ∈ Ω : | ̂ θ j ( ω ) -θ j | ⩽ η δ } and E ( δ ) = ⋂ j ⩽ J E j ( δ ) . We have

<!-- formula-not-decoded -->

because P ( E ( δ )) = P ( ∩ j ⩽ J E j ( δ )) = 1 -P ( ∪ j ⩽ J E j ( δ )) ⩾ 1 -∑ j ⩽ J P ( E j ( δ )) = 1 -Jδ , and setting δ ′ = Jδ leads to Equation (66). In what follows, we assume that E ( δ/J ) is true so | ̂ θ j -θ j | ⩽ η δ/J ( q ) for any j ∈ [ J ] . We denote by C = { j ∈ [ J ] : m j ( ̂ θ ) = n ⋆ j ( ̂ θ + η j ) &gt; 0 } the set of contributors, and to lighten notations we write

<!-- formula-not-decoded -->

for any j ∈ [ J ] . Note that ϑ ( ̂ θ ) is the actual weighted type within the coalition, whereas ˆ ϑ ( ̂ θ ) is the weighted type estimated by the aggregator. Consider first a contributor j ∈ C who is asked n ⋆ j ( ̂ θ + η j ) &gt; 0 samples. Their payoff under B ⋆ is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Second, consider j / ∈ C who is asked q ⩽ n -2( a/c )( ¯ θ -θ ) samples, and denote by r ∈ [ J ] the agent such that θ r = max k ∈C θ k . Observe that q ⩽ n ⋆ r ( ̂ θ + η r ) , so

<!-- formula-not-decoded -->

where the last inequality comes from r ∈ C and (67). Since j / ∈ C , we have θ j ⩾ ̂ θ j -η δ/J ( q ) ⩾ ̂ θ r + η δ/J ( q ) ⩾ θ r so u r ⩾ u j , and it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition 2. Assume H 1, H 2 and H 6. For any j ∈ [ J ] the estimator

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

satisfies H 5 with

<!-- formula-not-decoded -->

with probability 1 -δ/ 2 by H 2, H 1, H 6, and a union bound. Similarly,

<!-- formula-not-decoded -->

with probability 1 -δ/ 2 . Combining (68) and (70) along with an union bound yields

<!-- formula-not-decoded -->

with probability 1 -δ .

Example 4. Assume H 1, H 2, H 6 and H 7.

<!-- formula-not-decoded -->

(ii) In H 1, assume α δ = ln(1 /δ ) 1 / 2 , β = 2 RAD ( G ) and γ = 1 [Bousquet et al., 2003]. With ̂ θ ERM 0 ,j defined in Proposition 2, we have

<!-- formula-not-decoded -->

Proof. (i) Let j ∈ B . Observe that under H 7,we have

<!-- formula-not-decoded -->

Since the hypothesis class G is symmetric, we have

<!-- formula-not-decoded -->

(ii) Let j ∈ [ J ] , we have

<!-- formula-not-decoded -->

##