# Candidate-aware ranking: post-result diagnosis

Status: closed after candidate 1; this is not an acceptance-threshold change.

The frozen experiment appended one Train-only recent-100 ItemCF candidate to each
positive and assigned it label zero.  It was intended to reduce the mismatch
between a pointwise DeepFM training distribution and a two-stage serving pool.

Validation evidence rejects the hypothesis.  The best checkpoint (epoch 2) had
ROC-AUC 0.835648, below the valid V2 ranking baseline (0.864214).  In the
frozen end-to-end protocol, even reranking only the original two-tower Top-50
fell to Recall@10 0.091752, below the fixed-ranker reference 0.111306; adding
ItemCF candidates reduced it further to 0.061168.

The causal limitation is identifiable from the available labels.  The existing
ranking data contains observed positive interactions, observed low-preference
interactions used as hard negatives, and sampled unobserved items.  It does not
contain logged impressions of ItemCF candidates followed by non-clicks.  An
unconsumed I2I neighbour is therefore an *unlabeled plausible item*, not a
verified negative.  Giving all such items target zero creates false-negative
pressure.  Sampling-ratio, ItemCF-bound, or pool-cap tuning would not create
the missing counterfactual label, so it is not a justified second candidate.

Decision: close this ranking-negative-sampling direction early, retain all
artifacts and results, and do not deploy it.  The next independent hypothesis
family is multi-interest retrieval (MIND): the retrieval diagnosis, rather
than pointwise negative relabeling, is the appropriate location to test whether
one user vector compresses distinct interests.
