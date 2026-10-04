# Does MOFA+ actually help with missing data in multi-omic integration?

This project set out to answer one practical question. When you are integrating transcriptomic and proteomic data and some values are missing, is MOFA+ (or its newer relative, MOFA-Flex) actually a good choice for filling those gaps? Or would a plain, older imputation method do just as well or better?

Every method was tested the same way. A real value was hidden, the method guessed it, and the guess was compared against the truth. This was repeated across three real datasets, under different mixes of two reasons data goes missing in the first place. Everything below comes out of `evaluate_imputation_methods_clinical.ipynb`, the notebook that pulls together the results of the other benchmarking notebooks and produces the tables and figures shown here, except for the final section ("Beyond imputation"), which comes from `mofa_downstream_clinical.ipynb` and looks at what the two MOFA models actually find in the DLBCL data once they are trained on it.

## The setup 

Three matrices were tested, each with a different missing-data problem. All three are restricted to each dataset's 500 most-variable features (by observed-value variance) before anything else happens — a scope choice that matters here because it doesn't just shrink the matrix, it concentrates missingness: the most-variable features are disproportionately the ones already hit hardest by detection-limit dropout.

- **DLBCL transcriptome** (RNA): the full cohort is 332 patients, but 14 of them were never RNA-sequenced at all (~4% of the cohort, missing as entire patients, not as individual gene values) and are dropped before the matrix is even built — not imputed, just excluded. That leaves **318 patients**, each with a complete RNA profile, and it's this 318-patient matrix that the benchmark actually runs on. So there's **0% missing inside the matrix being tested**: One consequence: with nothing to calibrate a detection-limit curve from, this matrix's "MNAR" scenarios end up statistically indistinguishable from its MCAR ones.
- **DLBCL proteome** (332 patients, protein): **60% missing** in the 500 most-variable proteins used here — well above the ~33% missingness rate of the full ~6,900-protein panel, because the most-variable proteins are disproportionately the ones affected by detection-limit dropout. Low-abundance proteins are disproportionately the ones that go undetected.
- **CRC proteome** (997 patients, protein): **4% missing** in the 500 most-variable proteins used here (vs. ~0.3% across the full ~8,400-protein panel, same reason as above), with an artificial 30% masked in on top of that for the test, again biased toward low-abundance values.

For every matrix, a mix of two kinds of artificial gaps was created on top of whatever real missingness the matrix already had: some values hidden completely at random (MCAR), some hidden with a bias toward the lowest values, mimicking a detection limit (MNAR). The masking fraction itself was held fixed at 30%; what was swept was the *mix*, from 0% MNAR / 100% MCAR to the reverse, in 25% steps, repeated across five replicate runs at each setting — 25 scenarios per matrix. Twenty-one conventional imputation methods are compared against MOFA+ and MOFA-Flex on every scenario (Zero, LOD, Median and Regression were benchmarked too, but are left out of this evaluation). MOFA-Flex here always means the MOFA-Flex model with a STRING protein-network weight prior (GraphGP), stored as `MOFA-Flex-GraphGP` in the benchmark checkpoints; the plain MOFA-Flex model, without the network prior, is left out of this evaluation. Accuracy is reported as RMSE (root mean squared error) and NRMSE (RMSE scaled by each feature's own value range, then averaged across features) on the log2-scale values all three datasets already use, so lower is better on both.

Two methods, DreamAI and ADMIN, only finished 20 of the 25 CRC scenarios. Both rely on an R package that reliably crashes on the hardest condition (every hidden value drawn from the low end of the abundance range). Their average below only reflects the easier 20, which flatters them slightly. That is called out again wherever it actually changes a conclusion.

## How every method stacks up

Fig 1 shows RMSE across all 25 scenarios, one box per method, grouped by family: general-purpose methods in blue, MNAR/left-censored substitution methods in orange, R-backed methods in aqua, deep learning in yellow, MOFA+ and MOFA-Flex in their own colors. Fig 2 shows the same thing for NRMSE, which the evaluation notebook treats as the more trustworthy ranking metric, since it is not dominated by whichever features happen to sit at the largest scale.

![RMSE across the full grid, all methods, all three matrices](results/evaluation_figures/fig1_rmse_boxplot.png)

![NRMSE across the full grid, all methods, all three matrices](results/evaluation_figures/fig2_nrmse_boxplot.png)

By RMSE, MOFA+ is third on the DLBCL transcriptome (1.182, behind MsImpute at 1.172 and SoftImpute at 1.175 — a margin of about 0.01), fourth on the DLBCL proteome (0.517, behind MsImpute, SoftImpute, and DreamAI, all within about 0.02 of each other), and third among fully-covered methods on CRC (1.930, behind only DreamAI and ADMIN, which had the easier partial run described above).

By NRMSE the ranking shifts in MOFA+'s favor on two of the three matrices. On the transcriptome, MOFA+ (0.1605) edges ahead of SoftImpute (0.1608) into second place, though MsImpute (0.1602) still leads. On the DLBCL proteome, MOFA+ (0.1364) moves up to third, ahead of MICE (0.1365) and DreamAI (0.1440) — DreamAI's raw-RMSE edge comes from doing well on whichever features happen to sit at the largest scale, exactly the advantage NRMSE is built to remove. On CRC, the ranking doesn't change: MOFA+ stays third, behind DreamAI and ADMIN's partial coverage.

## MOFA+ against the best conventional method

The evaluation notebook checks this two ways: against whichever conventional method is genuinely fully covered on the exact same scenarios MOFA+ ran ("scenario-matched"), and against whichever conventional method has the single lowest average RMSE anywhere in the full grid, regardless of how much of that grid it actually completed ("full grid, all masking fractions"). The two do not always agree, and the disagreement itself is informative.

| Matrix | MOFA+ RMSE | Best conventional (scenario-matched) | Wins, scenario-matched | Best conventional (full grid, unfiltered) | Wins, full grid |
|---|---|---|---|---|---|
| DLBCL transcriptome | 1.182 | MsImpute, 1.172 | No | MsImpute, 1.172 | No |
| DLBCL proteome | 0.517 | MsImpute, 0.498 | No | MsImpute, 0.498 | No |
| CRC proteome | 1.930 | DAE, 1.931 | Yes | DreamAI, 1.803 | No |

On the transcriptome, MOFA+ now loses to MsImpute both ways — a reversal from an earlier run of this same benchmark, where MOFA+ had a narrow edge here. The margin is slim either way, about 0.01 RMSE. On the proteome, it simply loses to MsImpute, both ways, same conclusion as before. On CRC, the answer still depends on which baseline is used: against a method that actually finished every scenario MOFA+ did (DAE), MOFA+ wins, but only by about 0.0004 RMSE — essentially a tie. Against the flat minimum RMSE in the table, it loses, but only because that minimum belongs to DreamAI, which only completed 20 of the 25 scenarios and never attempted the hardest one. Whether MOFA+ "beats the best conventional method" on CRC still depends on whether a method gets credit for skipping the hard part.

## Does MOFA-Flex do any better?

MOFA-Flex was run on the full grid too, with genuine 25-of-25 coverage on every matrix. It doesn't beat the best conventional method on any matrix, scenario-matched or otherwise.

| Matrix | MOFA-Flex RMSE | Best conventional (scenario-matched) | Wins |
|---|---|---|---|
| DLBCL transcriptome | 1.266 | MsImpute, 1.172 | No |
| DLBCL proteome | 0.539 | MsImpute, 0.498 | No |
| CRC proteome | 1.972 | DAE, 1.931 | No |

Plain MOFA+ also beats MOFA-Flex on every one of the three matrices, in all 75 runs. The gap is small in absolute RMSE, but it is consistent in direction everywhere. The most likely reason is not the model itself. MOFA+ solves for its factors directly through closed-form coordinate ascent. MOFA-Flex estimates the same kind of model through stochastic variational inference, which adds its own training noise on top of whatever the underlying model contributes. The STRING protein-network prior is the better of the two MOFA-Flex set-ups tried (it beat MOFA-Flex without the prior in every run, which is why that set-up is the one reported), but it doesn't close the gap to MOFA+ on any matrix.

## Does it matter *why* a value went missing?

Every scenario mixes two kinds of gaps: some hidden completely at random (MCAR), some hidden with a bias toward low values (MNAR), the way real detection-limit missingness actually behaves. Fig 3 splits RMSE by which kind of gap a value came from, for the top 10 fully-covered methods on each matrix, plus MOFA+ and MOFA-Flex even if they miss that top 10.

![DLBCL transcriptome, RMSE by missingness mechanism](results/evaluation_figures/fig3_mechanism_split_dlbcl_rna.png)

![DLBCL proteome, RMSE by missingness mechanism](results/evaluation_figures/fig3_mechanism_split_dlbcl_protein.png)

![CRC proteome, RMSE by missingness mechanism](results/evaluation_figures/fig3_mechanism_split_crc_protein.png)



On CRC proteome, every method gets substantially worse on MNAR gaps, MOFA+ included: its RMSE goes from about 1.13 on the MCAR subset to about 2.59 on the MNAR subset, more than doubling. MOFA-Flex shows the same pattern, at a similar margin. MsImpute, SoftImpute, and MICE degrade the same way. All of these handle a missing value by leaving it out of the model fit, which only stays fair when whether a value is missing has nothing to do with what it would have been — the assumption (missing at random, in the sense Little and Rubin gave the term) that real detection-limit missingness violates. Hernández-Lobato and colleagues showed in 2014 that this kind of value-dependent missingness biases plain matrix-factorization models like these, and neither MOFA+ nor MOFA-Flex tests for it directly.

On the DLBCL proteome, the same methods do the opposite: MOFA+'s RMSE actually improves slightly on the MNAR subset (about 0.52 versus 0.51), and so does every other method tested there. This isn't evidence that MNAR missingness is secretly easy in mass spec proteomics — it's a consequence of this matrix already being ~60% missing for real, detection-limit reasons, before any synthetic masking happens. The truly lowest, hardest values are already `NaN` and excluded from the pool this benchmark draws synthetic MNAR gaps from, so the synthetic MNAR subset here is only mildly shifted toward low values but is also markedly less variable than the MCAR subset. Lower target variance mechanically caps how large RMSE can get, and that effect outweighs the "low values are harder" effect on this matrix. Real detection-limit censoring already removed the entries that would have made this task hard.

On the DLBCL transcriptome, there's effectively no difference between the MCAR and MNAR subsets for any method (MOFA+: 1.182 vs 1.181). That's expected, not a finding: this matrix has 0% real missingness once restricted to profiled patients, so the notebook's own MNAR-curve calibration has nothing to fit and falls back to an uninformative, MCAR-like weighting. The "MNAR" scenarios on this matrix are, statistically, just more MCAR scenarios.

Altogether, the "MNAR is harder" story is real, but only demonstrated on CRC. It reverses on the DLBCL proteome specifically because that matrix's real missingness has already stripped out the entries that would make the effect show up, and it's untested on the transcriptome because that matrix has no real missingness left to calibrate an MNAR curve from. Whether a benchmark like this one detects the MNAR penalty at all depends heavily on how much real detection-limit censoring the input data already had.

## Does the damage scale with how much of the masking is MNAR?

Fig 4 breaks RMSE down by masking fraction and MNAR fraction together, as a heatmap, for the top 5 fully-covered methods per matrix plus MOFA+ and MOFA-Flex. The masking fraction itself is fixed at 30% throughout this benchmark (see "The setup," above), so each heatmap really shows a profile across the five MNAR-fraction settings rather than a full two-way grid.

![DLBCL transcriptome, RMSE heatmap](results/evaluation_figures/fig4_rmse_heatmap_dlbcl_rna.png)

![DLBCL proteome, RMSE heatmap](results/evaluation_figures/fig4_rmse_heatmap_dlbcl_protein.png)

![CRC proteome, RMSE heatmap](results/evaluation_figures/fig4_rmse_heatmap_crc_protein.png)

The pattern tracks Fig 3, matrix by matrix, rather than holding uniformly. On CRC, error climbs steadily as the MNAR fraction rises, for every method shown, MOFA family included — RMSE for MOFA+_native goes from 1.31 at 0% MNAR to 2.47 at 100% MNAR, for example. On the DLBCL proteome, the trend runs the other way: RMSE for the same methods drifts down as the MNAR fraction increases (MOFA+_native: 0.53 at 0% MNAR to 0.51 at 100%), for the reason given above — this matrix's own real missingness has already removed the hardest low-abundance entries from what's left to mask. On the DLBCL transcriptome, RMSE barely moves across the MNAR-fraction sweep at all, consistent with there being no real MNAR signal to calibrate against in that matrix. Whether "more MNAR masking means more damage" depends entirely on how much real detection-limit censoring the underlying matrix already had — it is not a universal property of these methods, or of MNAR missingness in general.

## A consistency check: does correlation track RMSE the way it should?

Just for sanity check, a method with a better feature-centered correlation between its predictions and the truth should generally also have a lower RMSE. Fig 5 plots the two against each other, one panel per matrix, for the pilot scenario (30% masking, pure MCAR).

![Correlation versus RMSE, pilot scenario, all three matrices](results/evaluation_figures/fig5_corr_vs_rmse.png)

A method that lands far off the general trend in this plot would be one that tracks the right relative shape but is badly mis-scaled, or the reverse. Nothing here flags such a case among the results in hand. It is included mainly so a future re-run can be checked the same way.

## Where does the error concentrate by abundance?

Fig 6 looks at the pilot scenario, which is pure MCAR by design, and asks whether error is still concentrated at the low-abundance end even without any deliberate MNAR bias in that particular scenario. It shows mean absolute error by truth-abundance decile, for the top 6 methods on each matrix plus MOFA+ and MOFA-Flex.

![DLBCL transcriptome, error versus abundance](results/evaluation_figures/fig6_error_vs_abundance_dlbcl_rna.png)

![DLBCL proteome, error versus abundance](results/evaluation_figures/fig6_error_vs_abundance_dlbcl_protein.png)

![CRC proteome, error versus abundance](results/evaluation_figures/fig6_error_vs_abundance_crc_protein.png)

This figure is a separate, complementary check to Fig 3. Fig 3 shows the size of the MNAR penalty is matrix-dependent, not universal. Fig 6 asks a related but separate question: even under pure MCAR masking, does error still concentrate at the low-abundance end? For both protein matrices, yes — the lowest-abundance decile has by far the largest mean absolute error of any decile (DLBCL protein: 0.42 vs. ~0.30-0.35 in the middle deciles; CRC: 1.40 vs. ~0.68-0.88 elsewhere). For the transcriptome, no: error is elevated at both ends of the abundance range, not just the low end, consistent with there being no detection-limit-style link between abundance and missingness in RNA to begin with. So low-abundance values being harder to impute is a real pattern specific to the two proteomic matrices here — it doesn't depend on whether a given scenario's masking was deliberately biased toward them, but it isn't a universal property of every matrix tested.

## Which methods agree with each other?

Fig 7 looks at how similar different methods' own imputed values are to each other on the same masked entries, in the pilot scenario — a pairwise Pearson correlation between every pair of methods' guesses, not against the truth. This shows which methods behave like near-duplicates and which are doing something genuinely different, independent of which one happens to be more accurate.

![DLBCL transcriptome, method agreement](results/evaluation_figures/fig7_method_agreement_dlbcl_rna.png)

![DLBCL proteome, method agreement](results/evaluation_figures/fig7_method_agreement_dlbcl_protein.png)

![CRC proteome, method agreement](results/evaluation_figures/fig7_method_agreement_crc_protein.png)

The clearest pattern: every MOFA+ variant correlates almost perfectly with every other MOFA+ variant (MOFA+_native vs. MOFA+_joint: r=0.979 on the transcriptome, r=0.994 on the proteome) — unsurprising, since they share the same underlying factor model and differ only in how missing values are handled going in. MOFA+_native's least similar method on both protein matrices is MinProb (r=0.03 on DLBCL protein, r=0.09 on CRC), a left-censored substitution method that guesses low values almost regardless of context — about as different an imputation strategy from a joint factor model as this benchmark contains. On CRC, MOFA+_native's closest non-MOFA+ neighbor is DreamAI_MissForest (r=0.993), despite the two using unrelated approaches.

## Beyond imputation: what do the MOFA factors actually find?

Imputation accuracy is only half the reason to use MOFA+ or MOFA-Flex; the other half is the factors themselves. `mofa_downstream_clinical.ipynb` trains both models once on the real, unmasked DLBCL RNA + protein data and asks what those factors capture. CRC is left out, because its clinical metadata isn't available for this project.

The setup differs from the benchmark on purpose. Each view uses its 2,000 most-variable features (proteins must first be observed in at least 50% of patients, so the protein view isn't dominated by mostly-missing features), giving 4.2% missing in the RNA view (the 14 patients without RNA stay in, with that view missing) and 15.1% in the protein view. Both models get the same two views, 15 factors, the same per-view scaling, and three seeds each; seed 0 is the reference fit, and seeds 1 and 2 are only used to check reproducibility. MOFA-Flex is the same STRING GraphGP model as in the benchmark.

**Both models fit the data equally well, but split it very differently.** Taking all factors together, MOFA+ explains 47% of RNA and 42% of protein variance, MOFA-Flex 48% and 42%. MOFA+ concentrates that variance: Factor 1 alone carries 20% of RNA variance, Factors 2 and 3 about 10% of protein variance each, and most factors belong mainly to one view. MOFA-Flex spreads it evenly, 2-5% per factor, with all 15 factors active in both views.

![Variance explained per factor and per view, MOFA+ and MOFA-Flex](results/mofa_downstream/figures/fig1_variance_explained.png)

**MOFA+'s factors are reproducible; MOFA-Flex's individual factors are not, but its factor space is.** MOFA+ returns the same 15 factors from every seed. MOFA-Flex reproduces none of its 15 factors across seeds (best |r| 0.45-0.72), yet 13-14 of the 15 canonical correlations between seeds are at least 0.9, so the space the factors span is largely the same. The instability is rotation, not noise: with a plain Normal prior on the factors, MOFA-Flex has no reason to prefer one rotation over another. The same holds between the two models. 14 of the 15 MOFA+ factors are almost entirely captured by MOFA-Flex's factors taken together (R² 0.93-0.999), but no single MOFA-Flex factor correlates above |r| 0.57 with any MOFA+ factor.

![MOFA+ factors matched against MOFA-Flex factors](results/mofa_downstream/figures/fig2_mofaplus_vs_mofaflex_factors.png)

**MOFA+ puts known DLBCL biology on separate factors.** Every active factor was tested against cell of origin, genetic subtypes (LymphGen, genetic cluster), ecotype, and stromal and proliferation ranks, and against the MSigDB Hallmark gene sets via its weights:

| Axis | MOFA+ factor | Evidence |
|---|---|---|
| Cell of origin (ABC vs GCB) | Factor 5 | AUC 0.88; 14 of 15 marker genes load with the expected sign |
| Proliferation | Factor 4 | Spearman rho 0.54 with proliferation rank; E2F and MYC targets in the protein weights |
| Stroma / microenvironment | Factor 10 (also 3, 13) | rho 0.66 with stromal rank; EMT and angiogenesis gene sets |
| Lymphoma ecotype | Factor 1 | epsilon² 0.52; RNA-only, the largest factor |
| Unexplained protein programme | Factors 2 and 15 | protein-only; oxidative phosphorylation and fatty-acid metabolism; no clinical association |

Factors 2 and 15 explain a lot of protein variance but match no annotation. They could be a real metabolic programme, or something technical such as sample handling or MS batch, and should be checked against the proteomics batch/plate information before being read as biology.

![Factor-annotation associations, MOFA+ and MOFA-Flex](results/mofa_downstream/figures/fig3_factor_annotation_associations.png)

![Hallmark gene set enrichment of RNA factor weights](results/mofa_downstream/figures/fig5_hallmark_enrichment_rna.png)

**MOFA-Flex holds the same information, spread across factors.** Cell of origin, used here as a positive control, shows it most clearly: MOFA-Flex's best single factor separates ABC from GCB with an AUC of only 0.72, but all its factors together reach a cross-validated AUC of 0.90, level with MOFA+ (0.89 from all factors, 0.88 from Factor 5 alone).

**Sparsity on the factors doesn't fix MOFA-Flex's rotation problem; a varimax rotation does.** Three fixes were tried, all keeping the STRING GraphGP prior on the weights:

| | MOFA+ | MOFA-Flex | + Horseshoe factors | + SpikeSlab factors | varimax-rotated |
|---|---|---|---|---|---|
| Joint variance explained (RNA / protein) | 47% / 42% | 48% / 42% | 48% / 42% | 44% / 38% | 48% / 42% |
| Factors reproduced across seeds | 15/15 | 0/15 | 0/15 | 0/15 | 13/15 |
| Cell of origin, best single factor (AUC) | 0.88 | 0.72 | 0.72 | 0.69 | 0.87 |
| Best single-factor match to MOFA+ axes (\|r\|: cell of origin / proliferation / stroma / ecotype) | - | 0.57 / 0.46 / 0.50 / 0.50 | 0.57 / 0.46 / 0.51 / 0.50 | 0.50 / 0.38 / 0.44 / 0.45 | 0.77 / 0.73 / 0.81 / 0.98 |

A horseshoe prior on the factors has no measurable effect (with 2,000 features per view, the data outweighs a prior on the patient scores). A spike-and-slab prior on the factors is worse on every check. A varimax rotation of the plain MOFA-Flex fits, with no retraining and no change to the fit, makes 13 of 15 factors reproducible and lines them up with MOFA+'s: the ecotype factor almost exactly (r = 0.98), stroma, cell of origin and proliferation clearly. Sparsity on the weights together with the STRING prior would be the natural next test, but the installed MOFA-Flex doesn't support it.

![Cell-of-origin factor for every model variant](results/mofa_downstream/figures/fig8_cell_of_origin_all_models.png)

**No factor is associated with overall survival after multiple-testing correction.** Each factor went into a Cox model stratified by treatment (318 patients, 172 deaths), and again adjusted for cell of origin. The strongest signals point the expected way: the ABC end of the cell-of-origin factor goes with worse survival (MOFA+ Factor 5, HR 1.20 per SD, p = 0.016; MOFA-Flex Factor 10, HR 1.24, p = 0.004, FDR 0.07), and the proliferation factor (MOFA+ Factor 4, HR 0.83, p = 0.021) keeps its effect after adjusting for cell of origin (p = 0.027). This is a single cohort with no validation set, so it's a first look, not a prognostic model.

![Cox hazard ratios for overall survival, per factor](results/mofa_downstream/figures/fig6_cox_overall_survival.png)

Two caveats. Every analysis uses the 2,000 most-variable features per view, so pathway enrichment describes each factor within those features, not genome-wide. And the annotation, pathway and survival analyses cover MOFA+ and plain MOFA-Flex only, not the horseshoe, spike-and-slab or varimax variants.

## Conclusion 

MOFA+ is competitive, but not a standout, choice for filling in missing values across this transcriptomic and proteomic datasets. Across all three matrices, it lands third or fourth by RMSE among fully-covered methods (second or third by NRMSE), consistently within a few percent of MsImpute and SoftImpute, the two methods that lead almost every ranking here. It essentially ties the best conventional method on CRC (a margin of 0.0004 RMSE against DAE), and loses narrowly to MsImpute on both DLBCL matrices, including the transcriptome. That's a reasonable result for a method whose main selling point is joint multi-omics factor modeling, not imputation on its own.

Its weak spot is real, but conditional rather than universal. On CRC — the one matrix here with almost no real, pre-existing detection-limit censoring — MOFA+ (and every other method tested) loses substantial accuracy on values that were missing because they were too faint to detect: RMSE roughly doubles from the MCAR subset to the MNAR one. On the DLBCL proteome, which already carries ~60% real detection-limit missingness before this benchmark starts, the same methods do slightly better on the MNAR-flavored subset instead, because the real censoring has already stripped out the hardest, most extreme low values from what's left to test against. So the practical risk isn't "MOFA+ mishandles detection-limit missingness" in the abstract — it's that a method's measured robustness to MNAR gaps depends on how much real censoring the data already has, and can look artificially reassuring on data that's already heavily pre-censored. Anyone evaluating MOFA+ (or any of these methods) on their own proteomics data should check where it sits on that spectrum before trusting a benchmark result like this one at face value.

Between the two MOFA models tested, plain MOFA+ is the more accurate one for imputation specifically. MOFA-Flex loses to it, and to the best conventional method, on every matrix tested here. That does not mean it is the weaker choice overall. Its real advantage is the flexibility to plug in different priors and knowledge sources, such as the STRING network used here, which is a different question than raw imputation accuracy, the one this benchmark was built to answer.

The downstream analysis on DLBCL points the same way for interpretation. Both models explain about the same share of variance and hold the same information, but MOFA+ returns stable factors that each map onto a known DLBCL axis (cell of origin, proliferation, stroma, ecotype), while MOFA-Flex spreads those axes across factors that change from seed to seed. A varimax rotation recovers most of MOFA+'s interpretability from MOFA-Flex without retraining; sparsity priors on the factors do not. For reading biology off individual factors on this data, MOFA+ is the better-behaved model as it stands.

## Reproducing these figures

Run `benchmark_imputation_clinical.ipynb` (or the `.py` version — the two are kept in sync), then `evaluate_mofaplus_multiomic_clinical.ipynb` and `evaluate_mofaflex_graphgp_multiomic_clinical.ipynb` (the MOFA-Flex model reported here), then `evaluate_imputation_methods_clinical.ipynb` last. The "Beyond imputation" section comes from `mofa_downstream_clinical.ipynb`, which stands on its own: it reads the processed DLBCL data directly, saves its models, tables and figures under `results/mofa_downstream/`, and reloads the saved models on re-runs unless `RETRAIN = True`. Its first run needs internet access, to fetch STRING edges and the MSigDB Hallmark gene sets.

## References

Little, R. J. A. and Rubin, D. B. *Statistical Analysis with Missing Data*.

Hernández-Lobato, J. M., Houlsby, N., and Ghahramani, Z. (2014). Probabilistic Matrix Factorization with Non-random Missing Data. *Proceedings of the 31st International Conference on Machine Learning* (PMLR/AISTATS).

Ma, W. et al. (2020). NRMSE definition and use as a cross-feature-scale-fair imputation accuracy metric, as adopted in this benchmark's evaluation notebook.

Lazar, C. et al. (2016). Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative Proteomics Data Sets to Compare Imputation Strategies. *Journal of Proteome Research*, 15, 1116-1125.

Argelaguet, R. et al. (2018). Multi-Omics Factor Analysis, a framework for unsupervised integration of multi-omics data sets. *Molecular Systems Biology*, 14, e8124.

Argelaguet, R. et al. (2020). MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data. *Genome Biology*, 21, 111.

Qoku, A. et al. (2025). MOFA-FLEX: A Factor Model Framework for Integrating Omics Data with Prior Knowledge. *bioRxiv* 10.1101/2025.11.03.686250 (preprint, not yet peer reviewed).
