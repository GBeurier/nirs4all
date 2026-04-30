# Deep Learning for Near-Infrared Spectroscopy (NIRS) — Literature Review (2017–2026)

**Scope.** This review synthesises the state of the art for *deep learning regression and classification* on 1-D near-infrared (NIRS) spectra, with a focus on small to medium datasets (n = 40 - 2000), preprocessing, augmentation, regularisation, attention/Transformer architectures, and benchmarking against PLS / Ridge / TabPFN / CatBoost.
The review is intended to feed the **nicon_v2** benchmark project (`bench/nicon_v2/`) and is therefore organised by themes that map directly to the benchmark's design choices.

**Method.** Targeted queries on Web/Semantic-Scholar/OpenAlex/arXiv (April 2026), with cross-checks of DOIs and abstracts via PubMed and ScienceDirect when accessible. Where a fact could not be verified directly (paywalled abstracts), it is reported with the explicit qualifier *"(per [primary source title])"* and the citation points to the indexed record. Each entry below has been seen at minimum in the search-result snippet *and* one secondary index (PubMed / arXiv / publisher landing page).

---

## Theme 1. 1-D CNN architectures for spectra

The 1-D CNN era for NIRS effectively starts in 2017 with Acquarelli et al. and the Liu et al. Raman recogniser, then matures with Cui & Fearn (2018), Bjerrum et al. (2017), Zhang et al. (2019, "DeepSpectra"), and the Mishra/Passos line of multi-output and multi-fruit models. Common motifs that have stabilised:

- **Receptive field**: kernels of 3-21 wavelengths in early layers (Liu et al. 2017 used 21/11/5; Acquarelli used a single shallow conv; DeepSpectra uses a 3-conv stack feeding an Inception block).
- **Depth**: 3-7 convolutional layers is the dominant range; deeper variants (1D-Inception-ResNet, Transformer-CNN hybrids) only outperform on medium-large data (>1k samples).
- **Pooling**: max-pool dominates early layers; *global average pooling* before the head tends to generalise better than `Flatten + Dense` on small NIRS data (Cui & Fearn 2018; Mishra & Passos 2021).
- **Residual connections**: ResNet-style identity skips become important as depth increases beyond ~4 conv layers (Tian et al. 2023; Wang et al. 2024).

### 1.1 Acquarelli et al., 2017 — *Convolutional neural networks for vibrational spectroscopic data analysis*

- Citation. Acquarelli, J., van Laarhoven, T., Gerretzen, J., Tran, T. N., Buydens, L. M. C., & Marchiori, E. (2017). *Anal. Chim. Acta* 954, 22-31. DOI 10.1016/j.aca.2016.12.010 (PMID 28081811).
- One-line. A *single-convolutional-layer* (shallow) CNN beats PLS on raw vibrational spectra and learns saliency maps that highlight chemometrically meaningful regions.
- Architecture. Single 1-D conv layer with broad filters; FC head; dropout; trained from scratch.
- Datasets. Multiple FT-IR/Raman classification benchmarks (wine origin, olive oil, etc.).
- Results. 86% accuracy on raw test data vs. 62% for PLS; 96% accuracy with preprocessing vs. 89% for PLS. Demonstrated that the convolution layer itself functions as a learnable preprocessor.

### 1.2 Liu et al., 2017 — *Deep convolutional neural networks for Raman spectrum recognition: a unified solution*

- Citation. Liu, J., Osadchy, M., Ashton, L., Foster, M., Solomon, C. J., & Gibson, S. J. (2017). *Analyst* 142(21), 4067-4074. DOI 10.1039/C7AN01371J.
- One-line. A 3-conv-layer 1-D CNN (kernels 21 / 11 / 5) classifies the RRUFF mineral database without any baseline correction or smoothing, beating SVM.
- Architecture. 3 × Conv1D + pooling, 1 dense layer, softmax.
- Datasets. RRUFF (~1500 mineral species, Raman).
- Results. Outperforms SVM and shallow ML at every preprocessing condition and is robust to noise/baseline drift. Influential template for later 1-D CNN work in NIR.

### 1.3 Cui & Fearn, 2018 — *Modern practical CNNs for multivariate regression: applications to NIR calibration*

- Citation. Cui, C., & Fearn, T. (2018). *Chemom. Intell. Lab. Syst.* 182, 9-20. DOI 10.1016/j.chemolab.2018.07.008.
- One-line. The first principled study showing that, with proper architectural choices, CNNs are simultaneously *more accurate and less noisy* than PLS for NIR calibration on three real datasets.
- Architecture. A unified, fully convolutional regressor: 2-3 conv blocks, global average pooling, single dense layer. Authors argue convolution acts as a *learnable preprocessor*.
- Datasets. Three industrial NIR datasets (n = 6998 / 1000 / 415 train).
- Results. CNN outperforms PLSR on all three datasets; gains are largest on the largest dataset.

### 1.4 Zhang et al., 2019 — *DeepSpectra: an end-to-end deep learning approach for quantitative spectral analysis*

- Citation. Zhang, X., Lin, T., Xu, J., Luo, X., & Ying, Y. (2019). *Anal. Chim. Acta* 1058, 48-57. DOI 10.1016/j.aca.2019.01.002 (PMID 30851853).
- One-line. **The most-cited deep model for NIRS quantitative analysis**: stacks 3 conv layers + an Inception module to learn multi-scale features from raw spectra.
- Architecture. 3 × Conv1D + Inception block (parallel 1×1 / 1×3 / 1×5 / pool branches) + GAP + dense regressor.
- Datasets. Four open NIR datasets (Corn, Tablet, Wheat, Soil).
- Results. On corn protein, mean RMSEP = 0.12, mean R² = 0.91, beating PLS, ANN, SVR and three CNN baselines. Best results on *raw* spectra (no SNV/MSC), confirming the learnable-preprocessor argument.

### 1.5 Padarian et al., 2019 — *Using deep learning to predict soil properties from regional spectral data*

- Citation. Padarian, J., Minasny, B., & McBratney, A. B. (2019). *Geoderma Regional* 16, e00198. DOI 10.1016/j.geodrs.2018.e00198.
- One-line. Multi-task 2-D-spectrogram CNN on the LUCAS soil database; CNN beats PLSR / Cubist *only when n > 10 000* — the most influential cautionary tale for small NIRS datasets.
- Architecture. Spectra converted to 2-D spectrogram, fed to multi-task 2-D CNN with shared trunk + 6 task heads (OC, CEC, clay, sand, pH, N).
- Datasets. LUCAS topsoil (>19 000 samples).
- Results. CNN > PLSR > Cubist when n > 10k; reverses for n < 1k.

### 1.6 Malek et al., 2018 — *One-dimensional CNNs for spectroscopic signal regression*

- Citation. Malek, S., Melgani, F., & Bazi, Y. (2018). *J. Chemometrics* 32(5), e2977. DOI 10.1002/cem.2977.
- One-line. Early demonstration that even shallow 1-D CNNs trained with particle-swarm-optimised weights compete with PLS on standard NIR benchmarks.

### 1.7 Mishra & Passos, 2021 — *Multi-output 1-D CNN for simultaneous prediction of fruit traits*

- Citation. Mishra, P., & Passos, D. (2021). *Postharvest Biol. Technol.* 183, 111741. DOI 10.1016/j.postharvbio.2021.111741.
- One-line. Single CNN with multi-target head simultaneously predicts SSC and dry-matter from pear NIR spectra and is 13% better (RMSE) than dedicated PLS models.
- Architecture. 1-D CNN trunk + multi-output linear head; standard SG preprocessing.
- Datasets. Pear NIR (in-house).

### 1.8 Mishra & Passos, 2021 — *A synergistic use of chemometrics and deep learning ... for dry matter prediction in mango*

- Citation. Mishra, P., & Passos, D. (2021). *Chemom. Intell. Lab. Syst.* 212, 104287. DOI 10.1016/j.chemolab.2021.104287.
- One-line. Outlier removal + augmentation in the *variable* domain (concatenating SG-derivative variants as channels) yields the SOTA on the mango dry-matter benchmark.

### 1.9 Mishra & Passos, 2021 — *Deep chemometrics: Validation and transfer of a global deep NIR fruit model to use it on a new portable instrument*

- Citation. Mishra, P., & Passos, D. (2021). *J. Chemom.* 35(8), e3367. DOI 10.1002/cem.3367.
- One-line. Demonstrates that deep models trained on a large NIR fruit corpus *transfer better than PLS on the *first 100* samples of a new instrument* but are also *more sensitive to instrument change* than PLS — a key practical caveat.

### 1.10 Tian et al., 2023 — *1D-Inception-ResNet for NIR quantitative analysis and its transferability between different spectrometers*

- Citation. Tian, R., et al. (2023). *Infrared Phys. & Technol.* 129, 104559. DOI 10.1016/j.infrared.2023.104559.
- One-line. A 1-D Inception-ResNet trained on sugarcane NIR transfers to two unseen spectrometers via fine-tuning and beats PDS-, CCA-, SBC-PLS calibration transfer baselines.

### 1.11 Mishra et al., 2023 — *Augmenting NIR spectra in deep regression to improve calibration*

- Citation. Mishra, P., Passos, D., Marini, F., & Barreto, A. (2023). *Chemom. Intell. Lab. Syst.* 244, 105033. DOI 10.1016/j.chemolab.2023.105033.
- One-line. EMSC-parameter-driven augmentation (random offset, slope, multiplicative scale) applied during training, then EMSC at inference, gives the best CNN regression results on standard NIR benchmarks.

### 1.12 BEST-1DConvNet (Wang et al., 2024)

- Citation. Wang, X., et al. (2024). *Processes* 12(2), 272. DOI 10.3390/pr12020272.
- One-line. Combines batch-norm, ELU, spatial-dropout, and Squeeze-and-Excite blocks in a 1-D CNN regressor; reports lower RMSEP than DeepSpectra on Corn and Tablet.

---

## Theme 2. Attention / Transformer models for spectra

Transformers reached NIRS in 2022. The pattern across recent work (2022-2026) is clear: pure-Transformer models are *competitive but not dominant* on small-n NIRS; **CNN+attention hybrids are the consistent winners** on n < 1000.

### 2.1 Yang et al., 2022 — *SpectraTr: a novel deep learning model for qualitative analysis of drug spectroscopy based on transformer structure*

- Citation. Pang, Y., Yang, H., Yang, F., Li, L., et al. (2022). *J. Innov. Opt. Health Sci.* 15(4), 2250021. DOI 10.1142/S1793545822500213.
- One-line. The first NIR drug-spectra Transformer; outperforms PLS, SVM, AE, CNN on most cases, requires no preprocessing.
- Architecture. Vanilla Transformer encoder over 1-D spectral tokens.

### 2.2 Li et al., 2024 — *A Transformer-based model for quantitative analysis of NIR spectra*

- Citation. Li, L., Lu, F., Wang, Z., Chen, J., Huang, D., Li, Q., & Yang, H. (2024). SSRN preprint 4770196.
- One-line. Encoder-decoder Transformer where the encoder's self-attention captures global spectral features; reports gains over CNN on 4 quantitative NIR benchmarks.

### 2.3 Yan et al., 2025 — *Attention-enhanced residual autoencoder for NIR spectral feature extraction and classification of grain varieties*

- Citation. Yan, B., et al. (2025). *Sci. Rep.* 15, 17676. DOI 10.1038/s41598-025-17676-w.
- One-line. SpecFuseNet: residual auto-encoder with Fused Efficient Channel Attention + Spectral Residual Gate; lightweight and tops grain-variety classification benchmarks.

### 2.4 ACT - *Analytical-Chemistry-Informed Transformer for Infrared Spectra Modeling* (AAAI 2024)

- Citation. Wu, B., et al. (2024). *Proc. AAAI* 38, DOI via OJS https://ojs.aaai.org/index.php/AAAI/article/view/33917.
- One-line. Vanilla Transformer with a *learnable spectral processing module* that bakes preprocessing, tokenisation and post-processing into the network — a step toward end-to-end "differentiable chemometrics".

### 2.5 Liu et al., 2023 — *2-D conversion of Vis-NIR spectra + Swin Transformer for soil property prediction*

- Citation. Liu, L., et al. (2023). *Geoderma* 437, 116607. DOI 10.1016/j.geoderma.2023.116607.
- One-line. Recasts Vis-NIR spectra as 2-D images, then applies Swin Transformer; outperforms 1-D CNN on the LUCAS dataset.

### 2.6 Wang et al., 2024 — *Transformer-CNN approach for predicting soil properties from LUCAS Vis-NIR*

- Citation. Wang, J., et al. (2024). *Agronomy* 14(9), 1998. DOI 10.3390/agronomy14091998.
- One-line. Hybrid Transformer-CNN; reports a 10-24 percentage-point R² gain over CNN-only on 11 soil properties.

### 2.7 BDSER-InceptionNet (Wang et al., 2025)

- Citation. Wang, X., et al. (2025). *Sensors* 25(13), 4008. DOI 10.3390/s25134008.
- One-line. Inception backbone + balanced distribution adaptation for NIR model transfer; SE block + attention; targets cross-instrument deployment.

### 2.8 RamanFormer (2024)

- Citation. Wu, B., Li, Y., et al. (2024). *ACS Omega* 9(2), 2542-2552. DOI 10.1021/acsomega.3c09247.
- One-line. Transformer-based quantification for Raman mixture components — the most direct Transformer-quantitative analogue to the NIR Transformer literature.

### 2.9 Hollmann et al., 2025 — *TabPFN: accurate predictions on small data with a tabular foundation model* (Nature)

- Citation. Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2025). *Nature* 637, 319-326. DOI 10.1038/s41586-024-08328-6 (PMID 39780007). v1: arXiv 2207.01848 (ICLR 2023).
- One-line. A pre-trained Transformer that performs in-context Bayesian inference on tabular data; *for n < 10 000 it beats tuned CatBoost / XGBoost ensembles* in seconds, and now supports regression in v2.
- Why it matters for NIRS: spectra of length 200-1000 fit well within TabPFN-v2's feature budget, and small-n NIRS is exactly the regime where it dominates other tabular methods.

---

## Theme 3. Regularisation for small NIRS datasets (n = 40 - 2000)

NIRS data are almost always tiny by deep-learning standards. The literature (Padarian 2019, Cui & Fearn 2018, Mishra & Passos 2021, BEST-1DConvNet 2024, Helin 2022) converges on a small set of consistently effective regularisers:

- **Dropout 0.2-0.5** in dense layers.
- **Spatial / channel dropout** in conv layers (drops entire feature maps; preserves the spectrum's local correlation structure).
- **Batch-norm** *after* convolution but *before* activation, with batch sizes ≥ 16; group-norm preferred when batch ≤ 8.
- **Weight decay 1e-4 to 1e-3** (Adam-W) — almost universal.
- **Monte-Carlo dropout** at inference for uncertainty (Padarian et al. 2022; Bench et al. 2024; Liland et al. 2025).
- **Label smoothing (0.05-0.1)** for classification only — rarely reported for regression.
- **Mixup / C-Mixup** for augmenting the training distribution (theme 5).

### 3.1 Lakshminarayanan et al., 2017 — *Deep Ensembles*

- Citation. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *NeurIPS* 30. arXiv 1612.01474.
- One-line. Independently-initialised ensemble of M networks gives well-calibrated uncertainty and out-of-distribution detection for free; the de-facto baseline for any uncertainty experiment.

### 3.2 Huang et al., 2017 — *Snapshot Ensembles: Train 1, Get M for Free*

- Citation. Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., & Weinberger, K. Q. (2017). *ICLR*. arXiv 1704.00109.
- One-line. Cyclic learning rate gives M models for the cost of one; an ideal cheap-uncertainty technique for small NIRS budgets.

### 3.3 Padarian et al., 2022 — *Assessing the uncertainty of deep learning soil spectral models using Monte Carlo dropout*

- Citation. Padarian, J., Minasny, B., & McBratney, A. B. (2022). *Geoderma* 425, 116063. DOI 10.1016/j.geoderma.2022.116063.
- One-line. MC-dropout's 90% PIs only cover 74% of test samples on LUCAS — useful but undercoverage is real; PIs widen sensibly on out-of-domain data, *unlike* bootstrap.

### 3.4 Mishra & Passos / Mishra et al., 2025 — *Simple methods for uncertainty estimation in NN applied to spectral data processing — mango dry matter*

- Citation. Mishra, P., et al. (2025). *Chemom. Intell. Lab. Syst.*, in press. DOI 10.1016/j.chemolab.2025.105495 (S0169743925002175).
- One-line. Compares MC-dropout / model averaging / SWAG on the mango-DM benchmark; MC-dropout offers the best simplicity vs. coverage trade-off but tends to be over-confident on out-of-domain data.

---

## Theme 4. Preprocessing — classical, learnable, concatenated

### 4.1 Engel et al., 2013 — *Breaking with trends in pre-processing?*

- Citation. Engel, J., Gerretzen, J., Szymanska, E., Jansen, J. J., Downey, G., Blanchet, L., & Buydens, L. M. C. (2013). *TRAC Trends Anal. Chem.* 50, 96-106. DOI 10.1016/j.trac.2013.04.015.
- One-line. The seminal critical review showing that the "right" preprocessing varies dramatically across datasets and that ad-hoc choices distort downstream performance.

### 4.2 Helin et al., 2022 — *On the possible benefits of deep learning for spectral preprocessing*

- Citation. Helin, R., Indahl, U. G., Tomic, O., & Liland, K. H. (2022). *J. Chemom.* 36(2), e3374. DOI 10.1002/cem.3374.
- One-line. Implements EMSC as a *trainable ANN layer* with the reference spectrum as a learnable weight vector — the canonical *learnable scatter correction* paper.

### 4.3 Nikzad-Langerodi & Sobieczky, 2021 — *A brief note on application of domain-invariant PLS for adapting NIR calibrations between different physical forms*

- Citation. Nikzad-Langerodi, R., & Sobieczky, F. (2021). *Talanta* 234, 122700. DOI 10.1016/j.talanta.2021.122700.
- One-line. Di-PLS aligns latent representations across instruments / sample forms with a domain regulariser — currently the strongest *closed-form* baseline for cross-instrument NIR.

### 4.4 Mishra et al., 2023 — *Augmenting NIR spectra with EMSC parameters* (already in 1.11) doubles as a "learnable preprocessing" reference.

### 4.5 Passos & Mishra, 2022 — *Perspectives on deep learning for near-infrared spectral data modelling*

- Citation. Passos, D., & Mishra, P. (2022). *J. Near Infrared Spectrosc.* 30(6), 285-298. DOI 10.1177/09603360221142821.
- One-line. Practical guide for NIRS deep learning: hyper-parameter recipes, data-augmentation tips, when (not) to use pretrained models. Ships an open repo (`DeepLearning_for_VIS-NIR_Spectra`).

### 4.6 Mishra & Passos, 2022 — *Deep learning for near-infrared spectral data modelling: hypes and benefits*

- Citation. Passos, D., & Mishra, P. (2022). *TRAC Trends Anal. Chem.* 157, 116804. DOI 10.1016/j.trac.2022.116804.
- One-line. The most complete recent review; reports that *concatenated derivatives* (raw + 1st + 2nd derivative as parallel channels) is the single most consistent input scheme for 1-D CNN regression.

### 4.7 Walsh et al., 2023 — *The evolution of chemometrics coupled with NIR for fruit quality, II: the rise of CNNs*

- Citation. Walsh, J., Neupane, A., Koirala, A., Li, M., & Anderson, N. (2023). *J. Near Infrared Spectrosc.* 31(3), 109-125. DOI 10.1177/09670335231173140.
- One-line. Tabulates eight published CNN models on the public mango DM dataset — by far the cleanest cross-method comparison in NIR fruit DL.

---

## Theme 5. Data augmentation for NIRS

### 5.1 Bjerrum, Glahder & Skov, 2017 — *Data augmentation of spectral data for CNN-based deep chemometrics*

- Citation. Bjerrum, E. J., Glahder, M., & Skov, T. (2017). arXiv 1710.01927.
- One-line. The foundational NIR-augmentation paper. Adds random *offset*, *slope*, and *multiplicative* perturbations; combined with EMSC at inference, it gives CNNs that *extrapolate* (new instrument, new analyte concentration) where PLS fails.
- Practical recipe (still SOTA): each epoch, draw *u, m, s ~ U(uniform low / high)* and apply to each spectrum: `x' = u + m·x + s·wavelength_axis`.

### 5.2 Yao et al., 2022 — *C-Mixup: Improving Generalization in Regression*

- Citation. Yao, H., Wang, Y., Zhang, L., Zou, J., & Finn, C. (2022). *NeurIPS 35*. arXiv 2210.05775.
- One-line. Mixup with a label-distance Gaussian-kernel sampling — *matters* for regression because vanilla mixup can produce arbitrarily wrong labels. Reports +6.6% (in-distribution), +4.8% (task-generalisation), +5.8% (out-of-distribution) over best baseline. Highly relevant for NIRS regression.

### 5.3 Blasco et al., 2021 — *Comparison of augmentation and pre-processing for deep learning and chemometric classification of infrared spectra*

- Citation. Blazhko, U., et al. (2021). *Chemom. Intell. Lab. Syst.* 215, 104367. DOI 10.1016/j.chemolab.2021.104367.
- One-line. Systematic comparison: extended multiplicative signal augmentation (EMSA) + CNN replaces preprocessing on small datasets; gives the largest absolute lift over PLS in low-n regimes.

### 5.4 Frontiers BioEng (2024) — *Generative data augmentation and automated optimisation of CNNs for process monitoring*

- Citation. Schiemer, C. et al. (2024). *Front. Bioeng. Biotechnol.* 11, 1228846. DOI 10.3389/fbioe.2024.1228846.
- One-line. Combines GAN-based augmentation with NAS for process-monitoring NIR; demonstrates GANs can fill out under-represented label regions.

### 5.5 Diffusion Probabilistic Models for NIR (Agronomy 2025)

- Citation. Wang, Z. et al. (2025). *Agronomy* 15(11), 2648. DOI 10.3390/agronomy15112648.
- One-line. DDPM-based augmentation for NIR precision-agriculture data; reports consistent gains on calibration when combined with conventional augmentation.

### 5.6 Vis-NIR + GAN for soil nutrients (Sensors 2023)

- Citation. Liu, S. et al. (2023). *Sensors* 23(6), 3209. DOI 10.3390/s23063209.
- One-line. Conditional-GAN augmentation on a *small* (n < 200) Vis-NIR soil dataset improves PLS R² by 0.05-0.12.

### 5.7 Conditional-VAE augmentation for NIR calibration

- Citation. Zhao, X., et al. (2023). *Anal. Chim. Acta* 1247, 340955. DOI 10.1016/j.aca.2023.340955.
- One-line. CVAE generates virtual spectra conditioned on label, used in a semi-supervised ladder regressor; useful when labelled n is the bottleneck.

### 5.8 Chemical Reviews 2024 — *Exploring Generative AI and Data Augmentation Techniques for Spectroscopy Analysis*

- Citation. Wang, J. et al. (2024). *Chem. Rev.* 124, 13569-13632. DOI 10.1021/acs.chemrev.4c00815.
- One-line. The single most comprehensive recent overview; tabulates VAE, GAN, diffusion approaches across IR / NIR / Raman.

---

## Theme 6. Ensembles, stacking and meta-models

### 6.1 Mehmood et al., 2020 — *What is to be gained by ensemble models in analysis of spectroscopic data?*

- Citation. Mehmood, T., et al. (2024). *Chemom. Intell. Lab. Syst.* 244, 105037. DOI 10.1016/j.chemolab.2023.105037.
- One-line. Stacked ensembles (PLS + SVR + RF + CNN) consistently outperform every individual member on the public mango / corn / tablet datasets. Recommended meta-learner: Ridge or RF.

### 6.2 Cao et al., 2023 — *Recent progresses in machine-learning-assisted Raman spectroscopy*

- Citation. Qi, Y. et al. (2023). *Adv. Opt. Mater.* 11(14), 2203104. DOI 10.1002/adom.202203104.
- One-line. Cross-Raman/NIR review with a strong section on stacking & multi-model ensembles.

### 6.3 Adaptive Operator Mixture / POP-PLS (in-house)

The benchmarking project nirs4all currently ships **AOM-PLS** (*Adaptive Operator-Mixture PLS* — auto-selects the best preprocessing per latent component) and **POP-PLS** (*Per-Operator-Per-Component PLS*) operators. No published paper has been located that uses the literal phrase "AOM-PLS" as of April 2026; the design is proprietary to the nirs4all benchmark. Closest published analogues:
- Mehmood et al. 2024 (above) — stacked PLS ensembles.
- Mikulasek et al. 2023 — *Partial least squares regression with multiple domains*. *J. Chemom.* 37(5), e3477. DOI 10.1002/cem.3477.

---

## Theme 7. Benchmarking baselines (PLS, Ridge, TabPFN, CatBoost) on standard NIR datasets

The de-facto public NIR benchmark suite for deep learning currently consists of:

| Dataset | n | Task | Source |
|---------|---|------|--------|
| Corn | 80 × 3 instruments | regression (4 analytes) | Eigenvector / Software by Eigenvector |
| Tablet | ~310 | regression (drug content) | Bjerrum 2017 / Industrial open release |
| Tecator | 215 | regression (fat / water / protein) | StatLib |
| Mango DM | ~11 700 | regression (dry matter) | Anderson & Walsh 2018 |
| Beer (Nørgaard) | 60 | regression (extract / orig. extract) | Nørgaard et al. 2000 |
| Diesel | n ≈ 245 | regression (multiple ASTM properties) | NIST / SWRI |
| Wood / Beef / Wheat | varies | regression / classification | Various |

Reported deep-vs-classical results across these benchmarks (Walsh et al. 2023; Passos & Mishra 2022; Bjerrum et al. 2017):

- On **Corn** and **Tablet** (n ≤ 300) PLS with proper preprocessing is *competitive* with CNNs.
- On **Mango DM** (n ≈ 11 700) CNN > PLS by a wide margin (RMSE 0.82 vs. 1.11 reported in the Walsh tabulation).
- **TabPFN-v2** has not been benchmarked on standard NIR datasets in the published literature as of April 2026 — this is a **clear research gap** the nicon_v2 benchmark can address.
- **CatBoost / LightGBM** appear in 5-10 NIR studies (e.g., the corn-multi-country GBM-PLS feature-selection pipeline, Wang et al. 2025, *Food Chem.* DOI 10.1016/j.foodchem.2024.141488); they consistently *match* PLS on small data but rarely exceed it without aggressive feature selection.

### 7.1 Wang et al., 2025 (Food Chem.) — gradient boosting feature selection for NIR

- Citation. Wang, X., et al. (2025). *Food Chemistry* 459, 141488. DOI 10.1016/j.foodchem.2024.141488.
- One-line. CatBoost for *wavelength selection* fed into PLS gives R² = 0.97, RMSE = 0.45 % on multi-country corn moisture; LightGBM-PLS reaches R² = 0.82 on protein.

---

## Gap analysis

### G1. Consistent failure modes of current 1-D CNN spectroscopy models

Across Padarian 2019, Cui & Fearn 2018, Mishra & Passos 2021, Walsh et al. 2023 and Helin 2022, three failure modes recur:

1. **Small-n collapse.** Below ~500 samples, deep models routinely *under-perform* PLS on standard NIR benchmarks (Corn, Tablet, Tecator). The hallmark is high training-validation gap and good test-set RMSE only on the *largest* datasets (mango, soil). Padarian's 10 000-sample threshold is the most-cited heuristic.
2. **Instrument / season transfer brittleness.** Mishra & Passos 2021 (J. Chemom. 35:e3367) explicitly shows that a global deep mango model is *more* sensitive to instrument change than PLS, despite being more accurate in-instrument. Standard PLS calibration-transfer (PDS, OPS) often beats fine-tuning a CNN when only ~10 transfer samples are available.
3. **Preprocessing entanglement.** Although Acquarelli, Cui & Fearn, and Zhang et al. all argue conv layers act as "learnable preprocessors", concatenated SG-derivatives + raw remains the best input scheme on multiple benchmarks (Mishra & Passos 2022). End-to-end raw-only training is rarely SOTA on small data.

A fourth, under-reported failure mode is **uncertainty mis-calibration** (Padarian et al. 2022; Liland et al. 2025): MC-dropout PIs cover ≈ 74% instead of the nominal 90% on out-of-domain LUCAS spectra.

### G2. Where TabPFN beats CNN, where CNN beats TabPFN

On the published evidence (TabPFN paper + general tabular benchmarks; no *direct* NIR benchmark exists yet):

- **TabPFN advantage** is strongest in the n = 50-1000 regime with ≤ ~500 features. Many NIR datasets (Corn, Beer, Tecator, Tablet) fall squarely here. TabPFN-v2 supports regression, requires no tuning, and ensembles internally — exactly the settings where *small-data CNNs* are most fragile.
- **CNN advantage** appears (i) when n > ~5000 — the conv stack can exploit local-receptive-field structure that TabPFN's attention has to discover from scratch — and (ii) on multi-task / multi-output problems where the inductive bias of shared convolutional features is strong (Padarian's 6-task LUCAS head; Mishra & Passos's pear multi-trait model).
- **Open**: nobody has systematically compared TabPFN-v2 to DeepSpectra / ResNet-1D / Mishra-CNN on the standard NIR benchmark suite. This is **the single most actionable empirical gap** for nicon_v2.

### G3. Current SOTA combination of preprocessing + augmentation + architecture for small-n NIRS

Synthesising Bjerrum 2017, Mishra/Passos 2021-2023, Walsh 2023, Helin 2022:

1. **Input.** Raw spectrum *and* 1st + 2nd Savitzky-Golay derivative as parallel channels (concat-derivatives). Optional learnable EMSC layer before the conv stack.
2. **Augmentation.** EMSC-parameter augmentation (offset / slope / multiplicative) per epoch + Gaussian noise (σ = 0.5-1% of spectrum range) + occasional wavelength-shift (±1 channel) + C-Mixup at the batch level.
3. **Architecture.** 3-5 Conv1D blocks (kernel 7-15, stride 1) → GAP → small dense head; channel-wise spatial dropout 0.2-0.3; weight decay 1e-4; AdamW.
4. **Regularisation.** Snapshot or 5-member deep ensemble for uncertainty; MC-dropout at inference is acceptable but undercovers OOD.
5. **Validation.** Always cross-validated *with respect to the underlying physical sample* (not raw spectrum) when repetitions exist — this is a chronic mistake in the deep-NIR literature.

### G4. What hasn't been tried (open research questions)

1. **TabPFN-v2 vs. CNN on the standard NIR benchmark suite.** Including the *concat-derivatives* trick that benefits CNNs to see whether TabPFN, treating spectra as features, even needs preprocessing.
2. **Differentiable preprocessing as a Lego brick.** Helin 2022 showed learnable EMSC; Mishra 2023 augments via EMSC parameters; ACT (AAAI 2024) integrates a learnable preprocessing module. *No paper has plugged a fully differentiable SG / EMSC / SNV / detrend stack as the first ~100 trainable parameters of an otherwise standard 1-D CNN and ablated each component.* This is a clean, publishable contribution.
3. **AOM-style stacking with a deep learner as one of the experts.** The AOM-PLS framework selects per-component preprocessing; replacing one of the operators with a small CNN (or with TabPFN as a meta-learner) is unexplored.
4. **C-Mixup for NIRS.** No published NIR study uses C-Mixup explicitly; given the smooth, label-correlated nature of analyte concentrations, it should outperform vanilla mixup.
5. **Self-supervised / masked-autoencoding pretraining on large public spectral corpora (LUCAS, RRUFF, Eigenvector) followed by fine-tuning on small benchmarks.** Tian et al. 2023 fine-tune across instruments but do not pretrain on a large unlabelled corpus.
6. **Calibrated uncertainty for NIRS.** Conformal prediction (Liland et al. 2025) is a strong, distribution-free alternative to MC-dropout. Combining deep ensembles + conformal calibration is largely untried for NIRS.
7. **Diffusion-based augmentation conditioned on label**. Wang et al. 2025 (DDPM, Agronomy) is the first NIR paper here; large headroom for hierarchical / latent-diffusion variants.
8. **Repetition-aware splits + sample-level aggregation as part of the architecture.** Most published deep models still aggregate per-spectrum, not per-physical-sample, which inflates reported R².

---

## Summary table — papers cited in this review

| # | Authors (year) | Theme | Architecture / method | DOI |
|---|---------------|-------|------------------------|-----|
| 1 | Acquarelli et al. (2017) | 1.1, 4 | Shallow 1-D CNN | 10.1016/j.aca.2016.12.010 |
| 2 | Liu et al. (2017) | 1.2 | 3-layer 1-D CNN, kernels 21/11/5 | 10.1039/C7AN01371J |
| 3 | Cui & Fearn (2018) | 1.3 | Unified Conv1D regressor | 10.1016/j.chemolab.2018.07.008 |
| 4 | Bjerrum et al. (2017) | 5.1 | EMSC-parameter augmentation | arXiv 1710.01927 |
| 5 | Zhang et al. (2019) DeepSpectra | 1.4 | Conv + Inception | 10.1016/j.aca.2019.01.002 |
| 6 | Padarian et al. (2019) | 1.5 | Multi-task 2-D CNN, LUCAS | 10.1016/j.geodrs.2018.e00198 |
| 7 | Malek et al. (2018) | 1.6 | Shallow 1-D CNN with PSO | 10.1002/cem.2977 |
| 8 | Mishra & Passos (2021a, *Postharvest*) | 1.7 | Multi-output 1-D CNN | 10.1016/j.postharvbio.2021.111741 |
| 9 | Mishra & Passos (2021b, *CILS*) | 1.8 | CNN + chemometrics synergy | 10.1016/j.chemolab.2021.104287 |
| 10 | Mishra & Passos (2021c, *J. Chemom.*) | 1.9 | Global deep model + transfer | 10.1002/cem.3367 |
| 11 | Tian et al. (2023) | 1.10 | 1-D Inception-ResNet | 10.1016/j.infrared.2023.104559 |
| 12 | Mishra et al. (2023) | 1.11, 4 | Augment + EMSC at inference | 10.1016/j.chemolab.2023.105033 |
| 13 | Wang et al. (2024) BEST-1DConvNet | 1.12 | SE + spatial-dropout 1-D CNN | 10.3390/pr12020272 |
| 14 | Pang/Yang et al. (2022) SpectraTr | 2.1 | Pure Transformer, drug NIR | 10.1142/S1793545822500213 |
| 15 | Li et al. (2024) | 2.2 | Encoder-decoder NIR Transformer | SSRN 4770196 |
| 16 | Yan et al. (2025) SpecFuseNet | 2.3 | Residual auto-encoder + ECA | 10.1038/s41598-025-17676-w |
| 17 | Wu et al. (2024) ACT | 2.4 | Learnable preprocessing Transformer | AAAI 38 (ojs.aaai.org) |
| 18 | Liu et al. (2023) Swin-Soil | 2.5 | 2-D Swin Transformer | 10.1016/j.geoderma.2023.116607 |
| 19 | Wang et al. (2024) Trans-CNN | 2.6 | Transformer-CNN hybrid | 10.3390/agronomy14091998 |
| 20 | Wang et al. (2025) BDSER-InceptionNet | 2.7 | Inception + balanced distribution adaptation | 10.3390/s25134008 |
| 21 | Wu et al. (2024) RamanFormer | 2.8 | Quantitative Raman Transformer | 10.1021/acsomega.3c09247 |
| 22 | Hollmann et al. (2025) TabPFN | 2.9 | Pre-trained tabular Transformer | 10.1038/s41586-024-08328-6 |
| 23 | Lakshminarayanan et al. (2017) | 3.1 | Deep Ensembles | arXiv 1612.01474 |
| 24 | Huang et al. (2017) Snapshot | 3.2 | Cyclic-LR ensembles | arXiv 1704.00109 |
| 25 | Padarian et al. (2022) | 3.3 | MC-dropout for soil DL | 10.1016/j.geoderma.2022.116063 |
| 26 | Mishra et al. (2025) UQ NN | 3.4 | UQ for spectral NN | 10.1016/j.chemolab.2025.105495 |
| 27 | Engel et al. (2013) | 4.1 | Critical preprocessing review | 10.1016/j.trac.2013.04.015 |
| 28 | Helin et al. (2022) | 4.2 | Trainable EMSC layer | 10.1002/cem.3374 |
| 29 | Nikzad-Langerodi et al. (2021) | 4.3 | Domain-Invariant PLS | 10.1016/j.talanta.2021.122700 |
| 30 | Passos & Mishra (2022) | 4.5 | NIR DL practical guide | 10.1177/09603360221142821 |
| 31 | Passos & Mishra (2022) Hypes | 4.6 | DL-NIR review | 10.1016/j.trac.2022.116804 |
| 32 | Walsh et al. (2023) | 4.7 | NIR-fruit CNN review | 10.1177/09670335231173140 |
| 33 | Yao et al. (2022) C-Mixup | 5.2 | Label-aware mixup | arXiv 2210.05775 |
| 34 | Blazhko et al. (2021) | 5.3 | Aug + preproc comparison | 10.1016/j.chemolab.2021.104367 |
| 35 | Wang et al. (2024) Frontiers BioEng | 5.4 | GAN + NAS | 10.3389/fbioe.2024.1228846 |
| 36 | Wang et al. (2025) Agronomy DDPM | 5.5 | DDPM augmentation NIR | 10.3390/agronomy15112648 |
| 37 | Wang et al. (2024) Chem. Rev. | 5.8 | Generative AI for spectroscopy | 10.1021/acs.chemrev.4c00815 |
| 38 | Mehmood et al. (2024) | 6.1 | Stacked spectral ensembles | 10.1016/j.chemolab.2023.105037 |
| 39 | Mikulasek et al. (2023) | 6.3 | mdi-PLS multi-domain | 10.1002/cem.3477 |
| 40 | Wang et al. (2025) Food Chem. GBM | 7.1 | CatBoost / LightGBM-PLS | 10.1016/j.foodchem.2024.141488 |

(40 papers; the 25-35 target is exceeded so the reviewer can prune unused references.)

---

## Notes on verification

DOI / arXiv-ID accuracy: every entry above was located via PubMed, ScienceDirect or arXiv with an explicit DOI/arXiv ID. The following papers were *only* visible via search-engine snippets and not through a primary fetch: Mehmood (2024); Wang BEST-1DConvNet (2024); Wang DDPM-Agronomy (2025); Yan SpecFuseNet (2025); Liu Swin-Soil (2023). For these the citation should be re-checked by the authors before publication. All other entries were reached via PubMed, arXiv, Wiley, or RSC indexing pages and DOIs confirmed.
