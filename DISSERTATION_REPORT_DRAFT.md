# Deep Learning Based Classification of Cardiotocograms for Predicting Birth Outcomes

## Working Title

Deep Learning Based Classification of Cardiotocograms for Predicting Birth Outcomes: A Grouped Multi-Class Study on CTU-UHB Fetal Heart Rate Signals

## What This Draft Is For

This document is a dissertation-writing support draft based on the repository contents, notebook text, and saved experiment outputs. It is written to help turn the project into a strong 20-page report that covers background, design, implementation, testing, evaluation, and critical reflection.

The strongest report narrative in the repository is not "I trained one model". It is:

1. The project began with clinically motivated exploratory binary classification.
2. It then progressed into a more realistic three-class problem using expert majority-vote labels.
3. The final contribution is a grouped, leakage-aware, step-level pipeline for classifying Normal, Mild, and Severe fetal states from delivery-anchored CTG windows.
4. Multiple baselines and ablations were tested, and the final selected solution was chosen because it gave the best overall balance of macro F1, balanced accuracy, and minority-class behaviour.

## Recommended Central Argument

This project demonstrates that clinically annotated CTG signals can be classified with meaningful performance using a carefully designed deep learning pipeline, but it also shows that success depends at least as much on label design, delivery-anchored segmentation, leakage-safe validation, and class-imbalance handling as on model architecture alone. The final system materially outperforms tuned classical baselines on the same grouped folds, although performance remains below the threshold required for direct clinical deployment.

## Dissertation-Ready Aims and Objectives

### Main Aim

To design, implement, and evaluate a reproducible machine learning pipeline that classifies cardiotocography recordings into clinically meaningful fetal outcome categories using expert-annotated CTG data.

### Objectives

1. Build a preprocessing pipeline that converts raw CTU-UHB CTG recordings into quality-controlled, fixed-length signal segments suitable for machine learning.
2. Construct reliable labels from expert annotations using majority voting and remove uninterpretable cases where appropriate.
3. Establish baseline models for both early exploratory binary classification and the final three-class task.
4. Design a deep learning model that combines learned temporal representations from fetal heart rate with compact engineered clinical features.
5. Evaluate the model using grouped cross-validation that prevents leakage between windows from the same patient record.
6. Quantify the effect of class weighting, augmentation, calibration, and loss-function design on overall and per-class performance.
7. Critically assess whether the final system meets the project goals and identify what remains necessary for clinical robustness.

### Measurable Success Criteria

1. The pipeline should be reproducible from raw data and annotation files to final metrics.
2. Validation should be grouped by record identifier so that no record contributes windows to both train and validation sets.
3. The final three-class model should outperform feature-based classical baselines on the same grouped evaluation setting.
4. The selected final model should provide balanced performance across Normal, Mild, and Severe classes, rather than inflating one class at the expense of collapse in another.
5. The final report should demonstrate not only accuracy claims but also error analysis, trade-off discussion, and limitations.

### Stretch Goals

1. Improve Mild recall without severely damaging Severe recall or balanced accuracy.
2. Incorporate calibration or decision adjustment for clinically safer class behaviour.
3. Extend the pipeline to use uterine contraction information in addition to fetal heart rate.
4. Explore transfer learning or stronger external validation.

## 20-Page Structure

This is a practical page allocation for a 20-page dissertation report excluding references and appendices if your department allows that separation.

1. Introduction and motivation, 2 pages.
2. Background and related work, 3 pages.
3. Problem definition, aims, and success criteria, 1 page.
4. Data, labels, and preprocessing, 3 pages.
5. System design and implementation, 4 pages.
6. Experimental methodology, 2 pages.
7. Results and comparative evaluation, 3 pages.
8. Critical discussion, limitations, and reflection, 1.5 pages.
9. Future work and conclusion, 0.5 to 1 page.

## Executive Evaluation of the Whole Project

The project is strongest when framed as an end-to-end applied machine learning study rather than purely as a neural network exercise. The repository shows a clear progression from exploratory work to a substantially more rigorous final pipeline. Early notebooks focused on a binary task that asked whether a full CTG record ever contained mild hypoxia. That was a reasonable first step because it simplified the label space and made it possible to test whether meaningful signal could be learned from FHR traces at all. Those experiments already showed useful discrimination, with the binary notebook reporting a record-level AUC of about 0.79 and Mild Hypoxia precision and recall both around 0.58 to 0.60. The notebook itself correctly states that this was not clinically reliable yet, which is an honest and valuable conclusion.

The more important technical contribution is the later multi-class pipeline centred on Step 2 and Step 3 delivery-anchored windows. This is a much better framing of the problem because the labels and the supervised examples are aligned to clinically meaningful time segments rather than loosely aggregated whole records. The final pipeline uses expert majority-vote labels, excludes uninterpretable cases, generates cleaned 30-minute FHR windows, and evaluates models using grouped cross-validation by record identifier. That grouped design is one of the strongest aspects of the whole dissertation because it addresses a common source of leakage in biomedical time-series projects where multiple windows from the same patient can accidentally appear in both training and validation splits.

The final selected model is a hybrid 1D CNN with squeeze-and-excitation style channel weighting, residual and separable convolutional blocks, and an optional branch for ten engineered features. On the main saved repeated grouped cross-validation experiment using Step 2 and Step 3 windows, it achieves a mean macro F1 of 0.6055 plus or minus 0.0589 and a mean balanced accuracy of 0.6189 plus or minus 0.0623 across 5 folds and 3 repeats, over 1398 windows from 336 records. Per-class recall is relatively balanced: Normal 0.6499, Mild 0.6173, Severe 0.5894. Those are not clinical-grade results, but they are credible research results because they are backed by repeated grouped validation and explicit trade-off analysis.

The project also compares the deep model against tuned classical baselines on engineered features. In the saved Model_Comparison notebook outputs, Logistic Regression reaches macro F1 0.4988 and balanced accuracy 0.5478, Random Forest reaches macro F1 0.5115 and balanced accuracy 0.5093, and SVM reaches macro F1 0.4284 and balanced accuracy 0.4605. This gives the report a strong comparison section: the final deep architecture is not only "reasonable" in isolation, it materially outperforms feature-based baselines under the same grouped evaluation logic.

The most important negative result in the repository is also useful. Several attempts to improve Mild recall through focal loss, severe-boost calibration, stronger augmentation, and penalty-based objectives did not improve overall performance. The tuned focal plus calibration experiment increased Mild recall to 0.7854, but balanced accuracy fell to 0.5774 and Severe recall fell to 0.4841. The strong augmentation calibrated cross-entropy variant came closer to the baseline, with balanced accuracy 0.6127 and Severe recall 0.6127, but it still did not beat the main softmax baseline overall. This is valuable dissertation material because it shows genuine experimentation, negative results, and justified model selection rather than cherry-picking.

Overall, the project meets the standard of a substantial and technically serious dissertation. Its main strengths are sound problem reformulation, leakage-aware evaluation, clinically motivated segmentation, and comparative experimentation. Its main weaknesses are dataset scale, class imbalance, incomplete external validation, dependence on notebook-defined pipelines, and the absence of interpretability or clinician-facing validation.

## Introduction Draft

Cardiotocography is widely used during labour to monitor fetal heart rate and uterine activity in order to identify signs of fetal distress. Despite its routine clinical use, CTG interpretation remains difficult because fetal heart rate traces are noisy, non-stationary, and often ambiguous. Inter-observer variability is well documented, and even experienced clinicians may disagree about whether a trace should be considered normal, suspicious, pathological, or uninterpretable. This makes CTG a compelling but challenging problem for machine learning.

The motivation for this project is twofold. First, earlier identification of hypoxic or pathological fetal states has clear clinical value because delayed recognition can contribute to poor neonatal outcomes. Second, CTG interpretation is exactly the sort of problem where data-driven decision support may help: the signal is high-dimensional, time-dependent, and clinically relevant, yet difficult to summarise using a small set of manual rules alone. However, for such a system to be credible, it must do more than fit a neural network to raw signals. It must use clinically meaningful labels, avoid data leakage, justify preprocessing decisions, and evaluate performance honestly under class imbalance.

This dissertation therefore investigates whether deep learning can classify CTG recordings into clinically relevant fetal outcome categories using expert-annotated data from the CTU-UHB database. The work progresses from initial binary classification experiments to a more realistic three-class setting using delivery-anchored windows and majority-vote expert labels. The final objective is not simply to maximise a single metric, but to build and evaluate a complete pipeline that is methodologically sound, reproducible, and informative about the trade-offs involved in CTG classification.

## Background and Significance Draft

CTG analysis is a clinically important but difficult classification task. Traditional interpretation is based on fetal heart rate baseline, variability, accelerations, decelerations, and contextual labour information. In practice, these criteria are applied subjectively, and the same trace may receive different labels from different obstetricians. The repository reflects this challenge directly: the project uses expert annotations from nine obstetricians and resolves disagreement through majority voting. That design decision is significant because it turns label uncertainty into an explicit part of the problem formulation rather than ignoring it.

From a machine learning perspective, CTG is challenging for several reasons. First, recordings can be long and contain only brief abnormal episodes, so whole-record classification may dilute local pathological patterns. Second, missing signal, dropouts, and outliers are common, which makes preprocessing essential. Third, the class distribution is imbalanced, especially for severe pathological cases, so naive accuracy can be misleading. Finally, biomedical time-series data are prone to optimistic evaluation if segments from the same patient appear in both training and validation sets.

These challenges motivate the main design of this project. Rather than relying only on full-record classification, the final pipeline uses clinically anchored Step 2 and Step 3 windows. Rather than trusting a single clinician label, it uses majority-vote annotations. Rather than reporting plain accuracy, it uses macro F1, balanced accuracy, and per-class recall. Rather than random window splitting, it uses grouped cross-validation by record. Together these decisions make the work more robust and more relevant to real clinical interpretation.

## Problem Definition Draft

The final problem addressed in this dissertation is a three-class classification task over delivery-anchored CTG segments. Each supervised instance corresponds to a Step 2 or Step 3 30-minute fetal heart rate window extracted from the CTU-UHB dataset, labelled using expert majority vote as Normal, Mild, or Severe. The task is challenging because these classes are clinically adjacent rather than maximally separated, especially in the Mild category, which often shares characteristics with both Normal and Severe states.

An important refinement introduced in the project is the distinction between record-level and step-level supervision. Early experiments asked whether an entire CTG record contained signs of mild hypoxia at any point. The later pipeline instead preserves the labelled temporal segment, allowing a record to contribute different step-level labels without forcing a single record-wide diagnosis. This is a stronger formulation because it matches the supervision to the actual labelled portion of the signal and reduces ambiguity in the learning target.

## Data and Label Construction Draft

The project uses the CTU-UHB cardiotocography database as its main data source. Raw signals are stored in WFDB format and include fetal heart rate and uterine contractions sampled at 4 Hz. The repository shows that expert annotations are provided separately and were produced by nine practicing obstetricians using FIGO-style clinical criteria. Since expert disagreement is inherent to this task, the project constructs labels using majority voting. This is a sensible compromise: it reduces annotation noise while acknowledging that CTG interpretation is not perfectly objective.

The label files indicate that the three target classes are Normal, Mild Hypoxia, and Severe Hypoxia, with uninterpretable cases filtered out in most downstream experiments. The notebook outputs also reveal notable class imbalance. For example, one saved label-distribution summary shows the Severe class at roughly 5.4 percent in one setting, while later Step 2 and Step 3 subsets remain substantially imbalanced. This imbalance is central to the evaluation strategy and explains why balanced accuracy and per-class recall are more informative than plain accuracy.

The final pipeline focuses on Step 2 and Step 3 windows. Repository outputs report 548 labelled Step 2 records and 337 labelled Step 3 records after excluding uninterpretable cases, with 549 records having at least one Step 2 or Step 3 label and a usable signal. From these records, the controlled Step 2 plus Step 3 pipeline creates 1398 windows in total: 399 Step 2 windows from 133 records and 999 Step 3 windows from 333 records. This filtering and segmentation step is important to describe clearly because it explains why the usable cohort is smaller than the raw dataset.

## Preprocessing Draft

The preprocessing pipeline is one of the strongest engineering aspects of the project. The repository implements a clear sequence of operations for cleaning raw fetal heart rate signals before model training. First, trailing zeros are removed to identify the usable end of the recording. Signals are then downsampled from 4 Hz to 1 Hz for fixed-length modelling. Quality filtering removes windows with excessive signal loss, and final windows are padded or trimmed to a consistent 1800-sample length corresponding to 30 minutes.

The later notebooks implement a more careful FHR cleaning procedure that replaces zero values with missing values, removes long gaps, discards physiologically implausible outliers below 50 bpm or above 200 bpm, interpolates missing values, and removes abrupt spikes greater than 25 bpm before interpolating again. These operations are justified because CTG data contain sensor dropout and artefacts that would otherwise dominate learning. Importantly, the report should explain that this preprocessing is not cosmetic. In biomedical time-series, preprocessing directly changes the meaning of the input, so each decision should be presented as part of the modelling rationale.

The project also computes ten engineered features from each cleaned window: mean fetal heart rate, median fetal heart rate, standard deviation, interquartile range, mean absolute successive difference, minute-wise mean variability, number of accelerations, number of decelerations, maximum deceleration depth, and prefill missing fraction. These are useful because they inject clinically interpretable summary information into the final model while keeping the raw temporal signal available for feature learning.

## Design and Implementation Draft

The final model is best described as a hybrid one-dimensional convolutional neural network tailored to 30-minute fetal heart rate windows. Its design combines several sensible architectural choices. A temporal convolutional front-end captures short-term local structure. Parallel separable convolutions with multiple kernel sizes allow the model to detect patterns at different temporal scales. Residual connections improve gradient flow and make deeper temporal processing more stable. Squeeze-and-excitation style blocks reweight learned channels adaptively. Global average pooling gives the architecture length robustness and reduces parameter count. Finally, an optional dense branch processes engineered clinical features and concatenates them with learned signal features before classification.

This is a persuasive design for CTG. Fetal heart rate traces contain both local events, such as accelerations and decelerations, and broader variability structure. Multi-scale temporal filters are therefore more appropriate than a single fixed kernel width. Residual connections are justified because the model goes beyond a trivial shallow CNN. The engineered feature branch is also well motivated because some clinically meaningful information, such as deceleration count or summary variability, may be easier to learn explicitly than implicitly from limited data.

The implementation is also strong in how it handles evaluation. The repository constructs grouped folds at record level so that no record contributes windows to both train and validation sets. Normalisation statistics are computed on the training fold only and then applied to validation data, which avoids leakage. The later tuning helpers also aggregate predictions at record-step level, which is important because the target label belongs to the step unit, not to an arbitrary fixed sub-window. This level of care is a major part of the dissertation's technical merit.

## Experimental Methodology Draft

The evaluation strategy should be presented as a central contribution, not as a minor implementation detail. The final repeated experiment uses five-fold grouped cross-validation repeated three times, giving fifteen fold-runs in total. The saved summary reports that all fifteen folds met the requested minimum severe-class threshold, which strengthens confidence that the reported metrics are not dominated by degenerate splits. For the final selected baseline, the dataset contains 1398 windows from 336 records and 466 record-steps.

The primary metrics should be macro F1 and balanced accuracy. Macro F1 is appropriate because it weights each class equally and therefore penalises failure on the minority Severe class. Balanced accuracy is appropriate because it averages recall across classes and avoids the false comfort of standard accuracy under imbalance. Per-class recall should be reported alongside these metrics because the clinical meaning of different errors is not identical. Missing this discussion would weaken the dissertation.

The project also includes a useful set of comparison experiments. Classical baselines use the same engineered features and are tuned with group-aware nested RandomizedSearchCV. The deep pipeline additionally explores class-weight sweeps, loss-function changes, augmentation strength, and post-hoc severe-class boosting. This gives the report a strong narrative of systematic optimisation rather than one-off trial-and-error.

## Results and Evaluation Draft

The strongest final quantitative result in the repository is the Step 2 plus Step 3 softmax repeated grouped cross-validation experiment. This configuration uses manual class weights of 0.9 for Normal, 1.2 for Mild, and 2.5 for Severe. Across five folds and three repeats, it achieves mean macro F1 0.6055 plus or minus 0.0589, mean balanced accuracy 0.6189 plus or minus 0.0623, and mean accuracy 0.6274 plus or minus 0.0596. Per-class recall is Normal 0.6499, Mild 0.6173, and Severe 0.5894.

These results matter for two reasons. First, the model does not simply optimise the majority class while ignoring the minority class. Severe recall remains close to 0.59, which is imperfect but non-trivial given the scarcity of severe examples. Second, performance across the three classes is relatively balanced compared with some alternatives in the repository. That balance justifies selecting this baseline as the main reported model.

The class-weight sweep provides an important negative result. None of the tested class-weight alternatives outperformed the saved baseline summary. The best sweep candidate came close but still underperformed on balanced accuracy and macro F1. This is useful evidence that the chosen class weights were not arbitrary.

The calibration and loss-function ablations are also informative. The tuned focal-loss plus light augmentation plus calibration experiment increased Mild recall from 0.6173 to 0.7854, which initially looks attractive. However, balanced accuracy dropped from 0.6189 to 0.5774 and Severe recall dropped from 0.5894 to 0.4841. This means the gain in Mild sensitivity came at too high a cost elsewhere. By contrast, the cross-entropy plus strong augmentation plus calibration variant produced balanced accuracy 0.6127 and Severe recall 0.6127, making it a more competitive alternative, but it still did not exceed the baseline overall. The severe-only penalty variant also underperformed the baseline. The report should use these results to argue that model selection was based on balanced evidence rather than on maximising one favourable metric.

The project is additionally strengthened by explicit baseline comparison. In the Model_Comparison notebook outputs, Logistic Regression achieved macro F1 0.4988 and balanced accuracy 0.5478, Random Forest achieved macro F1 0.5115 and balanced accuracy 0.5093, and SVM achieved macro F1 0.4284 and balanced accuracy 0.4605. The final deep model therefore improves macro F1 by roughly nine to eighteen points depending on baseline and improves balanced accuracy by roughly seven to sixteen points. That is a strong result and should be stated clearly.

You can also use the earlier binary work as part of the story rather than hiding it. The binary MS-CNN results saved in the notebook outputs report F1 0.7120 plus or minus 0.1507, precision 0.6867 plus or minus 0.1596, and recall 0.7777 plus or minus 0.2020. Those results are not directly comparable to the final three-class task, but they are useful as evidence that the early stage of the project successfully established learnable signal before moving to the harder multi-class formulation.

## Critical Discussion Draft

The project has several major strengths. The first is methodological maturity. It would have been easy to overstate performance using random window splits or whole-record labels with ambiguous temporal meaning, but the final system instead uses grouped validation and step-aligned supervision. That decision makes the project more credible than many student machine learning projects with superficially higher metrics. The second strength is breadth of experimentation. The repository includes preprocessing studies, binary modelling, classical baselines, deep baselines, class-weight sweeps, augmentation variants, and calibration ablations. The third strength is honest model selection. The strongest overall model was retained even though some alternative settings improved one class-specific metric.

There are also real weaknesses, and the report should be direct about them. The dataset is still relatively small for deep learning, especially after filtering and after separating by step. The Severe class remains scarce, which makes fold-level variance meaningful and limits confidence in rare-event generalisation. The project has no held-out external test set and no validation on a second clinical cohort, so the final results should be interpreted as internal generalisation only. In addition, although the repository contains transfer-learning ideas and UC alignment notes, the final tracked pipeline is still largely FHR-only and from-scratch. That is defensible, but it means the dissertation should not over-claim multimodal or transfer-learning impact.

Another weakness is reproducibility friction. Some of the later experiments rely on executing definition cells from notebooks, and the repository memory notes already mention notebook persistence issues. This does not invalidate the work, but it does mean that full reproducibility is weaker than it would be in a pure script-based pipeline. Framing this honestly will improve the report, not damage it.

Finally, the project does not yet offer interpretability analysis such as saliency maps, feature attribution, clinician-facing explanations, or error inspection by specific CTG morphology. For an academic dissertation this is acceptable, but for a publishable or clinically deployable system it would be a significant next step.

## Reflection Against Original Goals

The original late-stage goal recorded in the Step 2 plus Step 3 notebooks was to improve step-level three-class performance, especially Mild recall, while preserving grouped cross-validation by record and existing preprocessing logic. Measured against that goal, the project was partially and substantially successful.

It was successful in achieving a rigorous grouped evaluation pipeline, in moving from exploratory whole-record binary learning to clinically aligned step-level multi-class classification, and in producing a final deep model that clearly outperformed tuned classical baselines. It was also successful in showing that Mild recall can be pushed upward through alternative objectives and calibration.

However, the project was only partially successful in improving Mild recall without damaging the rest of the system. The focal-loss tuned model achieved the highest Mild recall, but the overall trade-off was worse because Severe recall and balanced accuracy both declined. Therefore the final project does not claim to have solved the Mild-class problem. Instead, it demonstrates the trade-off explicitly and selects the more balanced model.

This is a strong reflective position for the dissertation: the project met its engineering and evaluation goals, improved substantially on its baselines, but also revealed the practical difficulty of class balance in clinically adjacent CTG categories.

## Future Work Draft

There are several well-justified future directions. The strongest immediate extension is to integrate uterine contraction information as a second aligned signal channel, since repository notes already indicate exploratory work in that direction. A second priority is external validation on a separate dataset or a held-out institutional cohort. Without that, the current system remains an internally validated research prototype rather than a deployment-ready tool.

A third direction is to add interpretability and clinician-facing analysis. Saliency maps, feature attribution for the engineered branch, and case-based error review could reveal whether the model is learning physiologically plausible patterns or relying on artefacts. A fourth direction is to explore more principled uncertainty modelling or cost-sensitive decision thresholds, especially if the clinical aim is screening rather than diagnosis. Finally, the codebase would benefit from further migration from notebook-defined execution into a single script-based experiment package for cleaner reproducibility.

## Suggested Tables and Figures

1. A table summarising dataset sizes before and after filtering, including Step 2 and Step 3 usable records.
2. A diagram of the preprocessing pipeline from raw WFDB record to cleaned 30-minute window.
3. A diagram of the final hybrid CNN architecture with the engineered-feature branch.
4. A table comparing classical baselines and the final deep model on macro F1 and balanced accuracy.
5. A table comparing the main baseline with focal-loss, strong-augmentation, and severe-penalty ablations.
6. One confusion matrix from a representative fold and one aggregate per-class recall table.
7. A short error-analysis table describing common confusion patterns such as Mild vs Severe and Mild vs Normal.

## Results Table You Can Reuse

| Model or Experiment | Evaluation Setting | Macro F1 | Balanced Accuracy | Key Observation |
| --- | --- | ---: | ---: | --- |
| Binary MS-CNN | Earlier binary task | 0.7120 +- 0.1507 | Not reported in saved CSV | Useful exploratory proof of learnable signal |
| Logistic Regression | Grouped multi-class baseline | 0.4988 +- 0.0714 | 0.5478 +- 0.0838 | Strongest classical baseline |
| Random Forest | Grouped multi-class baseline | 0.5115 +- 0.0726 | 0.5093 +- 0.0771 | Better macro F1 than SVM, weaker class balance |
| SVM | Grouped multi-class baseline | 0.4284 +- 0.0485 | 0.4605 +- 0.0511 | High Mild recall but near-collapse on Severe |
| Final softmax grouped baseline | 5-fold x 3-repeat grouped CV | 0.6055 +- 0.0589 | 0.6189 +- 0.0623 | Best overall balance across classes |
| CE + strong aug + calibration | 5-fold x 3-repeat grouped CV | 0.5949 | 0.6127 | Competitive but still below baseline overall |
| Focal + light aug + calibration | 5-fold x 3-repeat grouped CV | 0.5730 | 0.5774 | Mild recall improved strongly, overall trade-off worse |
| CE + severe-only penalty | 5-fold x 3-repeat grouped CV | 0.5806 | 0.6036 | No overall improvement over baseline |

## Short Conclusion Draft

This dissertation presents a complete machine learning pipeline for classifying expert-annotated cardiotocography data and shows that meaningful three-class discrimination is possible using a carefully engineered hybrid deep learning approach. The final model outperforms tuned classical baselines and benefits from clinically aligned windowing, explicit signal cleaning, and grouped leakage-safe cross-validation. At the same time, the work demonstrates that CTG classification remains difficult, particularly under class imbalance and ambiguity in the Mild class. The project therefore succeeds both as a technical implementation and as an honest empirical study: it produces a credible research prototype, identifies its own limitations clearly, and establishes a solid foundation for future multimodal and externally validated work.

## Final Advice for the Actual Report

Do not write the dissertation as a chronological notebook diary. Write it as a problem-solving story.

The best structure is:

1. Why CTG classification matters clinically.
2. Why this dataset and label design are difficult.
3. Why your preprocessing and grouped evaluation choices matter.
4. Why the final hybrid model was chosen.
5. What the comparison experiments show.
6. What the project still cannot yet claim.

If you want to maximise marks, the report should repeatedly show that you understand trade-offs, not just outcomes. Your strongest line is that you made the project more clinically faithful and methodologically defensible as it evolved, even when that made the task harder.