# PHANTOM // JOBINTEL — Project Presentation Outline
## Slide-by-Slide Defense Deck Structure (12 Slides)

---

### Slide 1: Title & Introduction
- **Project Title:** PHANTOM // JOBINTEL: AI-Powered Fake Job Advertisement Detection System
- **Tagline:** Detect Scams. Protect Job Seekers. Build a Safer Web.
- **Presenter:** Final Year B.Tech Computer Science & Engineering
- **Domain:** Applied Machine Learning, Natural Language Processing, Threat Intelligence

---

### Slide 2: Problem Statement & Real-World Impact
- Exponential rise in remote recruitment scams targeting students and professionals.
- Consequences: Financial theft, identity theft, credit fraud, and social security compromise.
- Shortcomings of existing defenses: Keyword search misses disguised postings; lack of remote URL analysis capability.

---

### Slide 3: Proposed Solution — The PHANTOM Architecture
- Dual-Mode Ingestion: Pasted text & Live job URL analysis.
- Multi-Stage Pipeline: Ingestion -> Security Sanitization -> TF-IDF Vectorization -> Machine Learning Inference -> Heuristic Red Flag Scan -> Token Explainability.
- Cybersecurity Intelligence UI: Streamlit with dark telemetry styling.

---

### Slide 4: Dataset Analysis & Provenance
- Kaggle Dataset: `fake_job_postings.csv` (17,880 total records).
- Real Class Breakdown: 17,014 Legitimate (95.16%) vs 866 Fraudulent (4.84%).
- Imbalance Challenge: ~19.65 : 1 ratio handled through stratified splitting and class weighting.
- Key features utilized: Multimodal combining of Title, Company Profile, Description, Requirements, Benefits, and Location.

---

### Slide 5: Natural Language Processing & Preprocessing
- Tokenization & Anonymization: `[URL]`, `[EMAIL]`, `[PHONE]`, `[CURRENCY]` tokens prevent overfitting on specific addresses.
- Feature Space: 10,000 max features, Unigrams and Bigrams.
- TF-IDF Scaling: Sublinear scaling ($1 + \ln(TF)$) and minimum document frequency cutoff of 2.

---

### Slide 6: Model Training & Algorithmic Comparison
- Candidate Classifiers:
  1. Logistic Regression (with Balanced Class Weights)
  2. Multinomial Naive Bayes ($\alpha=0.1$)
  3. Linear Support Vector Machine (Linear SVM with Platt Calibration)
- Strict methodology: 20% stratified test set (3,576 samples), no data leakage, pipeline encapsulation.

---

### Slide 7: Empirical Results & Champion Selection
- **Logistic Regression:** Fraud Recall 91.91%, Fraud Precision 70.98%, F1 80.10%
- **Multinomial Naive Bayes:** Fraud Recall 70.52%, Fraud Precision 84.14%, F1 76.73%
- **Linear SVM (Calibrated) [CHAMPION]:** Fraud Recall 82.66%, Fraud Precision **97.28%**, F1 **89.38%**, Accuracy **99.05%**
- Why Linear SVM Won: Best harmonic F1-score; extremely low false positive rate (only 4 false alarms across 3,403 legitimate posts).

---

### Slide 8: Calibration & Explainability Engine
- Platt Scaling: Fitting a sigmoid over cross-validation folds converts SVM margins into authentic risk probabilities.
- Feature Contribution: Linear weights multiplied by non-zero TF-IDF features explain which words drove the verdict.
- Transparency: Non-technical users see exact influential terms and confidence ratings.

---

### Slide 9: Defensive URL Extraction & SSRF Safeguards
- Threat: Attacker supplies internal IPs or cloud metadata URLs to probe server infrastructure.
- Countermeasures:
  - Scheme restriction (HTTP/HTTPS only).
  - Pre-flight DNS resolution with RFC 1918 private IP and link-local IP blocking.
  - Recursive redirect re-validation (up to 3 hops).
  - Response streaming with 2 MB cap against buffer exhaustion.

---

### Slide 10: Heuristic Red Flag Correlation
- Decoupled rule-based scanner detects overt scam indicators:
  - Free email domains for recruiters (e.g. `@gmail.com`).
  - Off-platform encrypted chats (Telegram, WhatsApp).
  - Advance check payment and equipment scam patterns.
  - Cryptocurrency wallet tasks.

---

### Slide 11: System Demonstration & Testing
- Automated Testing Suite: 26 pytest tests covering data loading, text preprocessing, heuristic rules, SSRF security, URL extraction, and end-to-end inference.
- 100% test pass rate in 2.5 seconds.
- Live demonstration of Pasted Mode and URL Mode.

---

### Slide 12: Conclusion & Future Roadmap
- Contributions: High-precision scam detection system with verified 89.38% Fraud F1 and defensive URL parsing.
- Future Enhancements: Transformer integration (BERT/RoBERTa), recruiter domain WHOIS age verification, and public corporate register API linking.
- Thank you! Questions & Discussion.
