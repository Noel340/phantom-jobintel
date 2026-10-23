# PHANTOM // JOBINTEL — Viva Voce Examination Questions & Answers
## Comprehensive Technical Q&A for B.Tech Computer Science and Engineering Defense

---

### Q1: What is the primary objective of PHANTOM // JOBINTEL?
**Answer:** The primary objective is to build an automated, machine-learning-driven threat intelligence system that classifies job recruitment postings as likely legitimate or potentially fraudulent, protecting job seekers from identity theft, financial fraud, and credential harvesting. It supports both manual text analysis and defensive live job URL extraction.

---

### Q2: What dataset did you use, and what are its key statistics?
**Answer:** We used the Kaggle Real or Fake Job Postings dataset (`fake_job_postings.csv`). It contains **17,880 total job postings** across 18 features (title, description, requirements, benefits, company profile, location, salary, etc.). The class distribution has **17,014 legitimate postings (95.16%)** and **866 fraudulent postings (4.84%)**, representing a severe class imbalance ratio of approximately **19.65 : 1**.

---

### Q3: Why is standard classification accuracy a deceptive metric in this project?
**Answer:** Because 95.16% of the postings are legitimate, a trivial baseline model that blindly predicts "Legitimate" for every input achieves a 95.16% overall accuracy while missing **100% of scams**. In cybersecurity, false negatives (failing to catch a scam ad) can lead to catastrophic financial or personal injury, so **Fraud Recall**, **Fraud Precision**, and **Fraud F1-Score** are far more critical metrics than raw accuracy.

---

### Q4: How did you handle the severe class imbalance during training?
**Answer:** 
1. Used **stratified train-test splitting (80/20)** to preserve the identical 95:5 class ratio in both training and test sets.
2. Applied **balanced class weighting** (`class_weight='balanced'`) in Logistic Regression and Linear Support Vector Classifier, which penalizes errors on the minority fraudulent class inversely proportional to their class frequency: $w_1 = \frac{N}{2 \times N_1}$.
3. Tuned the decision thresholds and prioritized the **Fraud F1-score** and **PR-AUC** during model selection.

---

### Q5: What three algorithms did you train and compare? What were the empirical results?
**Answer:** 
1. **Logistic Regression (Balanced Weights):** Fraud Recall: 91.91%, Fraud Precision: 70.98%, Fraud F1: 80.10%, Accuracy: 97.43%.
2. **Multinomial Naive Bayes ($\alpha=0.1$):** Fraud Recall: 70.52%, Fraud Precision: 84.14%, Fraud F1: 76.73%, Accuracy: 98.07%.
3. **Linear Support Vector Machine (Calibrated):** Fraud Recall: 82.66%, Fraud Precision: 97.28%, Fraud F1: 89.38%, Accuracy: 99.05%.

---

### Q6: Why did you choose Linear SVM (Calibrated) as the champion production model?
**Answer:** Linear SVM achieved the highest Fraud F1-score (**89.38%**) and an outstanding Fraud Precision of **97.28%** with only 4 false positives across 3,403 legitimate test posts. Furthermore, by wrapping it in `CalibratedClassifierCV(cv=3, method='sigmoid')`, we transformed its uncalibrated margin distances into mathematically valid posterior probabilities via Platt scaling without leaking test data.

---

### Q7: What is Platt Scaling, and why was it necessary for the SVM?
**Answer:** Linear SVM calculates a signed geometric distance $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$ from the separating hyperplane, which is not a probability. Platt scaling fits a logistic sigmoid function $P(y=1|f) = \frac{1}{1 + \exp(Af + B)}$ over decision scores using out-of-fold validation data to produce true calibrated probabilities between 0.0 and 1.0.

---

### Q8: What feature extraction technique was applied to the job posting text?
**Answer:** We used **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorization with:
- Maximum vocabulary size: 10,000 features.
- N-gram range: (1, 2) (both single words and two-word phrases like "wire transfer", "data entry").
- Sublinear term-frequency scaling ($1 + \ln(TF)$) to dampen the effect of heavily repeated tokens.
- Minimum document frequency cutoff of 2 to eliminate single-occurrence spelling noise.

---

### Q9: Why did you wrap the vectorizer and classifier in a scikit-learn Pipeline?
**Answer:** A `Pipeline` couples the feature vectorizer and classifier into a single serialized object. This prevents data leakage (fitting vocabulary strictly on training data) and guarantees that any new input during inference undergoes identical tokenization, transformation, and classification.

---

### Q10: How does your system explain its predictions to non-technical users?
**Answer:** For linear models, the prediction score is a linear combination of feature weights and TF-IDF values: $\sum (x_i \cdot w_i)$. In `src/explainability.py`, we multiply the non-zero TF-IDF features of the submitted ad by the model's learned coefficients to highlight the specific tokens that pushed the prediction toward fraud (e.g., "wire", "urgently", "payment") or toward legitimacy (e.g., "benefits", "health insurance", "requirements").

---

### Q11: What security threats arise when analyzing user-submitted URLs, and how did you mitigate them?
**Answer:** Allowing a server to fetch arbitrary user-supplied URLs exposes the application to **Server-Side Request Forgery (SSRF)**, where attackers force the server to access internal networks, localhost ports, or cloud metadata endpoints (`169.254.169.254`).
We mitigated this in `src/security.py` by:
- Enforcing an `http`/`https` scheme whitelist.
- Resolving DNS *before* making the HTTP request and verifying that resolved IPs do not belong to private (RFC 1918: 10.x, 172.16-31.x, 192.168.x), loopback (`127.0.0.1`), or link-local subnets.
- Validating redirects recursively in a loop.
- Limiting response sizes to 2 MB using chunked streaming to prevent DoS attacks.

---

### Q12: How does the webpage extraction handle different job website formats?
**Answer:** In `src/url_analyzer.py`, we employ a hierarchical fallback strategy:
1. **Structured Data:** Inspects `<script type="application/ld+json">` for schema.org `JobPosting` objects.
2. **Semantic DOM Fallback:** If JSON-LD is absent, parses HTML containers using CSS selectors and regex heuristics matching `job-description`, `article`, `main`, Open Graph tags, and heading tags.
3. Sanitizes extracted strings and formats them into the exact schema expected by the ML model.

---

### Q13: What heuristic red flag indicators does PHANTOM detect independently of the ML model?
**Answer:** In `src/predict.py`, an independent regex engine flags known scam patterns:
1. Recruiter using free email domains (`@gmail.com`, `@yahoo.com`, `@hotmail.com`) rather than a registered corporate domain.
2. Demanding interviews over off-platform encrypted apps (Telegram, WhatsApp).
3. Upfront check cashing or equipment purchase requirements.
4. Cryptocurrency / crypto wallet transfer duties.
5. Guaranteed instant employment with zero interview screening.

---

### Q14: How does PHANTOM ensure user privacy and safe deployment?
**Answer:** The system does not log or persist user PII or resumes to disk. All predictions are computed in-memory, user inputs are sanitized against control character injection, and no arbitrary shell or code execution is permitted on user-supplied strings.

---

### Q15: What are the main limitations of the current system, and how can they be improved in the future?
**Answer:**
1. **Sophisticated Copy-Paste Scams:** If a scammer copies legitimate job ad text word-for-word, the text model alone cannot detect the deception; this is why we include independent employer verification checklists.
2. **JavaScript-Heavy Portals:** Some single-page applications (SPAs) require headless browser rendering (e.g. Playwright) to extract DOM elements.
3. **Future Improvements:** Integrating transformer-based language models (BERT/RoBERTa), automated corporate domain WHOIS age lookups (scammers often use domains registered < 30 days ago), and linking official company registries (SEC EDGAR, Companies House).
