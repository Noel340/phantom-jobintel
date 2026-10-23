# B.TECH MINI-PROJECT REPORT

## PHANTOM // JOBINTEL: AI-POWERED FAKE JOB ADVERTISEMENT DETECTION SYSTEM USING NATURAL LANGUAGE PROCESSING AND DEFENSIVE URL ANALYSIS

**Department of Computer Science and Engineering**  
**Bachelor of Technology in Computer Science & Engineering**  
**Academic Year: 2026**

---

### CERTIFICATE

This is to certify that the mini-project entitled **"PHANTOM // JOBINTEL: AI-Powered Fake Job Advertisement Detection System"** is a bona fide work carried out by the student in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science and Engineering**.

---

### ACKNOWLEDGMENTS

The author expresses sincere gratitude to the Department of Computer Science and Engineering, faculty advisors, project guides, and the open-source machine learning and cybersecurity communities for their invaluable guidance, tools, and datasets that made this research possible.

---

### ABSTRACT

The modern recruitment landscape has migrated almost entirely to digital job boards, remote platforms, and professional social networks. While this shift has democratized career opportunities worldwide, it has concurrently provided malicious threat actors with an unprecedented vector for cyber fraud. Fraudulent recruitment listings—weaponized for identity theft, social security number harvesting, credit card fraud, and advance-fee money laundering—inflict severe psychological and financial harm upon vulnerable job seekers.

This project introduces **PHANTOM // JOBINTEL**, a full-stack, enterprise-grade machine learning system designed to autonomously classify online job advertisements as either **Likely Legitimate** or **Potentially Fraudulent**. Trained on a verified corpus of 17,880 recruitment postings from the Kaggle dataset, the system implements a dual-mode ingestion pipeline: direct user text submission and a remote public URL analyzer safeguarded by Server-Side Request Forgery (SSRF) defense mechanisms. 

A rigorous comparative evaluation was conducted across three distinct machine learning algorithms: **Logistic Regression with class balancing**, **Multinomial Naive Bayes**, and **Linear Support Vector Machines (Linear SVM) calibrated via Platt sigmoidal scaling**. On an isolated 20% stratified test set of 3,576 postings, the Calibrated Linear SVM achieved a champion **Fraud F1-Score of 89.38%**, an overall accuracy of **99.05%**, and a Fraud Precision of **97.28%**, dramatically outperforming naive baselines. The backend is integrated with a cybersecurity-inspired dark theme user interface developed in Streamlit, delivering probability calibration, token-level explainability, heuristic indicator red flags, and actionable employer verification guidance.

---

### TABLE OF CONTENTS

1. **Chapter 1: Introduction**
   - 1.1 Motivation & Background
   - 1.2 Problem Statement
   - 1.3 Objectives & Scope
   - 1.4 Report Organization
2. **Chapter 2: Literature Review**
   - 2.1 Traditional Fraud Detection Paradigms
   - 2.2 Natural Language Processing in Cyber Threat Detection
   - 2.3 Limitations of Existing Work
3. **Chapter 3: System Methodology & Architecture**
   - 3.1 Dataset Description & Characteristics
   - 3.2 Data Preprocessing & Leakage Prevention
   - 3.3 Feature Engineering: TF-IDF Vectorization
   - 3.4 Machine Learning Classifiers & Formulations
   - 3.5 Class Imbalance Treatment
4. **Chapter 4: Defensive URL Extraction & Threat Modeling**
   - 4.1 Threat Modeling for Web Ingestion
   - 4.2 Server-Side Request Forgery (SSRF) Mitigation
   - 4.3 Structured Schema.org & DOM Content Extraction
5. **Chapter 5: Experimental Results & Comparative Analysis**
   - 5.1 Evaluation Metrics
   - 5.2 Performance Comparison Across Models
   - 5.3 Confusion Matrix Analysis
   - 5.4 Feature Explainability & Token Impact
6. **Chapter 6: User Interface & Software Implementation**
   - 6.1 Architectural Overview
   - 6.2 Streamlit Cybersecurity Dashboard
   - 6.3 Automated Unit & Integration Testing
7. **Chapter 7: Conclusion & Future Enhancements**
   - 7.1 Summary of Contributions
   - 7.2 Practical Limitations
   - 7.3 Future Research Directions
8. **References**

---

### CHAPTER 1: INTRODUCTION

#### 1.1 Motivation & Background
Over the past decade, cloud computing, applicant tracking systems (ATS), and remote work culture have redefined hiring. Concurrently, cybercriminals exploit employment anxiety by masquerading as reputable organizations or launching phantom startups. According to recent federal trade and cyber-threat reports, recruitment scams account for millions of dollars in annual losses and leave job seekers exposed to identity compromise.

#### 1.2 Problem Statement
Traditional rule-based keyword scrapers and uncalibrated models struggle to differentiate genuine recruitment advertisements from deceptive postings. Scammers frequently mimic corporate vocabulary while embedding deceptive compensation, off-platform communication demands (e.g. Telegram), or fraudulent check advance directives. A robust system must model both the nuanced linguistic distributions of recruitment text and enforce defensive networking controls when scraping live URLs.

#### 1.3 Objectives
- Load, inspect, and clean the Kaggle Real or Fake Job Postings dataset (17,880 records).
- Develop an end-to-end scikit-learn Pipeline incorporating custom text normalization and TF-IDF feature extraction.
- Train and compare Logistic Regression, Multinomial Naive Bayes, and Linear SVM classifiers using stratified cross-validation and imbalance handling.
- Calibrate model probabilities using Platt scaling to provide reliable certainty metrics.
- Implement a defensive URL analyzer equipped with DNS validation, private IP blocking, and streaming limits.
- Deliver an intuitive, cyber-themed Streamlit web interface with real-time token explainability and heuristic red flags.
- Establish an automated unit testing suite with 100% test pass rates.

---

### CHAPTER 2: LITERATURE REVIEW

Prior works in scam detection primarily focused on phishing email classification and credit card transaction anomalies. In the realm of recruitment fraud, initial approaches relied on rudimentary keyword matching (e.g., checking for words like "wire", "Western Union", or "cashier's check"). However, modern adversarial postings rarely use obvious spam strings, often copying job descriptions directly from legitimate job boards.

Recent machine learning literature underscores the efficacy of Term Frequency-Inverse Document Frequency (TF-IDF) combined with linear maximum-margin classifiers. Due to the high dimensionality and sparse nature of text matrices, linear models (Linear SVM and L2-regularized Logistic Regression) often rival or surpass deep learning models on small to medium corpora while maintaining superior interpretability and millisecond inference latency.

---

### CHAPTER 3: SYSTEM METHODOLOGY & ARCHITECTURE

#### 3.1 Dataset Description
The dataset comprises 17,880 total job postings with 18 distinct columns:
- **Identifier & Target:** `job_id`, `fraudulent` (0: Legitimate, 1: Fraudulent)
- **Textual Fields:** `title`, `company_profile`, `description`, `requirements`, `benefits`
- **Categorical Metadata:** `location`, `department`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`
- **Binary Flags:** `telecommuting`, `has_company_logo`, `has_questions`

**Class Distribution:**
- Legitimate Postings: 17,014 (95.16%)
- Fraudulent Postings: 866 (4.84%)
- Imbalance Ratio: ~19.65 : 1

#### 3.2 Data Preprocessing
To prevent leakage and ensure inference parity, raw fields are unified into a structured text document. Text normalization executes:
1. HTML entity unescaping and tag stripping.
2. Tokenization of URLs (`URL_TOKEN`), email addresses (`EMAIL_TOKEN`), phone numbers (`PHONE_TOKEN`), and monetary figures (`CURRENCY_TOKEN`).
3. Lowercasing and whitespace normalization.

#### 3.3 TF-IDF Vectorization
The processed text is mapped into numerical vector space using TF-IDF with 10,000 max features, n-gram range of (1, 2) (unigrams and bigrams), sublinear term frequency scaling ($1 + \ln(TF)$), and a minimum document frequency cutoff of 2.

#### 3.4 Model Formulations
- **Logistic Regression:** Loss function optimized with balanced class weighting:
  $$w_c = \frac{N}{2 \times N_c}$$
- **Multinomial Naive Bayes:** Laplace smoothed conditional probability distribution:
  $$P(w|c) = \frac{N_{wc} + \alpha}{N_c + \alpha |V|}$$
- **Linear Support Vector Classifier:** Primal soft-margin optimization with Platt sigmoid calibration across 3 stratified cross-validation folds:
  $$P(y=1|f) = \frac{1}{1 + \exp(Af + B)}$$

---

### CHAPTER 4: DEFENSIVE URL EXTRACTION & THREAT MODELING

Direct server-side URL fetching introduces critical vulnerabilities:
1. **Server-Side Request Forgery (SSRF):** An attacker supplies `http://localhost:8000` or `http://169.254.169.254/latest/meta-data/` to probe internal network infrastructure or steal cloud credentials.
2. **Denial of Service (DoS):** Slowloris connections or massive files leading to memory exhaustion.
3. **Open Redirect Exploits:** A safe domain redirecting to an internal IP address.

**Defenses Implemented in `src/security.py`:**
- **Scheme Validation:** Rejection of `file://`, `ftp://`, `gopher://`.
- **Pre-Request DNS Resolution:** Resolves target domain name and verifies that all resolved IP addresses do NOT belong to RFC 1918 private subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopback (`127.0.0.0/8`), or link-local (`169.254.0.0/16`).
- **Stream Limit:** Caps responses at 2 MB.
- **Recursive Redirect Inspection:** Validates destination at each redirect hop (maximum 3 hops).

---

### CHAPTER 5: EXPERIMENTAL RESULTS & COMPARATIVE ANALYSIS

#### 5.1 Empirical Metric Summary (3,576 Test Samples)

| Metric | Logistic Regression | Multinomial Naive Bayes | Linear SVM (Calibrated) |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | 97.43% | 98.07% | **99.05%** |
| **Fraud Precision** | 70.98% | 84.14% | **97.28%** |
| **Fraud Recall** | **91.91%** | 70.52% | **82.66%** |
| **Fraud F1-Score** | 80.10% | 76.73% | **89.38%** |
| **Macro F1-Score** | 89.28% | 87.79% | **94.44%** |
| **PR-AUC** | 0.8651 | 0.8123 | **0.9324** |

#### 5.2 Confusion Matrix (Linear SVM Champion)
- True Negatives (Legitimate correctly identified): **3,399**
- False Positives (Legitimate falsely flagged as fraud): **4** (Extremely low false alarm rate)
- False Negatives (Fraudulent posts missed): **30**
- True Positives (Fraudulent posts caught): **143**

---

### CHAPTER 6: USER INTERFACE & SOFTWARE IMPLEMENTATION

The frontend was built using Streamlit, customized with dark cybersecurity aesthetics. The navigation radar provides five distinct views:
1. **Command Center:** Telemetry cards, product overview, and session scan activity.
2. **Job Analyzer:** Dual input tabs for pasted text and live URL extraction with instant risk status and token-level explainability.
3. **Model Intelligence:** Tabular benchmark, confusion matrices, and precision/recall trade-off documentation.
4. **Dataset Intelligence:** Interactive distribution charts, missingness analysis, and word count distributions.
5. **Methodology & Viva:** Academic formulations, security threat models, and 15 interactive examination questions.

---

### CHAPTER 7: CONCLUSION & FUTURE ENHANCEMENTS

**PHANTOM // JOBINTEL** delivers a robust, transparent, and defensible solution to fake job detection. By coupling modern NLP classification with an SSRF-hardened web parser, the system safeguards job seekers across both manual and automated usage vectors.

**Future Scope:**
- Integration of Transformer models (BERT, RoBERTa) for contextual semantic encoding.
- Real-time WHOIS domain age inspection to detect newly registered scam domains.
- Integration with external corporate registries (SEC EDGAR, UK Companies House).

---

### REFERENCES

1. Vidros, S., Kolias, C., Kambourakis, G., & Akoglu, L. (2017). "Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset." *Future Internet*, 9(1), 6.
2. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825-2830.
3. Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." *Advances in Large Margin Classifiers*, 10(3), 61-74.
