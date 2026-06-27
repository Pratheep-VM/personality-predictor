# Personality Predictor API (Production GitOps Deployment)

A production-grade, containerized classification engine that predicts whether a person is an **Introvert** or an **Extrovert** based on behavioral metrics. 

This repository implements a complete end-to-end machine learning and MLOps system—featuring stratified cross-validation, hyperparameter tuning, automated CI/CD via GitHub Actions with AWS OIDC, container orchestration using a lightweight Kubernetes cluster (`k3s`), and automated GitOps deployment via `ArgoCD`.

---

## 🚀 Live Endpoints & Resources

*   **Interactive Developer Portal (Scalar):** [http://54.166.202.139:30080/docs](http://54.166.202.139:30080/docs)
*   **Production API Prediction Endpoint:** `POST http://54.166.202.139:30080/predict`
*   **Training Notebook:** `notebook/training.ipynb`

---

## 🛠️ System Architecture

```text
                           ┌────────────────────────┐
                           │   Local Developer      │
                           └───────────┬────────────┘
                                       │ git push
                                       ▼
                           ┌────────────────────────┐
                           │    GitHub Repository   │
                           └───────────┬────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼ (CI Pipeline)                               ▼ (ArgoCD Sync)
  ┌────────────────────────┐                      ┌────────────────────────┐
  │  GitHub Actions Runner │                      │  ArgoCD Controller     │
  └─────────────┬──────────┘                      └───────────┬────────────┘
                │ AWS OIDC (Keyless)                          │ GitOps pull
                ▼                                             ▼
  ┌────────────────────────┐                      ┌────────────────────────┐
  │  Amazon ECR Registry   │                      │  Kubernetes (k3s)      │
  └─────────────┬──────────┘                      │  Cluster (AWS EC2)     │
                │                                 └───────────┬────────────┘
                │  imagePullSecrets                           │
                └─────────────────────────────────────────────┘