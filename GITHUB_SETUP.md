# Push this project to GitHub

Follow these steps to put the project on GitHub.

---

## 1. Create a new repository on GitHub

1. Go to [github.com](https://github.com) and sign in.
2. Click **New repository** (or **+** → **New repository**).
3. **Repository name:** e.g. `arxiv-cs-classification-lda` or `arxiv-cs-text-classification-lda`.
4. **Description (optional):** e.g. `Binary classification (Computational Linguistics) and LDA topic modelling on arXiv CS abstracts`.
5. Choose **Public**.
6. Do **not** add a README, .gitignore, or license (you already have them).
7. Click **Create repository**.

---

## 2. Push from your computer

Open a terminal in the **project folder** (the one that contains `README.md`, `code_34051201.ipynb`, `app/`, etc.) and run:

```bash
git init
git add .
git status
```

Check that `status` does **not** list:

- `train_set.csv`, `dev_set.csv`, `test_set.csv` (data)
- `venv/` or `env/`
- `models/*.joblib` or large files

If any of those appear, they should be ignored by `.gitignore`. Fix `.gitignore` if needed, then:

```bash
git commit -m "Initial commit: arXiv CS classification and LDA topic modelling"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your GitHub username and the repository name you chose.

---

## 3. Optional: deploy the Streamlit app

- **Streamlit Community Cloud:** Connect your GitHub repo, set the app file to `app/streamlit_app.py`, and deploy.  
- **Hugging Face Spaces:** Create a new Space, choose Streamlit, and point it to this repo and `app/streamlit_app.py`.

---

## 4. Optional: add a second remote (e.g. “portfolio”)

If you want a copy under a different repo name (e.g. for a portfolio):

```bash
git remote add portfolio https://github.com/YOUR_USERNAME/portfolio-repo-name.git
git push -u portfolio main
```

After this, your project is on GitHub and the README will show as the repo description.
