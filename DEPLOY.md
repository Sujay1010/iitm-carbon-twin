# Making the dashboard public

You end up with two things:
1. A **GitHub repository**: the master copy of the code that your institution can take over.
2. A **public website** on Streamlit Community Cloud (free), for example `https://iitm-carbon-twin.streamlit.app`.

The local-server version keeps working exactly as before (`streamlit run app.py`).

---

## Step 1: Put the code on GitHub (about 5 min, no command line needed)

1. Sign in at https://github.com (create a free account if needed).
2. Top right **+** → **New repository**.
   - Name: `iitm-carbon-twin` (or anything)
   - Visibility: **Public** (Streamlit's free tier needs this. Private works if you grant Streamlit access.)
   - Leave "Add a README" **unticked**.
   - Click **Create repository**.
3. On the empty repo page click **uploading an existing file**.
4. Unzip `iitm-carbon-twin.zip` on your computer, open the folder, select **everything inside it**
   (including the `.streamlit` folder; on Mac press Cmd+Shift+. to show hidden folders) and drag it into the page.
5. Click **Commit changes**.

Check that the repo shows `app.py`, `requirements.txt`, `pages/`, `models/`, `data/`, `.streamlit/`.

*Command-line alternative* (the zip already contains a git repo with one commit):
```bash
cd iitm-carbon-twin
git remote add origin https://github.com/<your-username>/iitm-carbon-twin.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Streamlit Community Cloud (about 5 min)

1. Go to https://share.streamlit.io and **Continue with GitHub**.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository: `<your-username>/iitm-carbon-twin`, Branch: `main`, Main file path: `app.py`.
4. Optional: **App URL** → pick a name such as `iitm-carbon-twin`.
5. **Advanced settings** → Python version **3.11** or **3.12**.
6. Click **Deploy**. The first build takes 3–6 minutes (geopandas and xgboost are large).

Share the resulting `https://….streamlit.app` link.

## Updating it later

Edit files on GitHub (pencil icon) or push from your computer. The public site redeploys
automatically within about a minute.

## Handing over to the institution

- On GitHub: **Settings → Collaborators** to add the institution's maintainer, or
  **Settings → Transfer ownership** to move the repo into the institution's GitHub organisation.
- On Streamlit Cloud, the new owner redeploys from the transferred repo (Step 2) under their own account.

## Known limitations of the public version

- **"Save changes" in the Advanced editor doesn't keep edits** on Streamlit Cloud, because its disk
  resets on restart. To change data permanently, edit `data/building_energy.csv` on GitHub instead,
  or use the local server.
- Free apps go to sleep after a few days without visitors. The first visitor clicks "Wake up" and waits about 30 s.
- If the ML page shows an XGBoost loading error, ask the colleague which `xgboost` version trained the
  models and pin it in `requirements.txt` (for example `xgboost==3.0.2`).
