# UNDRR Groundsource Starter

A very simple first version of a Groundsource-style extraction pipeline:

1. read documents from Databricks
2. send them to OpenAI
3. extract structured disaster-risk facts
4. save results to JSONL and CSV

This version is intentionally simple and beginner-friendly.

---

## 1. What you need

You need:

- an OpenAI API key
- a Databricks personal access token
- your Databricks **Server Hostname**
- your Databricks **HTTP Path** for a SQL warehouse or all-purpose compute
- Python 3.10 or 3.11

---

## 2. Create a project folder

Make a folder anywhere on your computer, for example:

```bash
mkdir undrr_groundsource_starter
cd undrr_groundsource_starter
```

Put these files inside it:

- `app.py`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `Dockerfile`

---

## 3. Create your `.env` file

Copy the example:

```bash
cp .env.example .env
```

Open `.env` and fill in your real keys.

Example:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4-mini
DATABRICKS_SERVER_HOSTNAME=adb-xxxxxxxxxxxx.x.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxxxxxxxxxxxxxx
DATABRICKS_TOKEN=dapi...
DATABRICKS_SOURCE_ASSET_TABLE=gold.asset_text
DATABRICKS_SOURCE_CHUNK_TABLE=gold.asset_text_chunks
LOG_LEVEL=INFO
```

Do not share `.env` and do not upload it to GitHub.

---

## 4. Run it locally without Docker

Create a Python virtual environment.

### Mac / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py --limit 3
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py --limit 3
```

If it works, you should get an `outputs/` folder containing:

- `document_extractions.jsonl`
- `extracted_records.csv`

---

## 5. What the script is doing

### Step 1
It connects to Databricks using your hostname, HTTP path, and token.

### Step 2
It reads rows from `gold.asset_text` and joins metadata from `gold.asset_text_chunks`.

### Step 3
For each document, it sends metadata + text to OpenAI.

### Step 4
OpenAI returns structured JSON with:

- a document summary
- a list of extracted records

### Step 5
The script saves the results locally.

---

## 6. Where to edit the prompt

Open `prompts.py` and look for:

```python
PROMPTS = (...)
```

That is the numbered prompt catalog sent to OpenAI. Prompt 01 is the baseline.

If you want a stricter schema, edit these classes:

- `EventExtraction`

---

## 7. First GitHub workflow

If this is your first time using GitHub, do this.

### Step A: initialize git

Inside your project folder:

```bash
git init
```

### Step B: check what files are there

```bash
git status
```

### Step C: add the files

```bash
git add app.py requirements.txt .env.example .gitignore Dockerfile README.md
```

### Step D: commit them

```bash
git commit -m "Initial simple UNDRR extractor"
```

### Step E: create an empty GitHub repository

Go to GitHub in your browser and create a new empty repo.
Do **not** add a README there because you already have one.

### Step F: connect local repo to GitHub

GitHub will show commands similar to:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

That uploads your project to GitHub.

Important:
- never commit `.env`
- if you ever accidentally commit a token, rotate it immediately

---

## 8. First Docker workflow

Docker lets you run the project inside a container.
That means the code runs in a clean, reproducible environment.

### Step A: install Docker Desktop

Install Docker Desktop and open it.
Make sure it is running.

### Step B: build the image

Inside the project folder:

```bash
docker build -t undrr-groundsource-starter .
```

### Step C: run the container

```bash
docker run --rm --env-file .env -v "$(pwd)/outputs:/app/outputs" undrr-groundsource-starter python app.py --limit 3
```

On Windows PowerShell, use:

```powershell
docker run --rm --env-file .env -v "${PWD}/outputs:/app/outputs" undrr-groundsource-starter python app.py --limit 3
```

This does three things:

- `--env-file .env` gives the container your keys and settings
- `-v ...` saves the output files to your computer
- `python app.py --limit 3` runs the script

---

## 9. Common beginner mistakes

### Mistake 1: missing `.env`
If the script says a variable is missing, your `.env` file is incomplete.

### Mistake 2: wrong Databricks HTTP path
You need the HTTP path of a SQL warehouse or supported all-purpose compute.

### Mistake 3: no table permissions
Your Databricks token must be allowed to read `gold.asset_text` and `gold.asset_text_chunks`.

### Mistake 4: sending too much text
If the prompt becomes too large or expensive, lower `--max-chars`.

### Mistake 5: committed secrets to GitHub
If this happens, rotate the key/token immediately and remove it from the repo history later.

---

## 10. What to improve next

Once this first version works, good next upgrades are:

1. chunking long documents instead of simple truncation
2. saving results back into a Databricks Delta table
3. adding retries and rate-limit handling
4. using a narrower schema for specific event extraction
5. using Batch API for large backfills
