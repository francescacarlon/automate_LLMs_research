import arxiv
import pandas as pd
import random
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()  # read .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- CONFIGURATION ----
category = "cs.AI"
years = range(2015, 2026)
papers_per_year = 5  # 🔼 You can safely increase this now
random.seed(42)  # reproducible random selection
output_file = "arxiv_csAI_2015_2025_with_Abstract_RQ.csv"

# ---- TEST SETTINGS ----
TEST_MODE = False
MAX_RQ_COUNT = 5  # ✅ Only generate RQs for first n. papers
rq_counter = 0    # to keep track of how many RQs were generated

# ---- LOAD EXISTING DATA (if any) ----
if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    existing_titles = set(df_existing["Title"].tolist())
    print(f"📁 Loaded {len(df_existing)} existing papers from {output_file}")
else:
    df_existing = pd.DataFrame()
    existing_titles = set()
    print("🆕 No existing file found — starting fresh.")

data = []

# ---- FUNCTION TO GENERATE RESEARCH QUESTION ----
def generate_research_question(title, abstract):
    prompt = f"""
You are an expert researcher. Given the title and abstract of a paper, generate a concise research question that captures the main problem the paper addresses.

Title: {title}
Abstract: {abstract}

Research Question:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        )
        rq = response.choices[0].message.content.strip()
        return rq
    except Exception as e:
        print(f"Error generating RQ for '{title}': {e}")
        return ""

# ---- FETCH PAPERS ----
for year in years:
    print(f"\n📅 Fetching papers for {year}...")
    query = f"cat:{category} AND submittedDate:[{year}01010000 TO {year}12312359]"

    try:
        search = arxiv.Search(
            query=query,
            max_results=50,  # fetch some papers to choose from
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = list(search.results())
    except arxiv.UnexpectedEmptyPageError:
        print(f"⚠️ Empty page for {year}, skipping")
        papers = []

    if not papers:
        print(f"⚠️ No papers found for {year}")
        continue

    # Randomly select n papers per year
    selected = random.sample(papers, min(papers_per_year, len(papers)))

    for result in selected:
        title = result.title.strip()

        # Skip if already fetched before
        if title in existing_titles:
            print(f"⏩ Skipping already saved paper: {title}")
            continue

        primary_cat = result.primary_category or ""
        main_field, _, sub_field = primary_cat.partition('.')
        abstract = result.summary.replace("\n", " ").strip()

        # ---- Conditionally generate RQs ----
        if (not TEST_MODE) or (rq_counter < MAX_RQ_COUNT):
            print(f"\n🔹 Generating RQ for paper {rq_counter + 1}: {title}")
            rq = generate_research_question(title, abstract)
            rq_counter += 1
            time.sleep(1)
        else:
            rq = ""

        data.append({
            "Title": title,
            "Abstract": abstract,
            "MainCategory": main_field,
            "SubCategory": sub_field,
            "Year": result.published.year,
            "ResearchQuestion": rq
        })

# ---- CREATE DATAFRAME ----
df_new = pd.DataFrame(data)
if not df_new.empty:
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_combined = df_existing.copy()

df_combined.drop_duplicates(subset="Title", inplace=True)

# ---- REASSIGN OR ADD RefNumber ----
if "RefNumber" in df_combined.columns:
    df_combined["RefNumber"] = range(1, len(df_combined) + 1)
else:
    df_combined.insert(0, "RefNumber", range(1, len(df_combined) + 1))


# ---- SAVE CSV ----
df_combined.to_csv(output_file, index=False)
print(f"\n✅ Total papers in dataset: {len(df_combined)}")
print(f"📄 Saved to {output_file}")
print(df_combined.tail(5))
