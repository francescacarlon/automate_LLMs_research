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
papers_per_year = 6  # 🔼 You can safely increase this now
random.seed(42)  # reproducible random selection
output_file = "arxiv_csAI_2015_2025_with_Abstract_RQ.csv"

# ---- TEST SETTINGS ----
TEST_MODE = False
MAX_RQ_COUNT = 5  # ✅ Only generate RQs for first n. papers
rq_counter = 0    # to keep track of how many RQs were generated

# ---- LOAD EXISTING DATA (if any) ----
if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    df_existing.columns = df_existing.columns.str.strip()  # remove accidental spaces
    if "Title" not in df_existing.columns:
        print("⚠️ Existing file does not contain a 'Title' column — starting fresh.")
        df_existing = pd.DataFrame(columns=["Title", "Abstract", "MainCategory", "SubCategory", "Year", "ResearchQuestion"])
    existing_titles = set(df_existing["Title"].astype(str).str.strip().str.replace("\n", " ", regex=False).tolist())
    print(f"📁 Loaded {len(df_existing)} existing papers from {output_file}")
else:
    df_existing = pd.DataFrame(columns=["Title", "Abstract", "MainCategory", "SubCategory", "Year", "ResearchQuestion"])
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

    # --- normalize titles in existing data
    df_existing["Title"] = df_existing["Title"].str.strip().str.replace("\n", " ", regex=False)

    # count how many papers already exist for this year
    existing_year_papers = df_existing[df_existing["Year"] == year]
    existing_count = len(existing_year_papers)

    if existing_count >= papers_per_year:
        print(f"✅ Already have {existing_count} papers for {year}, skipping.")
        continue

    needed = papers_per_year - existing_count
    print(f"📊 Need {needed} more papers for {year}.")

    query = f"cat:{category} AND submittedDate:[{year}01010000 TO {year}12312359]"

    try:
        search = arxiv.Search(
            query=query,
            max_results=100,  # fetch extra so we can pick cleanly
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = list(search.results())
    except arxiv.UnexpectedEmptyPageError:
        print(f"⚠️ Empty page for {year}, skipping.")
        continue

    if not papers:
        print(f"⚠️ No papers found for {year}")
        continue

    # clean all fetched titles before comparing
    fetched = [
        p for p in papers
        if p.title.strip().replace("\n", " ") not in df_existing["Title"].values
    ]

    # random sample exactly what we need
    selected = random.sample(fetched, min(needed, len(fetched)))

    for result in selected:
        title = result.title.strip().replace("\n", " ")

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
df_combined = pd.concat([df_existing, df_new], ignore_index=True)

# remove duplicates based on Title
df_combined.drop_duplicates(subset="Title", inplace=True)

# ---- keep only 5 papers per year ----
df_final = (
    df_combined
    .sort_values("Year")
    .groupby("Year", group_keys=False)
    .head(papers_per_year)
)

# ---- REASSIGN RefNumber ----
df_final["RefNumber"] = range(1, len(df_final) + 1)

# ---- ENFORCE COLUMN ORDER ----
desired_order = [
    "RefNumber",
    "Title",
    "Abstract",
    "MainCategory",
    "SubCategory",
    "Year",
    "ResearchQuestion"
]
df_final = df_final[desired_order]

# ---- SAVE CSV ----
df_final.to_csv(output_file, index=False)
print(f"\n✅ Final total: {len(df_final)} papers (5 per year)")
print(f"📄 Saved to {output_file}")
print(df_final.head(5))
