import arxiv
import pandas as pd
import random
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()  # read .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# print(os.getenv("OPENAI_API_KEY"))

# ---- CONFIGURATION ----
category = "cs.AI"
years = range(2015, 2026)
papers_per_year = 2  # -> 22 total (2015–2025 inclusive)
random.seed(42)  # reproducible random selection
output_file = "arxiv_csAI_2015_2025_with_Abstract_RQ.csv"

data = []

# ---- TEST SETTINGS ----
TEST_MODE = False
MAX_RQ_COUNT = 5  # ✅ Only generate RQs for first n. papers
rq_counter = 0    # to keep track of how many RQs were generated

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
    print(f"Fetching papers for {year}...")
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

    # Randomly select 2 papers per year
    selected = random.sample(papers, min(papers_per_year, len(papers)))

    for result in selected:
        primary_cat = result.primary_category or ""
        main_field, _, sub_field = primary_cat.partition('.')
        title = result.title.strip()
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
df = pd.DataFrame(data)
df.insert(0, "RefNumber", range(1, len(df) + 1))

# ---- SAVE CSV ----
df.to_csv(output_file, index=False)
print(f"\n✅ Collected {len(df)} papers.")
print(f"📄 Saved to {output_file}")
print(df.head(5))
