import arxiv
import pandas as pd
import random
import openai
import time

# ---- CONFIGURATION ----
category = "cs.AI"
years = range(2015, 2026)
papers_per_year = 2
random.seed(42)  # reproducible random selection
output_file = "arxiv_csAI_2015_2025_with_RQ.csv"
openai.api_key = "YOUR_OPENAI_API_KEY"

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
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # factual
            max_tokens=100
        )
        rq = response['choices'][0]['message']['content'].strip()
        return rq
    except Exception as e:
        print(f"Error generating RQ for {title}: {e}")
        return ""

# ---- FETCH PAPERS ----
for year in years:
    print(f"Fetching papers for {year}...")
    query = f"cat:{category} AND submittedDate:[{year}01010000 TO {year}12312359]"

    try:
        search = arxiv.Search(
            query=query,
            max_results=50,  # avoid empty pages
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = list(search.results())
    except arxiv.UnexpectedEmptyPageError:
        print(f"⚠️ Empty page for {year}, skipping")
        papers = []

    if not papers:
        print(f"⚠️ No papers found for {year}")
        continue

    # Randomly select papers_per_year papers
    selected = random.sample(papers, min(papers_per_year, len(papers)))

    for result in selected:
        primary_cat = result.primary_category or ""  # always cs.AI
        main_field, _, sub_field = primary_cat.partition('.')
        title = result.title.strip()
        abstract = result.summary.replace("\n", " ").strip()

        # Generate research question
        rq = generate_research_question(title, abstract)
        time.sleep(1)  # avoid rate limit

        data.append({
            "Title": title,
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
print(f"\n✅ Collected {len(df)} papers with Research Questions. Saved to {output_file}")
print(df)
