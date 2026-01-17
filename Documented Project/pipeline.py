import download
import filter
import enrich
import fix_title_description
import relevant_urls

def start_pipeline():
    # THE ONLY INPUT NEEDED
    date = input("Enter date to process (YYYYMMDD): ").strip()

    print(f"\n>> Step 1: Downloading...")
    download.main(date)

    print(f"\n>> Step 2: Filtering & Deduping...")
    filter.main(date)

    print(f"\n>> Step 3: Fetching Titles & Meta Tags...")
    enrich.main(date)

    print("\n>> Step 4: Cleaning Titles & Meta Tags...")
    fix_title_description.main(date)

    print("\n>> Step 5: Relevant URLs...")
    relevant_urls.main(date)

    print("\nALL STEPS COMPLETE")

if __name__ == "__main__":
    start_pipeline()