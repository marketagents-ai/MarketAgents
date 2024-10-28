import csv
from fuzzywuzzy import fuzz
import pandas as pd

def extract_occupation_details(employment_csv_path, education_csv_path, threshold=80):
    occupations = []
    
    # Read education data
    education_df = pd.read_csv(education_csv_path)
    education_df.set_index('2023 National Employment Matrix title', inplace=True)
    education_df = education_df.drop('2023 National Employment Matrix code', axis=1)

    with open(employment_csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Fuzzy match OCC_TITLE with education data
            best_match = None
            best_score = 0
            for title in education_df.index:
                score = fuzz.ratio(row['OCC_TITLE'], title)
                if score > best_score:
                    best_score = score
                    best_match = title

            if best_score >= threshold:
                education_data = education_df.loc[best_match].to_dict()
                occupation = {
                    'name': row['OCC_TITLE'],
                    'annual_income': row['A_MEDIAN'],
                    'hourly_wage': row['H_MEDIAN'],
                    'employment': row['TOT_EMP'],
                    'code': row['OCC_CODE'],
                    'percentile_10': row['A_PCT10'],
                    'percentile_25': row['A_PCT25'],
                    'percentile_75': row['A_PCT75'],
                    'percentile_90': row['A_PCT90'],
                    'education': education_data,
                    'matched_title': best_match,
                    'match_score': best_score
                }
                occupations.append(occupation)

    return occupations

def deduplicate_occupations(occupations, threshold=90):
    deduplicated = []
    for occupation in occupations:
        if not any(fuzz.ratio(occupation['name'], existing['name']) >= threshold for existing in deduplicated):
            deduplicated.append(occupation)
    return deduplicated

def format_occupation_details(occupations):
    formatted_output = ""
    for occupation in occupations:
        formatted_output += f"## {occupation['name']}\n\n"
        formatted_output += f"- Annual median income: ${occupation['annual_income']}\n"
        formatted_output += f"- Hourly median wage: ${occupation['hourly_wage']}\n"
        formatted_output += f"- Number employed: {occupation['employment']}\n"
        #formatted_output += f"- Occupation code: {occupation['code']}\n"
        #formatted_output += f"- 10th percentile wage: ${occupation['percentile_10']}\n"
        #formatted_output += f"- 25th percentile wage: ${occupation['percentile_25']}\n"
        #formatted_output += f"- 75th percentile wage: ${occupation['percentile_75']}\n"
        #formatted_output += f"- 90th percentile wage: ${occupation['percentile_90']}\n"
        #formatted_output += f"- Matched education title: {occupation['matched_title']}\n"
        #formatted_output += f"- Match score: {occupation['match_score']}\n"
        formatted_output += "\n"
        formatted_output += "- Education levels:\n"
        for level, percentage in occupation['education'].items():
            formatted_output += f"  - {level}: {percentage}%\n"
        
        formatted_output += "\n"
    
    return formatted_output

# Use local CSV files
employment_csv_path = r'national_M2023_dl.csv'
education_csv_path = r'education_2023.csv'

occupations = extract_occupation_details(employment_csv_path, education_csv_path)
deduplicated_occupations = deduplicate_occupations(occupations)
formatted_output = format_occupation_details(deduplicated_occupations)

# Save as a markdown file
with open('occupations.md', 'w') as file:
    file.write(formatted_output)

print("Occupations data has been saved to occupations.md")