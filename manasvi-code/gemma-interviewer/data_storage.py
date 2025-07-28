import csv
import os
from data_models import InterviewDataRow

CSV_FILE = 'feedback_reports.csv'

def save_report_to_csv(pydantic_rows: list[InterviewDataRow]):
    if not pydantic_rows:
        print("No validated data rows to save.")
        return

    headers = list(InterviewDataRow.model_fields.keys())
    file_exists = os.path.isfile(CSV_FILE)
    
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        
        for row_model in pydantic_rows:
            writer.writerow(row_model.model_dump(mode='json'))
            
    print(f"Report saved to {CSV_FILE} with Interview ID: {pydantic_rows[0].interview_id}")