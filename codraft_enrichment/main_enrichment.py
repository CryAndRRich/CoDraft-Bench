import os
import json
import time
import pandas as pd
from tqdm import tqdm

import openai
import instructor

from config.config_data import CONFIG_DATA
from .prompt import ENRICHMENT_PROMPTS
from .schemas import SCHEMA_MAP

def get_client(api_key):
    client = instructor.from_openai(
        openai.OpenAI(
            base_url=CONFIG_DATA.CODRAFT_CONFIG['base_url'],
            api_key=api_key
        ),
        mode=instructor.Mode.JSON,
    )
    return client

def process_batch(batch_data, client, version="v1"):
    try:
        batch_json = json.dumps(batch_data, indent=2)
        
        sys_message = ENRICHMENT_PROMPTS[version]["system"]
        user_message = ENRICHMENT_PROMPTS[version]["user"].format(batch_json=batch_json)
        ResponseSchema = SCHEMA_MAP[version] 

        return client.chat.completions.create(
            model="gemini-2.5-flash",
            response_model=ResponseSchema,
            messages=[
                {"role": "system", "content": sys_message},
                {"role": "user", "content": user_message}
            ],
            max_retries=3,
        )
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None

def run_enrichment(df, client, output_file_dir=None, version="v1"):
    print(f"RUN ENRICHMENT (Version: {version}) ---")
    results = []
    BATCH_SIZE = CONFIG_DATA.CODRAFT_CONFIG['batch_size']
    chunks = [df[i:i + BATCH_SIZE] for i in range(0, df.shape[0], BATCH_SIZE)]

    for batch_df in tqdm(chunks, desc="Processing Batches"):
        batch_input = batch_df[['Term', 'Class', 'Description', 'Sub_group']].to_dict(orient='records')
        
        response = process_batch(batch_input, client, version=version)
        
        if response and response.items:
            for item in response.items:
                results.append(item.model_dump()) 
        else:
            for row in batch_input:
                results.append({"original_term": row.get('Term', 'UNKNOWN'), "status": "ERROR"})

        time.sleep(1)

    final_df = pd.DataFrame(results)
    file_name = f"nice_expanded_results_{version}.csv"
    full_path = os.path.join(output_file_dir, file_name) if output_file_dir else file_name
    final_df.to_csv(full_path, index=False)
    print(f"\nSave to: {output_file_dir}")
