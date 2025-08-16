# scripts/mutuals_extract.py
import json
import subprocess
from pathlib import Path
import pandas as pd

def extract_mutual_ids(archive_path: str = "data/twitter_archive.json"):
    # Load archive
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    
    # Extract IDs
    following_ids = {item['following']['accountId'] for item in archive['following']}
    follower_ids = {item['follower']['accountId'] for item in archive['follower']}
    mutual_ids = list(following_ids & follower_ids)
    
    # Save outputs
    Path("data").mkdir(exist_ok=True)
    with open('data/mutual_ids.json', 'w') as f:
        json.dump(mutual_ids, f)
    with open('data/mutual_ids.txt', 'w') as f:
        f.write(','.join(mutual_ids))
    
    print(f"Saved {len(mutual_ids)} mutual IDs")
    return mutual_ids

def fetch_account_info(batch_size: int = 100):
    # Load mutual IDs
    try:
        with open('data/mutual_ids.json', 'r') as f:
            mutual_ids = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Run mutuals_extract.py first")
    
    # Batch processing
    batches = [mutual_ids[i:i+batch_size] 
                for i in range(0, len(mutual_ids), batch_size)]
    
    for i, batch in enumerate(batches, 1):
        batch_file = Path(f"data/mutuals_batch{i}.json")
        if batch_file.exists(): continue  # Skip existing batches
        
        print(f"Fetching batch {i}/{len(batches)}...")
        cmd = f"x-cli user --by-id --json {' '.join(batch)}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error batch {i}: {result.stderr}")
            continue
            
        with open(batch_file, 'w') as f:
            json.dump({"data": json.loads(result.stdout).get('data', [])}, f)
    
    print(f"Completed {len(batches)} batches")

def combine_and_export():
    # Find all batch files
    batch_files = sorted(Path("data").glob("mutuals_batch*.json"))
    if not batch_files:
        raise FileNotFoundError("No batch files found")
    
    # Combine data
    combined = []
    for file in batch_files:
        with open(file, 'r') as f:
            combined.extend(json.load(f).get('data', []))
    
    # Save JSON
    with open('data/mutuals_account_info.json', 'w') as f:
        json.dump({"data": combined}, f, indent=2)
    
    # Create CSV
    df = pd.DataFrame([{
        'id': u.get('id'),
        'username': u.get('username'),
        'name': u.get('name'),
        'followers': u.get('public_metrics', {}).get('followers_count'),
        'following': u.get('public_metrics', {}).get('following_count'),
        'tweets': u.get('public_metrics', {}).get('tweet_count'),
        'verified': u.get('verified'),
        'created_at': u.get('created_at'),
        'description': u.get('description')
    } for u in combined])
    
    df.to_csv('data/mutuals_summary.csv', index=False)
    print(f"Combined {len(df)} accounts into JSON and CSV")

if __name__ == "__main__":
    extract_mutual_ids()
    fetch_account_info()
    combine_and_export()