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
        if batch_file.exists():
            continue  # Skip existing batches
        
        print(f"Fetching batch {i}/{len(batches)}...")
        cmd = f"x-cli user --by-id --json --fields name location description affiliation most_recent_tweet_id public_metrics --expansions most_recent_tweet_id --tweet-fields created_at {' '.join(batch)}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error batch {i}: {result.stderr}")
            continue
            
        with open(batch_file, 'w') as f:
            response_data = json.loads(result.stdout)
            json.dump(response_data, f)
    
    print(f"Completed {len(batches)} batches")

def combine_and_export():
    # Find all batch files
    batch_files = sorted(Path("data").glob("mutuals_batch*.json"))
    if not batch_files:
        raise FileNotFoundError("No batch files found")
    
    # Combine data
    combined_users = []
    combined_tweets = []
    
    for file in batch_files:
        with open(file, 'r') as f:
            batch_data = json.load(f)
            combined_users.extend(batch_data.get('data', []))
            if 'includes' in batch_data and 'tweets' in batch_data['includes']:
                combined_tweets.extend(batch_data['includes']['tweets'])
    
    # Create lookup for tweet dates
    tweet_lookup = {tweet['id']: tweet.get('created_at') for tweet in combined_tweets}
    
    # Save JSON
    with open('data/mutuals_account_info.json', 'w') as f:
        json.dump({
            "data": combined_users,
            "includes": {"tweets": combined_tweets}
        }, f, indent=2)
    
    # Create CSV
    df = pd.DataFrame([{
        'id': u.get('id'),
        'username': u.get('username'),
        'name': u.get('name'),
        'location': u.get('location'),
        'description': u.get('description'),
        'affiliation': u.get('affiliation'),
        'most_recent_tweet_id': u.get('most_recent_tweet_id'),
        'most_recent_tweet_date': tweet_lookup.get(u.get('most_recent_tweet_id')),
        'followers_count': u.get('public_metrics', {}).get('followers_count')
    } for u in combined_users])
    
    df.to_csv('data/mutuals_summary.csv', index=False)
    print(f"Combined {len(df)} accounts into JSON and CSV")

if __name__ == "__main__":
    extract_mutual_ids()
    fetch_account_info()
    combine_and_export()