#!/usr/bin/env python3
"""
Classification Workflow Script (Run Once)

Uses the best model (selected by 01_model_selection.py) to classify tweets
for engagement analysis. Generates the dataset used in the final report.

Usage: uv run 02_classification_workflow.py
Prerequisites: Run 01_model_selection.py first to select model
Output: tweet_classifications.csv
"""

import json
import pandas as pd
import asyncio
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from litellm import Router
from pydantic import BaseModel
import random

# Load environment variables
load_dotenv()

class TweetClassification(BaseModel):
    humor_type: str
    topic_category: str  
    has_data_reference: bool
    shows_vulnerability: bool
    critique_type: str

def load_selected_model() -> str:
    """Load the model selected by the evaluation script."""
    try:
        with open('selected_model.txt', 'r') as f:
            first_line = f.readline()
            model = first_line.split(': ')[1].strip()
            return model
    except FileNotFoundError:
        print("❌ selected_model.txt not found. Run 01_model_selection.py first.")
        return None
    except Exception as e:
        print(f"❌ Error reading selected model: {e}")
        return None

def create_classification_router(selected_model: str) -> Router:
    """Create router for the selected model."""
    
    # Model configuration mapping
    model_configs = {
        "gpt-4o-mini": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "gemini-2.5-flash-lite": {
            "model": "openrouter/google/gemini-2.5-flash-lite", 
            "api_key": os.getenv("OPENROUTER_API_KEY")
        },
        "deepseek-chat": {
            "model": "openrouter/deepseek/deepseek-chat",
            "api_key": os.getenv("OPENROUTER_API_KEY")
        }
    }
    
    if selected_model not in model_configs:
        raise ValueError(f"Unknown model: {selected_model}")
    
    config = model_configs[selected_model]
    
    model_list = [{
        "model_name": "classifier",
        "litellm_params": {
            "model": config["model"],
            "api_key": config["api_key"],
            "max_parallel_requests": 10,
            "weight": 1,
        }
    }]

    return Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",
        num_retries=2,
        allowed_fails=3,
        cooldown_time=2,
        enable_pre_call_checks=True,
        default_max_parallel_requests=10,
        set_verbose=False,
    )

def create_classification_prompt() -> str:
    """Create the classification prompt."""
    return """
Analyze this tweet and classify it across several dimensions.

HUMOR CLASSIFICATION:
- "absurdist": Unexpected juxtapositions, surreal comparisons, treating mundane things as profound
- "self_deprecating": Making fun of oneself, admitting personal flaws/mistakes  
- "observational": Wry commentary on social situations, pointing out ironies
- "none": No humor detected

TOPIC CLASSIFICATION:
- "tech": Technology, AI, programming, software companies
- "housing": Real estate, zoning, urban planning, housing policy
- "religion": Faith, theology, religious communities, spirituality
- "politics": Government, elections, policy, political figures
- "social_commentary": Social issues, cultural criticism, gender/race dynamics
- "personal": Individual experiences, daily life, personal anecdotes
- "general": Doesn't clearly fit other categories

OTHER CLASSIFICATIONS:
- has_data_reference: References studies, data, research, statistics
- shows_vulnerability: Admits mistakes, uncertainty, learning, being wrong
- critique_type: "systemic" (broad systems), "institutional" (specific orgs), "personal" (individuals), "none"

Be precise and consistent in your classifications.
"""

def load_twitter_data() -> pd.DataFrame:
    """Load and prepare Twitter archive data."""
    
    with open('twitter_archive.json', 'r') as f:
        data = json.load(f)
    
    tweets = []
    for tweet_data in data['tweets']:
        tweet = tweet_data.get('tweet', tweet_data)
        
        retweet_count = int(tweet.get('retweet_count', 0))
        favorite_count = int(tweet.get('favorite_count', 0))
        weighted_engagement = favorite_count + (retweet_count * 10)
        
        tweet_info = {
            'id': tweet.get('id_str'),
            'text': tweet.get('full_text', ''),
            'created_at': tweet.get('created_at'),
            'retweet_count': retweet_count,
            'favorite_count': favorite_count,
            'weighted_engagement': weighted_engagement,
            'is_reply': tweet.get('in_reply_to_status_id') is not None,
            'reply_to_screen_name': tweet.get('in_reply_to_screen_name'),
            'char_count': len(tweet.get('full_text', '')),
        }
        
        # Only include tweets with sufficient content
        if len(tweet_info['text'].strip()) > 15:
            tweets.append(tweet_info)
    
    return pd.DataFrame(tweets)

def select_tweets_for_classification(df: pd.DataFrame, max_tweets: int = 1000) -> pd.DataFrame:
    """Select a representative sample of tweets for classification."""
    
    # Use stratified sampling across engagement quartiles for better representation
    df_sorted = df.sort_values('weighted_engagement', ascending=False)
    
    # Define engagement quartiles
    q1_end = len(df_sorted) // 4
    q2_end = len(df_sorted) // 2  
    q3_end = 3 * len(df_sorted) // 4
    
    q1 = df_sorted[:q1_end]  # Top quartile (highest engagement)
    q2 = df_sorted[q1_end:q2_end]  # Second quartile
    q3 = df_sorted[q2_end:q3_end]  # Third quartile  
    q4 = df_sorted[q3_end:]  # Bottom quartile (lowest engagement)
    
    # Sample proportionally from each quartile
    samples_per_quartile = max_tweets // 4
    
    selected = pd.concat([
        q1.sample(min(samples_per_quartile, len(q1)), random_state=42),
        q2.sample(min(samples_per_quartile, len(q2)), random_state=42),
        q3.sample(min(samples_per_quartile, len(q3)), random_state=42),
        q4.sample(min(samples_per_quartile, len(q4)), random_state=42)
    ])
    
    return selected.drop_duplicates(subset=['id'])

async def classify_tweet(tweet_text: str, router: Router) -> TweetClassification:
    """Classify a single tweet."""
    
    prompt = create_classification_prompt()
    
    messages = [
        {
            "role": "user",
            "content": f"{prompt}\n\nTweet to classify: \"{tweet_text}\""
        }
    ]

    try:
        response = await router.acompletion(
            model="classifier",
            messages=messages,
            temperature=0.0,
            response_format=TweetClassification
        )
        
        return TweetClassification.model_validate_json(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Classification error: {e}")
        # Return neutral classification on error
        return TweetClassification(
            humor_type="none",
            topic_category="general", 
            has_data_reference=False,
            shows_vulnerability=False,
            critique_type="none"
        )

async def classify_tweets_batch(tweets_df: pd.DataFrame, router: Router) -> pd.DataFrame:
    """Classify tweets in parallel with rate limiting."""
    
    print(f"🔄 Classifying {len(tweets_df)} tweets...")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(5)
    
    async def classify_with_limit(tweet_text: str) -> TweetClassification:
        async with semaphore:
            result = await classify_tweet(tweet_text, router)
            return result
    
    # Create tasks for all tweets
    tasks = [classify_with_limit(tweet['text']) for _, tweet in tweets_df.iterrows()]
    
    # Process in batches with progress updates
    classifications = []
    batch_size = 10
    
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        classifications.extend(batch_results)
        
        progress = min(i + batch_size, len(tasks))
        print(f"  Processed {progress}/{len(tasks)} tweets...")
        
        # Small delay between batches
        await asyncio.sleep(0.5)
    
    # Add classifications to dataframe
    classified_df = tweets_df.copy()
    for i, classification in enumerate(classifications):
        classified_df.at[classified_df.index[i], 'humor_type'] = classification.humor_type
        classified_df.at[classified_df.index[i], 'topic_category'] = classification.topic_category
        classified_df.at[classified_df.index[i], 'has_data_reference'] = classification.has_data_reference
        classified_df.at[classified_df.index[i], 'shows_vulnerability'] = classification.shows_vulnerability
        classified_df.at[classified_df.index[i], 'critique_type'] = classification.critique_type
    
    return classified_df

def save_classification_results(classified_df: pd.DataFrame, selected_model: str):
    """Save classification results with metadata."""
    
    # Save main results
    classified_df.to_csv('tweet_classifications.csv', index=False)
    print(f"💾 Classifications saved to: tweet_classifications.csv")
    
    # Save metadata
    metadata = {
        'model_used': selected_model,
        'total_tweets_classified': len(classified_df),
        'date_classified': pd.Timestamp.now().isoformat(),
        'methodology': 'Selected model based on GPT-5 agreement evaluation'
    }
    
    with open('classification_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Metadata saved to: classification_metadata.json")
    
    # Print summary statistics
    print(f"\n📊 CLASSIFICATION SUMMARY")
    print("="*40)
    print(f"Model used: {selected_model}")
    print(f"Tweets classified: {len(classified_df)}")
    
    print(f"\nHumor distribution:")
    humor_counts = classified_df['humor_type'].value_counts()
    for humor, count in humor_counts.items():
        print(f"  {humor}: {count} tweets")
    
    print(f"\nTopic distribution:")
    topic_counts = classified_df['topic_category'].value_counts()
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} tweets")
    
    print(f"\nEngagement by humor type:")
    humor_engagement = classified_df.groupby('humor_type')['weighted_engagement'].mean().round(0)
    for humor, avg_eng in humor_engagement.sort_values(ascending=False).items():
        print(f"  {humor}: {avg_eng:.0f} avg engagement")

async def main():
    """Run classification workflow."""
    
    print("🏷️  TWEET CLASSIFICATION WORKFLOW")
    print("="*40)
    
    # Load selected model
    selected_model = load_selected_model()
    if not selected_model:
        return
    
    print(f"Using model: {selected_model}")
    print(f"Selected via GPT-5 agreement evaluation")
    print()
    
    # Check API keys
    required_keys = ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        print(f"❌ Missing API keys: {missing_keys}")
        return
    
    # Load data
    print("📚 Loading Twitter data...")
    df = load_twitter_data()
    print(f"Loaded {len(df)} tweets from archive")
    
    # Select tweets for classification
    # Use a statistically meaningful sample size (n≈500 for robust analysis)
    selected_tweets = select_tweets_for_classification(df, max_tweets=500)
    print(f"Using stratified sample: {len(selected_tweets)} tweets for classification (balanced across engagement quartiles)")
    
    # Create router
    router = create_classification_router(selected_model)
    
    # Classify tweets
    classified_df = await classify_tweets_batch(selected_tweets, router)
    
    # Save results
    save_classification_results(classified_df, selected_model)
    
    print(f"\n✅ Classification workflow complete!")
    print(f"Ready for analysis in Quarto notebook.")

if __name__ == "__main__":
    asyncio.run(main())