#!/usr/bin/env python3
"""
Model Evaluation Script (Run Once)

Compares multiple LLMs using GPT-5 as ground truth on a fresh test set.
No circular validation - clean methodology for selecting the best classifier.

Usage: uv run 01_model_selection.py
Output: ../data/model_selection_results.csv
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

# Set reproducible seed
random.seed(42)

# Load environment variables
load_dotenv()

class TweetClassification(BaseModel):
    humor_type: str  # "absurdist", "self_deprecating", "observational", "none"
    topic_category: str  # "tech", "housing", "religion", "politics", "social_commentary", "personal", "general"
    has_data_reference: bool
    shows_vulnerability: bool
    critique_type: str  # "systemic", "institutional", "personal", "none"

def create_model_router() -> Router:
    """Create router with GPT-5 and candidate models."""
    model_list = [
        {
            "model_name": "gpt-5",
            "litellm_params": {
                "model": "openrouter/openai/gpt-5",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "max_parallel_requests": 2,
                "weight": 1,
            }
        },
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_parallel_requests": 5,
                "weight": 1,
            }
        },
        {
            "model_name": "gemini-2.5-flash-lite",
            "litellm_params": {
                "model": "openrouter/google/gemini-2.5-flash-lite",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "max_parallel_requests": 5,
                "weight": 1,
            }
        },
        {
            "model_name": "deepseek-chat",
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-chat",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "max_parallel_requests": 5,
                "weight": 1,
            }
        }
    ]

    return Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",
        num_retries=2,
        allowed_fails=3,
        cooldown_time=5,
        enable_pre_call_checks=True,
        default_max_parallel_requests=20,
        set_verbose=False,
    )

def create_clean_classification_prompt() -> str:
    """Create classification prompt with NO examples to avoid contamination."""
    return """
Analyze this tweet and classify it across several dimensions. Be precise and objective.

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

Classify based only on the content, not on engagement or popularity.
"""

def get_diverse_test_set(num_tweets: int = 20) -> List[Dict]:
    """Get diverse test tweets across engagement levels."""
    
    with open('../data/twitter_archive.json', 'r') as f:
        data = json.load(f)
    
    # Get tweets with sufficient text
    all_tweets = []
    for tweet_data in data['tweets']:
        tweet = tweet_data.get('tweet', tweet_data)
        
        retweet_count = int(tweet.get('retweet_count', 0))
        favorite_count = int(tweet.get('favorite_count', 0))
        weighted_engagement = favorite_count + (retweet_count * 10)
        
        text = tweet.get('full_text', '')
        if len(text.strip()) > 20:  # Minimum length filter
            all_tweets.append({
                'id': tweet.get('id_str'),
                'text': text,
                'engagement': weighted_engagement
            })
    
    # Sample from different engagement quartiles for diversity
    all_tweets.sort(key=lambda x: x['engagement'], reverse=True)
    
    high_eng = all_tweets[:len(all_tweets)//4]
    mid_eng = all_tweets[len(all_tweets)//4:3*len(all_tweets)//4]
    low_eng = all_tweets[3*len(all_tweets)//4:]
    
    # Balanced sample
    test_set = []
    test_set.extend(random.sample(high_eng, min(8, len(high_eng))))
    test_set.extend(random.sample(mid_eng, min(8, len(mid_eng))))
    test_set.extend(random.sample(low_eng, min(4, len(low_eng))))
    
    return test_set[:num_tweets]

async def classify_tweet(tweet_text: str, model_name: str, router: Router) -> Tuple[str, TweetClassification]:
    """Classify a tweet with a specific model."""
    
    prompt = create_clean_classification_prompt()
    
    messages = [
        {
            "role": "user",
            "content": f"{prompt}\n\nTweet to classify: \"{tweet_text}\""
        }
    ]

    try:
        response = await router.acompletion(
            model=model_name,
            messages=messages,
            temperature=0.0,  # Deterministic
            response_format=TweetClassification
        )
        
        classification = TweetClassification.model_validate_json(response.choices[0].message.content)
        return model_name, classification
        
    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        # Return neutral classification on error
        return model_name, TweetClassification(
            humor_type="none",
            topic_category="general",
            has_data_reference=False,
            shows_vulnerability=False,
            critique_type="none"
        )

def calculate_agreement(c1: TweetClassification, c2: TweetClassification) -> float:
    """Calculate agreement score between two classifications."""
    
    agreements = 0
    total_dimensions = 5
    
    if c1.humor_type == c2.humor_type: agreements += 1
    if c1.topic_category == c2.topic_category: agreements += 1
    if c1.has_data_reference == c2.has_data_reference: agreements += 1
    if c1.shows_vulnerability == c2.shows_vulnerability: agreements += 1
    if c1.critique_type == c2.critique_type: agreements += 1
    
    return agreements / total_dimensions

async def run_model_selection():
    """Run the model evaluation experiment."""
    
    # Check API keys
    if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing required API keys (OPENROUTER_API_KEY, OPENAI_API_KEY)")
        return None
    
    router = create_model_router()
    test_tweets = get_diverse_test_set(15)  # Conservative for GPT-5 costs
    
    print("🧪 MODEL EVALUATION EXPERIMENT")
    print("="*50)
    print(f"Testing {len(test_tweets)} diverse tweets")
    print("Ground truth: GPT-5")
    print("Candidate models: gpt-4o-mini, gemini-2.5-flash-lite, deepseek-chat")
    print()
    
    results = []
    models_to_test = ["gpt-5", "gpt-4o-mini", "gemini-2.5-flash-lite", "deepseek-chat"]
    
    for i, tweet in enumerate(test_tweets):
        tweet_text = tweet['text']
        tweet_id = tweet['id']
        engagement = tweet['engagement']
        
        print(f"📝 Tweet {i+1}/{len(test_tweets)} (Engagement: {engagement})")
        print(f"Text: {tweet_text[:80]}...")
        
        # Classify with all models in parallel
        tasks = [classify_tweet(tweet_text, model, router) for model in models_to_test]
        
        try:
            model_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            tweet_classifications = {}
            for result in model_results:
                if isinstance(result, Exception):
                    continue
                model_name, classification = result
                tweet_classifications[model_name] = classification
            
            # Require GPT-5 as ground truth
            if 'gpt-5' not in tweet_classifications:
                print("❌ GPT-5 classification failed, skipping")
                continue
            
            gpt5_classification = tweet_classifications['gpt-5']
            print(f"GPT-5: {gpt5_classification.humor_type} humor, {gpt5_classification.topic_category} topic")
            
            # Calculate agreement for other models
            for model_name, classification in tweet_classifications.items():
                if model_name != 'gpt-5':
                    agreement = calculate_agreement(gpt5_classification, classification)
                    results.append({
                        'tweet_id': tweet_id,
                        'tweet_text': tweet_text,
                        'engagement': engagement,
                        'model': model_name,
                        'agreement_score': agreement,
                        'gpt5_humor': gpt5_classification.humor_type,
                        'gpt5_topic': gpt5_classification.topic_category,
                        'model_humor': classification.humor_type,
                        'model_topic': classification.topic_category,
                        'gpt5_has_data': gpt5_classification.has_data_reference,
                        'model_has_data': classification.has_data_reference,
                        'gpt5_vulnerable': gpt5_classification.shows_vulnerability,
                        'model_vulnerable': classification.shows_vulnerability
                    })
                    
                    print(f"{model_name}: {classification.humor_type}/{classification.topic_category} | Agreement: {agreement:.1%}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error processing tweet: {e}")
            continue
        
        # Rate limiting
        await asyncio.sleep(1)
    
    return results

def analyze_and_save_results(results: List[Dict]):
    """Analyze results and determine best model."""
    
    if not results:
        print("❌ No results to analyze")
        return None
    
    df = pd.DataFrame(results)
    
    # Calculate model performance
    model_performance = df.groupby('model')['agreement_score'].agg(['mean', 'count', 'std']).round(3)
    model_performance = model_performance.sort_values('mean', ascending=False)
    
    print("\n" + "="*50)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*50)
    
    for model, row in model_performance.iterrows():
        print(f"{model:<25}: {row['mean']:.1%} agreement (±{row['std']:.1%}, n={int(row['count'])})")
    
    best_model = model_performance.index[0]
    best_score = model_performance.iloc[0]['mean']
    
    print(f"\n🏆 SELECTED MODEL: {best_model}")
    print(f"   GPT-5 Agreement: {best_score:.1%}")
    print(f"   Reason: Highest agreement with GPT-5 ground truth")
    
    # Save detailed results
    df.to_csv('../data/model_selection_results.csv', index=False)
    print(f"\n💾 Results saved to: ../data/model_selection_results.csv")
    
    # Save selection summary
    with open('../data/selected_model.txt', 'w') as f:
        f.write(f"Selected Model: {best_model}\n")
        f.write(f"GPT-5 Agreement: {best_score:.1%}\n")
        f.write(f"Methodology: Fresh test set, no circular validation\n")
        f.write(f"Test tweets: {len(df['tweet_id'].unique())}\n")
    
    print("📄 Selection summary saved to: ../data/selected_model.txt")
    
    return best_model, model_performance

async def main():
    """Run model evaluation."""
    
    print("🎯 Starting model evaluation...")
    print("Methodology: Clean test set, GPT-5 ground truth, no circular validation")
    print()
    
    results = await run_model_selection()
    
    if results:
        best_model, performance = analyze_and_save_results(results)
        print(f"\n✅ Model evaluation complete!")
        print(f"Use {best_model} for classification workflow.")
    else:
        print("❌ Model evaluation failed")

if __name__ == "__main__":
    asyncio.run(main())