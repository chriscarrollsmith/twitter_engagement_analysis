# Twitter Engagement Analysis

## Overview

This project analyzes my Twitter archive to identify content strategies that drive engagement, using robust methodology to avoid being misled by viral outliers that skew traditional averages.

To use this project, you will need to:

1. Download your Twitter archive from https://x.com/settings/download_your_data
2. Upload your data to the [Twitter Community Archive](https://www.community-archive.org/), which will process it and put it in a usable JSON format (warning: this will make your data public on the archive website!)
3. Download your data in JSON format from https://fabxmporizzqflnftavs.supabase.co/storage/v1/object/public/archives/{your_username}/archive.json
4. Save it to `data/twitter_archive.json`

## Results

**Key Finding**: **Observational humor provides the most consistent engagement** (1.4x advantage), while absurdist humor represents high-risk/high-reward viral potential.

📊 **Analysis Reports:**
- **[Content Engagement Analysis](outputs/content_engagement_analysis.md)** - Humor types and topic strategies
- **[Following Analysis](outputs/following_analysis.md)** - Analysis of my following and follower relationships
- **[Reply/Link/Media Analysis](outputs/reply_link_media_engagement_analysis.md)** - Media patterns and event studies

## Project Structure

```
Twitter/
├── scripts/                    # Core analysis scripts
│   ├── 01_model_selection.py   # Model evaluation and selection
│   └── 02_classification_workflow.py  # Tweet classification
├── notebooks/                  # Analysis notebooks
│   ├── content_engagement_analysis.qmd  # Main engagement analysis
│   ├── following_analysis.qmd  # Following analysis
│   ├── _quarto.yml             # Quarto configuration
│   └── reply_link_media_engagement_analysis.qmd  # Media/reply analysis
├── data/                       # Data files
│   ├── twitter_archive.json    # Original Twitter export
│   ├── model_selection_results.csv  # Model evaluation results
│   ├── tweet_classifications.csv    # Classified tweets
│   └── selected_model.txt      # Selected model for classification
├── utils/                        # Utility modules
│   └── analysis_utils.py       # Data loading and processing utilities
├── outputs/                    # Generated reports and figures
│   ├── content_engagement_analysis.md  # Rendered analysis report
│   └── *_files/                # Supporting figures
├── pyproject.toml          # Python dependencies
└── CLAUDE.md               # AI assistant instructions
```

## Usage

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- [x-cli](https://github.com/Promptly-Technologies-LLC/X-cli)
- Twitter archive data saved as `data/twitter_archive.json`

### 1. Install Dependencies
```bash
uv install
```

### 2. Set Up API Keys
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
MY_USER_ID=your_user_id
MY_SCREEN_NAME=your_screen_name
```

### 3. (Optional) Fetch Mutual Account Data
To analyze mutual follows, install x-cli and fetch account info:
```bash
# Install x-cli
uv tool install -U git+https://github.com/Promptly-Technologies-LLC/X-cli.git

# Extract and fetch mutual account info (handles batching and rate limits)
uv run scripts/mutuals_extract.py
```
Note: x-cli has a 100 API calls/month limit. The script saves progress and can resume if interrupted.

### 4. Run Analysis Pipeline

**Step 1: Model Evaluation**
```bash
cd scripts
uv run 01_model_selection.py
```
- Compares multiple LLMs using GPT-5 as ground truth
- Selects best model based on agreement scores
- Outputs: `../data/model_selection_results.csv`, `../data/selected_model.txt`

**Step 2: Tweet Classification**  
```bash
cd scripts
uv run 02_classification_workflow.py
```
- Uses selected model to classify tweets
- Generates engagement analysis dataset
- Outputs: `../data/tweet_classifications.csv`

**Step 3: Generate Reports**

```bash
uv run quarto render notebooks
```

## Key Findings

### Content Analysis

1. **Observational humor** delivers the most consistent engagement (1.4x advantage)
2. **Absurdist humor** has viral potential but inconsistent baseline performance  
3. **Topic choice** is secondary to humor type after controlling for outliers, but:
   - Housing and religion have been good niches for me
   - Twitter seems to like when I make use of data
   - Political posts performed better than I expected, though are best defused with observational humor
   - Personal topics are fine, but vulnerability is penalized

### Following Analysis

Moderate reciprocity rate (32.0%) suggests a balanced mix of mutual connections and one-way follows.

### Reply/Link/Media Analysis

My signup for the blue check upgrade roughly coincided with a major change in the algorithm, and you can see this in my data. This makes it a bit hard to disambiguate the effects of the upgrade from the effects of the algorithm change, but the fact that I have both pre- and post-upgrade data makes it possible to do some analysis using the post-upgrade data as the baseline.

1. **Algorithm changes** detected in my event study are basically consistent with what Elon Musk has shared about the new algorithm:
   - Links and media tweets performed much better than text-only tweets before the algorithm change, but about the same after.
   - Replies performed much better after the algorithm change.
   - Overall, I estimate that the new algorithm reduced my free-tier engagement by about 65%, with almost all of that penalty accruing to my link and media tweets.
2. **Blue check boost** detected by comparing my text-only tweets during the upgrade period to my text-only tweets after the upgrade (which is plausibly the baseline), seems to be on the order of about 15-20% more engagement for blue checks.

## Strategic Recommendations

- **Do** make use of data; **Don't** spend loads of time sourcing links and media to include in tweets.
- **Do** use observational humor to take the edge off of sensitive/controversial topics; **Don't** expect every absurdist joke to hit.
- **Do** post personal triumphs; **Don't** be too raw or self-deprecating.
- **Do** upgrade to blue check if a 15-20% boost is worth it to you; **Don't** expect it to make you a viral sensation.
