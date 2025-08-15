# Twitter Engagement Analysis

## Overview

This project analyzes my Twitter archive to identify content strategies that drive engagement, using robust methodology to avoid being misled by viral outliers that skew traditional averages.

**Key Finding**: **Observational humor provides the most consistent engagement** (1.4x advantage), while absurdist humor represents high-risk/high-reward viral potential.

📊 **Analysis Reports:**
- **[Content Engagement Analysis](outputs/content_engagement_analysis.md)** - Humor types and topic strategies
- **[Reply/Link/Media Analysis](outputs/reply_link_media_engagement_analysis.md)** - Media patterns and event studies

## Project Structure

```
Twitter/
├── scripts/                    # Core analysis scripts
│   ├── 01_model_selection.py   # Model evaluation and selection
│   └── 02_classification_workflow.py  # Tweet classification
├── notebooks/                  # Analysis notebooks
│   ├── content_engagement_analysis.qmd  # Main engagement analysis
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
└── Configuration files
    ├── pyproject.toml          # Python dependencies
    ├── _quarto.yml             # Quarto configuration
    └── CLAUDE.md               # AI assistant instructions
```

## Usage

### 1. Install Dependencies
```bash
uv install
```

### 2. Set Up API Keys
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
```

### 3. Run Analysis Pipeline

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

*Content Engagement Analysis:*
```bash
cd notebooks
quarto render content_engagement_analysis.qmd --to gfm
```
- Analyzes humor types and topic categories for engagement patterns
- Outputs: `../outputs/content_engagement_analysis.md`

*Reply/Link/Media Analysis:*
```bash
cd notebooks
quarto render reply_link_media_engagement_analysis.qmd --to gfm
```
- Analyzes engagement patterns for replies, links, and media content
- Includes temporal stability and blue-check upgrade event study
- Outputs: `../outputs/reply_link_media_engagement_analysis.md`

## Methodology Highlights

### **Methodologically Sound Approach**
- **Independent Ground Truth**: GPT-5 agreement evaluation for model selection
- **Outlier-Resistant Analysis**: Winsorized means (95th percentile cap) to handle skewed engagement data
- **Robustness Testing**: Multiple normalization methods to validate conclusions
- **No Circular Validation**: Fresh test sets with no prompt contamination
- **Confidence-Free Analysis**: Focus on actual performance over self-reported scores
- **Reproducible Pipeline**: Documented workflow with version control

### **Common Pitfalls Avoided**
- Using examples in prompts then testing on same examples
- Trusting LLM self-reported confidence scores
- Being misled by outlier-skewed averages in social media data
- Cherry-picking results or models
- Circular validation between training and testing

## Key Findings

1. **Observational humor** delivers the most consistent engagement (1.4x advantage)
2. **Absurdist humor** has viral potential but inconsistent baseline performance  
3. **Topic choice is secondary** to humor type after controlling for outliers
4. **Portfolio approach**: 70% consistent content, 30% viral experiments

## Strategic Recommendations

### Portfolio Strategy

**Baseline Content (70%): Observational Humor**
- Witty takes on everyday experiences, tech, and social commentary
- Goal: Consistent 1-5 engagement per tweet

**Viral Experiments (30%): Absurdist Humor** 
- "Personal disenrichment" style ironic inversions
- Religious/sacred metaphors for mundane activities
- Goal: Occasional viral breakthroughs (10+ engagement)

### Key Insight
**Topic choice matters less than humor type** - focus on areas where you can apply observational humor effectively rather than chasing specific topic categories.

## Technical Details

- **Model Selection**: Systematic LLM evaluation using GPT-5 ground truth
- **Outlier Handling**: Winsorized means to avoid viral tweet bias
- **Validation**: Robustness testing across multiple normalization methods
- **Reproducible**: Full pipeline with error handling and clear documentation

## Dependencies

Managed via `uv` package manager. See [Usage](#usage) section for setup instructions.

---

*This analysis demonstrates how to conduct methodologically sound content strategy research using modern LLM tools while avoiding common experimental design pitfalls, particularly the critical importance of robust outlier handling in skewed social media engagement data.*