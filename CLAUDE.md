# Twitter Engagement Analysis Project - Assistant Guidelines

## Project Overview

This project analyzes my downloaded Twitter archive (`data/twitter_archive.json`) to see what we can learn about engagement patterns using robust statistical methods and to identify effective content strategies.

The top-level keys in the JSON are:

- "account" (shape defined in `data/account_shape.json`)
- "community-tweet"
- "follower" (shape defined in `data/follower_shape.json`)
- "following" (shape defined in `data/following_shape.json`)
- "like" (shape defined in `data/like_shape.json`)
- "note-tweet"
- "profile" (shape defined in `data/profile_shape.json`)
- "tweets" (shape defined in `data/tweets_shape.json`)
- "upload-options"

## Key Technical Requirements

### Working Directory

- **ALWAYS** work from the project root directory (`/home/chris/software/Twitter`)
- **NEVER** change into subdirectories to run commands
- All paths in commands should be relative to the project root

### Python Execution

- **ALWAYS** run Python scripts with `uv run` (e.g., `uv run scripts/01_model_selection.py`)
- **ALWAYS** run Quarto renders with proper Python path: `uv run quarto render`
- The project uses `uv` for dependency management - never use pip directly

### Project Structure

```
Twitter/
├── scripts/        # Analysis pipeline scripts (run from this directory)
├── notebooks/      # Quarto analysis notebooks  
├── data/          # All data files (CSVs, JSON, parquet)
├── utils/         # Shared utility modules (analysis_utils.py)
├── outputs/       # Generated reports and figures
└── src/          # Legacy location - being migrated to utils/
```

### File Path Conventions

- Scripts access data with `../data/` prefix
- Notebooks access data with `../data/` prefix  
- utils module imported as `from utils.analysis_utils import ...`

## Analysis Pipeline

### 1. Model Selection (Run Once)

```bash
uv run scripts/01_model_selection.py
```

- Evaluates multiple LLMs against GPT-5 ground truth
- Outputs: `data/model_selection_results.csv`, `data/selected_model.txt`

### 2. Tweet Classification (Run Once)

```bash
uv run scripts/02_classification_workflow.py
```

- Uses selected model to classify all tweets
- Outputs: `data/tweet_classifications.csv`

### 3. Generate Analysis Reports

```bash
# Render all notebooks at once (outputs to outputs/ directory)
uv run quarto render notebooks
```

## Key Methodological Points

### Outlier Handling

- Social media data is heavily skewed - a few viral tweets can mislead averages
- Use **winsorized means** (95th percentile cap) for robust analysis
- Always compare mean vs median to understand skew

### Model Selection Approach

- Uses GPT-5 (a very large/smart model) as independent ground truth against which to test performance of small/cheap models

## Common Tasks

### Creating New Notebooks

1. **Create notebook file** in `notebooks/` directory
2. **Use relative paths** from notebook location:
   - Load data: `../data/your_file.json`
   - Import utils: `from utils.analysis_utils import ...`
3. **Configure from `notebooks/_quarto.yml`**
4. **Render from project root**:
   ```bash
   uv run quarto render notebooks/your_notebook.qmd
   ```
5. **Output location**: 
   - Markdown: `outputs/your_notebook.md`
   - Figures: `outputs/your_notebook_files/figure-commonmark/`

### Adding New Analysis

1. Create notebook in `notebooks/` directory following above guidelines
2. Import utilities: `from utils.analysis_utils import ...`
3. Load data from `../data/` paths
4. Follow winsorization approach for engagement metrics

### Updating Classifications

1. Modify prompt in `scripts/02_classification_workflow.py`
2. Re-run classification pipeline
3. Update dependent notebooks

### Testing Code Changes

```bash
# Run specific analysis (from project root)
uv run python -c "from utils.analysis_utils import load_archive; df = load_archive('data/twitter_archive.json'); print(df.shape)"

# Check notebook execution (from project root)
uv run quarto render notebooks/content_engagement_analysis.qmd --execute-debug
```

## Environment Variables

Required in `.env`:

- `OPENAI_API_KEY` - For GPT-4o-mini classification
- `OPENROUTER_API_KEY` - For model evaluation with GPT-5

## Important Notes

- Never commit API keys or `.env` file
- Archive data (`twitter_archive.json`) contains personal information
- Focus on methodological robustness over complexity
- Prefer editing existing files over creating new ones