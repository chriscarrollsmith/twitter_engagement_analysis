from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# --- Constants ---
# Replace with your actual user ID string if known; otherwise inferred at runtime
MY_USER_ID: str = "707078994852622336"
# Optionally provide your username to improve ID inference
MY_SCREEN_NAME: str = "christophcsmith"

# Account tier boundaries (inclusive start dates)
TIER_UPGRADED_START: str = "2023-09-12"
TIER_POST_UPGRADE_START: str = "2024-09-12"

# Analysis constants
WINSORIZE_THRESHOLD: float = 0.95
ROLLING_WINDOW_DAYS: int = 60

# Time handling
TIMEZONE: str = "UTC"


# --- Data loading utilities ---
def load_archive(path: str) -> pd.DataFrame:
    """Load Twitter archive JSON supporting JSONL, top-level list or object, and nested shapes.

    Strategy:
    1) Try JSON Lines via pandas. If rows wrap `tweet`, normalize that column.
    2) Fallback to `json.load` and normalize:
       - top-level list
       - top-level dict with `tweets` key (or itself a tweet dict)
    """
    # Try JSON Lines first
    try:
        df = pd.read_json(path, lines=True)
        if not df.empty:
            # JSONL case or accidental parse of a single JSON object as one line
            if 'tweet' in df.columns and isinstance(df.loc[0, 'tweet'], (dict,)):
                return pd.json_normalize(df['tweet'])
            if 'tweets' in df.columns and len(df) == 1:
                # Reconstruct the original dict and process via dict path
                obj_like = df.iloc[0].to_dict()
                if isinstance(obj_like.get('tweets'), list):
                    nested = pd.json_normalize(obj_like['tweets'])
                    if 'tweet' in nested.columns and isinstance(nested.loc[0, 'tweet'], (dict,)):
                        return pd.json_normalize(nested['tweet'])
                    return nested
            # If it's a flat dict-as-row, fall through to json.load path
        # If df is empty, fall through
    except ValueError:
        pass

    # Fallback to regular JSON
    with open(path, 'r') as f:
        obj = json.load(f)

    if isinstance(obj, list):
        df = pd.json_normalize(obj)
        # If rows wrap payload under a 'tweet' key, flatten it
        if 'tweet' in df.columns and isinstance(df.loc[0, 'tweet'], (dict,)):
            df = pd.json_normalize(df['tweet'])
        return df

    if isinstance(obj, dict):
        if 'tweets' in obj and isinstance(obj['tweets'], list):
            df = pd.json_normalize(obj['tweets'])
            if 'tweet' in df.columns and isinstance(df.loc[0, 'tweet'], (dict,)):
                df = pd.json_normalize(df['tweet'])
            return df
        # Already flat dict or unexpected key; normalize generically
        df = pd.json_normalize(obj)
        if 'tweet' in df.columns and isinstance(df.loc[0, 'tweet'], (dict,)):
            df = pd.json_normalize(df['tweet'])
        return df

    raise ValueError("Unsupported JSON structure in archive")


def ensure_string_id(series: pd.Series) -> pd.Series:
    """Safely cast ID-like columns to string to avoid precision issues."""
    return series.astype(str)


def infer_my_user_id(df: pd.DataFrame, username_hint: Optional[str] = None) -> str:
    """Infer the most likely user ID from available columns.

    Preference order:
    1) If `username_hint` or `MY_SCREEN_NAME` is available, map `user.screen_name` -> `user.id_str` (or `user.id`).
    2) Global mode of `user.id_str` (or `user.id`).
    3) Mode of `in_reply_to_user_id_str`.
    """
    # 1) Username-based inference
    screen_col_candidates = ['user.screen_name', 'user.screen_name_str', 'screen_name']
    id_col_candidates = ['user.id_str', 'user.id']
    hint = (username_hint or MY_SCREEN_NAME or '').lower()
    if hint:
        for screen_col in screen_col_candidates:
            if screen_col in df.columns:
                mask = df[screen_col].astype(str).str.lower() == hint
                if mask.any():
                    for id_col in id_col_candidates:
                        if id_col in df.columns:
                            ids = df.loc[mask, id_col].astype(str)
                            if not ids.empty:
                                return ids.mode(dropna=True).iat[0]
    # 2) Global mode of author ID
    for key in id_col_candidates:
        if key in df.columns:
            s = df[key].astype(str).value_counts()
            if not s.empty:
                return s.idxmax()
    # 3) Fallback: common reply target
    if 'in_reply_to_user_id_str' in df.columns:
        candidates = df['in_reply_to_user_id_str'].dropna().astype(str).value_counts()
        if not candidates.empty:
            return candidates.idxmax()
    raise ValueError("Could not infer MY_USER_ID; please set it explicitly in analysis_utils.py")


# --- Normalization helpers ---
def normalize_datetime(df: pd.DataFrame, tz: str = TIMEZONE) -> pd.DataFrame:
    """Create `post_datetime` as timezone-aware datetime and set as index."""
    df = df.copy()
    if 'created_at' in df.columns:
        df['post_datetime'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
    else:
        # Attempt alternate keys
        found = False
        for alt in ['tweet.created_at', 'time', 'date']:
            if alt in df.columns:
                df['post_datetime'] = pd.to_datetime(df[alt], errors='coerce', utc=True)
                found = True
                break
        if not found:
            # Generic fallback: find any column ending with 'created_at'
            candidates = [c for c in df.columns if str(c).endswith('created_at')]
            if candidates:
                df['post_datetime'] = pd.to_datetime(df[candidates[0]], errors='coerce', utc=True)
                found = True
        if not found:
            raise ValueError("No timestamp column found (e.g., created_at)")
    if tz and tz.upper() != 'UTC':
        df['post_datetime'] = df['post_datetime'].dt.tz_convert(tz)
    df = df.set_index('post_datetime', drop=False).sort_index()
    return df


# --- Detection utilities ---
def get_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def detect_retweet(row: pd.Series) -> bool:
    # Check various possible column names
    for col in ['retweeted', 'tweet.retweeted']:
        if col in row and pd.notna(row[col]):
            try:
                return bool(row[col])
            except Exception:
                pass
    for col in ['retweeted_status', 'tweet.retweeted_status']:
        if col in row and isinstance(row[col], dict):
            return True
    # Check text content
    text = row.get('full_text') or row.get('text') or row.get('tweet.full_text') or row.get('tweet.text') or ''
    return isinstance(text, str) and text.startswith('RT @')


def detect_quote(row: pd.Series) -> bool:
    # Check various possible column names
    for col in ['is_quote_status', 'tweet.is_quote_status']:
        if col in row and pd.notna(row[col]):
            try:
                return bool(row[col])
            except Exception:
                pass
    for col in ['quoted_status_id_str', 'tweet.quoted_status_id_str']:
        if col in row and pd.notna(row[col]):
            return True
    # Check entities for quote URLs
    entities = get_dict(row.get('entities')) or get_dict(row.get('tweet.entities.urls'))
    urls = entities.get('urls', []) or []
    for u in urls:
        expanded = (u.get('expanded_url') or '')
        if 'twitter.com' in expanded and '/status/' in expanded:
            return True
    return False


# --- Thread reconstruction ---
def reconstruct_threads(df: pd.DataFrame) -> pd.DataFrame:
    """Identify threads and position of each tweet within its thread.

    If the root tweet is not in the archive, the `thread_id` becomes the last known parent in the chain.
    """
    df_copy = df.copy()
    # Handle various id column names
    if 'id_str' not in df_copy.columns:
        if 'tweet.id_str' in df_copy.columns:
            df_copy['id_str'] = ensure_string_id(df_copy['tweet.id_str'])
        elif 'id' in df_copy.columns:
            df_copy['id_str'] = ensure_string_id(df_copy['id'])
        elif 'tweet.id' in df_copy.columns:
            df_copy['id_str'] = ensure_string_id(df_copy['tweet.id'])
        else:
            raise ValueError("No id column found")
    else:
        df_copy['id_str'] = ensure_string_id(df_copy['id_str'])

    # Handle reply column
    reply_col = 'in_reply_to_status_id_str'
    if reply_col not in df_copy.columns:
        if 'tweet.in_reply_to_status_id_str' in df_copy.columns:
            df_copy[reply_col] = df_copy['tweet.in_reply_to_status_id_str']
        else:
            df_copy[reply_col] = np.nan

    df_replying = df_copy[df_copy[reply_col].notna()].copy()
    # Parent map: child id -> parent id
    parent_map: Dict[str, str] = (
        df_replying.set_index('id_str')[reply_col].astype(str).to_dict()
    )

    thread_root_cache: Dict[str, str] = {}

    def find_thread_root(tweet_id: str) -> str:
        if tweet_id in thread_root_cache:
            return thread_root_cache[tweet_id]
        path: list[str] = [tweet_id]
        curr_id = tweet_id
        seen: set[str] = {tweet_id}
        while curr_id in parent_map:
            curr_id = parent_map[curr_id]
            if curr_id in seen:
                break
            path.append(curr_id)
            seen.add(curr_id)
        root_id = path[-1]
        for node in path:
            thread_root_cache[node] = root_id
        return root_id

    df_copy['thread_id'] = df_copy['id_str'].apply(find_thread_root)
    thread_groups = df_copy.sort_index().groupby('thread_id', sort=False)
    df_copy['thread_step_index'] = thread_groups.cumcount()
    df_copy['is_thread_starter'] = (df_copy['thread_step_index'] == 0)
    return df_copy


# --- Main feature engineering ---
def engineer_features(df: pd.DataFrame, my_user_id: Optional[str] = None) -> pd.DataFrame:
    """Apply feature engineering to raw tweet DataFrame.

    Returns a new DataFrame with detection flags, content features, time features, tier labels, and engagement metrics.
    """
    df_eng = df.copy()

    # Normalize timestamps first
    df_eng = normalize_datetime(df_eng)

    # Ensure id_str is string - handle nested columns
    if 'id_str' in df_eng.columns:
        df_eng['id_str'] = ensure_string_id(df_eng['id_str'])
    elif 'tweet.id_str' in df_eng.columns:
        df_eng['id_str'] = ensure_string_id(df_eng['tweet.id_str'])
    elif 'id' in df_eng.columns:
        df_eng['id_str'] = ensure_string_id(df_eng['id'])
    elif 'tweet.id' in df_eng.columns:
        df_eng['id_str'] = ensure_string_id(df_eng['tweet.id'])

    # Detection flags
    df_eng['is_retweet'] = df_eng.apply(detect_retweet, axis=1)
    df_eng['is_quote_tweet'] = df_eng.apply(detect_quote, axis=1)

    # Content features
    if 'full_text' in df_eng.columns:
        text_col = 'full_text'
    elif 'text' in df_eng.columns:
        text_col = 'text'
    elif 'tweet.full_text' in df_eng.columns:
        text_col = 'tweet.full_text'
    elif 'tweet.text' in df_eng.columns:
        text_col = 'tweet.text'
    else:
        # Find any column containing text
        text_candidates = [c for c in df_eng.columns if 'full_text' in str(c) or 'text' in str(c)]
        if text_candidates:
            text_col = text_candidates[0]
        else:
            raise ValueError("No text column found")
    
    df_eng[text_col] = df_eng[text_col].fillna("")
    df_eng['has_link'] = df_eng[text_col].str.contains(r'https://t.co/')

    def has_media_from_extended(x: Any) -> bool:
        if isinstance(x, dict):
            media = x.get('media', []) or []
            return len(media) > 0
        return False

    # Handle extended_entities with various column names
    if 'extended_entities' in df_eng.columns:
        df_eng['has_media'] = df_eng['extended_entities'].apply(has_media_from_extended)
    elif 'tweet.extended_entities.media' in df_eng.columns:
        # If we have nested media directly, check if it's not null
        df_eng['has_media'] = df_eng['tweet.extended_entities.media'].notna()
    else:
        df_eng['has_media'] = False
    
    # Handle entities
    if 'entities' in df_eng.columns:
        entities_series = df_eng['entities']
    else:
        # Use empty dict for missing entities
        entities_series = pd.Series([{}] * len(df_eng))
    df_eng['text_length_chars'] = df_eng[text_col].str.len()
    
    # Handle hashtags and mentions with nested columns
    if 'tweet.entities.hashtags' in df_eng.columns:
        df_eng['num_hashtags'] = df_eng['tweet.entities.hashtags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        df_eng['num_hashtags'] = entities_series.apply(lambda x: len(get_dict(x).get('hashtags', []) or []))
    
    if 'tweet.entities.user_mentions' in df_eng.columns:
        df_eng['num_mentions'] = df_eng['tweet.entities.user_mentions'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        df_eng['num_mentions'] = entities_series.apply(lambda x: len(get_dict(x).get('user_mentions', []) or []))
    
    df_eng['has_question_mark'] = df_eng[text_col].str.contains(r'\?')

    # Reply classification - handle nested columns
    if 'in_reply_to_user_id_str' in df_eng.columns:
        reply_user_col = 'in_reply_to_user_id_str'
    elif 'tweet.in_reply_to_user_id_str' in df_eng.columns:
        reply_user_col = 'tweet.in_reply_to_user_id_str'
    elif 'in_reply_to_user_id' in df_eng.columns:
        reply_user_col = 'in_reply_to_user_id'
    elif 'tweet.in_reply_to_user_id' in df_eng.columns:
        reply_user_col = 'tweet.in_reply_to_user_id'
    else:
        df_eng['reply_user_col_tmp'] = np.nan
        reply_user_col = 'reply_user_col_tmp'

    resolved_user_id = my_user_id or (None if MY_USER_ID == "YOUR_USER_ID_HERE" else MY_USER_ID)
    if resolved_user_id is None:
        try:
            resolved_user_id = infer_my_user_id(df_eng, username_hint=MY_SCREEN_NAME)
        except Exception:
            resolved_user_id = ""

    def classify_reply(row: pd.Series) -> str:
        val = row.get(reply_user_col)
        if pd.isna(val):
            return 'none'
        if str(val) == str(resolved_user_id) and resolved_user_id:
            return 'reply_own'
        return 'reply_other'

    df_eng['reply_type'] = df_eng.apply(classify_reply, axis=1)

    # Time features
    df_eng['weekday'] = df_eng.index.day_name()
    df_eng['hour_of_day'] = df_eng.index.hour
    df_eng['month'] = df_eng.index.to_period('M').astype(str)

    # Account tier assignment
    def assign_tier(ts: pd.Timestamp) -> str:
        # Ensure timezone-aware comparison
        ts = pd.Timestamp(ts)
        tier_start = pd.Timestamp(TIER_UPGRADED_START, tz=ts.tz if hasattr(ts, 'tz') else 'UTC')
        tier_end = pd.Timestamp(TIER_POST_UPGRADE_START, tz=ts.tz if hasattr(ts, 'tz') else 'UTC')
        
        if ts < tier_start:
            return 'tier_pre_upgrade'
        if tier_start <= ts <= tier_end:
            return 'tier_upgraded'
        return 'tier_post_upgrade'

    df_eng['account_tier'] = df_eng.index.to_series().apply(assign_tier)

    # Engagement metrics - handle nested columns
    if 'favorite_count' in df_eng.columns:
        df_eng['likes'] = pd.to_numeric(df_eng['favorite_count'], errors='coerce').fillna(0).astype(int)
    elif 'tweet.favorite_count' in df_eng.columns:
        df_eng['likes'] = pd.to_numeric(df_eng['tweet.favorite_count'], errors='coerce').fillna(0).astype(int)
    else:
        df_eng['likes'] = 0

    if 'retweet_count' in df_eng.columns:
        df_eng['retweets'] = pd.to_numeric(df_eng['retweet_count'], errors='coerce').fillna(0).astype(int)
    elif 'tweet.retweet_count' in df_eng.columns:
        df_eng['retweets'] = pd.to_numeric(df_eng['tweet.retweet_count'], errors='coerce').fillna(0).astype(int)
    else:
        df_eng['retweets'] = 0

    # Some archives do not include replies/bookmarks counts
    if 'reply_count' in df_eng.columns:
        df_eng['replies'] = pd.to_numeric(df_eng['reply_count'], errors='coerce').fillna(0).astype(int)
    else:
        df_eng['replies'] = 0
    
    if 'bookmark_count' in df_eng.columns:
        df_eng['bookmarks'] = pd.to_numeric(df_eng['bookmark_count'], errors='coerce').fillna(0).astype(int)
    else:
        df_eng['bookmarks'] = 0
    df_eng['total_engagement'] = df_eng[['likes', 'retweets', 'replies', 'bookmarks']].sum(axis=1)

    # Winsorize via quantile clipping to avoid masked arrays
    upper_q = df_eng['total_engagement'].quantile(WINSORIZE_THRESHOLD)
    df_eng['winsorized_engagement'] = df_eng['total_engagement'].clip(upper=upper_q)

    # Thread features
    df_eng = reconstruct_threads(df_eng)

    return df_eng


def create_core_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Return the core modeling sample per plan: exclude retweets and quote tweets; include none/reply_other."""
    mask = (
        (~df['is_retweet'])
        & (~df['is_quote_tweet'])
        & (df['reply_type'].isin(['none', 'reply_other']))
    )
    return df.loc[mask].copy()


