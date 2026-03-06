import requests
import json
import re
import os
import argparse
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove markdown quotes to deduplicate heavily quoted text segments
    text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
    # Remove HTML-style review comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Strip excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def fetch_github_issues(repo: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetches issues/PRs from a public GitHub repository.
    Handles basic pagination up to `limit` items.
    """
    if repo.startswith("http"):
        repo = repo.replace("https://github.com/", "").replace("http://github.com/", "")
    repo = repo.strip("/")
    
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": "all", "per_page": min(100, limit)}
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    print(f"Fetching up to {limit} issues from {repo}...")
    issues = []
    
    while url and len(issues) < limit:
        response = requests.get(url, params=params if not issues else None, headers=headers)
        response.raise_for_status()
        
        page_data = response.json()
        if not page_data:
            break
            
        issues.extend(page_data)
        
        # Check for next page in 'Link' header
        link_header = response.headers.get("Link", "")
        next_link = None
        if link_header:
            links = link_header.split(',')
            for link in links:
                if 'rel="next"' in link:
                    next_link = link[link.find('<')+1:link.find('>')]
                    break
        url = next_link
        
    return issues[:limit]

def normalize_and_deduplicate(raw_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes issues into a standard schema and removes exact duplicates
    or empty bodies.
    """
    seen_texts = set()
    normalized = []
    
    for issue in raw_issues:
        body = clean_text(issue.get("body", ""))
        
        # Skip empty items or items we've already seen perfectly matched
        if not body or body in seen_texts:
            continue
            
        seen_texts.add(body)
        
        # Determine source type
        is_pr = "pull_request" in issue
        source_type = "pull_request" if is_pr else "issue"
        
        normalized.append({
            "source_id": f"github_{source_type}_{issue['id']}",
            "text": body,
            "timestamp": issue.get("created_at", ""),
            "author": issue.get("user", {}).get("login", "unknown"),
            "metadata": {
                "title": issue.get("title", ""),
                "state": issue.get("state", ""),
                "url": issue.get("html_url", ""),
                "labels": [label["name"] for label in issue.get("labels", [])],
                "type": source_type
            }
        })
        
    return normalized

def main():
    parser = argparse.ArgumentParser(description="Ingest Phase: Download and normalize a public text corpus.")
    parser.add_argument("--repo", type=str, default="tiangolo/fastapi", help="GitHub repo to fetch issues from (e.g., owner/repo)")
    parser.add_argument("--limit", type=int, default=100, help="Max number of items to fetch")
    parser.add_argument("--output", type=str, default="data/corpus.jsonl", help="Output JSONL path")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    raw_data = fetch_github_issues(args.repo, args.limit)
    print(f"Fetched {len(raw_data)} raw items.")
    
    normalized_data = normalize_and_deduplicate(raw_data)
    print(f"After deduplication and filtering: {len(normalized_data)} items.")
    
    with open(args.output, "w", encoding="utf-8") as f:
        for item in normalized_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Successfully saved to {args.output}")

if __name__ == "__main__":
    main()
