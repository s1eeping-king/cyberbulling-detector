import torch
from transformers import pipeline
import json
import os
import time
from tqdm import tqdm

def classify_comments_sentiment():
    """
    Classify comments using the cardiffnlp/twitter-roberta-base-sentiment-latest model
    and save the results to a single JSON file in the roberta_classification directory.
    """
    print("Starting comment sentiment classification...")

    # Initialize the model
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

    sentiment_analyzer = pipeline("sentiment-analysis",
                                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                device=device)

    # Create output directory if it doesn't exist
    output_dir = "data/processed/roberta_classification"
    os.makedirs(output_dir, exist_ok=True)

    # Load comments from the dataset
    comments_path = "data/processed/integration/comments.json"
    if not os.path.exists(comments_path):
        print(f"Comments file not found: {comments_path}")
        return

    print(f"Loading comments from {comments_path}...")
    with open(comments_path, 'r', encoding='utf-8') as f:
        comments = json.load(f)

    print(f"Loaded {len(comments)} comments.")

    # Create a list to store all classification results
    all_classification_results = []

    # Process comments in batches to avoid memory issues (but save to a single file at the end)
    batch_size = 1000
    num_batches = (len(comments) + batch_size - 1) // batch_size

    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(comments))

        print(f"Processing batch {batch_idx + 1}/{num_batches} (comments {batch_start} to {batch_end-1})...")

        # Process each comment in the current batch
        for i in tqdm(range(batch_start, batch_end), desc=f"Batch {batch_idx + 1}"):
            comment = comments[i]
            comment_id = comment.get("commentId")
            text = comment.get("text", "").strip()

            # Skip empty comments
            if not text:
                all_classification_results.append({
                    "commentId": comment_id,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "error": "Empty text"
                })
                continue

            try:
                # Classify the comment
                result = sentiment_analyzer(text)[0]
                label = result["label"]
                score = result["score"]

                # Add the result to the list
                all_classification_results.append({
                    "commentId": comment_id,
                    "sentiment": label,
                    "confidence": score
                })
            except Exception as e:
                print(f"Error processing comment {comment_id}: {e}")
                # Add a default result for failed comments
                all_classification_results.append({
                    "commentId": comment_id,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "error": str(e)
                })

    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing completed in {processing_time:.2f} seconds.")

    # Save all results to a single file
    output_file = os.path.join(output_dir, "comments_sentiment.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_classification_results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    classify_comments_sentiment()
