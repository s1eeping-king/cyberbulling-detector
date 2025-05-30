import torch
from transformers import pipeline
import json
import os
import time
from tqdm import tqdm

def classify_videos():
    """
    Classify video descriptions using the cardiffnlp/twitter-roberta-base-offensive model
    and save the results to a single JSON file in the text_classification directory.
    """
    print("Starting video description classification...")

    # Initialize the model
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

    sentiment_analyzer = pipeline("text-classification",
                                model="cardiffnlp/twitter-roberta-base-offensive",
                                device=device)

    # Create output directory if it doesn't exist
    output_dir = "data/processed/integration/text_classification"
    os.makedirs(output_dir, exist_ok=True)

    # Load media sessions from the dataset
    media_path = "data/processed/integration/media_sessions.json"
    if not os.path.exists(media_path):
        print(f"Media sessions file not found: {media_path}")
        return

    print(f"Loading media sessions from {media_path}...")
    with open(media_path, 'r', encoding='utf-8') as f:
        media_sessions = json.load(f)

    print(f"Loaded {len(media_sessions)} media sessions.")

    # Create a list to store all classification results
    all_classification_results = []

    # Process media sessions in batches to avoid memory issues (but save to a single file at the end)
    batch_size = 1000
    num_batches = (len(media_sessions) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(media_sessions))
        batch_media = media_sessions[start_idx:end_idx]

        print(f"Processing batch {batch_idx + 1}/{num_batches} (media sessions {start_idx} to {end_idx})...")

        # Process each media session with progress bar
        for media in tqdm(batch_media, desc=f"Batch {batch_idx + 1}", unit="video"):
            # Get the media ID (postId in the JSON file)
            media_id = media.get("postId", "")

            # Get the description directly from the media session
            description = str(media.get("description", ""))

            if not description:
                # Skip empty descriptions
                all_classification_results.append({
                    "postId": media_id,
                    "classification": "not_offensive",
                    "confidence": 1.0
                })
                continue

            try:
                # Classify the description
                result = sentiment_analyzer(description)[0]
                label = result["label"]
                score = result["score"]

                # The model returns "offensive" or "non-offensive" directly
                offensive = "offensive" if label == "offensive" else "not_offensive"

                # Add the result to the list
                all_classification_results.append({
                    "postId": media_id,
                    "classification": offensive,
                    "confidence": score
                })
            except Exception as e:
                print(f"Error processing media {media_id}: {e}")
                # Add a default result for failed media
                all_classification_results.append({
                    "postId": media_id,
                    "classification": "not_offensive",
                    "confidence": 0.0,
                    "error": str(e)
                })

    # Save all results to a single file
    output_file = os.path.join(output_dir, "video_classifications.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_classification_results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_classification_results)} classification results to {output_file}")

    # Create a summary
    offensive_count = sum(1 for result in all_classification_results if result["classification"] == "offensive")
    not_offensive_count = len(all_classification_results) - offensive_count

    summary = {
        "total_videos": len(all_classification_results),
        "offensive_videos": offensive_count,
        "not_offensive_videos": not_offensive_count,
        "offensive_percentage": offensive_count / len(all_classification_results) * 100 if all_classification_results else 0
    }

    # Save the summary
    summary_file = os.path.join(output_dir, "video_classification_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Created summary in {summary_file}")
    print(f"Summary: {offensive_count} offensive videos out of {len(all_classification_results)} ({summary['offensive_percentage']:.2f}%)")

    print("Classification complete!")

if __name__ == "__main__":
    print("Script started...")
    start_time = time.time()
    classify_videos()
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")