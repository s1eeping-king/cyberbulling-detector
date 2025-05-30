import torch
from transformers import pipeline
import json
import os
import time
from tqdm import tqdm

def classify_users():
    """
    Classify user descriptions using the cardiffnlp/twitter-roberta-base-offensive model
    and save the results to a single JSON file in the text_classification directory.
    """
    print("Starting user description classification...")

    # Initialize the model
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

    sentiment_analyzer = pipeline("text-classification",
                                model="cardiffnlp/twitter-roberta-base-offensive",
                                device=device)

    # Create output directory if it doesn't exist
    output_dir = "data/processed/integration/text_classification"
    os.makedirs(output_dir, exist_ok=True)

    # Load users from the dataset
    users_path = "data/processed/integration/users.json"
    if not os.path.exists(users_path):
        print(f"Users file not found: {users_path}")
        return

    print(f"Loading users from {users_path}...")
    with open(users_path, 'r', encoding='utf-8') as f:
        users = json.load(f)

    print(f"Loaded {len(users)} users.")

    # Create a list to store all classification results
    all_classification_results = []

    # Process users in batches to avoid memory issues (but save to a single file at the end)
    batch_size = 1000
    num_batches = (len(users) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(users))
        batch_users = users[start_idx:end_idx]

        print(f"Processing batch {batch_idx + 1}/{num_batches} (users {start_idx} to {end_idx})...")

        # Process each user with progress bar
        for user in tqdm(batch_users, desc=f"Batch {batch_idx + 1}", unit="user"):
            # Get the user ID
            user_id = user.get("userId", "")

            # Get the description directly from the user
            description = str(user.get("description", ""))

            if not description:
                # Skip empty descriptions
                all_classification_results.append({
                    "userId": user_id,
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
                    "userId": user_id,
                    "classification": offensive,
                    "confidence": score
                })
            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                # Add a default result for failed users
                all_classification_results.append({
                    "userId": user_id,
                    "classification": "not_offensive",
                    "confidence": 0.0,
                    "error": str(e)
                })

    # Save all results to a single file
    output_file = os.path.join(output_dir, "user_classifications.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_classification_results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_classification_results)} classification results to {output_file}")

    # Create a summary
    offensive_count = sum(1 for result in all_classification_results if result["classification"] == "offensive")
    not_offensive_count = len(all_classification_results) - offensive_count

    summary = {
        "total_users": len(all_classification_results),
        "offensive_users": offensive_count,
        "not_offensive_users": not_offensive_count,
        "offensive_percentage": offensive_count / len(all_classification_results) * 100 if all_classification_results else 0
    }

    # Save the summary
    summary_file = os.path.join(output_dir, "user_classification_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Created summary in {summary_file}")
    print(f"Summary: {offensive_count} offensive users out of {len(all_classification_results)} ({summary['offensive_percentage']:.2f}%)")

    print("Classification complete!")

if __name__ == "__main__":
    print("Script started...")
    start_time = time.time()
    classify_users()
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")