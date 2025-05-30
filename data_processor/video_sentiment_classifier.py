import torch
from transformers import pipeline
import json
import os
import time
from tqdm import tqdm

def classify_video_descriptions_sentiment():
    """
    Classify video descriptions using the cardiffnlp/twitter-roberta-base-sentiment-latest model
    and save the results to a single JSON file in the roberta_classification directory.
    """
    print("Starting video description sentiment classification...")

    # Initialize the model
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

    sentiment_analyzer = pipeline("sentiment-analysis",
                                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                device=device)

    # Create output directory if it doesn't exist
    output_dir = "data/processed/roberta_classification"
    os.makedirs(output_dir, exist_ok=True)

    # Load media sessions from the dataset
    media_sessions_path = "data/processed/integration/media_sessions.json"
    if not os.path.exists(media_sessions_path):
        print(f"Media sessions file not found: {media_sessions_path}")
        return

    print(f"Loading media sessions from {media_sessions_path}...")
    with open(media_sessions_path, 'r', encoding='utf-8') as f:
        media_sessions = json.load(f)

    print(f"Loaded {len(media_sessions)} media sessions.")

    # Create a list to store all classification results
    all_classification_results = []

    # Process media sessions in batches to avoid memory issues (but save to a single file at the end)
    batch_size = 1000
    num_batches = (len(media_sessions) + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(media_sessions))
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} (media sessions {batch_start} to {batch_end-1})...")
        
        # Process each media session in the current batch
        for i in tqdm(range(batch_start, batch_end), desc=f"Batch {batch_idx + 1}"):
            media_session = media_sessions[i]
            post_id = media_session.get("postId")
            description = media_session.get("description", "").strip()
            
            # Skip empty descriptions
            if not description:
                all_classification_results.append({
                    "postId": post_id,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "error": "Empty description"
                })
                continue
            
            try:
                # Classify the description
                result = sentiment_analyzer(description)[0]
                label = result["label"]
                score = result["score"]

                # Add the result to the list
                all_classification_results.append({
                    "postId": post_id,
                    "sentiment": label,
                    "confidence": score
                })
            except Exception as e:
                print(f"Error processing media session {post_id}: {e}")
                # Add a default result for failed media sessions
                all_classification_results.append({
                    "postId": post_id,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "error": str(e)
                })
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing completed in {processing_time:.2f} seconds.")
    
    # Save all results to a single file
    output_file = os.path.join(output_dir, "video_descriptions_sentiment.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_classification_results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    classify_video_descriptions_sentiment()
