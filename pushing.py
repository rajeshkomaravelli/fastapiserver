from huggingface_hub import upload_folder

# Replace with your actual Hugging Face username/repo name
REPO_ID = "rajesh500759/allorganism"

# Upload the entire local folder to the repo root ("/")
upload_folder(
    folder_path="C:/Users/saich/OneDrive/Desktop\project\generalized_app\output (2).txt",      # Local folder path
    path_in_repo="",                # Upload to repo root
    repo_id=REPO_ID,                # Your Hugging Face repo
    repo_type="model",              # Could also be 'dataset' or 'space'
    allow_patterns=["*.pkl", "*.json", "*.csv", "*.cbm"],  # Include only these files
)