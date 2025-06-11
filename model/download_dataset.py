#!/usr/bin/env python3
"""
Download fake news datasets from Kaggle
"""

import os
import pandas as pd
import zipfile
import requests
from pathlib import Path

def setup_kaggle_credentials():
    """
    Check if Kaggle credentials are available
    """
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("⚠️  Kaggle credentials not found!")
        print("📥 To use Kaggle datasets, you need to:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place it in ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        print()
        return False
    
    return True

def download_fake_news_dataset():
    """
    Download a comprehensive fake news dataset
    """
    try:
        # Try to import kaggle
        import kaggle
        
        print("📥 Downloading fake news dataset from Kaggle...")
        
        # Download the "Fake and real news dataset"
        dataset_name = "clmentbisaillon/fake-and-real-news-dataset"
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name, 
            path=str(data_dir), 
            unzip=True
        )
        
        print("✅ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        return False

def download_backup_dataset():
    """
    Download a backup dataset if Kaggle isn't available
    """
    print("📥 Downloading backup dataset...")
    
    # Use a publicly available fake news dataset
    url = "https://raw.githubusercontent.com/KhaiVo0605/Fake_News_Detection/main/data/train.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        with open(data_dir / "train.csv", "w", encoding='utf-8') as f:
            f.write(response.text)
        
        print("✅ Backup dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading backup dataset: {e}")
        return False

def process_dataset():
    """
    Process the downloaded dataset into our format
    """
    data_dir = Path("data")
    
    # Look for different possible dataset files
    possible_files = [
        "Fake.csv",
        "True.csv", 
        "train.csv",
        "fake_or_real_news.csv"
    ]
    
    found_files = []
    for file in possible_files:
        if (data_dir / file).exists():
            found_files.append(file)
    
    if not found_files:
        print("❌ No dataset files found!")
        return False
    
    print(f"📊 Found dataset files: {found_files}")
    
    try:
        # If we have separate Fake.csv and True.csv files
        if "Fake.csv" in found_files and "True.csv" in found_files:
            print("🔄 Processing Fake.csv and True.csv...")
            
            fake_df = pd.read_csv(data_dir / "Fake.csv")
            true_df = pd.read_csv(data_dir / "True.csv")
            
            # Add labels
            fake_df['label'] = 'FAKE'
            true_df['label'] = 'REAL'
            
            # Combine datasets
            combined_df = pd.concat([fake_df, true_df], ignore_index=True)
            
            # Standardize columns
            if 'title' in combined_df.columns and 'text' in combined_df.columns:
                final_df = combined_df[['title', 'text', 'label']].copy()
            elif 'title' in combined_df.columns:
                combined_df['text'] = combined_df['title']
                final_df = combined_df[['title', 'text', 'label']].copy()
            else:
                print("❌ Required columns not found!")
                return False
                
        # If we have a single train.csv file
        elif "train.csv" in found_files:
            print("🔄 Processing train.csv...")
            
            df = pd.read_csv(data_dir / "train.csv")
            
            # Check column names and adapt
            if 'label' in df.columns:
                # Map numeric labels to text
                if df['label'].dtype in ['int64', 'float64']:
                    df['label'] = df['label'].map({0: 'REAL', 1: 'FAKE'})
                
                # Ensure we have required columns
                if 'title' not in df.columns and 'text' in df.columns:
                    df['title'] = df['text'].str[:50] + "..."
                elif 'text' not in df.columns and 'title' in df.columns:
                    df['text'] = df['title']
                
                final_df = df[['title', 'text', 'label']].copy()
            else:
                print("❌ Label column not found!")
                return False
        
        else:
            print("❌ Unknown dataset format!")
            return False
        
        # Clean and sample the data
        final_df = final_df.dropna()
        final_df = final_df[final_df['text'].str.len() > 10]  # Remove very short texts
        
        # Balance the dataset and sample for faster training
        fake_samples = final_df[final_df['label'] == 'FAKE'].sample(n=min(1000, len(final_df[final_df['label'] == 'FAKE'])), random_state=42)
        real_samples = final_df[final_df['label'] == 'REAL'].sample(n=min(1000, len(final_df[final_df['label'] == 'REAL'])), random_state=42)
        
        balanced_df = pd.concat([fake_samples, real_samples], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        # Save processed dataset
        output_file = data_dir / "fake_or_real_news.csv"
        balanced_df.to_csv(output_file, index=False)
        
        print(f"✅ Processed dataset saved to {output_file}")
        print(f"📊 Total samples: {len(balanced_df)}")
        print(f"📊 Real news: {len(balanced_df[balanced_df['label'] == 'REAL'])}")
        print(f"📊 Fake news: {len(balanced_df[balanced_df['label'] == 'FAKE'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return False

def main():
    """
    Main function to download and process dataset
    """
    print("🔄 Fake News Dataset Downloader")
    print("=" * 40)
    
    # Try Kaggle first
    if setup_kaggle_credentials():
        if download_fake_news_dataset():
            if process_dataset():
                print("✅ Kaggle dataset ready for training!")
                return True
    
    # Fallback to backup dataset
    print("\n🔄 Trying backup dataset...")
    if download_backup_dataset():
        if process_dataset():
            print("✅ Backup dataset ready for training!")
            return True
    
    print("❌ Failed to download any dataset!")
    return False

if __name__ == "__main__":
    main() 