import argparse
import sys
from training.trainer import Trainer
import os

def main():
    parser = argparse.ArgumentParser(description="Multimodal AI System for Aircraft Composite Damage")
    parser.add_argument('mode', choices=['train', 'ui'], help="Run mode: 'train' to start model training, 'ui' to launch the Streamlit frontend.")
    parser.add_argument('--config', default='configs/config.yaml', help="Path to config file.")
    parser.add_argument('--data_dir', default='data/processed', help="Path to preprocessed dataset for training.")

    args = parser.parse_args()

    if args.mode == 'train':
        print(f"Starting Training Process with config: {args.config}")
        trainer = Trainer(args.config)
        trainer.build_dataloaders(args.data_dir)
        trainer.run()
        print("Training completed.")
    elif args.mode == 'ui':
        print("Launching Streamlit UI...")
        # Start streamlit as a subprocess
        os.system("streamlit run ui/app.py")
        
if __name__ == "__main__":
    main()
