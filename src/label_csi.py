import pandas as pd
import argparse
import os
import logging

from module.logger import Logger

# Setup logging
Logger(log_file="label_csi.log")


class DatasetManager:
    """Handles operations on the unified dataset."""

    def __init__(self, output_csv):
        self.output_csv = output_csv

    def add_capture(self, input_csv, user, environment, position):
        """Add labeled capture to the dataset."""
        df_capture = pd.read_csv(input_csv)

        # Add labels to the capture
        df_capture['user'] = user
        df_capture['environment'] = environment
        df_capture['position'] = position

        # Check if the dataset file already exists
        if os.path.exists(self.output_csv):
            df_existing = pd.read_csv(self.output_csv)

            # Ensure df_existing is not empty before concatenation
            if not df_existing.empty:
                df_combined = pd.concat([df_existing, df_capture], ignore_index=True)
            else:
                df_combined = df_capture
        else:
            df_combined = df_capture

        # Save the updated dataset
        df_combined.to_csv(self.output_csv, index=False)
        logging.info(f"Capture added to dataset: {self.output_csv}")

    def remove_capture(self, user, environment, position):
        """Remove specific capture from the dataset."""
        if not os.path.exists(self.output_csv):
            logging.warning(f"Dataset not found: {self.output_csv}")
            return

        df_dataset = pd.read_csv(self.output_csv)
        df_filtered = df_dataset[
            ~((df_dataset['user'] == user) &
              (df_dataset['environment'] == environment) &
              (df_dataset['position'] == position))
        ]

        if len(df_filtered) == len(df_dataset):
            logging.warning(f"No matching capture found for user={user}, environment={environment}, position={position}.")
        else:
            df_filtered.to_csv(self.output_csv, index=False)
            logging.info(f"Capture removed from dataset: {self.output_csv}")

    def reset_dataset(self):
        """Reset the unified dataset by deleting the dataset file."""
        if os.path.exists(self.output_csv):
            os.remove(self.output_csv)
            logging.info(f"Dataset reset successfully: {self.output_csv}")
        else:
            logging.warning(f"No dataset found to reset: {self.output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Label CSI data and manage dataset')
    subparsers = parser.add_subparsers(dest='action', required=True)

    # Add capture
    add_parser = subparsers.add_parser('add', help='Add labeled capture to the dataset')
    add_parser.add_argument('-i', '--input', default='data/processed_csi.csv', help='Input CSV file (processed)')
    add_parser.add_argument('-u', '--user', required=True, help='User name')
    add_parser.add_argument('-e', '--environment', required=True, help='Capture environment')
    add_parser.add_argument('-p', '--position', required=True, help='Capture position')
    add_parser.add_argument('-o', '--output', default='dataset/dataset.csv', help='Unified dataset file')

    # Remove capture
    remove_parser = subparsers.add_parser('remove', help='Remove specific capture from the dataset')
    remove_parser.add_argument('-u', '--user', required=True, help='User name')
    remove_parser.add_argument('-e', '--environment', required=True, help='Capture environment')
    remove_parser.add_argument('-p', '--position', required=True, help='Capture position')
    remove_parser.add_argument('-o', '--output', default='dataset/dataset.csv', help='Unified dataset file')

    # Reset dataset
    reset_parser = subparsers.add_parser('reset', help='Reset the unified dataset')
    reset_parser.add_argument('-o', '--output', default='dataset/dataset.csv', help='Dataset file to reset')

    args = parser.parse_args()

    manager = DatasetManager(args.output)

    if args.action == 'add':
        manager.add_capture(args.input, args.user, args.environment, args.position)
    elif args.action == 'remove':
        manager.remove_capture(args.user, args.environment, args.position)
    elif args.action == 'reset':
        manager.reset_dataset()


if __name__ == "__main__":
    main()
