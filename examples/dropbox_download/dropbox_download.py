"""
python dropbox_download.py --url DROPBOX_URL --save_dir SAVE_DIR

e.g.,
DROPBOX_URL: https://www.dropbox.com/sh/8hha489c31yqkjm/AABhV8h7UHwVoMj8KLS3weE5a/200data?dl=0&subfolder_nav_tracking=1
SAVE_DIR: RelitDataset/200data
"""
import argparse
import wah


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    downloader = wah.DropboxDownloader()
    downloader(args.url, args.save_dir)
