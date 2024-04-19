import wah

url = "https://www.dropbox.com/sh/8hha489c31yqkjm/AABhV8h7UHwVoMj8KLS3weE5a/200data?dl=0&subfolder_nav_tracking=1"
save_dir = "RelitDataset/200data"


if __name__ == "__main__":
    downloader = wah.DropboxDownloader()
    downloader(url, save_dir)
