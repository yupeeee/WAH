import os
from tqdm import tqdm

from ..typing import Path
from ..utils.download_from_url import urlretrieve
from ..utils.zip import extract
from .driver import ChromeDriver

__all__ = [
    "DropboxDownloader",
]


class DropboxDownloader(ChromeDriver):
    def __init__(
        self,
        time_to_wait: int = 60,
        time_to_sleep: int = 10,
    ) -> None:
        super().__init__(time_to_wait, time_to_sleep)

        self.folder_name: str

    def __call__(
        self,
        url: Path,
        save_dir: Path,
    ) -> None:
        self.go(url=url)

        self.folder_name = self.get_text(
            find_element_by="XPATH",
            element_value='//*[@id="embedded-app"]/div/div/div/div/div[2]/div/div[2]/div[1]/div/div/nav/div/span/span/span/span/h1',
        )
        self.wait()

        # fetching all valid download urls
        elements = self.get_elements(
            find_elements_by="TAG_NAME",
            elements_value="a",
        )

        urls = []

        for element in elements:
            href = element.get_attribute("href")

            if "?dl=0" in href:
                urls.append(href.replace("?dl=0", "?dl=1"))

        # Error: no files found
        if len(urls) == 0:
            raise FileNotFoundError("No files to download.")

        # Proceed download
        save_dir = os.path.normpath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        disable = True if len(urls) > 1 else False

        for url in tqdm(
            urls,
            desc=f"Downloading {self.folder_name} to {save_dir}",
        ):
            fname = os.path.basename(url.replace("?dl=1", ""))

            # if file is folder, download as zip
            is_folder = False

            if len(os.path.splitext(fname)[-1]) == 0:
                is_folder = True
                fname += ".zip"

            save_path = os.path.join(save_dir, fname)
            urlretrieve(
                url=url,
                fpath=save_path,
                disable=disable,
            )

            # extract .zip files (folders)
            if is_folder:
                folder_name = os.path.splitext(fname)[0]
                extract(
                    save_path,
                    save_dir=os.path.join(save_dir, folder_name),
                    mode="r",
                )
                os.remove(save_path)

        self.close_driver()
