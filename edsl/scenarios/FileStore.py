import base64
import io
import tempfile
import mimetypes
import os
from typing import Dict, Any, IO, Optional

from edsl.scenarios.Scenario import Scenario
from edsl.utilities.remove_edsl_version import remove_edsl_version

from edsl.scenarios.file_methods import FileMethods
from typing import Dict, Any, IO, Optional, Literal
import requests
from urllib.parse import urlparse
import time
import subprocess
import google.generativeai as genai

from edsl import Scenario
from edsl.utilities.decorators import add_edsl_version, remove_edsl_version
from edsl.utilities.utilities import is_notebook


def view_docx(docx_path):
    import os
    import subprocess
    from IPython.display import HTML, display
    import base64

    if is_notebook():
        # For Jupyter notebooks, we'll need to convert DOCX to HTML
        # since browsers can't directly display DOCX files
        try:
            import mammoth

            with open(docx_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html = f"""
                <div style="width: 800px; height: 800px; padding: 20px; 
                           border: 1px solid #ccc; overflow-y: auto;">
                    {result.value}
                </div>
                """
                display(HTML(html))
        except ImportError:
            print("Please install mammoth for notebook preview: pip install mammoth")
        finally:
            return

    if os.path.exists(docx_path):
        try:
            if (os_name := os.name) == "posix":
                subprocess.run(["open", docx_path], check=True)  # macOS
            elif os_name == "nt":
                os.startfile(docx_path)  # Windows
            else:
                subprocess.run(["xdg-open", docx_path], check=True)  # Linux
        except Exception as e:
            print(f"Error opening DOCX: {e}")
    else:
        print("DOCX file was not found.")


def view_csv(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    return df


def view_html(html_path):
    import os
    import subprocess
    from IPython.display import IFrame, display, HTML

    if os.path.exists(html_path):
        if is_notebook():
            # Display the HTML inline in Jupyter Notebook
            display(IFrame(src=html_path, width=700, height=600))
            display(
                HTML(
                    f'<a href="{html_path}" target="_blank">Open HTML in a new tab</a>'
                )
            )
        else:
            try:
                if (os_name := os.name) == "posix":
                    # Open with the default browser on macOS
                    subprocess.run(["open", html_path], check=True)
                elif os_name == "nt":
                    # Open with the default browser on Windows
                    os.startfile(html_path)
                else:
                    # Open with the default browser on Linux
                    subprocess.run(["xdg-open", html_path], check=True)
            except Exception as e:
                print(f"Error opening HTML file: {e}")
    else:
        print("HTML file was not found.")


def view_html(html_path):
    import os
    from IPython.display import display, HTML

    if is_notebook():
        with open(html_path, "r") as f:
            html_content = f.read()
        display(HTML(html_content))
    else:
        if os.path.exists(html_path):
            try:
                if (os_name := os.name) == "posix":
                    subprocess.run(["open", html_path], check=True)
                elif os_name == "nt":
                    os.startfile(html_path)
                else:
                    subprocess.run(["xdg-open", html_path], check=True)
            except Exception as e:
                print(f"Error opening file: {e}")
        else:
            print("File was not created successfully.")


def view_pdf(pdf_path):
    import os
    import subprocess
    import os
    from IPython.display import HTML, display

    if is_notebook():
        # Convert to absolute path if needed
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        html = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="800px"
            height="800px"
            type="application/pdf"
        ></iframe>
        """
        display(HTML(html))
        return

    if os.path.exists(pdf_path):
        try:
            if (os_name := os.name) == "posix":
                # for cool kids
                subprocess.run(["open", pdf_path], check=True)  # macOS
            elif os_name == "nt":
                os.startfile(pdf_path)  # Windows
            else:
                subprocess.run(["xdg-open", pdf_path], check=True)  # Linux
        except Exception as e:
            print(f"Error opening PDF: {e}")
    else:
        print("PDF file was not created successfully.")


class FileStore(Scenario):
    __documentation__ = "https://docs.expectedparrot.com/en/latest/filestore.html"

    def __init__(
        self,
        path: Optional[str] = None,
        mime_type: Optional[str] = None,
        binary: Optional[bool] = None,
        suffix: Optional[str] = None,
        base64_string: Optional[str] = None,
        external_locations: Optional[Dict[str, str]] = None,
        extracted_text: Optional[str] = None,
        **kwargs,
    ):
        if path is None and "filename" in kwargs:
            path = kwargs["filename"]

        self._path = path  # Store the original path privately
        self._temp_path = None  # Track any generated temporary file

        self.suffix = suffix or path.split(".")[-1]
        self.binary = binary or False
        self.mime_type = (
            mime_type or mimetypes.guess_type(path)[0] or "application/octet-stream"
        )
        self.base64_string = base64_string or self.encode_file_to_base64_string(path)
        self.external_locations = external_locations or {}

        self.extracted_text = (
            self.extract_text() if extracted_text is None else extracted_text
        )

        super().__init__(
            {
                "path": path,
                "base64_string": self.base64_string,
                "binary": self.binary,
                "suffix": self.suffix,
                "mime_type": self.mime_type,
                "external_locations": self.external_locations,
                "extracted_text": self.extracted_text,
            }
        )

    @property
    def path(self) -> str:
        """
        Property that returns a valid path to the file content.
        If the original path doesn't exist, generates a temporary file from the base64 content.
        """
        # Check if original path exists and is accessible
        if self._path and os.path.isfile(self._path):
            return self._path

        # If we already have a valid temporary file, use it
        if self._temp_path and os.path.isfile(self._temp_path):
            return self._temp_path

        # Generate a new temporary file from base64 content
        self._temp_path = self.to_tempfile(self.suffix)
        return self._temp_path

    def __str__(self):
        return "FileStore: self.path"

    @classmethod
    def example(cls, example_type="txt"):
        file_methods_class = FileMethods.get_handler(example_type)
        if file_methods_class:
            return cls(file_methods_class().example())
        else:
            print(f"Example for {example_type} is not supported.")

    @property
    def size(self) -> int:
        if self.base64_string != None:
            return (len(self.base64_string) / 4.0) * 3  # from base64 to char size
        return os.path.getsize(self.path)

    def upload_google(self, refresh: bool = False) -> None:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        google_info = genai.upload_file(self.path, mime_type=self.mime_type)
        self.external_locations["google"] = google_info.to_dict()

    @classmethod
    @remove_edsl_version
    def from_dict(cls, d):
        # return cls(d["filename"], d["binary"], d["suffix"], d["base64_string"])
        return cls(**d)

    def __repr__(self):
        import reprlib

        r = reprlib.Repr()
        r.maxstring = 20  # Limit strings to 20 chars
        r.maxother = 30  # Limit other types to 30 chars

        params = ", ".join(f"{key}={r.repr(value)}" for key, value in self.data.items())
        return f"{self.__class__.__name__}({params})"

    def _repr_html_(self):
        parent_html = super()._repr_html_()
        from edsl.scenarios.ConstructDownloadLink import ConstructDownloadLink

        link = ConstructDownloadLink(self).html_create_link(self.path, style=None)
        return f"{parent_html}<br>{link}"

    def encode_file_to_base64_string(self, file_path: str):
        try:
            # Attempt to open the file in text mode
            with open(file_path, "r") as text_file:
                # Read the text data
                text_data = text_file.read()
                # Encode the text data to a base64 string
                base64_encoded_data = base64.b64encode(text_data.encode("utf-8"))
        except UnicodeDecodeError:
            # If reading as text fails, open the file in binary mode
            with open(file_path, "rb") as binary_file:
                # Read the binary data
                binary_data = binary_file.read()
                # Encode the binary data to a base64 string
                base64_encoded_data = base64.b64encode(binary_data)
                self.binary = True
        # Convert the base64 bytes to a string
        base64_string = base64_encoded_data.decode("utf-8")

        return base64_string

    def open(self) -> "IO":
        if self.binary:
            return self.base64_to_file(self.base64_string, is_binary=True)
        else:
            return self.base64_to_text_file(self.base64_string)

    def write(self, filename: Optional[str] = None) -> str:
        """
        Write the file content to disk, either to a specified filename or a temporary file.

        Args:
            filename (Optional[str]): The destination filename. If None, creates a temporary file.

        Returns:
            str: The path to the written file.
        """
        # Determine the mode based on binary flag
        mode = "wb" if self.binary else "w"

        # If no filename provided, create a temporary file
        if filename is None:
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(delete=False, suffix="." + self.suffix) as f:
                filename = f.name

        # Write the content using the appropriate mode
        try:
            with open(filename, mode) as f:
                content = self.open().read()
                # For text mode, ensure we're writing a string
                if not self.binary and isinstance(content, bytes):
                    content = content.decode("utf-8")
                f.write(content)
                print(f"File written to {filename}")
        except Exception as e:
            print(f"Error writing file: {e}")
            raise

        # return filename

    @staticmethod
    def base64_to_text_file(base64_string) -> "IO":
        # Decode the base64 string to bytes
        text_data_bytes = base64.b64decode(base64_string)

        # Convert bytes to string
        text_data = text_data_bytes.decode("utf-8")

        # Create a StringIO object from the text data
        text_file = io.StringIO(text_data)

        return text_file

    @staticmethod
    def base64_to_file(base64_string, is_binary=True):
        # Decode the base64 string to bytes
        file_data = base64.b64decode(base64_string)

        if is_binary:
            # Create a BytesIO object for binary data
            return io.BytesIO(file_data)
        else:
            # Convert bytes to string for text data
            text_data = file_data.decode("utf-8")
            # Create a StringIO object for text data
            return io.StringIO(text_data)

    @property
    def text(self):
        if self.binary:
            import warnings

            warnings.warn("This is a binary file.")
        else:
            return self.base64_to_text_file(self.base64_string).read()

    def to_tempfile(self, suffix=None):
        if suffix is None:
            suffix = self.suffix
        if self.binary:
            file_like_object = self.base64_to_file(
                self["base64_string"], is_binary=True
            )
        else:
            file_like_object = self.base64_to_text_file(self.base64_string)

        # Create a named temporary file
        mode = "wb" if self.binary else "w"
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix="." + suffix, mode=mode
        )

        if self.binary:
            temp_file.write(file_like_object.read())
        else:
            temp_file.write(file_like_object.read())

        temp_file.close()

        return temp_file.name

    def view(self) -> None:
        handler = FileMethods.get_handler(self.suffix)
        if handler:
            handler(self.path).view()
        else:
            print(f"Viewing of {self.suffix} files is not supported.")

    def extract_text(self) -> str:
        handler = FileMethods.get_handler(self.suffix)
        if handler and hasattr(handler, "extract_text"):
            return handler(self.path).extract_text()

        if not self.binary:
            return self.text

        return None
        # raise TypeError("No text method found for this file type.")

    def push(
        self, description: Optional[str] = None, visibility: str = "unlisted"
    ) -> dict:
        """
        Push the object to Coop.
        :param description: The description of the object to push.
        :param visibility: The visibility of the object to push.
        """
        scenario_version = Scenario.from_dict(self.to_dict())
        if description is None:
            description = "File: " + self.path
        info = scenario_version.push(description=description, visibility=visibility)
        return info

    @classmethod
    def pull(cls, uuid: str, expected_parrot_url: Optional[str] = None) -> "FileStore":
        """
        :param uuid: The UUID of the object to pull.
        :param expected_parrot_url: The URL of the Parrot server to use.
        :return: The object pulled from the Parrot server.
        """
        scenario_version = Scenario.pull(uuid, expected_parrot_url=expected_parrot_url)
        return cls.from_dict(scenario_version.to_dict())

    @classmethod
    def from_url(
        cls,
        url: str,
        download_path: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> "FileStore":
        """
        :param url: The URL of the file to download.
        :param download_path: The path to save the downloaded file.
        :param mime_type: The MIME type of the file. If None, it will be guessed from the file extension.
        """
        import requests
        from urllib.parse import urlparse

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # Get the filename from the URL if download_path is not provided
        if download_path is None:
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = "downloaded_file"
            # download_path = filename
            download_path = os.path.join(os.getcwd(), filename)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(download_path), exist_ok=True)

        # Write the file
        with open(download_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Create and return a new File instance
        return cls(download_path, mime_type=mime_type)

    def create_link(self, custom_filename=None, style=None):
        from edsl.scenarios.ConstructDownloadLink import ConstructDownloadLink

        return ConstructDownloadLink(self).create_link(custom_filename, style)
      
    @classmethod
    def from_url_screenshot(
        cls,
        url: str,
        full_page: bool = True,
        wait_until: Literal[
            "load", "domcontentloaded", "networkidle", "commit"
        ] = "networkidle",
        download_path: Optional[str] = None,
    ) -> "FileStore":
        """
        Takes a screenshot of a webpage and returns it as a FileStore object.

        :param url: The URL of the webpage to screenshot
        :param full_page: Whether to capture the full scrollable page or just the viewport
        :param wait_until: When to consider the page load finished. Options:
            - 'load': consider load finished when the load event is fired
            - 'domcontentloaded': consider load finished when the DOMContentLoaded event is fired
            - 'networkidle': consider load finished when there are no network connections for at least 500ms
            - 'commit': consider load finished when network response is received and the document started loading
        :param download_path: Optional path where to save the screenshot. If None, a temporary file will be used.
        :return: FileStore instance containing the screenshot
        """
        try:
            from playwright.async_api import async_playwright
            import asyncio
        except ImportError:
            raise ImportError(
                "Screenshot functionality requires additional dependencies.\n"
                "Install them with: pip install 'edsl[screenshot]'"
            )

        # Generate filename if not provided
        if download_path is None:
            download_path = os.path.join(
                os.getcwd(), f"screenshot_{int(time.time())}.png"
            )

        async def _take_screenshot():
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url, wait_until=wait_until)
                await page.screenshot(path=download_path, full_page=full_page)
                await browser.close()

        # Create a new event loop and run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_take_screenshot())
        loop.close()

        return cls(download_path, mime_type="image/png")


class CSVFileStore(FileStore):
    @classmethod
    def example(cls):
        from edsl.results.Results import Results

        r = Results.example()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            r.to_csv(filename=f.name)

        return cls(f.name)

    def view(self):
        import pandas as pd

        return pd.read_csv(self.to_tempfile())


class PDFFileStore(FileStore):
    def view(self):
        pdf_path = self.to_tempfile()
        print(f"PDF path: {pdf_path}")  # Print the path to ensure it exists
        import os
        import subprocess

        if os.path.exists(pdf_path):
            try:
                if os.name == "posix":
                    # for cool kids
                    subprocess.run(["open", pdf_path], check=True)  # macOS
                elif os.name == "nt":
                    os.startfile(pdf_path)  # Windows
                else:
                    subprocess.run(["xdg-open", pdf_path], check=True)  # Linux
            except Exception as e:
                print(f"Error opening PDF: {e}")
        else:
            print("PDF file was not created successfully.")

    @classmethod
    def example(cls):
        import textwrap

        pdf_string = textwrap.dedent(
            """\
        %PDF-1.4
        1 0 obj
        << /Type /Catalog /Pages 2 0 R >>
        endobj
        2 0 obj
        << /Type /Pages /Kids [3 0 R] /Count 1 >>
        endobj
        3 0 obj
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
        endobj
        4 0 obj
        << /Length 44 >>
        stream
        BT
        /F1 24 Tf
        100 700 Td
        (Hello, World!) Tj
        ET
        endstream
        endobj
        5 0 obj
        << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
        endobj
        6 0 obj
        << /ProcSet [/PDF /Text] /Font << /F1 5 0 R >> >>
        endobj
        xref
        0 7
        0000000000 65535 f 
        0000000010 00000 n 
        0000000053 00000 n 
        0000000100 00000 n 
        0000000173 00000 n 
        0000000232 00000 n 
        0000000272 00000 n 
        trailer
        << /Size 7 /Root 1 0 R >>
        startxref
        318
        %%EOF"""
        )
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_string.encode())

        return cls(f.name)


class PNGFileStore(FileStore):
    @classmethod
    def example(cls):
        import textwrap

        png_string = textwrap.dedent(
            """\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00\x00\x00\x01\x00\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDAT\x08\xd7c\x00\x01"""
        )
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_string.encode())

        return cls(f.name)

    def view(self):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img = mpimg.imread(self.to_tempfile())
        plt.imshow(img)
        plt.show()


class SQLiteFileStore(FileStore):
    @classmethod
    def example(cls):
        import sqlite3
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            conn = sqlite3.connect(f.name)
            c = conn.cursor()
            c.execute("""CREATE TABLE stocks (date text)""")
            conn.commit()

            return cls(f.name)

    def view(self):
        import subprocess
        import os

        sqlite_path = self.to_tempfile()
        os.system(f"sqlite3 {sqlite_path}")


class HTMLFileStore(FileStore):
    @classmethod
    def example(cls):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write("<html><body><h1>Test</h1></body></html>".encode())

        return cls(f.name)

    def view(self):
        import webbrowser

        html_path = self.to_tempfile()
        webbrowser.open("file://" + html_path)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # fs = FileStore.example("pdf")
    # fs.view()

    formats = FileMethods.supported_file_types()
    for file_type in formats:
        print("Now testinging", file_type)
        fs = FileStore.example(file_type)
        fs.view()
        input("Press Enter to continue...")

    # pdf_example.view()
    # FileStore(pdf_example).view()

    # pdf_methods = methods.get("pdf")
    # file = pdf_methods().example()
    # pdf_methods(file).view()

    # print(FileMethods._handlers)
