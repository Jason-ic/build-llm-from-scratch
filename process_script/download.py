import urllib.request
# url = ("https://raw.githubusercontent.com/rasbt/"
#  "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#  "the-verdict.txt")
# file_path = "/Users/youfangdajiankang/build-llm-from-scratch/embedding_text/the-verdict.txt"
# urllib.request.urlretrieve(url, file_path)

url = (
 "https://raw.githubusercontent.com/rasbt/"
 "LLMs-from-scratch/main/ch05/"
 "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)