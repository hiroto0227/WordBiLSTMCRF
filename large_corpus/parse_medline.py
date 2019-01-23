import argparse
import re
import os
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET

def parse_medline(input_path):
    abstracts = []
    with open(input_path) as f:
        for line in f:
            if re.search("<AbstractText>.*</AbstractText>", line):
                abstracts.append(re.sub("( *|\t*)</*AbstractText>", "", line))
    return abstracts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse PubMed.txt")
    parser.add_argument("--input-path", type=str, help="input path")
    parser.add_argument("--input-dir", type=str, help="input dir (must not input path and input dir at the same time)")
    parser.add_argument("--output-path", type=str, help="output path")
    opt = parser.parse_args()

    if opt.input_path:
        texts = parse_medline(opt.input_path)
    elif opt.input_dir:
        texts = []
        filenames = os.listdir(opt.input_dir)
        for filename in tqdm(filenames):
            print("\n{}".format(filename))
            texts.extend(parse_medline(os.path.join(opt.input_dir, filename)))

    with open(opt.output_path, "at", encoding="utf-8") as f:
        print(len(texts))
        f.write("".join(texts))
