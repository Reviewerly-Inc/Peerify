import re
import os

__all__ = [
    "transform_mmd_to_md",
    "remove_long_repeats",
    "remove_neurips_checklist",
    "clean_text_remove_long_repeats",
    "process_neurips_files",
]


def transform_mmd_to_md(input_dir: str, output_dir: str = None) -> None:
    if output_dir is None:
        output_dir = input_dir
    if output_dir != input_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".mmd"):
            old_path = os.path.join(input_dir, filename)
            base, _ = os.path.splitext(filename)
            new_path = os.path.join(output_dir, base + ".md")

            with open(old_path, "r", encoding="utf-8") as file:
                content = file.read()

            with open(new_path, "w", encoding="utf-8") as file:
                file.write(content.strip())

            if output_dir == input_dir:
                os.remove(old_path)


def clean_text_remove_long_repeats(text: str, char_thresh: int, punct_seq_thresh: int) -> str:
    text = re.sub(r"(.)\1{" + str(char_thresh) + r",}", lambda m: m.group(1) * char_thresh, text)
    text = re.sub(r"([^\w\s])\1{" + str(punct_seq_thresh) + r",}", lambda m: m.group(1) * punct_seq_thresh, text)
    return text


def remove_long_repeats(input_dir: str, output_dir: str = None,
                        char_thresh: int = 10, punct_seq_thresh: int = 10) -> None:
    if output_dir is None:
        output_dir = input_dir
    if output_dir != input_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            clean_text = clean_text_remove_long_repeats(content, char_thresh, punct_seq_thresh)
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(clean_text)


def remove_neurips_checklist(text: str) -> str:
    keyword = "NeurIPS Paper Checklist"
    index = text.find(keyword)
    if index != -1:
        return text[:index].rstrip()
    return text


def process_neurips_files(input_dir: str, output_dir: str = None) -> None:
    if output_dir is None:
        output_dir = input_dir
    if output_dir != input_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".md"):
            input_path = os.path.join(input_dir, filename)
            with open(input_path, "r", encoding="utf-8") as file:
                content = file.read()
            clean_content = remove_neurips_checklist(content)
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(clean_content)
