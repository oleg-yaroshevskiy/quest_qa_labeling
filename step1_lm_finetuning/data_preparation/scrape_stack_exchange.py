import os
import xml.etree.cElementTree as et
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pandas as pd
import requests
import wget
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_urls(main_url, path_to_dump):
    r = requests.get(main_url)
    soup = BeautifulSoup(r.content, "html.parser")

    listing_table = soup.find("table", class_="directory-listing-table")
    links = listing_table.findAll("a", href=True)

    # Write all links to a file
    link_list = "\n".join([main_url + l["href"] for l in links[1:]])
    with open(path_to_dump / "link_list.txt", "w") as f:
        f.write(link_list)

    return links


def download_and_unzip_data(main_url, links, path_to_dump):
    for link in tqdm(links):
        filename = str(link["href"])
        if filename[-1] != "/":
            print(f"Downloading {filename}...")
            url = main_url + filename
            filename = wget.download(url, out=str(path_to_dump))

            print(f"7z e {filename} -o{filename.rstrip('.7z')}")
            os.system(f"7z e {filename} -o{filename.rstrip('.7z')}")


def xml_to_pandas(root, columns, row_name="row"):
    df = None
    try:

        rows = root.findall(".//{}".format(row_name))

        xml_data = [[row.get(c) for c in columns] for row in rows]  # NESTED LIST

        df = pd.DataFrame(xml_data, columns=columns)
    except Exception as e:
        print("[xml_to_pandas] Exception: {}.".format(e))

    return df


def parse_xml_dump(pathes):
    stackexchange_dir, output_dir = pathes

    path = stackexchange_dir / "Users.xml"
    columns = ["Id", "Reputation", "DisplayName"]

    root = et.parse(path)
    user_df = xml_to_pandas(root, columns)
    user_df = user_df.rename(
        columns={
            "Reputation": "user_reputation",
            "DisplayName": "username",
            "Id": "OwnerUserId",
        }
    )

    path = stackexchange_dir / "Posts.xml"
    columns = [
        "AcceptedAnswerId",
        "AnswerCount",
        "Body",
        "ClosedDate",
        "CommentCount",
        "CreationDate",
        "FavoriteCount",
        "Id",
        "LastActivityDate",
        "OwnerUserId",
        "ParentId",
        "PostTypeId",
        "Score",
        "Title",
        "ViewCount",
    ]

    root = et.parse(path)
    posts_df = xml_to_pandas(root, columns)

    question_columns = [
        "Id",
        "CreationDate",
        "Score",
        "ViewCount",
        "Body",
        "OwnerUserId",
        "LastActivityDate",
        "Title",
        "AnswerCount",
        "CommentCount",
        "FavoriteCount",
        "AcceptedAnswerId",
        "ClosedDate",
    ]

    answer_columns = [
        "Id",
        "CreationDate",
        "Score",
        "Body",
        "OwnerUserId",
        "LastActivityDate",
        "CommentCount",
        "ParentId",
    ]

    question_df = posts_df[posts_df["PostTypeId"] == "1"][question_columns]
    answer_df = posts_df[posts_df["PostTypeId"] == "2"][answer_columns]

    answer_df = answer_df.merge(user_df, on="OwnerUserId")
    question_df = question_df.merge(user_df, on="OwnerUserId")

    answer_df.to_csv(output_dir / "answers.tsv", sep="\t", index=False)
    question_df.to_csv(output_dir / "questions.tsv", sep="\t", index=False)

    return question_df, answer_df


def parse_dumps(path_to_dump, out_dir):
    dumps = list(path_to_dump.glob("*com"))
    dumps = [path for path in dumps if ".meta" not in path.name]

    outputs = [out_dir / path.name for path in dumps]
    for path in outputs:
        if not path.exists():
            os.makedirs(str(path))

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(parse_xml_dump, zip(dumps, outputs)), total=len(dumps)))


def main():
    MAIN_URL = "https://archive.org/download/stackexchange/"

    # for demonstration, we download only some catalogs
    # change this if you'd like to download the whole dump
    SAMPLE = 5

    # Here we'll be storing the StackExchange dump
    PATH_TO_SX_DUMP = Path("input/sx_dump")
    OUT_DIR = PATH_TO_SX_DUMP / "stackexchange_parsed"

    if not os.path.exists(PATH_TO_SX_DUMP):
        os.makedirs(PATH_TO_SX_DUMP)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    links = get_urls(main_url=MAIN_URL, path_to_dump=PATH_TO_SX_DUMP)

    print(f"Fetched {len(links)} links to download data.")

    download_and_unzip_data(
        main_url=MAIN_URL, links=links[1:SAMPLE], path_to_dump=PATH_TO_SX_DUMP
    )

    parse_dumps(path_to_dump=PATH_TO_SX_DUMP, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
