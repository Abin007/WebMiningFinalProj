from bs4 import BeautifulSoup
from bs4.element import Comment
import re
import time
import requests
import csv
import glob


def tag_visible(element):
    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        "[document]",
    ]:
        return False
    if isinstance(element, Comment):
        return False
    return True


def runHTMLfiles(path):
    list_of_files = glob.glob(path)
    fw = open("jobs.csv", "w", encoding="utf8")
    for fileName in list_of_files:
        print(fileName[10:12])
        if fileName[10:12] == "DS":
            run(fileName, "Data Scientist", fw)
        elif fileName[10:12] == "SE":
            run(fileName, "Software Engineer", fw)
        elif fileName[10:12] == "DE":
            run(fileName, "Data Engineer", fw)
    fw.close()


def run(htmlpage, jobName, fw):
    with open(htmlpage, "r") as file:
        data = file.read()
    writer = csv.writer(fw, lineterminator="\n")
    soup = BeautifulSoup(data, "html5lib")
    jobAD = soup.findAll(text=True)
    HtmlText = ""
    visible_texts = filter(tag_visible, jobAD)
    HtmlText = u" ".join(t.strip() for t in visible_texts)
    HtmlText = HtmlText.lower()
    HtmlText = HtmlText.replace("\n", " ")
    HtmlText = HtmlText.replace(",", " ")

    if jobName == "Data Scientist":
        HtmlText = re.sub("data sci[a-z]+", " ", HtmlText, re.I)
    if jobName == "Data Engineer":
        HtmlText = re.sub("data eng[a-z]+", " ", HtmlText, re.I)
    if jobName == "Software Engineer":
        HtmlText = re.sub("software eng[a-z]+", " ", HtmlText, re.I)

    writer.writerow([HtmlText, jobName])


runHTMLfiles("./htmlZip/*.html")
