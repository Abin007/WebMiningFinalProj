from bs4 import BeautifulSoup
from bs4.element import Comment
import re
import time
import requests
import csv
import glob

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def run(htmlpage,jobName):
    jobs = []
    with open(htmlpage, "r") as file:
        data = file.read()
    fw = open("jobs.csv", "a+", encoding="utf8")
    writer = csv.writer(fw, lineterminator="\n")
    soup = BeautifulSoup(data, "html5lib")
    jobAD = soup.findAll(text=True)
    HtmlText = ''
    visible_texts = filter(tag_visible, jobAD) 
    HtmlText=u" ".join(t.strip() for t in visible_texts)
    HtmlText=HtmlText.replace(jobName," ")
    writer.writerow([HtmlText,jobName])


def runHTMLfiles(path):
    list_of_files = glob.glob(path)
    for fileName in list_of_files:
        if(fileName[11:13]=="DS"):
            run(fileName,"Data Scientist")
        elif(fileName[11:13]=="SE"):
            run(fileName,"Software Engineer")
        elif(fileName[11:13]=="DE"):
            run(fileName,"Data Engineer")



runHTMLfiles("./htmlTest/*.html")
