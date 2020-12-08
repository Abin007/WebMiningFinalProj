from typing import Text
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import csv
import os

j = 1


def getJobsAds(job, location):
    global j
    driver = webdriver.Chrome("./chromedriver")
    driver.maximize_window()
    locationsplit = location.split(" ")
    location = ("+").join(locationsplit)
    jobsplit = job.split(" ")
    jobConcat = ("+").join(jobsplit)
    folder = os.path.join(os.getcwd(), "htmlZip")
    try:
        os.makedirs(folder)
    except:
        print("file already exists")
    # fw = open("reviews.csv", "a+", encoding="utf8")
    # writer = csv.writer(fw, lineterminator="\n")
    driver.get(f"https://www.indeed.com/jobs?q={jobConcat}&l={location}")
    time.sleep(2)
    nextlink = True
    jobs = []
    done = False
    noofJobs = 0
    prevjobDesc = ""
    while nextlink:
        try:
            time.sleep(4)
            popButton = driver.find_element_by_id("popover-foreground")
            popButton.find_element_by_id("popover-x").click()
        except:
            print("No pop button")
        try:
            jobs = driver.find_elements_by_css_selector(
                "div.jobsearch-SerpJobCard.unifiedRow.row.result.clickcard"
            )
        except:
            print("No jobs found")
        noofJobs += len(jobs)
        print(f"Number of jobs added {noofJobs}")
        already_seen = set()
        if len(jobs) > 0:
            for i in range(len(jobs)):
                if jobs[i] in already_seen:
                    continue
                already_seen.add(jobs[i])
                jobDesc = "N/A"
                html = "N/A"
                try:
                    jobs[i].click()
                    time.sleep(4)
                    if len(driver.window_handles) > 1:
                        driver.switch_to.window(driver.window_handles[0])
                        time.sleep(4)
                    try:

                        iframe = driver.find_element_by_xpath(
                            "//iframe[@id='vjs-container-iframe']"
                        )
                        driver.switch_to.frame(iframe)
                        try:
                            jobDesc = driver.find_elements_by_css_selector(
                                "div.jobsearch-ViewJobLayout.jobsearch-ViewJobLayout--embedded"
                            )[0].text
                            jobDesc = jobDesc.replace("\n", " ")
                        except:
                            print("no Job Desc")

                        try:
                            html = driver.execute_script(
                                "return document.documentElement.outerHTML;"
                            )
                        except:
                            print("No Html")
                    except:
                        print("iFrame doesn't exist")
                    time.sleep(4)

                    if prevjobDesc == jobDesc:
                        continue

                    if jobDesc != "N/A" and html != "N/A":
                        prevjobDesc = jobDesc
                        # writer.writerow([jobDesc, job])
                        script_dir = os.path.dirname(__file__)
                        locname = "_".join(locationsplit)
                        jobname = ""
                        for i in jobsplit:
                            jobname += i[0]
                        rel_path = f"htmlZip/{jobname}_{locname}_{j}.html"
                        abs_file_path = os.path.join(script_dir, rel_path)
                        ad1 = open(abs_file_path, "w+", encoding="utf8")
                        j += 1
                        ad1.writelines(html)
                        ad1.close()
                except:
                    print("No Clickable Job AD")

                driver.switch_to.default_content()
                time.sleep(30)

        try:
            driver.find_element_by_xpath("//a[@aria-label='Next']").click()
            time.sleep(150)
        except:
            print("No Next")
            nextlink = False

    # fw.close()


getJobsAds("Data Scientist", "Maryland City, MD")
print(j)