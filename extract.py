from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import csv
import os


def getJobsSoftwareEngineer(location):
    driver = webdriver.Chrome("./chromedriver")
    location = location.split(" ")
    location = ("+").join(location)
    folder = os.path.join(os.getcwd(), "htmlZip")
    try:
        os.makedirs(folder)
    except:
        print("file already exists")
    fw = open("reviews.csv", "w", encoding="utf8")
    writer = csv.writer(fw, lineterminator="\n")
    driver.get(f"https://www.indeed.com/jobs?q=Data+Scientist&l={location}&radius=100")
    time.sleep(2)
    nextlink = True
    j = 0
    done = False
    while nextlink:
        try:
            time.sleep(2)
            driver.find_element_by_id("popover-x").click()
        except:
            print("No Button")

        jobs = driver.find_elements_by_css_selector(
            "div.jobsearch-SerpJobCard.unifiedRow.row.result.clickcard"
        )
        for i in range(len(jobs)):
            jobDesc = "N/A"
            html = "N/A"
            jobs[i].click()
            time.sleep(2)
            iframe = driver.find_element_by_xpath(
                "//iframe[@id='vjs-container-iframe']"
            )
            driver.switch_to.frame(iframe)
            time.sleep(2)
            try:
                html = driver.execute_script(
                    "return document.documentElement.outerHTML;"
                )
                jobDesc = driver.find_elements_by_css_selector(
                    "div.jobsearch-jobDescriptionText"
                )[0].text
                jobDesc = jobDesc.replace("\n", " ")
            except:
                print("no Text")
            if jobDesc != "N/A" and j <= 5000:
                writer.writerow([jobDesc, "Data Scientist"])
            else:
                done = True
                break

            if html != "N/A" and j <= 5000:
                script_dir = os.path.dirname(
                    __file__
                )  # <-- absolute dir the script is in
                rel_path = f"htmlZip/htmlAD{j}.html"
                abs_file_path = os.path.join(script_dir, rel_path)
                ad1 = open(abs_file_path, "w", encoding="utf8")
                j += 1
                ad1.writelines(html)
                ad1.close()
            driver.switch_to.default_content()
        try:
            driver.find_element_by_xpath("//a[@aria-label='Next']").click()
            time.sleep(2)
        except:
            print("No Next")
            nextlink = False

        if done == True:
            break
    fw.close()


getJobsSoftwareEngineer("San Diego, CA")
