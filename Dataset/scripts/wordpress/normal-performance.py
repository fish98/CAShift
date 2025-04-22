import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

import subprocess

options = Options()
# options.headless = False # For Debug
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# Wordpress IP
# must append '/' at the end
ip = "/"
# command = "minikube service wordpress --url"
# result = subprocess.run(command, shell=True, text=True, capture_output=True)
# if result.returncode == 0:
#     ip = f"{result.stdout.strip()}/"
#     print("IP:", ip)
# else:
#     print("Error: Wordpress IP not found")

# visit main page
driver.get(ip)
print("Init Page")
time.sleep(2)

def get_random_string(length):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def get_random_string_hard(length):
    letters = 'abcdefghijklmnopqrstuvwxyz!@#$%^&*()'
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def submit_random_form():
    driver.find_element(By.ID, 'comment').clear()
    driver.find_element(By.ID, 'comment').send_keys(f'{get_random_string_hard(random.randint(1, 1000))}')
    time.sleep(0.5)
    driver.find_element(By.ID, 'author').clear()
    driver.find_element(By.ID, 'author').send_keys(f'{get_random_string(random.randint(1, 100))}')
    time.sleep(0.5)
    driver.find_element(By.ID, 'email').clear()
    driver.find_element(By.ID, 'email').send_keys(f'{get_random_string(random.randint(1, 30))}@{get_random_string(random.randint(1, 30))}.com')
    time.sleep(0.5)
    driver.find_element(By.ID, 'url').clear()
    driver.find_element(By.ID, 'url').send_keys(f'{get_random_string(random.randint(1, 100))}.com')
    time.sleep(0.5)
    driver.find_element(By.ID, 'submit').click()
    time.sleep(4)

links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a')]

while(1):
    next_link = random.choice(links)
    if driver.current_url == ip:
        try:
            driver.get(next_link)
            print(f"Visit {next_link}")
            time.sleep(2)
            links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a')]
        except Exception as e:
            print(e)
    else:
        if "192.168" not in driver.current_url:
            try:
                driver.back()
                print(f"Return")
                time.sleep(2)
                links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a')]
            except Exception as e:
                print(e)
        else:
            pass

        if driver.find_elements(By.CSS_SELECTOR, 'input[type="submit"], button[type="submit"]'):
            if random.random() > 0.5:
                try:
                    submit_random_form()
                    print("Form submitted")
                    time.sleep(2)
                except Exception as e:
                    print(e)
        else:
            print("No form found")

        # visit
        if random.random() > 0.5:
            next_link = random.choice(links)
            try:
                driver.get(next_link)
                print(f"Visit {next_link}")
                time.sleep(2)
            except Exception as e:
                print(e)
        # return
        else:
            try:
                driver.back()
                print(f"Return")
                time.sleep(2)
                links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a')]
            except Exception as e:
                print(e)        

driver.quit()
