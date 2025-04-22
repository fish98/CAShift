import time
import sys
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

import subprocess

# check args
if len(sys.argv) != 2:
    print("Usage: python normal-performance.py [service url e.g, http://192.168.49.2:xxxx/]")
    sys.exit(1)

ip = sys.argv[1]

options = Options()
# options.headless = False # For Debug
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# Wordpress IP
# must append '/' at the end
# ip = "/"

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

def get_randome_email():
    return f'{get_random_string(random.randint(1, 30))}@{get_random_string(random.randint(1, 30))}.com'

def submit_random_login():
    driver.find_element(By.ID, 'username').clear()
    driver.find_element(By.ID, 'username').send_keys(f'{get_random_string(random.randint(1, 50))}')
    time.sleep(0.5)
    driver.find_element(By.ID, 'password').clear()
    driver.find_element(By.ID, 'password').send_keys(f'{get_random_string(random.randint(1, 50))}')
    time.sleep(0.5)
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"].btn.btn-primary')
    submit_button.click()
    time.sleep(2)

def submit_random_email():
    driver.find_element(By.ID, 'jform_email').clear()
    driver.find_element(By.ID, 'jform_email').send_keys(get_randome_email())
    time.sleep(0.5)
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"].btn.btn-primary.validate')
    submit_button.click()
    time.sleep(2)

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
        if "192.168" not in driver.current_url or "format=feed&type=rss" in driver.current_url:
            try:
                driver.back()
                print(f"Return")
                time.sleep(2)
                links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a')]
            except Exception as e:
                print(e)
        else:
            pass
        
        try:
            if driver.find_element(By.CSS_SELECTOR, 'button[type="submit"].btn.btn-primary'):
                if random.random() > 0.5:
                    try:
                        submit_random_login()
                        print("Login submitted")
                        time.sleep(2)
                    except Exception as e:
                        print(e)
            else:
                print("No Login Page found")

        except Exception as e:
            print("Button not found")
        
        try:
            if driver.find_element(By.CSS_SELECTOR, 'button[type="submit"].btn.btn-primary.validate'):
                if random.random() > 0.5:
                    try:
                        submit_random_email()
                        print("Email submitted")
                        time.sleep(2)
                    except Exception as e:
                        print(e)

        except Exception as e:
            print("Button not found")

        # visit
        if random.random() > 0.5:
            next_link = random.choice(links)
            try:
                driver.get(next_link)
                print(f"Visit {next_link}")
                time.sleep(2)
            except Exception as e:
                print(e)

        else:
            try:
                driver.back()
                print(f"Return")
                time.sleep(2)
                links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a')]
            except Exception as e:
                print(e)        

driver.quit()
