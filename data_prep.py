from bs4 import BeautifulSoup
import csv
import pandas as pd
from nltk.corpus import stopwords

# This function can also easily be used to extract comments instead of posts, just replace the posts html
# with a comments one

def extract_posts(posts_file):

    with open(posts_file, encoding="utf-8", errors='ignore') as infile:
        soup = BeautifulSoup(infile, "html.parser")

    comments = []

    for comment in soup.find_all('div', {"class": "_2pin"}):
        comments.append(comment.text)

    with open('fb_posts.csv', mode='a', encoding='utf-8', errors='ignore', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for comment in comments:
            writer.writerow([comment])

extract_posts("your_posts.html")

# Extracts message data from message html file

def get_messages(messages_file):
    with open(messages_file, encoding="utf-8", errors='ignore') as infile:
        soup = BeautifulSoup(infile, "html.parser")
        messages = []

    for message in soup.find_all('div', {"class": "pam _3-95 _2pi0 _2lej uiBoxWhite noborder"}):
        messages.append(message)

    soup2 = BeautifulSoup(str(messages), 'html.parser')
    for div in soup2.find_all('div', {"class": "_3-94 _2lem"}):
        div.decompose()

    cleaned = soup2

    filtered_messages = []

    # assuming you only want the messagse you've sent your name would go here
    for comment in cleaned:
        if 'YOUR NAME' in str(comment):
            chunk = comment.text.replace('YOUR NAME', '')
            filtered_messages.append(chunk)

    print(filtered_messages)

    with open('your_messages.csv', mode='a', encoding="utf-8", errors='ignore') as file:
        writer = csv.writer(file, delimiter=',')
        for comment in filtered_messages:
            writer.writerow([comment])

    with open("your_messages_text.txt", "a", encoding="utf-8") as text_file:
        text_file.write(str(filtered_messages))

# Insert messages here
get_messages("messages.html")

# Function to concatenate files

def file_concat(file_1, file_2, file_3):

    f1 = pd.read_csv(file_1)
    f2 = pd.read_csv(file_2)
    f3 = pd.read_csv(file_3)

    print(f1.head(5))

    complete = pd.concat([f1, f2, f3], sort=False)
    complete.to_csv("fb_complete.csv", index=False, encoding='utf-8-sig')

file_concat("your_comments.csv","your_messages.csv","your_posts.csv")

# Filter concatenated file

stop_words = set(stopwords.words('english'))

def filter_words(input_file, output_file):

    data = open(input_file, errors="ignore", encoding="utf-8").read()

    words = data.split()

    for w in words:
        if not w in stop_words:
            file = open(output_file, mode='a', errors="ignore", encoding="utf-8")
            file.write("{} {}".format(w, ""))
            file.close()

filter_words("fb_complete.csv", "filtered_words.txt")