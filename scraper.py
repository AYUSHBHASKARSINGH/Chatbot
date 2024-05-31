import requests
from bs4 import BeautifulSoup
import pandas as pd
import os



def scrape_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    header = soup.find('header', class_='td-post-title')
    if header:
        h1 = header.find('h1', class_='entry-title')
        if h1:
            h1_text = h1.get_text(strip=True)

    else:
        div1 = soup.find_all('div', class_='tdb-block-inner')
        for head in div1:
            # print(head)
            header1 = head.find('h1', class_='tdb-title-text')
            if header1:
                # print("jvjh")
                h1_text = header1.get_text(strip=True)
                break
            else:
                h1_text = 'No header element with class "td-post-title" found.'




# Extract the content
    content_div = soup.find('div', class_='td-post-content')
    text_elements = []
    if content_div:
        for element in content_div.children:
            if element.name == 'ol':
                for li in element.find_all('li'):
                    text_elements.append(li.get_text(strip=True))
            elif element.name == 'p':
                text_elements.append(element.get_text(strip=True))
            elif element.name == 'ul':
                for li in element.find_all('li'):
                    text_elements.append(li.get_text(strip=True))

        full_text = "\n".join(text_elements)


    if len(text_elements) == 0:
        content_div2 = soup.find_all('div', class_='tdb-block-inner')
        text_elements1 = []
        for content_div1 in content_div2:
            if content_div1:
                for element in content_div1.children:
                    if element.name == 'ol':
                        for li in element.find_all('li'):
                            text_elements1.append(li.get_text(strip=True))
                    elif element.name == 'p':
                        text_elements1.append(element.get_text(strip=True))
                full_text = "\n".join(text_elements1)
            else:
                full_text = 'No div element with class "td-post-content" found.'

    return h1_text, full_text


# Function to save content to a file
# def save_to_file(url_id, title, content):
#     filename = f"{url_id}.txt"
#     with open(filename, 'w', encoding='utf-8') as file:
#         file.write(title + "\n\n" + content)

# print(scrape_content("https://insights.blackcoffer.com/how-advertisement-increase-your-market-value/"))



def save_to_file(url_id, title, content):
    data_folder = "Data"
    os.makedirs(data_folder, exist_ok=True)

    filename = os.path.join(data_folder, f"{url_id}.txt") 
    with open(filename, 'w', encoding='utf-8') as file:
        content1 = content.replace("\n", " ")
        file.write(title + " " + content1)

xlsx_file = 'Input.xlsx'  
df = pd.read_excel(xlsx_file)

for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    print(f"Processing URL_ID: {url_id}, URL: {url}")

    title, content = scrape_content(url)

    save_to_file(url_id, title, content)

print("Content extraction and saving completed.")