import requests
from requests.utils import requote_uri
from bs4 import BeautifulSoup
import csv
import pandas as pd
import os
import json

wiktionary_base_url =  "https://en.wiktionary.org/wiki/Category:"

offensive_category = "English_offensive_terms"

headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"}


'''
    Given relative href, return full wiktionary url for that page
'''
def create_full_wiki_page_url(href):
    safe_href = requote_uri(href)
    return f'https://en.wiktionary.org{safe_href}#English'


'''
    Be mindful of mode, if file exists, most likely want to append
'''
def write_csv_file(file_path, data, overwriteFlag = False):
    mode = 'w' if overwriteFlag else 'a'
    with open(file_path, mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
    if mode == 'a':
        print(f'Appended {file_path}')
    else:
        print(f'Successfully created CSV file at {file_path}')

'''
    Helper function to get paging data, note: the first time the function is called, the url will have a different structure
    Output: Returns next page url (can be None), pages (actual html content from BeautifulSoup)
'''
def process_category_page(url, category, current_page_ct, isFirstPage = False):
    if isFirstPage:
        full_url = f'{wiktionary_base_url}{category}'
    else:
        full_url = url

    print(f'Making request to {full_url}')
    r = requests.get(full_url, headers=headers)

    assert r.status_code == 200

    soup = BeautifulSoup(r.content, 'html5lib')

    ## Try and figure out how many pages are in the category
    page_ct_div = soup.find('div', attrs={'id' : 'mw-pages'})

    if page_ct_div is None:
        ## kind of hacky, skips the remaining entries on the last page.
        return None, []

    page_ct_div_text = page_ct_div.find('p').getText()

    ## Gonna need this for paging, if i decide to automate the paging. -> Might be able to page by getting the href from the (Next) link tag
    total_page_ct = page_ct_div_text.split(" ")[10]
    total_page_ct = int(total_page_ct.replace(',', ''))


    print(f'Current pages shown {current_page_ct}, out of {total_page_ct} total pages')

    nxt_url = None
    if (total_page_ct >= current_page_ct):
        ## Should mean there is a next button to find :)
        nxt_anchor_tags = page_ct_div.find_all('a')
        if isFirstPage:
            # print(nxt_anchor_tags)
            # os.abort()
            nxt_link = nxt_anchor_tags[0]['href']
        else:
            print(nxt_anchor_tags)
            nxt_link = nxt_anchor_tags[1]['href']
        
        ## 
        nxt_url = f'https://en.wiktionary.org{requote_uri(nxt_link)}'
    
    pages = soup.find_all('div', attrs={'class' : 'mw-category-group'})

    pages = pages[3:] ## exclude first three bc they're subcategories

    return nxt_url, pages

def process_category_headers(pages, data_folder, overwriteFlag):
    ## array to use for writing entry data to csv file,
    ## should have structure [page, url] , note: url will be base_url + href 
    ## However, we're creating a separate csv file for each heading letter, so need to clear array after writing the csv file
    data = [['Page', 'Url']]

    no_files_written = 0
    total_entries = 0
    for page in pages:
        page_start_letter = page.h3.get_text()
        print(f'Pages starting with {page_start_letter}')
        csv_file_path = os.path.join(data_folder, page_start_letter + '.csv')
        print(csv_file_path)

        ## li is the actual listing for the entry with the content we want
        for li in page.ul.find_all("li"):
            entry_anchor = li.find("a")
            entry_title = entry_anchor.get_text()
            if 'Citation' in entry_title:
                print(f'Skipping {entry_title} because it is a citation')
                return
            entry_href = entry_anchor["href"]

            data.append([entry_title, create_full_wiki_page_url(entry_href)])
            total_entries += 1
        
        ## Have all the pages for the current heading.
        write_csv_file(csv_file_path, data, overwriteFlag)
        no_files_written += 1

        ## assume the write function works, so reset the data array
        data = [['Page', 'Url']]
    
    print(f'Finished processing categories and wrote {no_files_written} csv files to the folder {data_folder}')
    return total_entries

def debug_clear_categories_folder(data_folder):
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)

def concat_csv_files_in_data_folder(data_folder, category):
    new_csv_filename = f'{category.split('_')[0]}-page-entries.csv'
    new_csv_file_path = os.path.join(data_folder, new_csv_filename)

    csv_paths = [
        os.path.join(data_folder, fn)
        for fn in os.listdir(data_folder)
        if fn.endswith('.csv')
    ]

    if not csv_paths:
        print(f'No CSV files were found in {data_folder}')
        return
    
    combined = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)
    combined.to_csv(new_csv_file_path, index=False)
    print(f'Combined {len(csv_paths)} files → "{new_csv_filename}"')


'''
     maybe I want to create a csv file per starting letter i.e. a-offensive_english.csv, b-offensive-english.csv, 
'''
def fetch_categories(category, data_folder):
    ## Clear the data folder each iteration for now, get fresh data each time this function is called.
    debug_clear_categories_folder(data_folder)

    isFirstPage = True
    current_url = wiktionary_base_url

    ct = 0
    while (current_url is not None):
        nxt_url, pages = process_category_page(current_url, category, ct, isFirstPage)
        current_url = nxt_url
        ct += process_category_headers(pages, data_folder, overwriteFlag=isFirstPage)
        print(f'Next Url: {nxt_url}')
       

        if isFirstPage:
            isFirstPage = False

    ## Eventually will create a function to append all the csv files into one file, to make processing later on easier.
    print(f'Finished writing {ct} page entries.') 


'''
    Check the content of the entry 
'''
def parse_entry_content(entry_name, url):

    r = requests.get(url, headers=headers)
    assert r.status_code == 200

    soup = BeautifulSoup(r.content, 'html5lib')

    ## Parse Etymology
    entry_content = soup.find('div', attrs={'class' : 'mw-content-ltr mw-parser-output'})
    etymology = entry_content.find_all('p')[0].get_text() 
 
    ## Parse parts of speech (can be multiple) - Just brute force with a list of known parts of speech?
    ## might work because the id of <h3> is the part of speech.
    known_parts_of_speech = ['Noun', 'Verb', 'Adjective', 'Adverb', 'Phrase', 'Proper noun', 'Interjection', 'Proverb']
    '''
        for each part of speech, there is a definition, then some example usages. 
        need to get an object, something like 
        { 
            word : 'test'
            etymology : 'Big Blah'
            parts_of_speech : [
                 {
                    part_of_speech : Noun,
                    user_definition : 'blah blah',
                    example_usages : ['foo', 'bar']
                }
            ]
        }
    '''
    h3_tags = entry_content.find_all('h3')
    parts_of_speech = [
        tag.get_text()
        for tag in h3_tags
        if tag.get_text() in known_parts_of_speech
    ]
    ## assume there is at least one part of speech, probably a safe bet?
    ## ol tag has user definition and example usages, that corresponds to first part of speech
    ol_tags = entry_content.find_all('ol')

    ## part of speech was populated in order as it appears on the page.
    iteration = 0
    parts_of_speech_data = []
    for ol_tag in ol_tags:
        ## skip the references ol.
        if ol_tag.find_parent('div', attrs={'class' : 'mw-references-wrap'}):
            break

        if iteration >= len(parts_of_speech):
            print(f'Skipping part of speech out of bounds')
            break

        part_of_speech = parts_of_speech[iteration]

        ## user_definition is in li text
        li_tag = ol_tag.find('li')
        user_definition = li_tag.get_text().split(".")[0]
        

        ## example usages is in a ul tag in the li tag

        ex_usages = []
        example_usage_ul_tag = li_tag.find('ul')
        if example_usage_ul_tag is None:
            print(f'Skipping {entry_name} because no examples were provided.')
            return None


        example_usage_li_tags = example_usage_ul_tag.find_all('li')
        for example_usage_li_tag in example_usage_li_tags:
            quotation_div = example_usage_li_tag.find('div', attrs={'class' : 'h-quotation'})
            if quotation_div is not None:
                ex_usages.append(quotation_div.get_text())
            else:
                ## find a <dd> tag?
                dd_tag = example_usage_li_tag.find('dd')
                if dd_tag is not None:
                    ex_usages.append(dd_tag.get_text())

            
        parts_of_speech_data.append({'part_of_speech' : part_of_speech, 'user_definition' : user_definition, 'example_usages' : ex_usages})

        iteration += 1
    
    if len(parts_of_speech_data) <= 0:
        print(f'Skipping {entry_name} because no valid parts of speech were found.')
        return None
    
    return {
        'word' : entry_name,
        'etymology' : etymology,
        'parts_of_speech' : parts_of_speech_data
    }


##fetch_categories(offensive_category, "offensive_categories")
## want to create one big csv for now (only need to execute this function once)
## concat_csv_files_in_data_folder('offensive', 'offensive_categories') -> move this around later.
##data = parse_entry_content('41%','https://en.wiktionary.org/wiki/41%25#English')

'''
    Writes the parsed page / entry content to a file with jsonl format (separated by \n)
'''
def parse_entries_csv(filepath, jsonl_path, csv_output_path):
    df = pd.read_csv(filepath)

    write_csv_header = not os.path.exists(csv_output_path)

    with open(jsonl_path, 'w', encoding='utf-8') as jf, \
        open(csv_output_path, 'a', newline='', encoding='utf-8') as cf:

        csv_writer = None

        for idx, row in df.iterrows():
            print(f'On Page -> {row['Page']}')
            word_data = parse_entry_content(row['Page'], row['Url'])
            
            if word_data is None:
                continue

            json.dump(word_data, jf, ensure_ascii=False)
            jf.write('\n')

            if csv_writer is None:
                fieldnames = list(word_data.keys())
                csv_writer = csv.DictWriter(cf, fieldnames = fieldnames)
                if write_csv_header:
                    csv_writer.writeheader()
                    write_csv_header = False
            
            csv_writer.writerow(word_data)
            
    print(f'Finished writing {jsonl_path} and {csv_output_path}')    






parse_entries_csv("offensive\\offensive-page-entries.csv", 'offensive\\offensive-parsed-page-content.json', 'offensive\\offensive-parsed-page-content.csv')