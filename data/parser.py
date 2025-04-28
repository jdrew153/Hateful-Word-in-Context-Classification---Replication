import json
import requests
from requests.utils import requote_uri
import re
import argparse

argp = argparse.ArgumentParser()
argp.add_argument("--word", default=None, help="Type a word to search it's definition(s)")
args = argp.parse_args()

## Using another dictionary api bc Wiktionary python module wasn't working
dict_api_base = 'https://api.dictionaryapi.dev/api/v2/entries/en/'


def lookup_word_defs(word, debug_print = False):
    ## Takes in a word as a string and concatenates the word with the base url for the dict api
    ## Returns an array of the definitions for the parts of speech the word may have
    ##  -> Only returns the array of meanings if response is 200 else just return empty array

    ## clean up word just in case.
    clean_word = re.sub(r"\s+", "", word)
    url = f"{dict_api_base}{clean_word}"

    response = requests.get(requote_uri(url))

    if response.status_code != 200:
        print(f'Invalid response from request to: {url} -> {response.status_code}')
        return []
    
    ## should have a valid response with json at this point
    meanings = response.json()[0]['meanings']

    # if (debug_print):
    #     print(json.dumps(meanings, indent=2, ensure_ascii=False))
    
    ## Meanings has json structure of [{ partOfSpeech: '', definitions: [ definition: '' ]}]
    ## --> Want to return [{ partOfSpeech : '', definition: '' }], where definition is the top definition from the api response
    ret = []
    for meaning in meanings:
        partOfSpeech = meaning['partOfSpeech']
        top_def = meaning['definitions'][0]['definition']

        ret.append({'partOfSpeech' : partOfSpeech, 'definition' : top_def})
    
    if (debug_print):
        print(json.dumps(ret, indent=2, ensure_ascii=False))
    
    return ret

## kind of just for testing purposes i guess
assert args.word is not None
lookup_word_defs(args.word, debug_print=True)




