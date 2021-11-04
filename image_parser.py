from html.parser import HTMLParser
import os


class ImageParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.img_url = ''

    def handle_data(self, data):
        print(data)
        if data.find('src') != -1:
            self.img_url = data[data.find('src'):data.find('src') + 121]

def lazy_parser(text):
    results = []
    for idx, chr in enumerate(text):
        if chr == "s":
            if text[idx+1] == "r":
                if text[idx+2] == "c":
                    results.append(text[idx+5:idx+121])

    return results