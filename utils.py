import unicodedata
import re


def clean_name(name, stopwords=[]):
    name=name.upper()
    name = name.replace('&','Y')
    name = name.replace('&','Y')
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
    name=name.replace("'S",'S')
    name=name.replace('-',' ')
    name=name.replace("'",' ')
    name=re.sub('[^A-Za-z0-9ñ\s]+', '', name) #remove special characters
    name=re.sub('\s{2}', ' ', name) #replace 2 white spaces to 1
    name=re.sub('\s{3}', ' ', name) #replace 2 white spaces to 1
    words=name.split()
    words=[w for w in words if w not in stopwords]
    name=" ".join(words)
    name=name.strip() #remove white spaces
    name = name.replace('7UP', '7 UP')
    return str(name)