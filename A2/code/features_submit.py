####################
# FINAL SUBMISSION #
####################
import re
import sys
import numpy as np
#from gensim.models import Word2Vec
#model = Word2Vec.load("300features_0minwords_10context")


if (len(sys.argv) < 3):
    print("USAGE\t"),
    print("<ARG1>: train.txt to add feature to")
    print("<ARG2>: output.txt to write output to")
    print("<ARG3>: 1 if training and 0 if test")
    exit()


model_train = sys.argv[1]
f = open(model_train, "r")

l = f.readlines()
l = map(lambda i: i.strip(), l)
f.close()

f = open("./thesaurus/names_surnames.txt", "r");names=  f.readlines(); f.close(); 
names = map(lambda i: i.strip().lower(), names)
names = set(names)

f= open("./thesaurus/newPlacesInNCR.txt", "r"); locations = f.readlines(); f.close();
locations = map(lambda i: i.strip().lower(), locations)
locations = filter(lambda i: not i.isdigit(), locations)
locations = set(locations)

f= open("./thesaurus/wordsEn.txt", "r"); engWords = f.readlines(); f.close();
engWords = map(lambda i: i.strip().lower(), engWords)
engWords = set(engWords)



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def substring_feature(train_example, position):
    examine = train_example[position].lower()
    return examine[1:] if len(examine) > 1 else -1

def other_substring_feature(train_example, position):
    examine = train_example[position].lower()
    return examine[:-1] if len(examine) > 1 else -1

def get_tweet(i, train):
    tweet = []

    for j in xrange(i, len(l)):
        if not l[j]:
            break
        else:
            p = l[j].split(" ")

            if train:    
                x, _ = l[j].split(" ")
            else:
                x = l[j]
            tweet.append(x)

    return tweet, j


def generate_tweets(train):
    # type: () -> list[list]
    i = 0
    tweets = []
    while True:
        tweet, i = get_tweet(i, train)
        # print(i)
        i += 1
        tweets.append(tweet)
        if i == len(l):
            break
    print(len(tweets))
    # print(tweets)
    return tweets


# Token level features

def web_url(stringInp):
    return "http" in stringInp or "www" in stringInp or "https" in stringInp


def is_wordVectorid(train_example, position):
    examine = train_example[position].lower()
    try:
        similar = model.most_similar(examine)[:3]
        similar_strings = " ".join(map(lambda i: i[0], similar))
        return similar_strings
    except KeyError:
        return ""

def is_englishWord(train_example, position):
    examine = train_example[position].lower()
    return examine in engWords

def is_webUrl(train_example, position):
    return web_url(train_example[position])

def is_location_keyword(train_example, position):
    examine = train_example[position].lower()

    def is_place_inNCR(stringInp):
        if stringInp in locations:
            return True 

    return is_place_inNCR(examine) 


def is_builder_name(train_example, position):
    possible_builders = ["amrapali", "supertech", "unitech", "parsvnath", "parsavnath", "parshvanath", "omaxe", "unitech", "ansal", "ansalapi", "raheja", "dlf", "exotica"]
    examine = train_example[position].lower()
    return examine in possible_builders

def is_location(train_example, position):

    def is_sector(stringInp):
        return ("sec" in stringInp) and not "http" in stringInp and not stringInp in ["secure", "security", "second"]


    possible_locations = ["noida", "faridabad", "greater", "highway", "amrapali",  "city",  "dadri", "east", "west", "yamuna", "north", "south", "supertech", "unitech","parsvnath", "parsavnath", "parshvanath", "omaxe", "unitech", "ansalapi", "ansal", "raheja", "dlf", "prateek" , "exotica", "wave"]

    examine = train_example[position].lower()
    check = map(lambda location: location in  examine, possible_locations)
    return  not web_url(examine) and (any(check) or is_sector(examine) or examine == "ncr" or re.findall(r'\bgaur\b', examine) or re.findall(r'\bpark\b', examine) )
    
def is_hashtag(train_example, position):

    return (not web_url(train_example[position]))  and train_example[position][0] == "#"

def is_name(train_example, position):

    def next_phone():
        return position + 1 < len(train_example) and is_phone_number(train_example, position+1)
    examine = train_example[position].lower()
    next_is_phone = examine in ["call", "contact", "details", "cont", "plz", "number", "no."]
    callRegex = re.search(r"\bcall[-@]?", examine) or re.search(r"\bcont[-@]?", examine)
    

    next_is_phone =  callRegex or next_is_phone

    #return (y_prev == "N" or next_phone()) and not next_is_phone and not y == "T" and (len(train_example[position]) > 1 ) and not web_url(train_example[position].lower()) and not is_location(train_example, position, y, y_prev)

    return not next_is_phone and  (len(train_example[position]) > 1 ) and not web_url(train_example[position].lower()) and not is_location(train_example, position)

def is_name_repo(train_example, position):

    examine = train_example[position].lower()
    return examine in names and not is_location(train_example, position, ) and not is_location_keyword(train_example, position) 

def is_price(train_example, position):

    examine = train_example[position].lower()
    
    price_regex = re.compile(r'\b(\d{1,})?(lac|crore|cr|lakh)s?\b')
    price_regex2 = re.compile(r'\d{2,}[ /-]?rs')
    price_regex3 = re.compile(r'\b(\d{2,})?rs(\d{2,})?\b')



    rupees_quantity = re.search(price_regex2, examine) or re.search(price_regex3, examine)
    def custom_check():
        return position+1 >= len(train_example) or re.search(price_regex,train_example[position+1].lower())
    
    def custom_check2():
        def check(stringInp):
            #return ("sec" in stringInp) and not "http" in stringInp and not stringInp in ["secure", "security", "second"]
            r = re.compile(r'\bsec(tor)?\b')
            return not re.search(r, stringInp)
        
        return position-1<0 or check(train_example[position-1].lower()) 

    return ("/-" in examine or re.search(price_regex, examine) or rupees_quantity or ((is_number(examine) or (is_number(examine[1:]) and (len(examine) > 2 or custom_check() ))) and custom_check2()))   and  not is_phone_number(train_example, position) and (len(examine) > 1 or custom_check()) and not is_land_area(train_example, position) and not is_webUrl(train_example, position) 


def is_phone_number(train_example, position):
    telephone_regex = re.findall(r'(?:\+?@?\d{2}[ -]?)?\d{10}', train_example[position])
    return telephone_regex



def is_other(train_example, position):

    def garbage_token(stringInp):
        #return len(train_example[position]) == 1 and not_area_or_price_or_cost(y) and not_area_or_price_or_cost(y_prev)  and not train_example[position].isdigit()
    
        return len(train_example[position]) == 1 and  not  train_example[position].isdigit()

    def not_area_or_price_or_cost(stringInp):
        return stringInp not in ["LA", "C", "P"]


    stringObserve = train_example[position].lower()
    return web_url(stringObserve) or garbage_token(stringObserve) and not is_price(train_example, position) and not is_location(train_example, position) and not is_cost(train_example, position)

def is_land_area(train_example, position):

    def is_sq_or_ft(stringInp):
        keywords = ["sq", "ft", "meter", "acre", "sqm", "mtr", "square", "feet", "sqr", "size", "sz", "carpet", "area"]
        return any(map(lambda keyword: keyword in stringInp, keywords))

    def check_prev():
        re1 = re.compile(r'\d{2,}sq')
        re2 = re.compile(r'\d{2,}mtr')
        examine_prev = train_example[position-1].lower() if position -1 > 0 else ""
        return position - 1> 0 and examine_prev in  ["sq", "sqr", "square", "sqft"] or re.match(re1, examine_prev) or re.match(re1, examine_prev)

    def check_if_magnitude():
        return position+1 < len(train_example) and ( train_example[position].isdigit() and is_sq_or_ft(train_example[position+1].lower()))


    def check_not_cost():
        curr_iter = position
        iter_round = 1
        while iter_round < 4:
            if curr_iter < 0: return True
            else:
                examine_prev = train_example[curr_iter].lower()
                if (examine_prev in ["per", "/"] or "rs" in examine_prev or "per" in examine_prev or "pr" in examine_prev): return False
            iter_round += 1
            curr_iter -=1 
        return True

    examine = train_example[position].lower()
    keywords = ["per", "pr", "/"]
    any_match = map(lambda i: i not in  examine, keywords)
    any_match = all(any_match)

    return  not "http" in train_example[position].lower() and check_not_cost()   and (is_sq_or_ft(train_example[position].lower()) or (train_example[position] == '.' and check_prev()) or check_if_magnitude())

def is_other_second(train_example, position):

    listOfKeyWords = ["available", "rent", "office", "shop", "for", "sale", "girls", "family", "bachelors", "inclusive",  "download", "forwarded","very" ]
    examine = train_example[position].lower()
    return examine in listOfKeyWords 


def is_near(train_example, position):
    curr_pos = position
    itr = 0
    while (itr < 5):
        if curr_pos < 0: break
        else:
            examine = train_example[curr_pos].lower()
            if examine == "near" and (itr == 0 or is_location(train_example, curr_pos+1)): return True 

        curr_pos -=1
        itr +=1
    
    return False

def is_cost(train_example, position):

    def custom_check(examine):
        l = ["/", "per", "pr"]
        check_if_any = any(map(lambda i: i in examine, l))
        return check_if_any and (if_area(examine) or (position+1 < len(train_example) and if_area(train_example[position+1].lower())))

    def if_area(examine):
        areas = ["sq", "ft", "meter", "acre", "sqm", "mtr", "square", "feet", "sqr", "size", "sz", "gaj"]
        check_if_any_sec = any(map(lambda i: i in examine, areas))
        return check_if_any_sec
    def is_amount(stringInp):
        is_num = is_number(stringInp)
        r = re.compile(r'(@?\d{3,}-?/?-?k?)?(cr|lac|crore|k)?')

        return is_num or re.search(r, stringInp)  

    def template():
        return position-1 > 0 and if_area(train_example[position-1].lower()) and is_cheap(train_example, position)

    def check_cost(direction):
        curr_iter = position
        iter_round = 1
        while iter_round < 4:
            if curr_iter < 0 or curr_iter >= len(train_example):
                break
            else:
                examine_prev = train_example[curr_iter].lower()
                if (examine_prev in ["/", "per", "pr"] or custom_check(examine_prev) ): 
                        return  curr_iter + 1 < len(train_example) and if_area(train_example[curr_iter+1].lower())
            iter_round += 1
            curr_iter += direction 
        return False

    r2 = re.compile(r'(@?\d{3,}-?/?-?k?)\s*(cr|lac|crore|k)?\s*(/|per|pr)\s*(sq|meter|mtr|acre|gaj)')
    examine = train_example[position].lower()
    return not web_url(examine) and (  check_cost(-1) or ((is_amount(examine) and check_cost(1))) or re.search(r2, examine)) and not (train_example[position] == '/') 
   


def is_cheap(train_example, position):

    def range_cheap(i): return i < 50000 and i > 1000
    def check_next():
        return position + 1 < len(train_example) and (train_example[position+1].lower() not in ["cr", "lac",  "lacs",  "sq", "ft", "meter", "acre", "sqm", "mtr", "square", "feet", "sqr", "size", "sz"]) #or (train_example[position+1].lower() == "all" and position+2 < len(train_example) and train_example[position+2].lower() in "inclusive"))

    examine = train_example[position].lower()
    return (examine.isdigit() and range_cheap(int(examine)) or (examine[1:].isdigit() and range_cheap(int(examine[1:])))) and check_next() and not is_land_area(train_example, position) and not is_location(train_example, position) and not is_attribute(train_example, position) 

def is_beginning(train_example, position):
    return position < 2

def is_end(train_example, position):
    return len(train_example) - position < 2

def is_attribute(train_example, position):
    examine = train_example[position].lower()

    listOfKeyWords = ["furnished", "bhk", "parking", "ready","move", "residential", "swimming", "penthouse", "color", "kitchen", "bedroom", "toilet", "wooden", "work", "occupied", "facing", "floor", "industrial", "family", "bachelors"]
    def custom_f_fourth():
        return position+1 < len(train_example) and train_example[position+1].lower() == "floor" and train_example[position][:-2].isdigit()
    
    def custom_f_third():
        return position+1 < len(train_example) and train_example[position].lower() == "fully" and train_example[position+1].lower() == "furnished"
    def custom_f_second():
        return position -1 > 0 and train_example[position-1].lower() == "ready" and re.search(r"\bto\b", examine)
    def custom_f():
        return position + 1 < len(train_example) and  train_example[position].isdigit() and train_example[position+1].lower() == "bhk"
    return (examine in listOfKeyWords) or  custom_f_second() or custom_f() or custom_f_third() or custom_f_fourth() 

def is_rent(train_example, position):
    for i,token in enumerate(train_example):
        if i < position and position < i+3:
                if token.lower() == "rent":
                    return True

    return False


def write_features(f,data, featureSet, featureName, train):

    trainExampleIndex = 0
    pos_curr = 0

    for line in data:
        if not line:
            f.write("\r\n")
            trainExampleIndex += 1
            pos_curr = 0

        else:
            if train:
                x, y = line.strip().split(" ")
            else:
                x = line.strip()
            on_features = []
            for feature in featureSet:
                if feature == "substring_feature":
                    ft = substring_feature(tweets[trainExampleIndex], pos_curr)
                    ft2 = other_substring_feature(tweets[trainExampleIndex], pos_curr)
                    if ft != -1:
                        on_features.append(ft)

                    if ft2 != -1:
                        on_features.append(ft2)
                elif feature == "WordVec":
                    cluster_id = is_wordVectorid(tweets[trainExampleIndex], pos_curr)
                    on_features.append(str(cluster_id))
                elif featureSet[feature](tweets[trainExampleIndex], pos_curr):
                    on_features.append(featureName[feature])

            pos_curr += 1
            on_features = " ".join(on_features)
            if on_features:
                if train:
                    f.write(x + " " + on_features + " " + y + "\r\n")
                else:
                    f.write(x + " " + on_features  + "\r\n")
            else:
                if train:
                    f.write(x + " " + y + "\r\n")
                else:
                    f.write(x + "\r\n")

if __name__ == '__main__':

    model_path = sys.argv[2]
    train = int(sys.argv[3])
    f2 = open(model_path, "w")

    tweets = generate_tweets(train)
    
    featureSet = {'is_location': is_location, "is_land_area": is_land_area, 'is_price': is_price, "is_phone_number" : is_phone_number, "is_other": is_other, "is_other_second" : is_other_second, "is_attribute" : is_attribute, "is_hashtag": is_hashtag, "is_location_keyword" : is_location_keyword, "is_name_repo" : is_name_repo, "is_english_word" : is_englishWord, "is_builder_name" : is_builder_name, "is_near" : is_near,  "is_cost" : is_cost, "is_cheap" : is_cheap, "is_rent" : is_rent}
    featureName = {'is_location': "LOCATION", 'is_price': "PRICE_OR_COST",  "is_land_area" :  "LAND", "is_phone_number": "PHONE", "is_other" : "OTHER",  "is_other_second": "OTHER_SECOND", "is_attribute" : "ATTRIBUTE", "is_hashtag" : "HASHTAG" , "is_location_keyword": "LOCATION_KEY", "is_name_repo" : "NAME_REPO", "is_english_word": "ENGLISH_WORD", "is_builder_name": " BUILDER", "is_near": "NEAR", "is_cost" : "PROBAB_COST", "is_cheap" : "CHEAP", "is_rent" : "RENT"}

    write_features(f2, l, featureSet, featureName, train) 
    f2.close()
