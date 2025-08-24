import re
str_folderPath="D:jeeva\notes"

print(str_folderPath)

str_folderPath=r"D:jeeva\notes"

print(str_folderPath)
re_search = re.search("pattern","string contains the pattern")
print(re_search)

re_search = re.search("pattern","string does not contains the ")
print(re_search)

sara_string = r"sara was able to fine the item needed quickly"
print(sara_string)
replace_sara_sarah = re.sub("sara","sarah", sara_string)
print(replace_sara_sarah)


customer_reviews = ["this is first review",
                    "this is second review",
                    "third review by",
                    "fourth review",
                    "this is fifth review",
                    "this fifth review",
                    "this i sixth review",
                    "I want this review",
                    "he needed this review",
                    "We wanted this review",
                    "she need this review"
                   ]
sarahs_review = []
pattern_to_find = r"this is?"
for string in customer_reviews:
    if(re.search(pattern_to_find,string)):
        sarahs_review.append(string)

print(sarahs_review)


a_reviews = []
pattern_to_find= r"^f"
for string in customer_reviews:
    if(re.search(pattern_to_find,string)):
        a_reviews.append(string)
print(a_reviews)

y_reviews = []
pattern_to_find= r"y$"
for string in customer_reviews:
    if(re.search(pattern_to_find,string)):
        y_reviews.append(string)
print(y_reviews)

needwant_reviews = []
pattern_to_find= r"(need|want)ed"
for string in customer_reviews:
    if(re.search(pattern_to_find,string)):
        needwant_reviews.append(string)
print(needwant_reviews)

# refer this example in google doces
np_punct_reviews = []
pattern_to_find= r"[^\w\s]"
for string in customer_reviews:
    if(re.search(pattern_to_find,string)):
        needwant_reviews.append(string)
print(needwant_reviews)