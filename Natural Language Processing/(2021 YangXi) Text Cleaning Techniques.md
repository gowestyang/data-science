* lower / upper case
* Punctuation (punkt from nltk)
    * Represent interrupting punctuation marks (, ! . ?) as a single special word
    * Remove non-interrupting punctuation marks, such as " ' < >
    * Collapse multi-sign markds, such as ... !! ???
* Numbers
    * Keep relevant numbers, for example, area code
    * If there are too many unique numbers, replace all number with a special token, such as </NUMBER>
* Special characters
* Special words: Emoji (emoji), hash tags, etc

* short words
* TFIDF
* stem / lem