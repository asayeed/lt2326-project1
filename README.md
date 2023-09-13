# lt2326-project1
In-class project attempt #1

## Proposal/Brainstorming
Linguistic motivations (understand "language" as such)
Literary patterns
Understanding NLP modeling x3
Modeling human understanding
modeling MT
Practical applications (Free-text survey analysis)

What about literary analysis?
* Authorship attribution (not sufficiently multimodal)
* Book cover OCR
* Genre classification by book cover

We will take a book cover image dataset and train a classifier on various features from the book covers, particularly genre/category.  
Our scientific question is: do book images contain enough information to recover other feature of the work? In other words: can you
really not judge a book by its cover?  This is an important problem because we would like to confirm our intuition that cover art
styles are somewhat predictable by genre, so it has potential significance in the area of literary studies. Possibly unsurprisingly
this topic has been extensively discussed in the literature, most recently by Rashid et al. (2023).  We will be using the data from 
the Book Covers Kaggle challenge for this task, which contains ISBN and genre for a large number of books.  We will 
access the covers data through the OpenLibrary API.  We will extract features from the cover via a convolutional network and classify
using a softmax layer.  We will evaluate this classification straightforwardly using accuracy, precision, and recall.  
