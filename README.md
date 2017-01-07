# robotebert
* A machine learning application that generates crowd-sourced movie reviews via Twitter sentiment analysis 
* A humble homage to the late great everyman's movie critic :)

* Basic Approach:
    + Use labeled external twitter data to train a TFIDF Vectorizer and Ridge Logistic Regression Classifier
    + Pull the topN box office movies for each day
    + Pull tweets that mention each movie from that day
    + Generate tweet-based sentiment metrics for each movie for each day
    + Use a combination of self-training and active learning to augment the training data and improve the model over time

* Draws data from:
    + [VADER Annotated Tweet Data](https://github.com/cjhutto/vaderSentiment)
    + [Box Office Mojo](http://www.boxofficemojo.com/)
    + [OMDB API](http://www.omdbapi.com/)
    + [Twitter Search API](https://dev.twitter.com/rest/public/search)
    
* To-Do
    + Explore other initial training sets (esp. those with emojis)
    + Track self-training and active learning progress wrt validation metrics
    + develop a web-based front-end with visualizations via Flask/Heroku

