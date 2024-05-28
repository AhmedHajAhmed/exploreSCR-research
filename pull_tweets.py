import csv
import datetime
import tweepy
import os


TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAALZ5sQEAAAAAKA1wuYYiFyTF15QbEGSiL4KcgbM%3DR2tK4tIUaOO7RFPp6JqposBVSC1tTjs5QTxaTkrY2H4ZaTLcuZ"
username = "aboalmout"


def get_tweets(username: str):

    client = tweepy.Client(TWITTER_BEARER_TOKEN)
    user_id = client.get_user(username=username).data.id
    responses = tweepy.Paginator(client.get_users_tweets, user_id, max_results=100, limit=10)
    tweets_list = [["link", "username", "tweet"]]
    currentime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    counter = 0

    for response in responses:
        counter += 1
        try:
            for tweet in response.data:  # see any individual tweet by id at: twitter.com/anyuser/status/TWEET_ID_HERE
                tweets_list.append([f"https://twitter.com/anyuser/status/{tweet.id}", username, tweet.text])
        except Exception as e:
            print(e)

    with open(f"tweets_{username}_{currentime}.csv", "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(tweets_list)

    print("Done!")




if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    get_tweets(username)




