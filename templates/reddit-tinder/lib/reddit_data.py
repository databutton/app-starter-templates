from datetime import datetime

import pandas as pd
import requests


def get_data(hours=1):
    now = int(datetime.today().timestamp())
    t24 = now - hours * 3600
    rq = requests.get(
        "https://api.pushshift.io/reddit/search/submission",
        params={
            "subreddit": "datascience",
            "sort": "desc",
            "sort_type": "created_utc",
            "after": t24,
            "before": now,
            "size": 50,
        },
    )
    if rq.ok:
        data = rq.json()["data"]
        lst = [
            [
                post["title"],
                post["selftext"],
                post["full_link"],
                post["num_comments"],
                post["score"],
                post["retrieved_on"],
            ]
            for post in data
        ]
        df = pd.DataFrame(data, columns=["title", "selftext", "full_link", "num_comments", "score", "retrieved_on"])
        df["time"] = pd.to_datetime(df["retrieved_on"], unit="s")
        return df
    else:
        return rq.ok
