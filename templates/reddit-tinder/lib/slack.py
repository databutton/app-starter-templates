slack_token = "xoxb-1722552209095-3471711466385-pzYvMcKCT3jZWFA7XCjzyafp"
slack_channel = "#user-research"
slack_icon_emoji = ":see_no_evil:"
slack_user_name = "karper"


import json
import requests


def post_message_to_slack(text, blocks=None):
    return requests.post(
        "https://slack.com/api/chat.postMessage",
        {
            "token": slack_token,
            "channel": slack_channel,
            "text": text,
            "icon_emoji": slack_icon_emoji,
            "username": slack_user_name,
            "blocks": json.dumps(blocks) if blocks else None,
        },
    ).json()
