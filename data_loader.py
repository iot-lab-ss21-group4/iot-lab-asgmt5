import numpy as np
import json
import pandas as pd
import requests as requests

scroll_open_timeout = "1m"
consumer_scroll_api_template = "https://131.159.35.132:443/api/consumers/consume/{}/_search?scroll={}"
entries_per_request = 1000


def load_data(consumer_id: int, consumer_key: str) -> pd.DataFrame:
    """
    Uses the scroll API to get all data from an index
    Args:
        consumer_id: id of the consumer
        consumer_key: bearer token of the consumer. Must start with 'Bearer ey'

    Returns: Count and timestamp in a pandas data frame

    """
    consumer_scroll_api_header = {
        "Authorization": consumer_key,
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    consumer_scroll_api = consumer_scroll_api_template.format(consumer_id, scroll_open_timeout)
    response = requests.get(
        url=consumer_scroll_api,
        headers=consumer_scroll_api_header,
        verify=False,
        data=json.dumps({"size": entries_per_request, "query": {"match_all": {}}}),
    )
    payload = response.json()
    scroll_id = payload["body"]["_scroll_id"]
    hits = payload["body"]["hits"]
    total_hits = hits["total"]
    while True:
        if total_hits <= len(hits["hits"]):
            break
        response = requests.get(
            url=consumer_scroll_api,
            headers=consumer_scroll_api_header,
            verify=False,
            data=json.dumps({"scroll_id": scroll_id, "scroll": scroll_open_timeout}),
        )
        next_hits = response.json()["body"]["hits"]
        hits["hits"].extend(next_hits["hits"])
        hits["total"] = next_hits["total"]
        scroll_id = payload["body"]["_scroll_id"]

    hits_list = hits["hits"]
    data_range = range(total_hits)
    counts_df = pd.DataFrame(index=data_range, columns=["t", "count"])
    for i in data_range:
        timestamp_ms, value = hits_list[i]["_source"]["timestamp"], hits_list[i]["_source"]["value"]
        # Filter / Change ts values here
        counts_df.loc[i] = [int(np.round(timestamp_ms / 1000)), value]

    return counts_df
