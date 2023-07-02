import json
from tqdm import tqdm

if __name__ == "__main__":
    best_comments = dict()

    with open('./data/full_comments.jsonl', 'r') as comment_f:
        for line in tqdm(comment_f):
            line_json = json.loads(line)
            parent_id = line_json["parent_id"].split("_")[1]
            comment_text = line_json["body"]
            score = line_json["score"]

            if parent_id in best_comments.keys():
                if best_comments[parent_id]["score"] < score:
                    best_comments[parent_id] = line_json
            else:
                best_comments[parent_id] = line_json
    print("Total Comments Registered", len(best_comments))

    nice_posts = []
    with open('./data/full_posts.jsonl', 'r') as f:
        for line in tqdm(f):
            line_json = json.loads(line)
            curr_id = line_json["id"]
            if curr_id in best_comments.keys():
                if best_comments[curr_id]["score"] <= 3:
                    continue

                curr_comment = best_comments[curr_id]
                post_text = line_json.get('title', '') + '\n' + line_json.get('selftext', '')
                combined_json = {
                    "input":post_text,
                    "instruction": "",
                    "output": curr_comment["body"]
                }
                nice_posts.append(combined_json)
    with open('./data/reddit_data.jsonl', 'w') as data_f:
        json.dump(nice_posts, data_f, indent=2)
                # with open('./data/reddit_data.jsonl', 'a') as data_f:
                    # combined_json = {
                    #     "post_id": curr_id,
                    #     "post_score": line_json["score"],
                    #     "post_text": post_text,
                    #     "comment_id": curr_comment["id"],
                    #     "comment_text": curr_comment["body"],
                    #     "comment_score": curr_comment["score"]
                    # }
                    # data_f.write(json.dumps(combined_json) + "\n")
                    