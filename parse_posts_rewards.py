import json
from tqdm import tqdm
import numpy as np

TOP_K = 20
INSTRUCTIONS = [
    "Generate a response based on this Reddit post on r/uwaterloo",
    "What would you comment as a uwaterloo student on this post to get the most upvotes",
    "Respond to this post as if you were a UW student",
    "From the perspective of a Waterloo student, how do you feel about this post",
    "What would you say as a UW student on this post",
    "As a student at the University of Waterloo, what do you think about this post?",
    "Imagine you are a University of Waterloo student. What would you say to this post?",
    "This post was made on the subreddit r/uwaterloo. Imagine you are a student and avid redditor and write a response to this post",
    "What might you say to this reddit post if you were a student at the University of Waterloo and an avid redditor?",
    "You are a University of Waterloo student and frequent redditor. Someone made the following post on the subreddit r/uwaterloo. Write a response to it.",
    "How do you react to this post as a redditor and a student at Waterloo?",
    "Imagine you are studying at the University of Waterloo. How would you reply to this post to get a lot of upvotes?",
    "As a Waterloo student and a reddit user, what is your opinion on this post?",
    "Put yourself in the shoes of a UW student and a reddit fan. How do you comment on this post?",
    "You are a student at Waterloo and you love reddit. What is your take on this post?",
    "You are studying at the University of Waterloo and you are active on reddit. How do you respond to this post?",
    "Imagine you are a University of Waterloo student and you see this reddit post. Write a comment that will get many upvotes on it.",
    "You are a University of Waterloo student. Write a comment to this reddit post to gain maximum reddit karma.",
    "You are a frequent visitor to the subreddit r/uwaterloo and you see this post. Write a comment that will get a lot of karma in response to this post.",
    "What are you thoughts as a student at the University of Waterloo about this reddit post? Write them as a comment that will gain lots of reddit karma."
]

if __name__ == "__main__":
    best_comments = dict()

    with open('./data/full_comments.jsonl', 'r') as comment_f:
        for line in tqdm(comment_f):
            line_json = json.loads(line)
            parent_id = line_json["parent_id"].split("_")[1]
            comment_text = line_json["body"]
            score = line_json["score"]

            if "moderator" in line_json.get("author").lower() or line_json.get('distinguished') == 'moderator':
                continue

            if line_json.get("body") == "[deleted]":
                continue

            if parent_id in best_comments.keys():
                if len(best_comments[parent_id]) < TOP_K:
                    best_comments[parent_id].append(line_json)
                    best_comments[parent_id] = sorted(best_comments[parent_id], key=lambda x: x['score'], reverse=True)
                elif best_comments[parent_id][TOP_K-1]["score"] < score:
                    best_comments[parent_id][TOP_K-1] = line_json
                    best_comments[parent_id] = sorted(best_comments[parent_id], key=lambda x: x['score'], reverse=True)
            else:
                best_comments[parent_id] = [line_json]
    print("Total Comments Registered", len(best_comments))

    nice_posts = []
    with open('./data/full_posts.jsonl', 'r') as f:
        for line in tqdm(f):
            line_json = json.loads(line)
            curr_id = line_json["id"]

            if line_json.get("selftext") == "[deleted]":
                continue 

            if curr_id in best_comments.keys():
                post_text = line_json.get('title', '') + '\n' + line_json.get('selftext', '')

                for _ in range(3):
                    a, b = sorted(np.random.choice(range(len(best_comments[curr_id])), size=2))
                    if a==b: continue
                    comment_chosen = best_comments[curr_id][a]
                    comment_rejected = best_comments[curr_id][b]
                    combined_json = {
                        "input": post_text,
                        "instruction": np.random.choice(INSTRUCTIONS),
                        "chosen": comment_chosen,
                        "rejected": comment_rejected
                    }
                    nice_posts.append(combined_json)                

    with open('./data/reddit_data_rewards.jsonl', 'w') as data_f:
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
