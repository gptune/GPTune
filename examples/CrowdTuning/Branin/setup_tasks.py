import random
import json

if __name__ == "__main__":

    json_data = {
        "source_tasks": [],
        "target_tasks": []
    }

    num_source_tasks = 10
    num_target_tasks = 10

    a_min, a_max = 0.5, 1.5
    b_min, b_max = 0.1, 0.15
    c_min, c_max = 1.0, 2.0
    r_min, r_max = 5.0, 7.0
    s_min, s_max = 8.0, 12.0
    t_min, t_max = 0.03, 0.05

    random.seed(0)

    for i in range(num_source_tasks):
        a = round(random.uniform(a_min, a_max),2)
        b = round(random.uniform(b_min, b_max),2)
        c = round(random.uniform(c_min, c_max),2)
        r = round(random.uniform(r_min, r_max),2)
        s = round(random.uniform(s_min, s_max),2)
        t = round(random.uniform(t_min, t_max),2)

        source_task = {
            "a": a,
            "b": b,
            "c": c,
            "r": r,
            "s": s,
            "t": t
        }

        json_data["source_tasks"].append(source_task)

    for j in range(num_target_tasks):
        a = round(random.uniform(a_min, a_max),2)
        b = round(random.uniform(b_min, b_max),2)
        c = round(random.uniform(c_min, c_max),2)
        r = round(random.uniform(r_min, r_max),2)
        s = round(random.uniform(s_min, s_max),2)
        t = round(random.uniform(t_min, t_max),2)

        target_task = {
            "a": a,
            "b": b,
            "c": c,
            "r": r,
            "s": s,
            "t": t
        }

        json_data["target_tasks"].append(target_task)

    with open("tasklist.json", "w") as f_out:
        json.dump(json_data, f_out, indent=2)

