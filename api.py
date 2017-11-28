#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from flask import Flask, request, jsonify
from utils import normalize_name_for_querying_vector_model,normalize_name_for_space_name, normalize_display_name
from topKSubAreas import TopKSubAreas
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/topics")
def topics():
    start_query = time.time()

    #get parameters from request
    area_name = request.args.get('area', 'machine_learning')
    context = request.args.get('context', 'computer_science')
    k = request.args.get('k', 10, int)
    weighting_mode = request.args.get("weighting_mode", 0, int) #0 is simple version
    compute_mode = request.args.get("compute_mode", 0, int) #0 is mixed wiki, acm and mag
    method = request.args.get("method", "origin")
    confidence = request.args.get("confidence", 0.0, float)
    has_parent = request.args.get("has_parent", 0, int)
    if has_parent >= 1:
        has_parent = True
    else:
        has_parent = False

    # preprocess
    area_name = area_name.strip()
    context = context.strip()
    method = method.strip()

    area_name = normalize_name_for_space_name(area_name)
    context = normalize_name_for_space_name(context)

    #area_name = normalize_name_for_querying_vector_model(area_name)
    #context = normalize_name_for_querying_vector_model(context)


    # get class instance
    topK_subAreas = TopKSubAreas(area = area_name, context = context, k = k, threshold = confidence,
                                 weighting_mode = weighting_mode, compute_mode = compute_mode,
                                 method=method, has_parent=has_parent)
    ranked_scores = topK_subAreas.getResult()
    #print(ranked_scores)

    end_query = time.time()
    r = {'area': normalize_display_name(area_name), 'result': ranked_scores[0], 'parents': ranked_scores[1], 'time':end_query-start_query}
    #return r
    return jsonify(r)
    # return

def main():
    app.run(host="0.0.0.0", port=5097)

if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))
