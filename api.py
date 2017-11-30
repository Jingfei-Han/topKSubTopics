#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from flask import Flask, request, jsonify
from utils import normalize_name_for_space_name, normalize_display_name
from utils import get_hierarchy, get_midDict
from topKSubAreas import TopKSubAreas
from flask_cors import CORS
import json
from collections import defaultdict, OrderedDict
import copy

app = Flask(__name__)
CORS(app)

app.config["JSON_SORT_KEYS"] = False

def get_topk(area_name, context, k, kp, weighting_mode, compute_mode, method, confidence, has_parents, has_children):
    topK_subAreas = TopKSubAreas(area = area_name, context = context, k = k, kp = kp, threshold = confidence,
                                 weighting_mode = weighting_mode, compute_mode = compute_mode,
                                 method=method, has_parents=has_parents, has_children=has_children)
    ranked_scores = topK_subAreas.getResult()
    return ranked_scores

#hierarchy:
@app.route("/hierarchy")
def hierarchy():
    start_query = time.time()

    #get parameters from request
    area_name = request.args.get('area', 'machine_learning')
    context = request.args.get('context', 'computer_science')
    k = request.args.get('k', 10, int)
    kp = request.args.get("kp", 3, int)
    weighting_mode = request.args.get("weighting_mode", 0, int) #0 is simple version
    compute_mode = request.args.get("compute_mode", 0, int) #0 is mixed wiki, acm and mag
    method = request.args.get("method", "origin")
    confidence = request.args.get("confidence", 0.0, float)
    has_parents = request.args.get("has_parents", 0, int)
    has_children = request.args.get("has_children", 1, int)
    depth_of_tree = request.args.get("depth", 1, int) #depth: the depth of hierarchical tree
    depth_of_parents = request.args.get("depth_p", 1, int) #depth_p: the depth of parents tree

    if has_parents >= 1:
        has_parents = True
    else:
        has_parents = False

    if has_children >=1:
        has_children = True
    else:
        has_children = False

    # preprocess
    area_name = area_name.strip()
    context = context.strip()
    method = method.strip()

    area_name = normalize_name_for_space_name(area_name)
    context = normalize_name_for_space_name(context)
    display_name = normalize_display_name(area_name)
    if has_children:
        dic = get_midDict(area_name = area_name, context = context, k = k, kp = kp, weighting_mode = weighting_mode,
                          compute_mode= compute_mode, method = method, confidence= confidence,
                          depth_of_tree= depth_of_tree, c_or_p = "children")
        tmp = dic[display_name]
        tmp.pop("level")
        root = copy.deepcopy(dic[display_name])
        showTopics = set()
        root = get_hierarchy(data = dic, root = root, name = display_name, n = 0,
                             depth = depth_of_tree, showTopics= showTopics, c_or_p= "children")
    if has_parents:
        dic_p = get_midDict(area_name = area_name, context = context, k = k, kp = kp, weighting_mode = weighting_mode,
                          compute_mode= compute_mode, method = method, confidence= confidence,
                          depth_of_tree= depth_of_parents, c_or_p = "parents")
        tmp = dic_p[display_name]
        tmp.pop("level")
        root_p = copy.deepcopy(dic_p[display_name])
        showTopics_p = set()
        root_p = get_hierarchy(data = dic_p, root = root_p, name = display_name, n = 0,
                             depth = depth_of_tree, showTopics= showTopics_p, c_or_p= "parents")

    root_res = defaultdict(OrderedDict)
    root_res["name"] = display_name
    if has_children:
        root_res["children"] = root["children"]
    if has_parents:
        root_res["parents"] = root_p["parents"]

    end_query = time.time()
    #compute_time = end_query - start_query
    return jsonify(root_res)

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

    # get class instance
    topK_subAreas = TopKSubAreas(area = area_name, context = context, k = k, threshold = confidence,
                                 weighting_mode = weighting_mode, compute_mode = compute_mode,
                                 method=method, has_parent=has_parent)
    ranked_scores = topK_subAreas.getResult()

    end_query = time.time()
    #compute_time = end_query - start_query
    if has_parent:
        r = {'area': normalize_display_name(area_name), 'result': ranked_scores[0], 'parents': ranked_scores[1]}
    else:
        r = {'area': normalize_display_name(area_name), 'result': ranked_scores[0]}

    return jsonify(r)

def main():
    app.run(host="0.0.0.0", port=5097)

if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))
