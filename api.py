#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from flask import Flask, request, jsonify
from utils import normalize_name_for_space_name, normalize_display_name
from topKSubAreas import TopKSubAreas
from flask_cors import CORS
import json
from collections import defaultdict, OrderedDict
import copy
import os

app = Flask(__name__)
CORS(app)

app.config["JSON_SORT_KEYS"] = False

def get_topk(area_name, context, k, kp, weighting_mode, compute_mode, method, confidence, has_parents, has_children):
    topK_subAreas = TopKSubAreas(area = area_name, context = context, k = k, kp = kp, threshold = confidence,
                                 weighting_mode = weighting_mode, compute_mode = compute_mode,
                                 method=method, has_parents=has_parents, has_children=has_children)
    ranked_scores = topK_subAreas.getResult()
    return ranked_scores

def get_hierarchy(data, root, name, n, depth, showTopics, c_or_p="children"):
    if len(root[c_or_p]) == 0:
        root[c_or_p] = []
        return root
    if n == depth:
        root[c_or_p] = []
        return root
    else:
        for i in range(len(root[c_or_p])-1, -1, -1):
            name = root[c_or_p][i]
            if name not in showTopics:
                showTopics.add(name)
            else:
                del root[c_or_p][i]
                continue
            tmp = data[name]
            try:
                tmp.pop("level")
            except:
                pass
            root_tmp = copy.deepcopy(tmp)
            root[c_or_p][i] = get_hierarchy(data, root_tmp, name, n+1, depth, showTopics, c_or_p=c_or_p)
    return root

def get_midDict(area_name, context, k, kp, weighting_mode, compute_mode, method, confidence, depth_of_tree, c_or_p="children"):

    c_or_p = c_or_p.lower().strip()
    if c_or_p != "children":
        c_or_p = "parents"
    cur_index, has_parents, has_children = (0, False, True) if c_or_p == "children" else (1, True, False)

    result = get_topk(area_name, context, k, kp, weighting_mode, compute_mode, method, confidence, has_parents, has_children)
    display_name = normalize_display_name(area_name)
    """save to an OrderedDefaultDict -- children"""
    dic = defaultdict(OrderedDict)

    dic[display_name]["name"] = display_name
    dic[display_name]["level"] = 0
    dic[display_name][c_or_p] = result[cur_index]

    for depth in range(1, depth_of_tree + 1):
        for topic in list(dic):
            if dic[topic]["level"] == depth - 1:
                for subtopic in dic[topic][c_or_p]:
                    # subtopic is a display name
                    if subtopic not in dic.keys():
                        tmp = get_topk(subtopic, context, k, kp, weighting_mode, compute_mode, method, confidence, has_parents=has_parents, has_children=has_children)
                        dic[subtopic]["name"] = subtopic
                        dic[subtopic]["level"] = depth
                        dic[subtopic][c_or_p] = tmp[cur_index]

    def preprocess(c_or_p):
        #preprocess: handle the same subtopic
        #Record childset
        childRecord = [display_name]
        for curParent in childRecord:
            if curParent in dic:
                dic[curParent][c_or_p] = dic[curParent][c_or_p][::-1] # reverse, in order to delete
                childList = dic[curParent][c_or_p]
                for i in range(len(childList)-1, -1, -1):
                    curChild = childList[i]
                    if curChild in childRecord:
                        del childList[i]
                    else:
                        childRecord.append(curChild)
                dic[curParent][c_or_p] = dic[curParent][c_or_p][::-1]
            else:
                continue
        return dic

    dic = preprocess(c_or_p)

    return dic

#hierarchy:
@app.route("/topics")
def topics():
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

    # restrict k, kp <= 30, depth_of_tree, depth_of_parents <=5
    k = min(k, 30)
    kp = min(kp, 30)
    depth_of_tree = min(depth_of_tree, 5)
    depth_of_parents = min(depth_of_parents, 5)

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

    #cache name
    filename = "_".join("{}" for i in range(12)).format(area_name, context, k, kp, confidence, weighting_mode, compute_mode, method,
                            depth_of_tree, depth_of_parents, has_children, has_parents)
    filename = os.path.join(cachePath, filename)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            result = json.load(f, object_pairs_hook=OrderedDict)
            return jsonify(result)

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
                             depth = depth_of_parents, showTopics= showTopics_p, c_or_p= "parents")

    root_res = defaultdict(OrderedDict)
    root_res["name"] = display_name
    len_chidren = 0
    len_parents = 0
    if has_children:
        root_res["children"] = root["children"]
        len_chidren = len(root_res["children"])
    if has_parents:
        root_res["parents"] = root_p["parents"]
        len_parents = len(root_p["parents"])

    if not os.path.exists(filename):
        if len_chidren !=0 or len_parents != 0:
            with open(filename, "w") as f:
                json.dump(root_res, f, indent=4)

    end_query = time.time()
    #compute_time = end_query - start_query
    return jsonify(root_res)

def main():
    app.run(host="0.0.0.0", port=5098)

if __name__ == '__main__':
    start_t = time.time()
    global cachePath
    cachePath = "./.cache/"
    #cachePath = "./.cache_debug/"
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))
