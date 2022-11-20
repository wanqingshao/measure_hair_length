import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, graph
from skimage.morphology import skeletonize
from skimage import morphology
from skan import draw, Skeleton, summarize
import skan
import sys
import optparse
import os
import glob
import re
import pandas as pd

class OptionParser(optparse.OptionParser):
    """
    Adding a method for required arguments.
    Taken from:
    http://www.python.org/doc/2.3/lib/optparse-extending-examples.html
    """
    def check_required(self, opt):
        option = self.get_option(opt)

        # Assumes the option's 'default' is set to None!
        if getattr(self.values, option.dest) is None:
            print("%s option not supplied" % (option), file=sys.stderr)
            self.print_help()
            sys.exit(1)



def load_and_process_image(image_name, folder_name):
    image_orig = cv.imread(folder_name + '/' +image_name)
    grey = cv.cvtColor(image_orig, cv.COLOR_BGR2GRAY)
    th2 = cv.adaptiveThreshold(grey,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY_INV,21,4)
    skeleton1 =  (skeletonize(th2, method='lee')/255).astype(bool)
    skeleton1 = morphology.remove_small_objects(skeleton1, 20, connectivity=30)
    g = Skeleton(skeleton1)
    sk = skeleton1.copy()
    stats = skan.summarize(g)
    thres_min_size = 5
    for ii in range(stats.shape[0]):
        if (stats.loc[ii, 'branch-distance'] < thres_min_size
                and stats.loc[ii, 'branch-type'] == 1):
            # grab NumPy indexing coordinates, ignoring endpoints
            integer_coords = tuple(
                g.path_coordinates(ii)[1:-1].T.astype(int)
            )
            # remove the branch
            sk[integer_coords] = False
    skclean =  morphology.remove_small_objects(skeleton1, 100, connectivity=30)  

    return image_orig, th2, skclean


def output_original_image(image_orig, th2, skclean, image_name, folder_name):
    fig, axes = plt.subplots(1,3, figsize=(18, 4))
    axes[0].imshow(image_orig)
    axes[1].imshow(th2, cmap='gray', vmin=0, vmax=255)
    axes[2].imshow(skclean * 255, cmap='gray', vmin=0, vmax=255)
    fig.savefig(folder_name + "/" + image_name + "_original_grey_and_skeleton.tif")


def angle(v1, v2):
    x_diff = v2[0] - v1[0]
    y_diff = v2[1] - v1[1]
    rad = np.arctan2(y_diff, x_diff)
    degree = np.rad2deg(rad)
    if degree < 0 :
        degree = 360 + degree
    return degree

def get_nearby(to_resolve_starts, to_resolve_ends, anchor_path_coord_end):
    starts_dist = [(x[0] - anchor_path_coord_end[0])**2 + (x[1] - anchor_path_coord_end[1])**2 for x in to_resolve_starts]
    end_dist = [(x[0] - anchor_path_coord_end[0])**2 + (x[1] - anchor_path_coord_end[1])**2 for x in to_resolve_ends]
    starts_of_interest = [i for i in range(len(starts_dist)) if starts_dist[i] <= 4]
    ends_of_interest = [i for i in range(len(end_dist)) if end_dist[i] <= 4]
    return starts_of_interest, ends_of_interest


def search_next_path(anchor_path_coord_end, to_resolve_df, to_resolve_starts, to_resolve_ends, anchor_direct, checked_single_start,new_g, delta, angle_cutoff):
    starts_of_interest, ends_of_interest = get_nearby(to_resolve_starts, to_resolve_ends, anchor_path_coord_end)
    to_review = to_resolve_df.iloc[list(set(starts_of_interest +ends_of_interest))]
    #to_review = to_review.loc[to_resolve_df["node-id-src"] != anchor_start_node]
    to_review = to_review[~to_review.index.isin(checked_single_start)]
    last_path = checked_single_start[-1] 
    last_start_0, last_start_1, last_end_0, last_end_1 = to_resolve_df.loc[last_path, "image-coord-src-0"], to_resolve_df.loc[last_path, "image-coord-src-1"], to_resolve_df.loc[last_path, "image-coord-dst-0"], to_resolve_df.loc[last_path, "image-coord-dst-1"]
    same_start_end = to_review.loc[(to_review["image-coord-src-0"] == last_start_0) &
                                   (to_review["image-coord-src-1"] == last_start_1)&
                                   (to_review["image-coord-dst-0"] == last_end_0)&
                                   (to_review["image-coord-dst-1"] == last_end_1)]
    opp_start_end = to_review.loc[(to_review["image-coord-src-0"] == last_end_0) &
                                   (to_review["image-coord-src-1"] == last_end_1)&
                                   (to_review["image-coord-dst-0"] == last_start_0)&
                                   (to_review["image-coord-dst-1"] == last_start_1)]  
    index_to_rm = list(same_start_end.index) + list(opp_start_end.index)
    to_review = to_review[~to_review.index.isin(index_to_rm)]
    if(to_review.shape[0]>0):
        #print("to be reviewed:")
        #print(to_review.index)
        to_review_angles = []
        to_review_lengths = []
        to_review_path_type = []
        for rr in range(to_review.shape[0]):
            rr_index = to_review.index[rr]
            review_path_coord = new_g.path_coordinates(rr_index)
            direction = find_mid_direction(rr_index, anchor_path_coord_end, to_resolve_df, to_resolve_starts, to_resolve_ends)
            if(len(review_path_coord) >= delta):
                review_path_coord_start, review_path_coord_start_delta = get_path_node(direction, rr_index, delta, to_review, review_path_coord, node_type = "start")
                review_direct = angle(review_path_coord_start, review_path_coord_start_delta )
                path_type = "extended"
            elif(len(review_path_coord) < delta and to_review["branch-type"].to_list()[rr] != 1):
                review_direct = anchor_direct + 30
                path_type = "small_junc"
            else:
                review_direct = 999
                path_type = "small_tail"
            to_review_lengths.append(to_review.loc[rr_index, "branch-distance"])
            to_review_angles.append(review_direct)
            to_review_path_type.append(path_type)
        #print("review angles")
        #print(to_review_angles)
        diff_to_anchor = abs((to_review_angles - anchor_direct))
        most_similar_index = np.where(diff_to_anchor == diff_to_anchor.min())[0]
        most_similar_diff = diff_to_anchor[most_similar_index[0]]
        if most_similar_diff < angle_cutoff:
            if len(most_similar_index) == 1:
                to_extend = to_review.index[most_similar_index[0]]
                path_type = to_review_path_type[most_similar_index[0]]
            else:
                selected_index = ""
                max_length = -1
                for i in range(len(to_review_lengths)):
                    if i in most_similar_index:
                        if to_review_lengths[i] > max_length:
                            max_length  = to_review_lengths[i]
                            selected_index = i
                to_extend = to_review.index[selected_index]
                path_type = to_review_path_type[selected_index]                  
        else:
            to_extend = ""
            path_type = "no_good"
    else:
        to_extend = ""
        path_type = "end"
        most_similar_diff = 0
    return to_extend, path_type, most_similar_diff


def find_tip_direction(ai, to_resolve_df):
    ai_df = to_resolve_df.loc[ai]
    ai_start = ai_df["node-id-src"]
    ai_end = ai_df["node-id-dst"]
    to_resolve_start_to_ai_end = to_resolve_df[to_resolve_df["node-id-src"] == ai_end]
    to_resolve_end_to_ai_start = to_resolve_df[to_resolve_df["node-id-dst"] == ai_start]
    to_resolve_start_to_ai_start = to_resolve_df[to_resolve_df["node-id-src"] == ai_start]
    to_resolve_end_to_ai_end = to_resolve_df[to_resolve_df["node-id-dst"] == ai_end]
    
    if(len(to_resolve_start_to_ai_end) >0):
        direction = "start_to_end"
    elif(len(to_resolve_end_to_ai_start) >0):
        direction = "end_to_start"
    elif(len(to_resolve_start_to_ai_start) >1):
        direction = "end_to_start"
    else:
        direction = "start_to_end"
    return direction

def find_mid_direction(to_extend, last_end, to_resolve_df, to_resolve_starts, to_resolve_ends):
    to_extend_df = to_resolve_df.loc[to_extend]
    to_extend_src = [to_resolve_df.loc[to_extend, "image-coord-src-0"], to_resolve_df.loc[to_extend, "image-coord-src-1"]]
    to_extend_dst = [to_resolve_df.loc[to_extend, "image-coord-dst-0"], to_resolve_df.loc[to_extend, "image-coord-dst-1"]]
    src_dist =sum((np.array(to_extend_src) - np.array(last_end))**2)
    dst_dist =sum((np.array(to_extend_dst) - np.array(last_end))**2)
    if(src_dist < dst_dist):
        direction = "start_to_end"
    else:
        direction = "end_to_start"
    return direction

def get_path_node(direction, ai, delta, to_resolve_df, anchor_path_coord, node_type = "end"):
    if node_type == "end":
        if direction == "start_to_end":
            anchor_path_coord_end = anchor_path_coord[-1]
            delta_new = np.min([delta, len(anchor_path_coord)])
            anchor_path_coord_end_delta = anchor_path_coord[-1 * delta_new]
            if anchor_path_coord_end[0] != to_resolve_df.loc[ai, "image-coord-dst-0"]:
                anchor_path_coord_end = anchor_path_coord[0]
                anchor_path_coord_end_delta = anchor_path_coord[delta_new -1]
        else:
            anchor_path_coord_end = anchor_path_coord[0]
            delta_new = np.min([delta, len(anchor_path_coord)])
            anchor_path_coord_end_delta = anchor_path_coord[delta_new -1]
            if anchor_path_coord_end[0] != to_resolve_df.loc[ai, "image-coord-src-0"]:
                anchor_path_coord_end = anchor_path_coord[0]
                anchor_path_coord_end_delta = anchor_path_coord[delta_new -1]
    else:
        if direction == "end_to_start":
            anchor_path_coord_end = anchor_path_coord[-1]
            delta_new = np.min([delta, len(anchor_path_coord)])
            anchor_path_coord_end_delta = anchor_path_coord[-1 * delta_new]
            if anchor_path_coord_end[0] != to_resolve_df.loc[ai, "image-coord-dst-0"]:
                anchor_path_coord_end = anchor_path_coord[0]
                anchor_path_coord_end_delta = anchor_path_coord[delta_new -1]
        else:
            anchor_path_coord_end = anchor_path_coord[0]
            delta_new = np.min([delta, len(anchor_path_coord)])
            anchor_path_coord_end_delta = anchor_path_coord[delta_new -1]
            if anchor_path_coord_end[0] != to_resolve_df.loc[ai, "image-coord-src-0"]:
                anchor_path_coord_end = anchor_path_coord[0]
                anchor_path_coord_end_delta = anchor_path_coord[delta_new -1]        
    return anchor_path_coord_end, anchor_path_coord_end_delta

 
        
def resolve_path(to_resolve, new_stats, new_g, delta,  angle_cutoff):
    path_list = []
    for ri in to_resolve:
        #print(ri)
        to_resolve_df = new_stats[new_stats["skeleton-id"] == ri]
        #to_resolve_df = dedup(to_resolve_df)
        to_resolve_tips = to_resolve_df[(to_resolve_df["branch-type"] == 1) & (to_resolve_df["branch-distance"] >=delta)].index
        to_resolve_starts = list(zip(to_resolve_df["image-coord-src-0"], to_resolve_df["image-coord-src-1"]))
        to_resolve_ends = list(zip(to_resolve_df["image-coord-dst-0"], to_resolve_df["image-coord-dst-1"]))
        checked_start = []
        for ai in to_resolve_tips:
            if not(ai in checked_start):
                checked_start.append(ai)
                checked_single_start = [ai]
                #print("ri:" + str(ri) + "\tai:" + str(ai))
                to_extend = "start"
                path_type = "start"
                ind_path = [ai]
                total_angle = []
                direction = find_tip_direction(ai, to_resolve_df)
                anchor_path_coord = new_g.path_coordinates(ai)
                anchor_path_coord_end, anchor_path_coord_end_delta = get_path_node(direction, ai, delta, to_resolve_df, anchor_path_coord)          
                anchor_direct = angle(anchor_path_coord_end_delta, anchor_path_coord_end)
                #print("anchor direct " + str(anchor_direct))
                while to_extend != "":
                    to_extend, path_type, angle_diff  = search_next_path(anchor_path_coord_end, to_resolve_df, to_resolve_starts, to_resolve_ends, anchor_direct, checked_single_start, new_g, delta, angle_cutoff)
                    #print(to_extend, path_type)
                    checked_single_start.append(to_extend)
                    if(to_extend != ""):
                        ind_path.append(to_extend)
                        total_angle.append(angle_diff)
                        direction = find_mid_direction(to_extend, anchor_path_coord_end, to_resolve_df, to_resolve_starts, to_resolve_ends)
                        #print(direction)
                        anchor_path_coord = new_g.path_coordinates(to_extend)
                        anchor_path_coord_end, anchor_path_coord_end_delta = get_path_node(direction, to_extend, delta, to_resolve_df, anchor_path_coord)
                        if path_type != "small_junc":
                            anchor_direct = angle(anchor_path_coord_end_delta, anchor_path_coord_end)
                    #print("anchor direct " + str(anchor_direct))
                    #checked_start.append(to_extend)
                    #print(checked_start)
                #checked_start.append(to_extend)
                path_list.append((ind_path, total_angle))
    return path_list

def select_best(path_list, new_stats, dedup = False):
    all_paths = []
    for k in path_list:
        all_paths.extend(k[0])
    k_d = {x:all_paths.count(x) for x in all_paths}    
    path_decreasing = sorted(k_d, key=k_d.get, reverse=True)
    selected_path_list = []
    checked_path_list = []
    for ind_p in path_decreasing:
        if new_stats.loc[ind_p, "branch-distance"] > 40 :
            p_collect = []
            angle_collect = []
            for path in path_list:
                if (not(dedup) and (ind_p in path[0])) or (dedup and (ind_p in path[0]) and not(path in checked_path_list)):
                    start_status = new_stats.loc[path[0][0]]["branch-type"] == 1
                    end_status = new_stats.loc[path[0][-1]]["branch-type"] == 1
                    p_collect.append(path)
                    average_angle = np.mean(path[1])
                    if not(start_status) or not(end_status):
                        average_angle += 90
                    if np.isnan(average_angle):
                        average_angle = 999
                    angle_collect.append(average_angle)
            if len(p_collect) > 0 :
                angle_min = np.argmin(angle_collect)
                selected_path = p_collect[angle_min]
                selected_path_list.append(selected_path)
                checked_path_list.extend(p_collect)
    return selected_path_list


def collect_length(resovled_path_list_uniq, new_stats, image_shape, min_length, include_edge):
    path_to_plot = []
    length_collection = []
    for path in resovled_path_list_uniq:
        ind_path_list = path[0]
        ind_path_stats = new_stats.loc[ind_path_list]
        total_length = sum(ind_path_stats["branch-distance"])
        if total_length >= min_length:
            if not include_edge:
                min_start_0 = min(ind_path_stats["image-coord-src-0"])
                max_start_0 = max(ind_path_stats["image-coord-src-0"])
                min_start_1 = min(ind_path_stats["image-coord-src-1"])
                max_start_1 = max(ind_path_stats["image-coord-src-1"])
                min_end_0 = min(ind_path_stats["image-coord-dst-0"])
                max_end_0 = max(ind_path_stats["image-coord-dst-0"])
                min_end_1 = min(ind_path_stats["image-coord-dst-1"])
                max_end_1 = max(ind_path_stats["image-coord-dst-1"])
                coord_min = min(min_start_0, min_start_1, min_end_0, min_end_1)
                coord_0_max = max(max_start_0, max_end_0)
                coord_1_max = max(max_start_1, max_end_1)
                if coord_min > 10 and coord_0_max < (image_shape[0] - 10) and coord_1_max < (image_shape[1] - 10):
                    path_to_plot.append(ind_path_list)
                    length_collection.append(total_length)
    return path_to_plot, length_collection
     
    

def output_path_plot(input_image_name, output_image_base, in_file_dir, out_file_dir, min_length,include_edge, delta = 10, angle_cutoff = 80):
    image_orig, th2, skclean = load_and_process_image(input_image_name, in_file_dir)
    image_shape = image_orig.shape
    output_original_image(image_orig, th2, skclean, output_image_base, out_file_dir)
    new_g = Skeleton(skclean)
    new_stats = summarize(new_g)
    all_ids = new_stats["skeleton-id"].to_list()
    d = {x:all_ids.count(x) for x in all_ids}
    to_resolve = [k for k,v in d.items() if v > 2]
    singleton = [k for k,v in d.items() if v == 1]
    singleton_path = list(new_stats[new_stats["skeleton-id"].isin(singleton)].index)
    resovled_path_list = resolve_path(to_resolve, new_stats, new_g, delta = delta,  angle_cutoff = angle_cutoff)
    resovled_path_list_best  = select_best(resovled_path_list, new_stats)
    resovled_path_list_uniq  = select_best(resovled_path_list_best, new_stats, dedup = True)
    resovled_path_list_uniq.extend([([x], 0) for x in singleton_path])
    path_to_plot, length_collection = collect_length(resovled_path_list_uniq, new_stats, image_shape, min_length, include_edge)
    for i in range(len(path_to_plot)):
        path = path_to_plot[i]
        img1 = cv.cvtColor(th2, cv.COLOR_GRAY2RGB)
        for p in path:
            coord = new_g.path_coordinates(p)
            for v in coord:
                img1[int(v[0]), int(v[1])] = [255, 0, 0]
                try:
                    img1[(int(v[0])-3):(int(v[0])+3), (int(v[1])-3):(int(v[1])+3)] = [255, 0, 0]
                except:
                    pass
        plt.imsave(out_file_dir +"/" + output_image_base + "_" + str(i) + ".png", img1)
    image_number = re.sub("[A-Z].*", "", output_image_base, flags=re.I)
    image_type = re.sub(".*[0-9]", "", output_image_base)
    length_dic = {"sample_id":image_number, "image_type":image_type, "hair_id":[i for i in range(len(length_collection))], "length":length_collection}
    return length_dic
       

    
def main():
    opt_parser = OptionParser()
    opt_parser.add_option("--in_file_dir",
                          dest="in_file_dir",
                          type="string",
                          help="in_file_dir",
                          default=None)    
    opt_parser.add_option("--out_file_dir",
                          dest="out_file_dir",
                          type="string",
                          help="out_file_dir",
                          default=None)
    opt_parser.add_option("--min_length",
                          dest="min_length",
                          type="int",
                          help="min_length",
                          default=200)
    opt_parser.add_option("--include_edge",
                          dest="include_edge",
                          action="store_true",
                          help="include_edge",
                          default=False)
                
    (options, args) = opt_parser.parse_args()

    # validate the command line arguments
    opt_parser.check_required("--in_file_dir")
    opt_parser.check_required("--out_file_dir")
    
    in_file_dir = options.in_file_dir
    out_file_dir = options.out_file_dir
    min_length = options.min_length
    include_edge = options.include_edge
    
    os.makedirs(out_file_dir, exist_ok=True)
    
    all_images = glob.glob( in_file_dir + "/*tif")
    all_lengths = []
    for image in all_images:
        image_name =  re.sub(".*\/", "", image)
        output_image_base = re.sub(".*\/|.tif", "", image)
        print("Processing " + image_name)
        length_dic = output_path_plot(image_name, output_image_base, in_file_dir, out_file_dir, min_length,include_edge,delta = 15, angle_cutoff = 80)
        length_df = pd.DataFrame.from_dict(length_dic)
        all_lengths.append(length_df)
    all_length_df = pd.concat(all_lengths, ignore_index = True)
    all_length_df.to_csv(out_file_dir + "/outut_length.csv", index = False)
        
if __name__ == "__main__": main()

    