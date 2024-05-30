import os
import sys
sys.path.append('/media/brownradai/ssd_2t/covid_cxr/region_surv')
sys.path.append('/home/brownradai/Projects/covid_cxr/RSPNet/')
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from utils.file_and_folder_operations import *
ANATOMICAL_REGIONS = {
    "right lung": 0,
    "right upper lung zone": 1,
    "right mid lung zone": 2,
    "right lower lung zone": 3,
    "right hilar structures": 4,
    "right apical zone": 5,
    "right costophrenic angle": 6, # x
    "right hemidiaphragm": 7,
    "left lung": 8,
    "left upper lung zone": 9,
    "left mid lung zone": 10,
    "left lower lung zone": 11,
    "left hilar structures": 12,
    "left apical zone": 13,
    "left costophrenic angle": 14, # x
    "left hemidiaphragm": 15,
    "trachea": 16,
    "spine": 17,
    "right clavicle": 18,
    "left clavicle": 19,
    "aortic arch": 20,
    "mediastinum": 21,
    "upper mediastinum": 22,
    "svc": 23,
    "cardiac silhouette": 24,
    "cavoatrial junction": 25, # x
    "right atrium": 26,
    "carina": 27,
    "abdomen": 28,
}

COVID_REGIONS = {
    # 'Lines/tubes':[-1],
    'Lungs':[0,1,2,3,4,5,7, 8,9,10,11,12,13,15],
    'Pleura':[0,8],
    'Heart and mediastinum':[16, 20, 21, 22, 23, 24, 26, 27],
    'Bones':[17, 18, 19],
    # 'Others':[-1],
    }


def plot_box(box, ax, clr, linestyle, region_detected=True):
    x0, y0, x1, y1 = box
    h = y1 - y0
    w = x1 - x0
    ax.add_artist(
        plt.Rectangle(xy=(x0, y0), height=h, width=w, fill=False, color=clr, linewidth=1, linestyle=linestyle)
    )

    # add an annotation to the gt box, that the pred box does not exist (i.e. the corresponding region was not detected)
    if not region_detected:
        ax.annotate("not detected", (x0, y0), color=clr, weight="bold", fontsize=10)


def plot_detections(
    image,
    bboxes,
    id_
):
    regions_sets = COVID_REGIONS
    bboxes = np.concatenate([bboxes, np.array([[1,1,510,510]])], axis=0)
    regions_sets['Bones'].append(29)
    for num_region_set, region_k in enumerate(regions_sets):
        region_idx = regions_sets[region_k]
        fig = plt.figure(figsize=(8, 8), dpi=400)
        ax = plt.gca()

        plt.imshow(image, cmap="gray")
        plt.axis("off")

        region_colors = ["b", "g", "r", "c", "m", "y", 'lime',   'cornflowerblue', 'limegreen', 'tomato', 'aquamarine',  'violet', 'lemonchiffon', 'palegreen']

        if num_region_set == 4:
            region_colors.pop()


        for region_index, color in zip(region_idx, region_colors):
            # box_gt and box_pred are both [List[float]] of len 4
            box_gt = bboxes[region_index].tolist()

            plot_box(box_gt, ax, clr=color, linestyle="solid", region_detected=True)

        # using writer.add_figure does not correctly display the region_set_text in tensorboard
        # so instead, fig is first saved as a png file to memory via BytesIO
        # (this also saves the region_set_text correctly in the png when bbox_inches="tight" is set)
        # then the png is loaded from memory and the 4th channel (alpha channel) is discarded
        # finally, writer.add_image is used to display the image in tensorboard
        
        fig.savefig(join(bboxes_plot_img_path, id_, str(num_region_set)+'.jpg'), bbox_inches="tight")
        plt.close(fig)

if __name__=='__main__':
    path_internal_dataset = "/home/brownradai/Projects/covid_cxr/covid_cxr_data/preprocess/"
    # path_runs = "/home/brownradai/Public/hhd_3t_shared/covid_cxr_RSPNet"
    # path_internal_dataset = "/media/brownradai/ssd_2t/covid_cxr/covid_cxr_data/preprocess"
    for setname in ['brown', 'penn']:
        bboxes_plot_img_path = join(path_internal_dataset, setname+'_img_box_400dpi')
        maybe_mkdir_p(bboxes_plot_img_path)
        img_all = np.load(os.path.join(path_internal_dataset, setname+'_imgs_255_corr.npy')).astype(np.float32) 
        coord_all = np.load(os.path.join(path_internal_dataset, setname+'_bboxes.npy')).astype(np.float32) 
        if setname == 'brown':
            datasets_df = pd.read_excel(os.path.join(path_internal_dataset, setname+'_table_w_report_split_corr.xlsx'))
        elif setname=='penn':
            datasets_df = pd.read_excel(os.path.join(path_internal_dataset, setname+'_data_corr.xlsx'), index_col=0)
        ids_list = []
        curr_dict_list = []
        for i in range(len(img_all)):
            study_id = int(datasets_df.iloc[i]['Accession Number'])
            id_ = str(study_id)
            # if id_ == '53673759':
            if id_ in ids_list:
                id_ = str(study_id)+'_a'
                if id_ in ids_list:
                    id_ = str(study_id)+'_b'
                    if id_ in ids_list:
                        id_ = str(study_id)+'_c'
                        if id_ in ids_list:
                            id_ = str(study_id)+'_d'
                            if id_ in ids_list:
                                id_ = str(study_id)+'_e'
                                if id_ in ids_list:
                                    id_ = str(study_id)+'_f'
                                    if id_ in ids_list:
                                        id_ = str(study_id)+'_g'
                                        if id_ in ids_list:
                                            id_ = str(study_id)+'_h'
            if id_ == '53781498' or id_ == '53838559' or id_ == '53680988':
                ids_list.append(id_)
                maybe_mkdir_p(join(bboxes_plot_img_path, id_))
                image = img_all[i]
                bboxes = coord_all[i]
                plot_detections(image, bboxes, id_)