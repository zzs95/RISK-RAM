import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import spacy
import numpy as np
import pandas as pd
import torch
import torchvision
from dataset.constants import REPORT_KEYS, COVID_REGIONS, ANATOMICAL_REGIONS_
from evaluate_utils.write_utils import write_all_losses_and_scores_to_tensorboard_feat
from dataset.create_image_report_dataloader import get_data_loaders
from models.risk_model.region_SP_model import RegionSPModel
from configs.risk_prediction_config import *
from path_datasets_and_weights import path_runs
from utils.utils import write_config, seed_everything
from utils.file_and_folder_operations import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sksurv.metrics import concordance_index_censored
import cv2
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

class GradCAM():
    '''
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al. 
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    '''
    def __init__(self, model, target_layers, dim=1280, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers
        self.dim = dim
        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)
        
        self.activations = []
        self.grads = []
        
    def forward_hook(self, module, input, output):
        # self.activations.append(input[0])
        self.activations.append(output[0])
        
    def backward_hook(self, module, grad_input, grad_output):
        # self.grads.append(grad_input[0].detach())
        self.grads.append(grad_output[0].detach())
        
    def calculate_cam(self, model_input):
        self.model.eval()
        
        # forward
        y_c = self.model.risk_predict_grad(model_input)

        # backward
        self.model.zero_grad()
        y_c.backward()
        
        # get activations and gradients
        activations = self.activations[0].cpu().data.numpy().squeeze()
        grads = self.grads[0].cpu().data.numpy().squeeze()
        
        # calculate weights
        weights = np.mean(grads.reshape(grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1, 1)
        cam = (weights * activations).sum(axis=0)
        cam = np.maximum(cam, 0) # ReLU
        cam = cam / cam.max()
        return cam
    
    @staticmethod
    def show_cam_on_image(image, cam):
        # image: [H,W,C]
        image = image[:,:,None].repeat(3, 2)
        h, w = image.shape[:2]
        
        cam_resized = cv2.resize(cam, (h,w))
        cam_resized = cam_resized / cam_resized.max()
        heatmap = cv2.applyColorMap((255*cam_resized).astype(np.uint8), cv2.COLORMAP_JET) # [H,W,C]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap / heatmap.max()
        
        return cam_resized, heatmap 
    
def plot_box(box, ax, clr, linestyle, linewidth=1, region_detected=True):
    x0, y0, x1, y1 = box
    h = y1 - y0
    w = x1 - x0
    ax.add_artist(
        plt.Rectangle(xy=(x0, y0), height=h, width=w, fill=False, color=clr, linewidth=linewidth, linestyle=linestyle)
    )

    # add an annotation to the gt box, that the pred box does not exist (i.e. the corresponding region was not detected)
    if not region_detected:
        ax.annotate("not detected", (x0, y0), color=clr, weight="bold", fontsize=10)


def plot_img_hm(image, hm):
    image = image[:,:,None].repeat(3, 2)
    image = (image - image.min()) / (image.max() - image.min())
    heatmap = hm / hm.max()
    result = 0.4*heatmap + 0.6*image
    result = result / result.max()
    return result
    
    
    
def gen_model(
    model,
    train_dl,
    device,
    run_folder_path,
    cfg,
    set_name
):
    model.eval()
    gen_text = {
        "idxs": [],
        "time_label": [],
        'event_label': [],
        'risk_score': [],
        'risk_region_order':[],
        'risk_region_value':[],
        
    }
    maybe_mkdir_p(join(run_folder_path, set_name))
    for num_batch, batch in enumerate(train_dl):
        # if num_batch != 127:
        #     continue
        if batch['accNum'] != 53680988:
            continue
        print('has')
        print(num_batch)
        case_path = join(run_folder_path, set_name, str(batch['idxs'][0].numpy()))
        maybe_mkdir_p(case_path)
        
        img_path = join(run_folder_path, set_name, 'images')
        hm_path = join(run_folder_path, set_name, 'heatmaps')
        img_hm_path = join(run_folder_path, set_name, 'img_hms')
        r_hm_path = join(run_folder_path, set_name, 'region_hms')
        boxes_path = join(run_folder_path, set_name, 'boxes')
        maybe_mkdir_p(img_path)
        maybe_mkdir_p(hm_path)
        maybe_mkdir_p(img_hm_path)
        maybe_mkdir_p(r_hm_path)
        maybe_mkdir_p(boxes_path)
        
        torch.cuda.empty_cache()
        batch_size = batch['clin_feat'].shape[0]
        with torch.no_grad(): 
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                risk_score = model.risk_predict(cfg, batch, device)

        grad_cam = GradCAM(model, model.img_encoder.encoder.layer4[-1])
        cam = grad_cam.calculate_cam(batch)
        image = batch['images'][0,0].numpy()
        cam_resized, heatmap = GradCAM.show_cam_on_image(image, cam)      
        
        
        out_size = 128
        region_cams = torchvision.ops.roi_align(input=torch.tensor(cam_resized)[None, None], boxes=batch['boxes'], output_size=out_size, spatial_scale=1, aligned=False)
        region_cams = region_cams[:,0]
        
        region_hms = torchvision.ops.roi_align(input=torch.tensor(heatmap.transpose([2,0,1]))[None].float(), boxes=batch['boxes'], output_size=out_size, spatial_scale=1, aligned=False)
        region_hms = region_hms.permute(0,2,3,1).numpy()
        
        region_images = torchvision.ops.roi_align(input=batch['images'], boxes=batch['boxes'], output_size=out_size, spatial_scale=1, aligned=False)
        region_images = region_images[:,0].numpy()
        region_name_list = list(ANATOMICAL_REGIONS_.keys())        
        dpi = 300
        plt.figure(figsize=(15, 15), dpi=dpi)
        plt.axis("off")
        plt.imshow(image)
        plt.savefig(join(case_path, 'image.jpg'))
        # plt.savefig(join(img_path, str(batch['idxs'][0].numpy()) + '.jpg'))
        
        plt.figure(figsize=(15, 15), dpi=dpi)
        plt.axis("off")
        plt.imshow(heatmap)
        plt.savefig(join(case_path, 'heatmap.jpg'))  
        # plt.savefig(join(hm_path, str(batch['idxs'][0].numpy()) + '.jpg'))
        
        plt.figure(figsize=(15, 15), dpi=dpi)
        plt.axis("off")
        plt.imshow(plot_img_hm(image, heatmap))
        plt.savefig(join(case_path, 'img_hm.jpg'))
        # plt.savefig(join(img_hm_path, str(batch['idxs'][0].numpy()) + '.jpg'))


        fig, axs = plt.subplots(5, 6, figsize=(15, 15), dpi=dpi)
        for i in range(5):
            for j in range(6):
                idx = j+ i*6 
                if idx < 29:
                    ax = axs[i, j]
                    hms_ = region_hms[idx]
                    imgs_ = region_images[idx]
                    img_hm = plot_img_hm(imgs_, hms_)
                    ax.imshow(img_hm)  
                    ax.axis("off")
                    ax.set_title(f'{idx + 1} {region_name_list[idx]}')  
        for idx in range(29):
            fig, ax =  plt.subplots(figsize=(3, 3), dpi=dpi)
            plt.axis("off")
            hms_ = region_hms[idx]
            imgs_ = region_images[idx]
            img_hm = plot_img_hm(imgs_, hms_)
            plt.imshow(img_hm)  
            ax.axis("off")
            ax.set_title(f'{idx + 1} {region_name_list[idx]}')
            plt.savefig(str(idx) + '.jpg') 
            plt.close('all')

        plt.tight_layout()
        plt.savefig(join(case_path, 'region_hms.jpg'))
        # plt.savefig(join(r_hm_path, str(batch['idxs'][0].numpy()) + '.jpg'))
        
        bboxes = batch['boxes'][0].numpy()
        fig = plt.figure(figsize=(15, 15), dpi=dpi)
        ax = plt.gca()

        image = cv2.imread('127_a.jpg')[:,:,0]
        bboxes = bboxes /224 * 512
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        for i in range(29):
            box_gt = bboxes[i].tolist()
            color = 'orangered'
            linewidth = 1
            if i == 12:
                color = 'lime'
                linewidth = 2
            elif i == 10:
                color = 'b'
                linewidth = 2
            elif i == 27:
                color = 'g'
                linewidth = 2
            elif i == 25:
                color = 'c'
                linewidth = 2
            elif i == 20:
                color = 'm'
                linewidth = 2
            
            plot_box(box_gt, ax, clr=color, linestyle="solid", linewidth=linewidth, region_detected=True)
        plt.savefig(join(case_path, 'boxes.jpg'))       
        # plt.savefig(join(boxes_path, str(batch['idxs'][0].numpy()) + '.jpg'))
        plt.close('all')

        cam_weight = region_cams.reshape(29,-1).sum(-1)
        # w = batch['boxes'][0][:,3] - batch['boxes'][0][:,1]
        # h = batch['boxes'][0][:,2] - batch['boxes'][0][:,0]
        # cam_weight = cam_weight / (w*h) # do not size normalize, bcz the larger hm in a large space will get a high score
        cam_weight = cam_weight / cam_weight.max()
        cam_index = torch.sort(cam_weight, descending=True)[1] 
        cam_weight = cam_weight * risk_score[0].cpu()
        order_names = ''
        for i in range(10):
            idx = cam_index[i]
            order_names += region_name_list[idx]
            order_names += ', '
        
        order_values = ''
        for i in range(10):
            idx = cam_index[i]
            order_values += str(cam_weight[idx].numpy())[:7]
            order_values += ', '
            print(f'{cam_weight[idx].numpy():.3f}')
                    
        gen_text["idxs"].extend(batch["idxs"].numpy())
        gen_text["time_label"].extend(batch["time_label"].cpu().numpy())
        gen_text["event_label"].extend(batch["event_label"].cpu().numpy())
        gen_text["risk_score"].extend(risk_score[:,0].cpu().numpy())
        gen_text["risk_region_order"].extend([order_names])
        gen_text["risk_region_value"].extend([order_values])
        
    pd.DataFrame.from_dict(gen_text).to_excel(os.path.join(run_folder_path, set_name+'.xlsx'))
    return None

def get_model(device):
    model = RegionSPModel()
    CHECKPOINT = '/home/brownradai/Public/hhd_3t_shared/covid_cxr_RSPNet/base_SP_model_seed45/run_1/45/checkpoints/epoch_99.pt'
    sur_checkpoint = torch.load(CHECKPOINT, map_location=torch.device("cpu"))['model']

    model.load_state_dict(sur_checkpoint)
    model.to(device, non_blocking=True)
    
    del sur_checkpoint
    return model


def create_run_folder(SEED):
    run_folder_path = os.path.join(path_runs, exp_name, f"revieaw_run_{RUN}",)
    checkpoints_folder_path = os.path.join(run_folder_path, "checkpoints")
    tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")
    
    if os.path.exists(run_folder_path):
        log.info(f"Folder to save run {RUN} already exists at {run_folder_path}. ")
        # if not RESUME_TRAINING:
        #     log.info(f"Delete the folder {run_folder_path}.")
        #     if local_rank == 0:           
        #         shutil.rmtree(run_folder_path)
    maybe_mkdir_p(run_folder_path)
    maybe_mkdir_p(checkpoints_folder_path)
    maybe_mkdir_p(tensorboard_folder_path)
    log.info(f"Run {RUN} folder created at {run_folder_path}.")
        
    config_file_path = os.path.join(run_folder_path, "run_config.txt")
    config_parameters = {
        "COMMENT": RUN_COMMENT,
        "SEED": SEED,
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        "EVALUATE_EVERY_K_EPOCHS": EVALUATE_EVERY_K_EPOCHS,
        "EPOCHS": epochs,
        "MULTI_GPU": MULTI_GPU,
        "BATCH_SIZE": BATCH_SIZE,
        "checkpoints_folder_path": checkpoints_folder_path
    }
    return run_folder_path, tensorboard_folder_path, config_file_path, config_parameters

BATCH_SIZE = 1
def main():
    (run_folder_path, tensorboard_folder_path, config_file_path, config_parameters) = create_run_folder(SEED)
    seed_everything(config_parameters['SEED'])
    data_seed = 45
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # model, text_sur_predictor, text_encoder = get_model(device)
    model = get_model(device)
    
    # train_loader, val_loader, test_loader, gpt_tokenizer, train_sampler = get_data_loaders(setname='brown', batch_size=config_parameters['BATCH_SIZE'], 
    #                                                                                        image_input_size=IMAGE_INPUT_SIZE, is_token=False, random_state_i=data_seed,
    #                                                                                        worker_seed=config_parameters['SEED'], )
    # config_parameters["TRAIN total NUM"] = len(train_loader.dataset)
    # config_parameters["TRAIN index NUMs"] = train_loader.dataset.tokenized_dataset['index']
    # config_parameters["VAL total NUM"] = len(val_loader.dataset)
    # config_parameters["VAL index NUMs"] = val_loader.dataset.tokenized_dataset['index']
    # config_parameters["TEST total NUM"] = len(test_loader.dataset)
    # config_parameters["TEST index NUMs"] = test_loader.dataset.tokenized_dataset['index']
    
    test_loader, gpt_tokenizer = get_data_loaders(setname='brown', batch_size=config_parameters['BATCH_SIZE'], 
                                                    image_input_size=IMAGE_INPUT_SIZE, is_token=False, random_state_i=data_seed,
                                                    worker_seed=config_parameters['SEED'], return_all=True )

    log.info("Starting generating!")
    gen_model(
        model=model,
        train_dl=test_loader,# test_loader,
        device=device,
        run_folder_path=run_folder_path,
        cfg=config_parameters,
        set_name='brown'
    )

    test_loader, gpt_tokenizer = get_data_loaders(setname='penn', batch_size=config_parameters['BATCH_SIZE'], image_input_size=IMAGE_INPUT_SIZE, return_all=True, 
                                                                                           random_state_i=config_parameters['SEED'])
    config_parameters["Penn TEST total NUM"] = len(test_loader.dataset)
    config_parameters["Penn TEST index NUMs"] = test_loader.dataset.tokenized_dataset['index']
    write_config(config_file_path, config_parameters)
    # gen_model(
    #     model=model,
    #     train_dl=test_loader,
    #     device=device,
    #     run_folder_path=run_folder_path,
    #     cfg=config_parameters,
    #     set_name='penn'
    # )

if __name__ == "__main__":
    exp_name = 'gen_region_risk'
    main()
