from ast import Lambda
import numpy as np
import pdb
import os
from custom_optims.radam import RAdam
from models.model_ABMIL import ABMIL
from models.model_DeepMISL import DeepMISL
from models.model_MLPOmics import MLPOmics
from models.model_MLPWSI import MLPWSI
from models.model_SNNOmics import SNNOmics
from models.model_MaskedOmics import MaskedOmics
from models.model_MCATPathways import MCATPathways
from models.model_SurvPath import SurvPath
from models.model_SurvPath_with_nystrom import SurvPath_with_nystrom
from models.model_TMIL import TMIL
from models.model_DoubleMediatorSurvPath import DoubleMediatorSurvPath
# from models.model_motcat import MCATPathwaysMotCat
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from tqdm import tqdm

from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)


#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss

import torch.optim as optim



def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually
    
    Args:
        - datasets : tuple
        - cur : Int 
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split,val_split


def _init_loss_function(args):
    r"""
    Init the survival loss function
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss
    
    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):
    
    print('\nInit Model...', end=' ')
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0
    
    # omics baselines
    if args.modality == "mlp_per_path":

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "dropout" : args.encoder_dropout, "num_classes" : args.n_classes,
        }
        model = MaskedOmics(**model_dict)

    elif args.modality == "omics":

        model_dict = {
             "input_dim" : omics_input_dim, "projection_dim": 64, "dropout": args.encoder_dropout
        }
        model = MLPOmics(**model_dict)

    elif args.modality == "snn":

        model_dict = {
             "omic_input_dim" : omics_input_dim, 
        }
        model = SNNOmics(**model_dict)

    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = ABMIL(**model_dict)

    # unimodal and multimodal baselines
    elif args.modality in ["deepmisl_wsi", "deepmisl_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = DeepMISL(**model_dict)

    elif args.modality == "mlp_wsi":
        
        model_dict = {
            "wsi_embedding_dim":args.encoding_dim, "input_dim_omics":omics_input_dim, "dropout":args.encoder_dropout,
            "device": args.device

        }
        model = MLPWSI(**model_dict)

    elif args.modality in ["transmil_wsi", "transmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = TMIL(**model_dict)

    elif args.modality == "coattn":

        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCATPathways(**model_dict)

    elif args.modality == "coattn_motcat":

        model_dict = {
            'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,
            "ot_reg":0.1, "ot_tau":0.5, "ot_impl":"pot-uot-l2"
        }
        model = MCATPathwaysMotCat(**model_dict)

    # survpath 
    elif args.modality == "survpath":

        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes}

        if args.use_nystrom:
            model = SurvPath_with_nystrom(**model_dict)
        else:
            model = SurvPath(**model_dict)
            
    elif args.modality == "ccd2_survpath": # 改.......
        from models.model_CCD import CCD_SurvPath
        
        # 构造通路先验概率（基于生物学重要性）
        pathway_priors = torch.tensor([
            0.12, 0.15, 0.10, 0.08, 0.12, 0.10, 0.08, 0.09, 0.08, 0.08  # 前10个通路的先验
            # 可以根据实际通路数量调整
        ] + [0.01] * (len(args.omic_sizes) - 10))  # 其余通路设较小先验
        
        # 归一化
        pathway_priors = pathway_priors / pathway_priors.sum()
        
        model_dict = {
            'omic_sizes': args.omic_sizes, 
            'num_classes': args.n_classes,
            'pathway_prior_probs': pathway_priors,
        }
        model = CCD_SurvPath(**model_dict)
        
    elif args.modality == "ccd_survpath":
        from models.model_CCD2 import CCD_SurvPath
        from utils.pathway_utils import build_pathway_gene_matrix, get_pathway_prior_probs
        
        # 构造通路-基因关系矩阵
        all_gene_names = list(args.dataset_factory.all_modalities["rna"].columns)
        pathway_gene_matrix = build_pathway_gene_matrix(
            args.dataset_factory.omic_names, 
            all_gene_names
        )
        
        model_dict = {
            'omic_sizes': args.omic_sizes, 
            'raw_gene_dim': len(all_gene_names),  # 原始基因维度
            'num_classes': args.n_classes,
            'pathway_gene_matrix': pathway_gene_matrix,  # 通路-基因关系矩阵
            # 'omic_names': args.dataset_factory.omic_names
        }
        model = CCD_SurvPath(**model_dict)
        
    elif args.modality == "double_mediator_survpath":
        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes}
        model = DoubleMediatorSurvPath(**model_dict)

    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)

    return model

def _init_loaders(args, train_split, val_split):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')

    return train_loader,val_loader

def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(modality, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
    
    """
    
    if modality in ["mlp_per_path", "omics", "snn"]:
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["coattn", "coattn_motcat"]:
        
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        mask = mask.to(device)

    elif modality in ["survpath", "double_mediator_survpath"]:
        if modality in ["survpath", "ccd2_survpath"]:
            data_WSI = data[0].to(device)
            if data[6][0,0] == 1:
                mask = None
            else:
                mask = data[6].to(device)
        else:
            data_WSI = None
            mask = None

        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        
        

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        
    elif modality == "ccd_survpath":
        # data = (patch_features, raw_gene_tensor, omic_list, label, event_time, c, clinical_data, mask)
        data_WSI = data[0].to(device)
        raw_gene_data = data[1].to(device)  # 新增：原始基因数据
        
        data_omics = []
        for item in data[2][0]:  # 分组的通路数据
            data_omics.append(item.to(device))
        
        if data[7][0,0] == 1:
            mask = None
        else:
            mask = data[7].to(device)

        y_disc, event_time, censor, clinical_data_list = data[3], data[4], data[5], data[6]
        
        return data_WSI, raw_gene_data, y_disc, event_time, censor, data_omics, clinical_data_list, mask
    
        
    # elif modality == "double_mediator_survpath":
    #     data_WSI = None
    #     mask = None
    #     data_omics = []
    #     for item in data[1][0]:
    #         data_omics.append(item.to(device))
    #     y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        
    else:
        raise ValueError('Unsupported modality:', modality)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask

def _process_data_and_forward(model, modality, device, data):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
    
    """
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

    if modality in ["coattn", "coattn_motcat"]:  
        
        out = model(
            x_path=data_WSI, 
            x_omic1=data_omics[0], 
            x_omic2=data_omics[1], 
            x_omic3=data_omics[2], 
            x_omic4=data_omics[3], 
            x_omic5=data_omics[4], 
            x_omic6=data_omics[5]
            )  

    elif modality in ['survpath', 'ccd2_survpath']:
        input_args = {"x_path": data_WSI.to(device)}
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
        input_args["return_attn"] = False
        out = model(**input_args)
        
    elif modality == 'ccd_survpath':
        data_WSI, raw_gene_data, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

        # 构造输入参数
        input_args = {
            "x_path": data_WSI.to(device),
            "raw_gene_expr": raw_gene_data.to(device)  # 新增：原始基因数据
        }
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
        input_args["return_attn"] = False
        
        out = model(**input_args)
        
        return out, y_disc, event_time, censor, clinical_data_list

    elif modality == 'double_mediator_survpath':
        input_args = {}
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
        input_args["return_attn"] = False
        out = model(**input_args)
        
    else:
        out = model(
            data_omics = data_omics, 
            data_WSI = data_WSI, 
            mask = mask
            )
        
    if len(out.shape) == 1:
            out = out.unsqueeze(0)
    return out, y_disc, event_time, censor, clinical_data_list


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    # progress_bar = tqdm(loader, desc=f"Epoch {epoch}", total=len(loader))

    # one epoch
    # for batch_idx, data in enumerate(progress_bar):
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)
        
        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor) 
        loss_value = loss.item()
        loss = loss / y_disc.shape[0]
        
        risk, _ = _calculate_risk(h)

        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

        total_loss += loss_value 

        loss.backward()
        optimizer.step()
        scheduler.step()
        # progress_bar.set_postfix(loss=f"{loss_value:.4f}")

        if (batch_idx % 20) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
        
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    return c_index, total_loss

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.
    
    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc

def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None):
    r"""
    Run a validation loop on the trained model 
    
    Args: 
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
        - survival_train : np.array
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Validation Progress"):

            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

            if modality in ["coattn", "coattn_motcat"]:  
                h = model(
                    x_path=data_WSI, 
                    x_omic1=data_omics[0], 
                    x_omic2=data_omics[1], 
                    x_omic3=data_omics[2], 
                    x_omic4=data_omics[3], 
                    x_omic5=data_omics[4], 
                    x_omic6=data_omics[5]
                )  
            elif modality in ["survpath", "double_mediator_survpath", "ccd_survpath"]:
                if modality in ["survpath", "ccd_survpath"]:
                    input_args = {"x_path": data_WSI.to(device)}
                else:
                    input_args = {}
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                
                h = model(**input_args)
                
            else:
                h = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]


            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss


def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs

    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    return lr_scheduler

def _step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    all_survival = _extract_survival_metadata(train_loader, val_loader)
    
    for epoch in range(args.max_epochs):
        _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)
        # _, val_cindex, _, _, _, _, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
        print('Val loss:', total_loss, ', val_c_index:', val_cindex)
    # save the trained model
    # torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
    
    print('Final Val c-index: {:.4f}'.format(val_cindex))
    # print('Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
    #     val_cindex, 
    #     val_cindex_ipcw,
    #     val_IBS,
    #     val_iauc
    #     ))

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)

def _train_val(datasets, cur, args):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    #----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)
    
    #----> init loss function
    loss_fn = _init_loss_function(args)

    #----> init model
    model = _init_model(args)
    
    #---> init optimizer
    optimizer = _init_optim(args, model)

    #---> init loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler 
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    #---> do train val
    results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss)

def _train_loop_survival_ccd(epoch, model, modality, loader, optimizer, scheduler, loss_fn, alpha=0.1, beta=0.5):
    """
    CCD模型的训练循环，包含多任务损失
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []

    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()
        
        data_WSI, raw_gene_data, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
        # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument index in method wrapper_CUDA_gather)
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)
        
        # 构造输入参数
        input_args = {"x_path": data_WSI.to(device)} # [B, 4000, dim]
        for i in range(len(data_omics)):
            input_args['x_omic%d' % (i+1)] = data_omics[i].type(torch.FloatTensor).to(device) # [B, 4000, dim]
        input_args["return_attn"] = False
        input_args["raw_gene_expr"] = raw_gene_data.type(torch.FloatTensor).to(device) 
        
        # 获取各分支预测用于多任务学习
        gene_logits, wsi_logits, fusion_logits = model.forward(return_branches=True, **input_args)
        # fusion_logits = model.forward(return_branches=False, **input_args)
        
        # 计算多任务损失
        
        loss_pathway = loss_fn(h=gene_logits, y=y_disc, t=event_time, c=censor)
        loss_wsi = loss_fn(h=wsi_logits, y=y_disc, t=event_time, c=censor) 
        loss_fusion = loss_fn(h=fusion_logits, y=y_disc, t=event_time, c=censor)
        
        # 总损失（类似原文的多任务设置）
        total_batch_loss = loss_fusion + alpha * loss_pathway + beta * loss_wsi
        
        loss_value = total_batch_loss.item()
        total_batch_loss = total_batch_loss / y_disc.shape[0]
        
        # 计算风险评分
        risk, _ = _calculate_risk(fusion_logits)
        
        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
            all_risk_scores, all_censorships, all_event_times, all_clinical_data, 
            event_time, censor, risk, clinical_data_list
        )

        total_loss += loss_value 

        total_batch_loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx % 20) == 0:
            print("batch: {}, loss: {:.3f} (Gene: {:.3f}, wsi: {:.3f}, fusion: {:.3f})".format(
                batch_idx, total_batch_loss.item(), 
                loss_pathway.item(), loss_wsi.item(), loss_fusion.item()
            ))
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    return c_index, total_loss

def _summary_ccd(dataset_factory, model, modality, loader, loss_fn, survival_train=None, use_counterfactual=True):
    """
    CCD模型的评估函数，支持反事实推理
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.
    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Validation Progress"):
            data_WSI, raw_gene_expr, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

            # 构造输入参数
            input_args = {"x_path": data_WSI.to(device)}
            for i in range(len(data_omics)):
                # print('i:', i)
                input_args['x_omic%d' % int(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
            input_args["return_attn"] = False
            input_args["raw_gene_expr"] = raw_gene_expr.type(torch.FloatTensor).to(device)
            if use_counterfactual:
                # 使用反事实推理进行去偏预测
                h = model.counterfactual_inference(**input_args)
            else:
                # 使用常规预测
                h = model.forward(**input_args)
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            
            # 放在同一个设备上
            h = h.to(device)
            y_disc = y_disc.to(device)
            event_time = event_time.to(device)
            censor = censor.to(device)
            
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]

            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
                all_risk_scores, all_censorships, all_event_times, all_clinical_data, 
                event_time, censor, risk, clinical_data_list
            )
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    # 构造患者结果字典
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss

def _step_ccd0(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    """
    CCD模型的训练和验证主循环
    """
    all_survival = _extract_survival_metadata(train_loader, val_loader)
    best_debiased_results = 0
    for epoch in range(args.max_epochs):
        # 使用CCD特定的训练循环
        _train_loop_survival_ccd(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)
    
        # 保存训练好的模型
        # torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
        # 每个epoch 输出一次 分别评估常规预测和去偏预测
        print("=== 评估常规预测 ===")
        results_dict_regular, val_cindex_regular, val_cindex_ipcw_regular, val_BS_regular, val_IBS_regular, val_iauc_regular, total_loss_regular = _summary_ccd(
            args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, use_counterfactual=False
        )
        
        print("=== 评估去偏预测 ===")
        results_dict_debiased, val_cindex_debiased, val_cindex_ipcw_debiased, val_BS_debiased, val_IBS_debiased, val_iauc_debiased, total_loss_debiased = _summary_ccd(
            args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, use_counterfactual=True
        )
        # 将最好的去偏预测结果 保存起来并 return
        if val_cindex_debiased > best_debiased_results:
            best_debiased_results = val_cindex_debiased
            best_debiased_results_dict = results_dict_debiased
            best_debiased_results_val_cindex = val_cindex_debiased
            best_debiased_results_val_cindex_ipcw = val_cindex_ipcw_debiased
            best_debiased_results_val_BS = val_BS_debiased
            best_debiased_results_val_IBS = val_IBS_debiased
            best_debiased_results_val_iauc = val_iauc_debiased
            best_debiased_results_total_loss = total_loss_debiased
            
        print('best_debiased_results in epoch {}: Regular C-index: {:.4f} | Debiased C-index: {:.4f} | Improvement: {:.4f}'.format(
            epoch,
            best_val_cindex_regular, val_cindex_debiased, val_cindex_debiased - val_cindex_regular
        ))
        
    return best_debiased_results_dict, (best_debiased_results_val_cindex, best_debiased_results_val_cindex_ipcw, best_debiased_results_val_BS, best_debiased_results_val_IBS, best_debiased_results_val_iauc, best_debiased_results_total_loss)

    # 返回去偏后的结果作为主要结果
    # return results_dict_debiased, (val_cindex_debiased, val_cindex_ipcw_debiased, val_BS_debiased, val_IBS_debiased, val_iauc_debiased, total_loss_debiased)

def _step_ccd(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    """
    CCD模型的训练和验证主循环
    """
    all_survival = _extract_survival_metadata(train_loader, val_loader)
    
    # 初始化最佳结果追踪（以去偏结果为准）
    best_debiased_cindex = 0
    
    # 初始化最佳去偏结果对应的所有变量
    best_debiased_results_dict = None
    best_debiased_results_val_cindex = 0
    best_debiased_results_val_cindex_ipcw = 0
    best_debiased_results_val_BS = 0
    best_debiased_results_val_IBS = 0
    best_debiased_results_val_iauc = 0
    best_debiased_results_total_loss = 0
    
    # 初始化最佳去偏结果对应的常规预测结果
    best_regular_results_dict = None
    best_regular_results_val_cindex = 0
    best_regular_results_val_cindex_ipcw = 0
    best_regular_results_val_BS = 0
    best_regular_results_val_IBS = 0
    best_regular_results_val_iauc = 0
    best_regular_results_total_loss = 0
    
    for epoch in range(args.max_epochs):
        print(f"\n=== Epoch {epoch+1}/{args.max_epochs} ===")
        
        # 使用CCD特定的训练循环
        _train_loop_survival_ccd(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)
    
        # 保存训练好的模型
        # torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
        # 每个epoch 输出一次 分别评估常规预测和去偏预测
        print("=== 评估常规预测 ===")
        results_dict_regular, val_cindex_regular, val_cindex_ipcw_regular, val_BS_regular, val_IBS_regular, val_iauc_regular, total_loss_regular = _summary_ccd(
            args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, use_counterfactual=False
        )
        
        # 输出常规预测结果
        # print(f"常规预测结果 - C-index: {val_cindex_regular:.4f}, C-index(IPCW): {val_cindex_ipcw_regular:.4f}, "
        #       f"BS: {val_BS_regular:.4f}, IBS: {val_IBS_regular:.4f}, iAUC: {val_iauc_regular:.4f}, Loss: {total_loss_regular:.4f}")
        
        print("=== 评估去偏预测 ===")
        results_dict_debiased, val_cindex_debiased, val_cindex_ipcw_debiased, val_BS_debiased, val_IBS_debiased, val_iauc_debiased, total_loss_debiased = _summary_ccd(
            args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, use_counterfactual=True
        )
        
        # 输出去偏预测结果
        # print(f"去偏预测结果 - C-index: {val_cindex_debiased:.4f}, C-index(IPCW): {val_cindex_ipcw_debiased:.4f}, "
        #       f"BS: {val_BS_debiased:.4f}, IBS: {val_IBS_debiased:.4f}, iAUC: {val_iauc_debiased:.4f}, Loss: {total_loss_debiased:.4f}")
        
        # 计算改进幅度
        improvement = val_cindex_debiased - val_cindex_regular
        print(f"去偏改进: {improvement:.4f} (去偏 - 常规)")
        
        # 以去偏预测结果为准，当找到更好的去偏结果时，同时保存该epoch的常规预测和去偏预测结果
        if val_cindex_debiased > best_debiased_cindex:
            best_debiased_cindex = val_cindex_debiased
            
            # 保存最佳去偏结果
            best_debiased_results_dict = results_dict_debiased
            best_debiased_results_val_cindex = val_cindex_debiased
            best_debiased_results_val_cindex_ipcw = val_cindex_ipcw_debiased
            best_debiased_results_val_BS = val_BS_debiased
            best_debiased_results_val_IBS = val_IBS_debiased
            best_debiased_results_val_iauc = val_iauc_debiased
            best_debiased_results_total_loss = total_loss_debiased
            
            # 同时保存该epoch对应的常规预测结果
            best_regular_results_dict = results_dict_regular
            best_regular_results_val_cindex = val_cindex_regular
            best_regular_results_val_cindex_ipcw = val_cindex_ipcw_regular
            best_regular_results_val_BS = val_BS_regular
            best_regular_results_val_IBS = val_IBS_regular
            best_regular_results_val_iauc = val_iauc_regular
            best_regular_results_total_loss = total_loss_regular
            
            print(f"*** 新的最佳去偏结果! Epoch {epoch+1} - 去偏C-index: {val_cindex_debiased:.4f}, 对应常规C-index: {val_cindex_regular:.4f} ***")
            
        # 输出当前最佳结果比较（同一epoch的常规vs去偏）
        if best_debiased_cindex > 0:  # 确保已有最佳结果
            improvement = best_debiased_results_val_cindex - best_regular_results_val_cindex
            print(f'当前最佳模型结果 - 常规C-index: {best_regular_results_val_cindex:.4f} | 去偏C-index: {best_debiased_results_val_cindex:.4f} | 改进: {improvement:.4f}')
        print("-" * 80)
        
    # 训练结束后输出最终总结（同一模型状态下的常规vs去偏比较）
    print(f"\n=== 训练完成总结 ===")
    if best_debiased_cindex > 0:
        final_improvement = best_debiased_results_val_cindex - best_regular_results_val_cindex
        print(f'最佳模型（以去偏结果为准）:')
        print(f'  常规预测 C-index: {best_regular_results_val_cindex:.4f}')
        print(f'  去偏预测 C-index: {best_debiased_results_val_cindex:.4f}')
        print(f'  去偏改进幅度: {final_improvement:.4f}')
        print(f'  改进百分比: {(final_improvement/best_regular_results_val_cindex*100):.2f}%')
    else:
        print("未找到有效的去偏预测结果")
    
    return best_debiased_results_dict, (best_debiased_results_val_cindex, best_debiased_results_val_cindex_ipcw, best_debiased_results_val_BS, best_debiased_results_val_IBS, best_debiased_results_val_iauc, best_debiased_results_total_loss)

def _train_val_ccd(datasets, cur, args):
    """
    CCD模型的完整训练验证流程
    """
    # 获取数据分割
    train_split, val_split = _get_splits(datasets, cur, args)
    
    # 初始化损失函数
    loss_fn = _init_loss_function(args)

    # 初始化CCD模型
    model = _init_model(args)
    
    # 初始化优化器
    optimizer = _init_optim(args, model)

    # 初始化数据加载器
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # 学习率调度器
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    # 执行训练验证
    results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = _step_ccd(
        cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader
    )

    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss)
