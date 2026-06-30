import argparse
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "utils"))
sys.path.append(os.path.join(base_dir, "dataset"))
sys.path.append(os.path.join(base_dir, "model"))
import h5py
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import RMLgeneral, RMLval, RMLtest
import numpy as np
import random
from torch.utils.data import DataLoader
from IQFormerLite import IQFormerLite
from IQFormer import IQFormer
from MCFormer import MCformer
from AMCNET import AMC_Net
from MCLDNN import MCLDNN
from PETCGDNN import PETCGDNN
from FEA_T128 import FEA_T as FEA_T128
from FEA_T1024 import FEA_T as FEA_T1024
import torch
from torch import nn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from plot_tSNE import plot_tsne
from traintest import train_epoch, test_epoch, val_epoch
from model_report import get_report_batch, adjust_inputs, build_multi_dtype_report, print_multi_dtype_report, format_multi_dtype_report
import time
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_confusion_matrix(cm, database, SNR, save_dir, labels=[]):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=labels, yticklabels=labels, cbar=False,
                square=True,
                annot_kws={"fontsize": 20})
    plt.title(database+' SNR='+str(SNR)+'dB')
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{database}_{SNR}.pdf'), bbox_inches='tight', dpi=450)
    plt.close()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def infer_seq_length(sample):
    if isinstance(sample, np.ndarray):
        arr = sample
    else:
        arr = np.array(sample)
    if arr.ndim == 1:
        return int(arr.shape[0])
    if arr.ndim >= 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            return int(arr.shape[1])
        if arr.shape[1] == 2 and arr.shape[0] != 2:
            return int(arr.shape[0])
        return int(arr.shape[-1])
    return 128


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RML SMY model')
    # Dataset
    parser.add_argument('--database_path', type=str, default="./dataset")
    parser.add_argument('--database_choose', type=str, default="2016.10b")
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--minSNR', type=int, default=-20) # 2016 -20-18 2018 -20-30
    parser.add_argument('--maxSNR', type=int, default=18)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss', type=str, default='CCE', help='Cross Entropy Loss')
    parser.add_argument('--aux_mode', type=str, default='none', choices=['none', 'stft', 'conv', 'kan'],
                        help='Auxiliary feature mode: none (IQ-only), stft (STFT-IQ fusion), conv (Conv band-IQ fusion), kan (KAN filter bank)')
    parser.add_argument('--band_k', type=int, default=32, help='Number of frequency bands for Conv/KAN-based feature extraction')
    parser.add_argument('--kernel_size', type=int, default=31, help='Kernel size for KAN filterbank')
    parser.add_argument('--grid_size', type=int, default=4, help='Grid size for KAN filterbank')
    parser.add_argument('--grid_range', type=float, nargs=2, default=[-2.0, 2.0], help='Grid range for KAN filterbank')
    parser.add_argument('--lkf_variant', type=str, default='full',
                        choices=['full', 'conv', 'base_only', 'rbf_only', 'bspline'],
                        help='LKF ablation variant used when aux_mode=kan')
    parser.add_argument('--report_only', action='store_true', help='Generate model report and exit')
    parser.add_argument('--report_batch', type=int, default=1, help='Batch size for model report')
    parser.add_argument('--report_length', type=int, default=128, help='Input length for model report')

    # model
    parser.add_argument('--model', type=str, default='IQFormerLite',
                        choices=['IQFormerLite', 'IQFormer', 'MCFormer', 'AMCNET', 'MCLDNN', 'PETCGDNN', 'FEA_T128', 'FEA_T1024'],
                        help='Model to use')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--dry_run', action='store_true', help='Run a single batch for verification')
    parser.add_argument('--skip_post_test_artifacts', action='store_true',
                        help='Exit after writing Test_ACC.csv to skip confusion matrices and t-SNE plots')
    parser.add_argument('--save_stage_checkpoints', action='store_true',
                        help='Save extra epoch checkpoints for LKF interpretability analysis')
    parser.add_argument('--stage_epochs', type=str, default='0,5,15,best,final',
                        help='Comma-separated epoch tags to save when --save_stage_checkpoints is enabled. Supports integers, best, final.')
    parser.add_argument('--comment', type=str, default='IQFormer',
                        help='Comment to describe the saved model')
    if not os.path.exists('save_models'):
        os.mkdir('save_models')
    args = parser.parse_args()

    if not os.path.exists('logs'):
        os.mkdir('logs')
    args = parser.parse_args()

    # define model saving path
    model_tag = 'model_{}_{}_{}_{}'.format(args.database_choose, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('save_models', model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # set log sub-directories
    cm_save_path = os.path.join('logs', model_tag, 'confusionMatrix')
    if not os.path.exists(cm_save_path):
        os.makedirs(cm_save_path)
    
    tsne_save_path = os.path.join('logs', model_tag, 'tsne')
    if not os.path.exists(tsne_save_path):
        os.makedirs(tsne_save_path)

    # GPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    set_seed(args.seed)

    # Load dataset
    dataset_aux_mode = args.aux_mode
    if args.model == 'IQFormer':
        dataset_aux_mode = 'stft'

    train_dataset = [[],[],[]]
    val_dataset = [[],[],[]]
    test_dataset = [[],[],[]]
    input_length = 128
    if args.database_choose == '2019':
        raise ValueError("HisarMod2019 dataset is no longer supported.")
    else: 
        if args.database_choose[-1] == 'a':
            data = pd.read_pickle(os.path.join(args.database_path, 'RML2016.10a.pkl'))
            classes = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'AM-SSB', 'WBFM']
        else:
            classes = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB','WBFM']
            data = pd.read_pickle(os.path.join(args.database_path, 'RML2016.10b.dat'))
        train_dataset = [[],[],[]]
        val_dataset = [[],[],[]]
        test_dataset = [[],[],[]]
        for item in data.items():
            (label, SNR), samples = item
            if SNR < args.minSNR or SNR > args.maxSNR or label not in classes:
                continue
            labels = np.full(len(samples), classes.index(label))
            SNR = np.full(len(samples), SNR)
            X, x, Y, y, SNR_tr, SNR_te = train_test_split(samples, labels, SNR, test_size=args.test_size,
                                                        random_state=233,
                                                        stratify=labels)
            train, val, train_label, val_label, SNR_tr, SNR_va = train_test_split(X, Y, SNR_tr, test_size=0.25,
                                                                                random_state=233,
                                                                                stratify=Y)
            train_dataset[0].extend(train)
            train_dataset[1].extend(train_label)
            train_dataset[2].extend(SNR_tr)
            val_dataset[0].extend(val)
            val_dataset[1].extend(val_label)
            val_dataset[2].extend(SNR_va)
            test_dataset[0].extend(x)
            test_dataset[1].extend(y)
            test_dataset[2].extend(SNR_te)
        train_dataset = RMLgeneral(np.array(train_dataset[0]),np.array(train_dataset[1]),np.array(train_dataset[2]), aux_mode=dataset_aux_mode)
        val_dataset = RMLtest(np.array(val_dataset[0]),np.array(val_dataset[1]),np.array(val_dataset[2]), aux_mode=dataset_aux_mode)
        test_dataset = RMLtest(np.array(test_dataset[0]),np.array(test_dataset[1]),np.array(test_dataset[2]), aux_mode=dataset_aux_mode)
        if len(train_dataset) > 0:
            input_length = infer_seq_length(train_dataset.samples[0])
    print(f'train_size:{len(train_dataset)}\tval_size:{len(val_dataset)}\t')
    
    if args.dry_run:
        print("Dry run mode: Truncating datasets to 2 batches...")
        limit = args.batch_size * 2
        train_dataset.samples = train_dataset.samples[:limit]
        train_dataset.label = train_dataset.label[:limit]
        train_dataset.SNR = train_dataset.SNR[:limit]
        val_dataset.samples = val_dataset.samples[:limit]
        val_dataset.label = val_dataset.label[:limit]
        val_dataset.SNR = val_dataset.SNR[:limit]
        test_dataset.samples = test_dataset.samples[:limit]
        test_dataset.label = test_dataset.label[:limit]
        test_dataset.SNR = test_dataset.SNR[:limit]
        print(f'Dry run sizes -> train:{len(train_dataset)} val:{len(val_dataset)}')

    # Training Dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # Testing Dataloader
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    del train_dataset
    del val_dataset
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    if args.database_choose == '2016.10b':
        num_classes = 10
    else:
        num_classes = 11
    
    if args.model == 'IQFormerLite':
        if args.database_choose in ['2016.10a','2016.10b']:
            model = IQFormerLite([2,3,2], embed_dims=[64,64,64],
                    mlp_ratios=1,
                    act_layer=nn.GELU,
                    num_classes=num_classes,
                    down_patch_size=3, down_stride=2, down_pad=1,
                    drop_rate=0.2, drop_path_rate=0.,
                    use_layer_scale=False, layer_scale_init_value=1e-5,
                    fork_feat=False,
                    vit_num=1,
                    aux_mode=args.aux_mode,
                    band_k=args.band_k,
                    kernel_size=args.kernel_size,
                    grid_size=args.grid_size,
                    grid_range=tuple(args.grid_range),
                    lkf_variant=args.lkf_variant)
        else:
            # model = IQFormerLite([3,3,3], embed_dims=[64,64,64],
            #     mlp_ratios=4,
            #     act_layer=nn.GELU,
            #     num_classes=num_classes,
            #     down_patch_size=3, down_stride=2, down_pad=1,
            #     drop_rate=0.2, drop_path_rate=0.2,
            #     use_layer_scale=False, layer_scale_init_value=1e-5,
            #     fork_feat=False,
            #     vit_num=1,
            #     aux_mode=args.aux_mode,
            #     band_k=args.band_k,
            #     kernel_size=args.kernel_size,
            #     grid_size=args.grid_size,
            #     grid_range=tuple(args.grid_range))
            model = IQFormerLite([2,3,2], embed_dims=[64,64,64],
                mlp_ratios=1,
                act_layer=nn.GELU,
                num_classes=num_classes,
                down_patch_size=3, down_stride=4, down_pad=3,
                drop_rate=0.2, drop_path_rate=0.2,
                use_layer_scale=False, layer_scale_init_value=1e-5,
                fork_feat=False,
                vit_num=1,
                aux_mode=args.aux_mode,
                band_k=args.band_k,
                kernel_size=args.kernel_size,
                grid_size=args.grid_size,
                grid_range=tuple(args.grid_range),
                lkf_variant=args.lkf_variant)
    elif args.model == 'IQFormer':
        if args.database_choose in ['2016.10a','2016.10b']:
            model = IQFormer([1,2,1], embed_dims=[64,64,64],
                mlp_ratios=4,
                act_layer=nn.GELU,
                num_classes=num_classes,
                down_patch_size=3, down_stride=2, down_pad=1,
                drop_rate=0.2, drop_path_rate=0.,
                use_layer_scale=False, layer_scale_init_value=1e-5,
                fork_feat=False,
                vit_num=1,)
        else:
            model = IQFormer([3,3,3], embed_dims=[64,64,64],
                mlp_ratios=4,
                act_layer=nn.GELU,
                num_classes=num_classes,
                down_patch_size=3, down_stride=2, down_pad=1,
                drop_rate=0.2, drop_path_rate=0.2,
                use_layer_scale=False, layer_scale_init_value=1e-5,
                fork_feat=False,
                vit_num=1)
    elif args.model == 'MCFormer':
        model = MCformer(num_classes=num_classes)
    elif args.model == 'AMCNET':
        model = AMC_Net(num_classes=num_classes, sig_len=input_length)
    elif args.model == 'MCLDNN':
        model = MCLDNN(num_classes=num_classes, frame_length=input_length)
    elif args.model == 'PETCGDNN':
        model = PETCGDNN(num_classes=num_classes, frame_length=input_length)
    elif args.model == 'FEA_T128':
        model = FEA_T128(num_class=num_classes, seq_length=input_length)
    elif args.model == 'FEA_T1024':
        model = FEA_T1024(num_class=num_classes, seq_length=input_length)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    # criterion
    criterion = nn.CrossEntropyLoss()
    # AdamW optimizer
    optimizer1 = torch.optim.AdamW(model.parameters(),  lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer1, 'min', factor=0.5, patience=3, min_lr=5e-5)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    if True:
        batch_x, batch_stft = get_report_batch(train_loader, args.aux_mode, device, force_stft=(args.model == 'IQFormer'))
        report_length = args.report_length
        if args.aux_mode == 'stft' and report_length != input_length:
            report_length = input_length
        elif args.model in ['PETCGDNN', 'MCLDNN', 'AMCNET', 'FEA_T128', 'FEA_T1024'] and report_length != input_length:
            report_length = input_length
        batch_x, batch_stft = adjust_inputs(batch_x, batch_stft, batch_size=args.report_batch, length=report_length)
        reports = build_multi_dtype_report(model, batch_x, batch_stft, device, args.aux_mode, ['fp32'])
        report_text = format_multi_dtype_report(reports)
        print(report_text, end="")
        report_path = os.path.join('logs', model_tag, 'model_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        if args.report_only:
            sys.exit(0)

    # Training and testing
    epochs_without_improvement = 0
    patience = 10
    num_epochs = args.num_epochs
    stage_tokens = [token.strip().lower() for token in args.stage_epochs.split(',') if token.strip()]
    stage_epoch_ids = set()
    for token in stage_tokens:
        if token.isdigit():
            stage_epoch_ids.add(int(token))
    stage_ckpt_dir = os.path.join(model_save_path, 'stage_checkpoints')
    if args.save_stage_checkpoints and not os.path.exists(stage_ckpt_dir):
        os.makedirs(stage_ckpt_dir)
    writer = SummaryWriter('logs/{}'.format(model_tag))
    for epoch in range(num_epochs):
        train_loss, train_ACC, train_true, train_pred = train_epoch(epoch, train_loader, model, args.minSNR,
                                                                    args.maxSNR,optimizer1, criterion, device, aux_mode=args.aux_mode)
                                                                     
        writer.add_scalar('train_loss', train_loss, epoch)
        val_loss, val_ACC, val_true, val_pred, model_v = val_epoch(epoch, val_loader, model, args.minSNR,
                                                                   args.maxSNR,
                                                                   scheduler, criterion, device, aux_mode=args.aux_mode)
        writer.add_scalar('val_loss', val_loss, epoch)
        if epoch == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'weight.pt'))
            max_acc = val_ACC['Avg']
            if args.save_stage_checkpoints and 'best' in stage_tokens:
                torch.save(model.state_dict(), os.path.join(stage_ckpt_dir, 'epoch_best.pt'))

        else:
            if max_acc < val_ACC['Avg']:
                torch.save(model.state_dict(), os.path.join(model_save_path, 'weight.pt'))
                if args.save_stage_checkpoints and 'best' in stage_tokens:
                    torch.save(model.state_dict(), os.path.join(stage_ckpt_dir, 'epoch_best.pt'))
                avg = val_ACC['Avg']
                print(f'max_acc:{max_acc}=====>{avg}')
                max_acc = val_ACC['Avg']
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f'epochs_without_improvement:{epochs_without_improvement}/{patience}')
        tracc = pd.DataFrame.from_dict(train_ACC, orient='index', columns=['0']).reset_index(names='SNR')
        vaacc = pd.DataFrame.from_dict(val_ACC, orient='index', columns=['0']).reset_index(names='SNR')
        if epoch == 0:
            tracc.to_csv(f'logs/{model_tag}/Train_Epoch.csv', index=False)
            vaacc.to_csv(f'logs/{model_tag}/Val_Epoch.csv', index=False)
        else:
            tracc = pd.Series(tracc['0'])
            vaacc = pd.Series(vaacc['0'])
            tr = pd.read_csv(f'logs/{model_tag}/Train_Epoch.csv')
            tr.insert(tr.shape[1], f'{epoch}', tracc)
            va = pd.read_csv(f'logs/{model_tag}/Val_Epoch.csv')
            va.insert(va.shape[1], f'{epoch}', vaacc)
            tr.to_csv(f'logs/{model_tag}/Train_Epoch.csv', index=False)
            va.to_csv(f'logs/{model_tag}/Val_Epoch.csv', index=False)

        train_CM = confusion_matrix(train_true, train_pred)
        train_row_sums = train_CM.sum(axis=1, keepdims=True)
        traincm = np.divide(train_CM.astype('float'), train_row_sums, out=np.zeros_like(train_CM, dtype=float), where=train_row_sums != 0)
        traincm = np.around(traincm, decimals=2)
        val_CM = confusion_matrix(val_true, val_pred)
        val_row_sums = val_CM.sum(axis=1, keepdims=True)
        valcm = np.divide(val_CM.astype('float'), val_row_sums, out=np.zeros_like(val_CM, dtype=float), where=val_row_sums != 0)
        valcm = np.around(valcm, decimals=2)

        #   plotCM
        plot_confusion_matrix(traincm, args.database_choose, 'all', cm_save_path, labels=classes)
        plot_confusion_matrix(valcm, args.database_choose, 'all', cm_save_path, labels=classes)
        if args.save_stage_checkpoints and epoch in stage_epoch_ids:
            torch.save(model.state_dict(), os.path.join(stage_ckpt_dir, f'epoch_{epoch}.pt'))
        if epochs_without_improvement == patience:
            print(f'Early stopping at epoch {epoch}')
            break
    del train_loader
    del val_loader
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'weight.pt')))
    if args.save_stage_checkpoints and 'final' in stage_tokens:
        torch.save(model.state_dict(), os.path.join(stage_ckpt_dir, 'epoch_final.pt'))
    start = time.time()
    test_true, test_pred, test_SNR = test_epoch(0, test_loader, model, device, aux_mode=args.aux_mode)
    end = time.time()
    used = end - start
    print('Avg_test_time:', used / len(test_pred))
    pred = torch.stack(test_pred).cpu().data.numpy().argmax(1).tolist()
    true = torch.stack(test_true).cpu().data.numpy().tolist()
    test_SNR = [int(i) for i in test_SNR]
    SNR = dict([(key, 0) for key in range(args.minSNR, args.maxSNR + 1, 2)])
    SNR_true = dict([(key, 0) for key in range(args.minSNR, args.maxSNR + 1, 2)])
    for slice in range(len(pred)):
        if (type(test_SNR[slice])).__name__ == 'list':
            test_SNR[slice] = test_SNR[slice][0]
        if pred[slice] == true[slice]:  
            SNR[test_SNR[slice]] = SNR.get(test_SNR[slice]) + 1
            SNR_true[test_SNR[slice]] = SNR_true.get(test_SNR[slice]) + 1
        else:
            SNR[test_SNR[slice]] = SNR.get(test_SNR[slice]) + 1
    avg_true = 0
    avg_all = 0
    for key in range(args.minSNR, args.maxSNR + 1, 2):
        avg_all += SNR[key]
        avg_true += SNR_true[key]
        if SNR[key] > 0:
            SNR[key] = SNR_true[key] / float(SNR[key])
        else:
            SNR[key] = 0.0
    if avg_all > 0:
        SNR['Avg'] = avg_true / float(avg_all)
    else:
        SNR['Avg'] = 0.0
    Avg = SNR['Avg']
    print(f'test_acc={Avg}')
    # Save test accuracy
    testacc = pd.DataFrame.from_dict(SNR, orient='index', columns=['0']).reset_index(names='SNR')
    testacc.to_csv(f'logs/{model_tag}/Test_ACC.csv', index=False)
    if args.skip_post_test_artifacts:
        sys.exit(0)
    mod_dic = {}
    for snr in range(args.minSNR,args.maxSNR+1,2):
        SNR_cm = [i for i in zip(test_SNR, pred, true) if i[0] == snr]
        if len(SNR_cm) == 0:
            continue
        true_cm = []
        pred_cm = []
        true_cls = np.zeros(num_classes)
        all = np.zeros(num_classes)
        for i in SNR_cm:
            pred_cm.append(i[1])
            true_cm.append(i[2])
            if i[1] == i[2]:
                true_cls[i[1]] = true_cls[i[1]] + 1
                all[i[1]] = all[i[1]] + 1
            else:
                all[i[2]] = all[i[2]] + 1
        Cls_ACC = {cls: (x / y if y > 0 else 0.0) for cls, x, y in zip(classes, true_cls, all)}
        mod_dic[snr] = list(Cls_ACC.values())
        modacc = pd.DataFrame.from_dict(mod_dic, orient='index', columns=classes).reset_index(names='SNR')
        modacc.to_csv(f'logs/{model_tag}/Test_mod_SNR.csv', index=False)
        test_CM = confusion_matrix(true_cm, pred_cm)
        test_row_sums = test_CM.sum(axis=1, keepdims=True)
        testcm = np.divide(test_CM.astype('float'), test_row_sums, out=np.zeros_like(test_CM, dtype=float), where=test_row_sums != 0)
        testcm = np.around(testcm, decimals=2)
        plot_confusion_matrix(testcm, args.database_choose, snr, cm_save_path, labels=classes)
        SNR_tsne_ = [i for i in
                zip(test_SNR, torch.stack(test_pred).cpu().data.numpy(), torch.stack(test_true).cpu().data.numpy()) if
                i[0] == snr]
        _, pred_0, true_0 = zip(*SNR_tsne_)
        plot_tsne(np.array(list(pred_0)), np.array(list(true_0)), args.database_choose, snr, classes, tsne_save_path)
