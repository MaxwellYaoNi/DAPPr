import os, torch, torchvision, functools
from torch.optim import AdamW
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from tqdm import tqdm
import utils
from sklearn.metrics import average_precision_score
import numpy as np
import DAPPr, EDL

@torch.no_grad()
def get_logits(model, loader):
    device = next(model.parameters()).device
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, labels in loader:
            all_logits.append(model(x.to(device, non_blocking=True)))
            all_labels.append(labels.to(device, non_blocking=True))
    return torch.cat(all_logits), torch.cat(all_labels)

@torch.no_grad()
def evaluate_val_acc(model, val_loader):
    logits, labels = get_logits(model, val_loader)  # reuse
    return (logits.argmax(1) == labels).float().mean().item() * 100

def aupr_ood(unc_id, unc_ood):
    unc_id = np.nan_to_num(unc_id)
    unc_ood = np.nan_to_num(unc_ood)

    bin_labels = np.concatenate([np.ones(unc_id.shape[0]), np.zeros(unc_ood.shape[0])])
    scores     = -np.concatenate((unc_id, unc_ood))

    return average_precision_score(bin_labels, scores)

@torch.no_grad()
def evaluate_test_metrics(model, test_loader, *ood_loaders, uncertainty_func=DAPPr.DAPPr_uncertainty):
    id_logits, id_labels = get_logits(model, test_loader)
    ood_logits_list = [get_logits(model, dl)[0] for dl in ood_loaders]

    # compute uncertainty scores
    to_numpy = lambda unc: {k: v.cpu().numpy() for k, v in unc.items()}
    id_unc = to_numpy(uncertainty_func(id_logits))
    ood_unc_list = [to_numpy(uncertainty_func(ood_logits)) for ood_logits in ood_logits_list]

    # test accuracy
    is_correct = (id_logits.argmax(1) == id_labels).cpu().numpy().astype(int)
    metrics = {'test_acc': round(is_correct.mean() * 100, 4)}

    # confidence for in-distribution
    for name, uncertainty in id_unc.items():
        metrics[f'Conf_AUPR_{name}'] = round(average_precision_score(is_correct, -uncertainty) * 100, 4)

    # OOD detection
    for i, ood_unc in enumerate(ood_unc_list):
        for name in id_unc:
            metrics[f'OOD{i+1}_AUPR_{name}'] = round(aupr_ood(id_unc[name], ood_unc[name]) * 100, 4)
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    group = parser.add_argument_group('training arguments')
    group.add_argument('--dataset',         type=str,   default='CUB', help='dataset name')
    group.add_argument('--bs',              type=int,   default=256)
    group.add_argument('--lr',              type=float, default=2e-3)
    group.add_argument('--epochs',          type=int,   default=200)
    group.add_argument('--seed',            type=int,   default=42)
    group.add_argument('--num_workers',     type=int,   default=8)
    group.add_argument('--test_every',      type=int,   default=10, help='testing after specific epochs')
    group.add_argument('--out_dir',         type=str,   default='outs')
    group.add_argument('--data_root',       type=str,   default='data')
    group.add_argument('--hdf5', action='store_true',   default=False, help='whether to use hdf5')

    group = parser.add_argument_group('uncertainty modeling arguments')
    group.add_argument('--method',          type=str,   default='DAPPr', choices=['DAPPr', 'EDL'],
                       help='uncertainty modeling method')
    group.add_argument('--lamb',            type=float, default=2e-4)
    group.add_argument('--lamb_schedule',   type=str,   default='warmup', choices=['warmup', 'linear'],)

    args = parser.parse_args()

    #### Set the working directory
    name_configs = [args.method,
                    args.dataset,
                    f'seed{args.seed}'  if args.seed != 42 else None]

    base_sub_path  = '_'.join([nc for nc in name_configs if nc is not None])
    args.save_path = os.path.join(args.out_dir, base_sub_path)
    utils.ensure_dirs(args.save_path)
    utils.set_seed(args.seed)
    print(args)

    #### Prepare Dataset
    train_dl,      val_dl, test_dl     = utils.get_id_dataloader(args.data_root, args.dataset, args.bs,
                                                                 args.num_workers, args.hdf5)
    imagenet_o_dl, dtd_dl, place365_dl = utils.get_ood_dataloader(args.data_root,
                                                                  64 if args.dataset == 'TinyImageNet' else 224,
                                                                  args.bs, args.num_workers, args.hdf5)
    class_dim = utils.get_classes_num(args.dataset)

    #### Prepare model and optimizer
    model = torchvision.models.resnet50(weights=None, num_classes=class_dim)
    if args.dataset == 'TinyImageNet':
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = torch.nn.Identity()

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineLRScheduler(opt, t_initial=args.epochs, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)

    ######################## select uncertainty modeling method #####################
    METHODS = {'DAPPr': (DAPPr.DAPPr_loss, functools.partial(DAPPr.DAPPr_uncertainty, include_entropy=True)),
               'EDL'  : (EDL.EDL_loss, functools.partial(EDL.EDL_uncertainty, include_entropy=True))}
    criterion, unc_func = METHODS[args.method]
    #################################################################################

    #### training
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    test_logger = utils.MetricsLogger(args.save_path, True, 1)
    best_acc = 0.
    best_results = {}

    for ep in tqdm(range(args.epochs)):
        model.train()
        factor = min(1., ep / 10.) if args.lamb_schedule == 'warmup' else ep / args.epochs
        lamb_reg = args.lamb * factor
        for x, labels in train_dl:
            opt.zero_grad()
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, labels, lamb_reg)
            loss.backward()
            opt.step()
        scheduler.step(ep)

        #### testing
        ep = ep + 1
        if ep % args.test_every == 0 or (ep == args.epochs):
            model.eval()
            val_acc = evaluate_val_acc(model, val_dl)
            test_metrics = evaluate_test_metrics(model, test_dl, imagenet_o_dl, dtd_dl, place365_dl, uncertainty_func=unc_func)
            log_item = {'epoch': ep, 'val_acc': round(val_acc, 4)}
            log_item.update(test_metrics)

            if val_acc >= best_acc:
                best_acc = val_acc
                # Select results based on the best validation accuracy.
                best_results = test_metrics
                torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))

            test_logger.log(**log_item)

    print('final results:', best_results)
