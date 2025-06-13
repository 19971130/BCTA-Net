import os
import torch


def save_all(epoch, model, optimizer, scheduler,
             hparams, best_metric, name):
    torch.save({
        'end_epoch': epoch,
        'best_ssim': best_metric['ssim']['value'],
        'ssim_epoch': best_metric['ssim']['epoch'],
        'best_psnr': best_metric['psnr']['value'],
        'psnr_epoch': best_metric['psnr']['epoch'],
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, os.path.join(
        hparams['train']['save_dir'],
        hparams['train']['model_name'],
        hparams['train']['task_name'],
        'ckpt', name + '.pth'
        )
    )


def load_all(hparams, name, model, optimizer, scheduler, best_metric):
    ckpt_path = os.path.join(hparams['train']['save_dir'],
                             hparams['train']['model_name'],
                             hparams['train']['task_name'],
                             'ckpt', name + '.pth')
    ckpt_info = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt_info['model'])
    optimizer.load_state_dict(ckpt_info['optimizer'])
    scheduler.load_state_dict(ckpt_info['scheduler'])
    best_metric['ssim']['value'] = ckpt_info['best_ssim']
    best_metric['ssim']['epoch'] = ckpt_info['ssim_epoch']
    best_metric['psnr']['value'] = ckpt_info['best_psnr']
    best_metric['psnr']['epoch'] = ckpt_info['psnr_epoch']
    return ckpt_info['end_epoch']
