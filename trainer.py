from utils import AverageMeter,ProgressMeter
import torch
import time

def train(train_loader, model, nce_criterion, mse_criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    nce_losses = AverageMeter('NCE Loss', ':.4e')
    mse_losses = AverageMeter('MSE Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, nce_losses,mse_losses,losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.mode.lower() == "id":
            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
            output, target = model(im_q=images[0], im_k=images[1])
            loss = nce_criterion(output, target)
            nce_losses.update(loss.item(), images[0].size(0))
            losses.update(loss.item(), images[0].size(0))


        elif args.mode.lower() == "caid":
            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
                images[2] = images[2].cuda(args.gpu, non_blocking=True)
            output, target, rec_output = model(im_q=images[0], im_k=images[1])
            nce_loss = nce_criterion(output, target)
            mse_loss = mse_criterion(rec_output, images[2])
            loss = args.contrastive_weight * nce_loss + args.mse_weight * mse_loss

            nce_losses.update(nce_loss.item(), images[0].size(0))
            mse_losses.update(mse_loss.item(), images[0].size(0))
            losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, nce_criterion, mse_criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    nce_losses = AverageMeter('NCE Loss', ':.4e')
    mse_losses = AverageMeter('MSE Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, nce_losses,mse_losses,losses],
        prefix="Validation: ")

    model.eval()
    counter = torch.zeros((2,), device=torch.device(f'cuda:{args.rank}'))

    end = time.time()
    for i, (images) in enumerate(val_loader):
        with torch.no_grad():
        # measure data loading time
            data_time.update(time.time() - end)
            if args.mode.lower() == "id":
                if args.gpu is not None:
                    images[0] = images[0].cuda(args.gpu, non_blocking=True)
                    images[1] = images[1].cuda(args.gpu, non_blocking=True)
                output, target = model(im_q=images[0], im_k=images[1])
                loss = nce_criterion(output, target)
                nce_losses.update(loss.item(), images[0].size(0))
                losses.update(loss.item(), images[0].size(0))
            elif args.mode.lower() == "caid":
                if args.gpu is not None:
                    images[0] = images[0].cuda(args.gpu, non_blocking=True)
                    images[1] = images[1].cuda(args.gpu, non_blocking=True)
                    images[2] = images[2].cuda(args.gpu, non_blocking=True)
                output, target, rec_output = model(im_q=images[0], im_k=images[1])
                nce_loss = nce_criterion(output, target)
                mse_loss = mse_criterion(rec_output, images[2])
                loss = args.contrastive_weight * nce_loss + args.mse_weight * mse_loss

                nce_losses.update(nce_loss.item(), images[0].size(0))
                mse_losses.update(mse_loss.item(), images[0].size(0))
                losses.update(loss.item(), images[0].size(0))

            counter[0] += loss.item()
            counter[1] += 1

        # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return counter
