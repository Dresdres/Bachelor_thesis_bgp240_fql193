import config
import torch
import torch.optim as optim

from CNN import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YOLOloss

torch.backends.cudnn.benchmark = True

def train_func(train_loader, model, optimizer, loss_func, scaler, scaled_anchors, train_eval_loader):
    loop = tqdm(train_loader, leave = True)
    losses = []
    val_losses = []
    mean_loss = 0
    for batch_idx, (x, y) in enumerate(loop):
        model.train()
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_func(out[0], y0, scaled_anchors[0]) 
                + loss_func(out[1], y1, scaled_anchors[1])
                + loss_func(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar

        mean_loss = sum(losses)/len(losses)
        loop.set_postfix(loss=mean_loss)
    print("training loss:", mean_loss)
    loop_val = tqdm(train_eval_loader, leave = True)
    mean_val_loss = 0
    for idx, (x,y) in enumerate(loop_val):
        model.eval()
        with torch.set_grad_enabled(False):
            x = x.to(config.DEVICE)
            y0_val, y1_val, y2_val = (
                y[0].to(config.DEVICE),
                y[1].to(config.DEVICE),
                y[2].to(config.DEVICE)
            )
            val_out = model(x)
            val_loss = (
                    loss_func(val_out[0], y0_val, scaled_anchors[0])
                    + loss_func(val_out[1], y1_val, scaled_anchors[1])
                    + loss_func(val_out[2], y2_val, scaled_anchors[2])
            )
        val_losses.append(val_loss.item())
        mean_val_loss = sum(val_losses)/len(val_losses)
        loop_val.set_postfix(loss=mean_val_loss)
    print("validation loss:", mean_val_loss)
    return mean_loss


def main():
    model = YOLOv3(num_classes = config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path =config.DATASET + "/LS_train_10Jun.csv", 
        test_csv_path  =config.DATASET + "/LS_test_10Jun.csv",
        val_csv_path   =config.DATASET + "/LS_val_10Jun.csv"
    )
    loss_func = YOLOloss()
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE,
        )

    scaled_anchors = (torch.tensor(config.ANCHORS)*torch.tensor(config.S).
                      unsqueeze(1).unsqueeze(2).repeat(1,3,2)).to(config.DEVICE)

    epoch_total = 0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_total += 1
        print("Total epochs: ", epoch_total)
        mean_loss = train_func(train_loader, model, optimizer, loss_func, scaler, scaled_anchors, train_eval_loader)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename= config.CHECKPOINT_FILE)


        if epoch_total % 2 == 0 and mean_loss < 1.5:
        #if epoch_total % 2 == 0:
            print("On test loader:")
            check_class_accuracy(model, test_loader, config.CONF_THRESHOLD)

            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval=mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model_name = "model4/modelLS_2cls_e_" + str(epoch_total) + ".h5"
            torch.save(model, model_name)
    torch.save(model, model_name)

if __name__ == '__main__':
    main()
