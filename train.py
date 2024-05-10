import argparse
from re import I
from model import ModelFactory
from solver import Solver
from loss import get_loss_fn
from VSR_datasets import *

description='Video Super Resolution pytorch implementation'

parser = argparse.ArgumentParser(description=description)

parser.add_argument('-m', '--model', metavar='M', type=str, default='TDAN',
                    help='network architecture. Default TDAN')
parser.add_argument('-s', '--scale', metavar='S', type=int, default=4,
                    help='interpolation scale. Default 3')
parser.add_argument('--train-set', metavar='T', type=str, default='data/VSR_data/train',
                    help='data set for training. Default train')
parser.add_argument('--test-set', metavar='NAME', type=str, default='data/VSR_data/val/Temple',
                    help='dataset for testing. Default Temple from SPMC  dataset')
parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=64,
                    help='batch size used for training. Default 100')
parser.add_argument('-l', '--learning-rate', metavar='L', type=float, default=1e-4,
                    help='learning rate used for training. Default 1e-3')
parser.add_argument('-n', '--num-epochs', metavar='N', type=int, default=600,
                    help='number of training epochs. Default 600')
parser.add_argument('-f', '--fine-tune', dest='fine_tune', action='store_true',
                    help='fine tune the model under check_point dir,\
                    instead of training from scratch. Default False')          #是否接着训练
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                    help='print training information. Default False')
parser.add_argument('-g', '--gpu', metavar='G', type=str, default='0',
                    help='GPU numbers')
parser.add_argument('-cp', '--checkpoint', metavar='CP', type=str, default='/checkpoint',
                    help='network architecture. Default False')
parser.add_argument('-le', '--last-epoch', metavar='LE', type=int, default=0,
                    help='the last epoch for subsequent training')
parser.add_argument('-ss', '--scales', metavar='SS', type=str, default='2,3,4',
                    help='Training scales')
parser.add_argument('-p', '--patch-size', metavar='P', type=int, default=48,
                    help='patch size used for training. Default 48')

args = parser.parse_args()

def get_full_path(scale, train_set):
    """
    Get full path of data based on configs and target path
    example: data/interpolation/test/set5/3x
    """
    scale_path = str(scale) + 'x'
    return os.path.join('preprocessed_data', train_set, scale_path)
    
def display_config():
    print('############################################################')
    print('# Video Super Resolution - Pytorch implementation          #')

    print('############################################################')
    print('')
    print('-------YOUR SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" %(str(arg), str(getattr(args, arg))))
    print('')


def main():
    display_config()
    print('Contructing dataset...')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(args.scales)

    # train_dataset  = VSR_Dataset(dir=args.train_set, trans = transforms.Compose([RandomCrop(48, args.scale), DataAug(), ToTensor()]))
    model_factory = ModelFactory()
    model = model_factory.create_model(args.model, scale = args.scales)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(1.0 * params / (1000 * 1000))
    train_dataset  = VSR_TrainDataset(dir=args.train_set, lr_img_sz=args.patch_size, scale = args.scales, model_name = model.name)
    
    loss_fn = get_loss_fn(model.name)
    print(loss_fn)

    check_point = os.path.join(args.checkpoint, model.name, str(args.scale) + 'x')
    # if not os.path.exists(check_point):
    #     os.makedirs(check_point)

    solver = Solver(model, check_point, model.name, loss_fn=loss_fn, batch_size=args.batch_size,
                    num_epochs=args.num_epochs, learning_rate=args.learning_rate,
                    fine_tune=args.fine_tune, verbose=args.verbose, last_epoch = args.last_epoch, scales = args.scales)

    print('===> Start Train')
    print("==================================================")
    print("Method: %s || Scale: %d || Epoch totle: %d)"%(args.model, args.scale, args.num_epochs))
    scales = args.scales.split(",") if args.scales else [4]
    scales = [float(s) for s in scales]
    val_dataset  = VSR_TestDataset(dir=args.test_set, scale = scales, model_name = model.name)
    solver.train(train_dataset, val_dataset)
    #dataset = "model"
    #solver.test(dataset)
if __name__ == '__main__':
    main()

