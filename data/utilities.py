import yaml


def LoadConfig(opt):
    # Load data from YAML file
    with open(opt.cfg, 'r') as file:
        config = yaml.safe_load(file)

    opt.batch_size = config['batch_size']
    opt.nc = config['nc']
    opt.img_channels = config['img_channels']
    opt.epochs = config['epochs']
    opt.lr = config['lr']
    opt.image_size = config['image_size']
    opt.train_csv = config['train_csv']
    opt.test_csv = config['test_csv']
    opt.imgs_path = config['imgs_path']
    opt.workers = config['workers']
    opt.device = config['device']
    opt.shuffle = config['shuffle']
    opt.classes = config['classes']
    return opt