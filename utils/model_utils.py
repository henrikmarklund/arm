import models

def get_model(args, image_shape):

    num_channels, W, H = image_shape

    print("num channels: ", num_channels)

    if args.dataset == 'mnist':
        num_classes = 10
    elif args.dataset in ('celeba'):
        num_classes = 2
    elif args.dataset == 'femnist':
        num_classes = 62

    # This is the one used in the paper
    model = models.ContextualConvNet(num_channels, n_context_channels=args.n_context_channels,
             num_classes=num_classes, support_size=args.support_size, use_context=args.use_context,
                                     prediction_net=args.prediction_net,
                                     pretrained=args.pretrained, context_net=args.context_net)

    return model
