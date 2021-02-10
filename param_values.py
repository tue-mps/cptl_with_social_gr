#todo
def set_default_values(args, also_hyper_params=True):
    # -set default-values for certain arguments based on chosen scenario & experiment
    args.tasks = 10 if args.tasks is None else args.tasks
    args.iters = 100 if args.iters is None else args.iters
    args.lr = 0.001 if args.lr is None else args.lr
    if also_hyper_params:
        args.si_c = 5. if args.si_c is None else args.si_c
        args.ewc_lambda = 500. if args.ewc_lambda is None else args.ewc_lambda
        if hasattr(args, 'o_lambda'):
            args.o_lambda = 1000. if args.o_lambda is None else args.o_lambda
        args.gamma = 0.9 if args.gamma is None else args.gamma
    return args