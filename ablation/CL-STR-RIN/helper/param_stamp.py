


def get_param_stamp(args, model_name, verbose=True, replay=False, replay_model_name=None):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:      "+model_stamp)

    # -for hyper-parameters
    hyper_stamp = "{i_e}{num}-lr{lr}{lrg}-b{bsz}-rb{rbsz}-z{z_dim}-{optim}".format(
        i_e="e" if args.iters is None else "i", num=args.epochs if args.iters is None else args.iters, lr=args.lr,
        lrg=("" if args.lr==args.lr_gen else "-lrG{}".format(args.lr_gen)) if hasattr(args, "lr_gen") else "",
        bsz=args.batch_size, rbsz=args.replay_batch_size, z_dim=args.z_dim, optim=args.optimizer,
    )
    if verbose:
        print(" --> hyper-params:   "+ hyper_stamp)

    # -for replay
    if replay:
        replay_stamp = "{rep}{KD}{agem}{model}{gi}".format(
            rep=args.replay,
            KD="",
            agem="",
            model="" if (replay_model_name is None) else "-{}".format(replay_model_name),
            gi="-gi{}".format(args.gen_iters) if (
                hasattr(args, "gen_iters") and (replay_model_name is not None) and (not args.iters==args.gen_iters)
            ) else ""
        )
        if verbose:
            print(" --> replay:      " + replay_stamp)
    replay_stamp = "--{}".format(replay_stamp) if replay else ""

    # --> combine
    param_stamp = "{model_stamp}---{hyper_stamp}---{replay_stamp}{seed}".format(
        model_stamp=model_stamp, hyper_stamp=hyper_stamp, replay_stamp=replay_stamp,
        seed="-s{}".format(args.seed) if not args.seed==0 else "",
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp
