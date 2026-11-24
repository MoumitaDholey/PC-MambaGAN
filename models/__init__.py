def create_model(opt):
    print(opt.model)
    if opt.model == 'i2i_mamba_one':
        from .i2i_mamba_one import I2IMamba_model
        model = I2IMamba_model()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)

    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
