from libreasr.lib.inference.imports import *
from libreasr.lib.defaults import TORCH_NUM_CPU_THREADS

# we need
# - config
# - transforms
# - model (+pretrained weights)
# - language (+tokenizer)

# we don't need
# - Databunch
# - Learner


def load_stuff(lang, config_path=None, **kwargs):

    # use more threads for PyTorch
    torch.set_num_threads(TORCH_NUM_CPU_THREADS)

    # grab stuff
    conf, lang, m, tfms = parse_and_apply_config(
        inference=True, lang=lang, path=config_path, **kwargs
    )

    # arguments for transforms
    # TODO: put this into a function
    tfms_args = OrderedDict(
        lang=lang,
        channels=conf["channels"],
        sr=conf["sr"],
        target_sr=conf["sr"],
        win_length=conf["win_length"],
        hop_length=conf["hop_length"],
        delta_win_length=conf["delta_win_length"],
        deltas=conf["deltas"],
        n_foward_frames=conf["n_forward_frames"],
        mfcc_args=conf["mfcc_args"],
        melkwargs=conf["melkwargs"],
        use_extra_features=False,
    )

    # create tfms
    tfm_args = OrderedDict(
        **tfms_args,
        random=False,
        label_state=None,
    )
    x_tfm = Pipeline(preload_tfms(tfms[0], tfm_args))
    x_tfm_stream = Pipeline(preload_tfms(tfms[1], tfm_args))

    print("[inference] Model and Pipeline set up.")
    print("[inference] Transforms:", x_tfm)
    print("[inference] Stream Transforms:", x_tfm_stream)

    return conf, lang, m, x_tfm, x_tfm_stream
