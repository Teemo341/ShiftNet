from shiftdm.hack import disable_verbosity, enable_sliced_attention


save_memory = True
if save_memory:
    enable_sliced_attention()

disable_verbosity()
