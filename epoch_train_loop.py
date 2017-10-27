import config


@config.option("max_epochs", "Maximum number of epochs to iterate.",
               type=config.int)
def loop(get_data_stream, item_action, epoch_callback, max_epochs):
    epoch_count = 0
    epoch_callback(epoch_count)
    while True:
        for x in get_data_stream():
            item_action(x)
        epoch_count += 1
        stop = epoch_callback(epoch_count)
        if stop or epoch_count == max_epochs:
            break
