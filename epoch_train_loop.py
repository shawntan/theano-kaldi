import config
@config.option("max_epochs","Maximum number of epochs to iterate.",type=config.int)
def loop(get_data_stream,item_action,epoch_callback,max_epochs):
    epoch_count = 0
    while True:
        for x in get_data_stream():
            item_action(x)
        epoch_count += 1
        if epoch_callback(epoch_count) or epoch_count == max_epochs:
            break
