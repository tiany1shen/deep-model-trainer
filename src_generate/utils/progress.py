def progress_bar(current, total, length):
    done = int(current / total * length)
    remain = length - done
    return "|" + ">" * done + "=" * remain + "|"

def ratio(current, total):
    rate = f"{current}/{total}".rjust(2 * len(str(total)) + 1, " ")
    percent = f"{current/total*100:.2f}%".rjust(7, " ")
    return rate + " [" + percent + "]"

def print_progress(current_epoch, total_epoch, current_iter, total_iter, length=10):
    
    epoch = f"EPOCH: {progress_bar(current_epoch, total_epoch, length)} {ratio(current_epoch, total_epoch)}"
    itera = f"ITER: {progress_bar(current_iter, total_iter, length)} {ratio(current_iter, total_iter)}"
    return " " + epoch + " | " + itera