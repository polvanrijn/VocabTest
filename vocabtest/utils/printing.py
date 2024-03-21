def bold(string):
    return '\033[1m' + string + '\033[0m'

def red(string):
    return '\033[91m' + string + '\033[0m'

def green(string):
    return '\033[92m' + string + '\033[0m'

def orange(string):
    return '\033[93m' + string + '\033[0m'

def warning(string):
    print(bold(orange(string)))

def error(string):
    print(bold(red(string)))
    raise Exception(string)

def success(string):
    print(bold(green(string)))

def info(string):
    print(bold(string))



def print_summary(msg, count, total, type='print'):
    full_message = f'{msg}: {count}/{total} ({count / total * 100:.1f}%)'
    if type == 'print':
        info(full_message)
    elif type == 'warning':
        warning(full_message)
    else:
        raise NotImplementedError()