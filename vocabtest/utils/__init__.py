import pexpect

def clean(words):
    # TODO: clean words
    import openai
    import os

    openai.api_key = open(os.path.expanduser(os.path.join("~/", ".openai"))).read().strip()

    words = ["poltavkacultuur", "maastrichtenaar", "bezoekerscentrum", "kinderpornografie"]
    word_list_string = "- " + "\n- ".join(words)

    prompt = f"""
    Indicate if the following words in Dutch are offensive:
    {word_list_string}
    Return a list with 0 and 1s. Use 1 if the word is offensive, and 0 if it’s not. Do not give explanations, just return the answers as a valid Python list.
    """

    # auto complete prompt with gpt4
    openai

def cmd(command, timeout=60 * 10, verbose=True):
    """
    Run a command in the shell with a timeout and print the output
    """
    p = pexpect.spawn(command, timeout=timeout)
    lines = []
    while not p.eof():
        line = p.readline().decode("utf-8")
        lines.append(line)
        if verbose:
            print(line, end="")
    p.close()
    return lines