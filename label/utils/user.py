def user_question(msg):
    user_answer = input(msg + " [y/n]: ")
    while True:

        if user_answer == "y" or user_answer == "yes":
            return True
        if user_answer == "n" or user_answer == "no":
            return False

        user_answer = input(" - chose between (y)es or (n)o: ")


def user_input(msg):
    user_answer = input(msg + ": ")
    while True:

        if isfloat(user_answer):
            return float(user_answer)
        elif user_answer == "":
            return None

        user_answer = input(" - insert a number: ")


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
