import random
random.seed(2022)

s = ["Akmarzhan", "Lucca", "Alexander", "Anosha", "Mahmud", "Jakub"]
random.shuffle(s)
random.shuffle(s)

def report(algo, students: list):
    print(f"{algo}: {students[0]} & {students[1]}")

report("PPO", s[:2])
report("SAC", s[2:4])
report("DDPG", s[4:6])