with open("./data/soc-sign-bitcoinalpha.csv") as file:
    input = file.read()

links = input.splitlines()
nodes = set()
for link in links:
    source, target, _, _ = link.split(",")
    nodes.add(source)
    nodes.add(target)
print(len(nodes))
