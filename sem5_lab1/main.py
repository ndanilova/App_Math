import json

from pulp import *
with open('input.json', 'r') as file:
    # Parse JSON data
    data = json.load(file)
goal = data["goal"]
variables = []
for i in range(len(data["f"])):
    variables.append("x" + str(i + 1))


f_coefs = []
for i in range(len(data["f"])):
    f_coefs.append(data["f"][i])

for i in range(len(variables)):
    variables[i] = pulp.LpVariable(variables[i], lowBound=0) # я без понятия как сделать чтоб нижняя граница могла тоже указываться любая

output_value = ""
if goal == "max":
    output_value = "Максимальное"
    problem = pulp.LpProblem('0', LpMaximize)
elif goal == "min":
    output_value = "Минимальное"
    problem = pulp.LpProblem('0', LpMaximize)
else:
    raise Exception("goal is not defined")
f_sum = 0
for i in range(len(f_coefs)):
    f_sum += f_coefs[i] * variables[i]
problem += f_sum
for i in range(len(data["constraints"])):
    coefs = []
    expression = data["constraints"][i]['type']
    b = int(data["constraints"][i]["b"])
    sum = 0
    for j in range(len(data["constraints"][i]["coefs"])):
        coefs.append(data["constraints"][i]["coefs"][j])
    for j in range(len(variables)):
        sum += coefs[j] * variables[j]
    if expression == "lte":
        problem += sum <= b
    if expression == "eq":
        problem += sum == b
    if expression == "gte":
        problem += sum >= b

problem.solve()
print("Результат:")
for variable in problem.variables():
    print(variable.name, "=", variable.varValue)
print(output_value + " значение функции:")
print(value(problem.objective))



