import pandas as pd

with open("app.log","r") as file:
    logs = file.readlines()
report = {}
for log in logs:
    if "Module=" not in log:
        continue
    module = log.split("Module=")[1]
    if module not in report:
        report[module] = {
            "ERROR":0,
            "WARNING":0
        }
    if "ERROR" in log:
        report[module]["ERROR"] += 1
    elif "WARNING" in log:
        report[module]["WARNING"] += 1


rows = []

for module,counts in report.items():
    rows.append([module,counts["ERROR"],counts["WARNING"]])

df = pd.DataFrame(rows,columns=["Module","Error Count","Warning Count"])


df.to_csv("log_report.csv",index=False)
print("CSV Report Generated Successfully")

