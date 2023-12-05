import pickle
from collections import Counter

pkl_name = "profiles/profiles-agent_num=500-top_p=0.7-temp=1.5.pkl"
with open(pkl_name, "rb") as f:
    profiles = pickle.load(f)

names = []
gender = []
race = []
education = []
age = []
occupation = []
religion = []
pb = []

for p in profiles:
    names.append(p['Name'].lower().strip())
    gender.append(p['Gender'].lower().strip())
    race.append(p['Race'].lower().strip())
    education.append(p['Education'].lower().strip())
    age.append(p['Age'].lower().strip())
    occupation.append(p['Occupation'].lower().strip())
    religion.append(p['Religion'].lower().strip())
    pb.append(p['Political belief'].lower().strip())

names_counter = Counter(names)
gender_counter = Counter(gender)
race_counter = Counter(race)
education_counter = Counter(education)
age_counter = Counter(age)
occupation_counter = Counter(occupation)
religion_counter = Counter(religion)
pb_counter = Counter(pb)

print(names_counter)
print()
print(gender_counter)
print()
print(race_counter)
print()
print(education_counter)
print()
print(age_counter)
print()
print(occupation_counter)
print()
print(religion_counter)
print()
print(pb_counter)
print()
# plt.pie(names)
# plt.savefig(f"{pkl_name}.png")
