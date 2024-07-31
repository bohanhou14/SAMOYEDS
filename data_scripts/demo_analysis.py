import pickle
from collections import Counter

# pkl_name = "profiles/profiles-state-United States=attitude=probably_no_or_definitely_no-agent_num=500-top_p=0.7-temp=1.5.pkl"
# pkl_name = "profiles/profiles-agent_num=500-top_p=0.7-temp=1.5.pkl"
pkl_name = "profiles/profiles-state-United States=attitude=probably_no_or_definitely_no-agent_num=500-top_p=0.7-temp=1.5.pkl"

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

def counter_to_ordered_list(counter):
    ordered_keys = []
    ordered_values = []
    for key in counter.keys():
        ordered_keys.append(key)
        ordered_values.append(counter[key])
    return ordered_keys, ordered_values

# print(names_counter)
ordered_keys, ordered_values = counter_to_ordered_list(names_counter)
print(ordered_keys)
print(ordered_values)
print()

ordered_keys, ordered_values = counter_to_ordered_list(gender_counter)
print(ordered_keys)
print(ordered_values)
print()

ordered_keys, ordered_values = counter_to_ordered_list(race_counter)
print(ordered_keys)
print(ordered_values)
print()

ordered_keys, ordered_values = counter_to_ordered_list(education_counter)
print(ordered_keys)
print(ordered_values)
print()

ordered_keys, ordered_values = counter_to_ordered_list(occupation_counter)
print(ordered_keys)
print(ordered_values)
print()

ordered_keys, ordered_values = counter_to_ordered_list(age_counter)
print(ordered_keys)
print(ordered_values)
print()

ordered_keys, ordered_values = counter_to_ordered_list(religion_counter)
print(ordered_keys)
print(ordered_values)
print()

ordered_keys, ordered_values = counter_to_ordered_list(pb_counter)
print(ordered_keys)
print(ordered_values)
print()
# plt.pie(names)
# plt.savefig(f"{pkl_name}.png")
