import numpy as np
import argparse
import pickle

age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
age_dist = [10.26, 15.78, 16.58, 17.18, 17.95,15.17, 7.08]
edu_labels = ["Less than High School", "High School", "Some College", "2 year Bachelor Degree", "4 year Bachelor Degree", "Master's Degree", "Professional Degree", "Doctorate Degree"]
edu_dist = [4.39, 18.79, 25.30,11.05, 22.83, 3.12, 2.45, 12.06]
gen_labels = ["Male", "Female", "Non-binary"]
gen_dist = [44.38, 51.29, 4.33]
race_labels = ["White", "Black", "Hispanic", "Asian", "Native American", "Pacific Islander", "Multiple Race", "Other Race"]
race_dist = [54.08, 13.81, 5.40, 2.39, 0.74, 0.20, 4.44, 18.93]
ocu_labels = [
    "Community and Social Service", 
    "Education, library occupation", 
    "Arts, entertainment, media",
    "Healthcare practitioners",
    "Healthcare support",
    "Protective service",
    "Food preparation and serving",
    "Building and grounds cleaning & maintenance",
    "Personal care and service",
    "Sales",
    "Office and administrative support",
    "Construction",
    "Installation, maintenance, repair",
    "Production",
    "Transportation and material moving",
    "Other occupation",
    "Unemployed"
]

ocu_dist = [
    2.23, 4.73, 1.82, 4.50, 2.98, 
    0.88, 3.77, 1.34, 1.14, 4.99, 
    6.06, 1.45, 2.02, 1.65, 2.60, 
    15.03, 42.81
]

# https://news.gallup.com/poll/1690/Religion.aspx
rel_labels = ["Protestant", "Christian (non-specific)", "Catholic", "Jewish", "Mormon", "Other religion (Buddist, Muslim, etc)", "Atheist", "No answer"]
rel_dist = [33, 11, 22, 2, 1, 6, 22, 3]

# https://news.gallup.com/poll/15370/party-affiliation.aspx
pol_labels = ["Republican", "Independents", "Democrats"]
pol_dist = [30, 41, 28]

def profile_gen():
    profile = {}
    age = np.random.choice(age_labels, p=age_dist/np.sum(age_dist))
    if "-" in age:
        age_start = int(age.split("-")[0])
        age_end = int(age.split("-")[1])
    else:
        age_start = 75
        age_end = 100
    profile["Age"] = np.random.randint(age_start, age_end)

    profile["Education"] = np.random.choice(edu_labels, p=edu_dist/np.sum(edu_dist))
    profile["Gender"] = np.random.choice(gen_labels, p=gen_dist/np.sum(gen_dist))
    profile["Race"] = np.random.choice(race_labels, p=race_dist/np.sum(race_dist))
    profile["Occupation"] = np.random.choice(ocu_labels, p=ocu_dist/np.sum(ocu_dist))
    profile["Religion"] = np.random.choice(rel_labels, p=rel_dist/np.sum(rel_dist))
    profile["Political belief"] = np.random.choice(pol_labels, p=pol_dist/np.sum(pol_dist))
    return profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_profiles", type=int)
    parser.add_argument("--output", type=str, default="profiles.pkl")
    args = parser.parse_args()
    profiles = []
    for i in range(args.num_profiles):
        profiles.append(profile_gen())
    output = args.output.split(".pkl")[0] + f"-num={args.num_profiles}.pkl" 
    with open(output, "wb") as f:
        pickle.dump(profiles, f)
        f.close()

    

