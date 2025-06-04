# Midterm (30%), Assignments (10%), Quizzes (15%)
midterm = [
    27.41666667, 37.14772727, 46.29166667, 42.82575758, 34.22727273,
    36.99621212, 44.83333333, 30.70075758, 39.66666667, 43.51136364,
    42.21212121, 39.64393939, 36.47727273, 31.13636364, 33.00757576,
    36.49621212, 41.14015152, 36.35606061, 18.71212121, 43.49621212,
    40.52272727, 39.77651515, 37.29166667, 40.67045455, 40.34469697,
    39.56818182, 43.07575758, 42.59090909, 41.64015152, 42.83333333,
    24.84090909
]

assignments = [
    3.916666667, 9.875, 9.791666667, 9.416666667, 9.25, 9.791666667, 9.833333333,
    5.541666667, 9.666666667, 9.875, 9.666666667, 9.916666667, 10, 9.5, 9.166666667,
    9.291666667, 9.708333333, 9.833333333, 3.166666667, 9.041666667, 10, 9.958333333,
    9.791666667, 9.125, 9.958333333, 10, 9.916666667, 9.75, 9.708333333, 9.083333333, 0
]

quizzes = [
    5, 5.272727273, 8, 6.909090909, 5.727272727, 4.454545455, 8, 3.909090909, 7,
    7.636363636, 6.545454545, 5.727272727, 6.727272727, 2.636363636, 5.090909091,
    6.454545455, 5.181818182, 5.272727273, 0.5454545455, 7.454545455, 6.272727273,
    5.818181818, 5, 6.545454545, 4.636363636, 6.818181818, 6.909090909, 7.090909091,
    6.181818182, 7, 1.090909091
]

names = [
"Muhammad Waleed Hussain", "Muhammad Miqdad Ahmad", "Ahmad Waleed Akhtar", "Murtaza Khalid",
"Muhammad Mateen Amjad", "Muhammad Arham", "Abdullah Farrukh Sanani", "Fatima Waseem",
"Aitsam Atif", "Muhammad Saad Ramzan", "Hadia Ahmad", "Maheen",
"Ariba Mumtaz", "Muhammad Saad", "Huzaifa Khan", "Syed Kifayat Ur Rahman",
"Syeda Maham Batool","Muhammad Moiz Ahmad","Syed Muhammad Atif","Maryam Aftab",
"Umair Ul Hassan", "Alishba Zahid", "Shawaiz Asghar", "Hoor Ul Ain Amir",
"Abdullah Zahid", "Muhammad Khizar Naveed", "Muhammad Hassan Arif", "Usman Shakir",
"Muhammad Haseeb", "Rameeza Rahim", "Atika Tahir ",
]

# Weights
w_mid = 0.30
w_assign = 0.10
w_quiz = 0.10

# Calculate weighted total (excluding final)
weighted_total = [
    round(m * w_mid + a * w_assign + q * w_quiz, 2)
    for m, a, q in zip(midterm, assignments, quizzes)
]

# Print individual scores
for i, score in enumerate(weighted_total, 1):
    print(f"{i-1}: {names[i-1]}: {score} / 50 (before final)")

# Calculate and print average
print("\n\n\n")
print(f"{names[1]}: {weighted_total[1]} / 50 (before final)")
print(f"{names[16]}: {weighted_total[16]} / 50 (before final)")
print(f"{names[11]}: {weighted_total[11]} / 50 (before final)")
print(f"{names[13]}: {weighted_total[13]} / 50 (before final)")
average_score = round(sum(weighted_total) / len(weighted_total), 2)
print(f"\nAverage (before final): {average_score} / 50")
