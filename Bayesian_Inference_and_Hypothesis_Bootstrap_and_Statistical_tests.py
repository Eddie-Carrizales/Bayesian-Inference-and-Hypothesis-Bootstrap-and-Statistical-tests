import numpy as np
import scipy.stats as stats
import pandas as pd
from tabulate import tabulate
from itertools import product
import math

# Name: Edilberto Carrizales
# Language: Python
# Language version: 3.9

def problem1():
    print("\n----------------------- Problem 1. Chi-Square Test -----------------------")
    # Chi-Square Tests
    # Given:
    sources = [1, 3, 12, 45, 71, 41, 9, 2, 1]
    events = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    standard_dev = 2.0049

    # number of samples
    n = sum(sources)

    # calculated weighted values
    weighted_list = []
    for i in range(len(sources)):
        weighted_list.append(sources[i] * events[i])

    weighted_values_df = pd.DataFrame({"Sources": sources, "Events": events, "Weighted": weighted_list})
    print(tabulate(weighted_values_df, headers='keys', tablefmt='psql', numalign="center", showindex= "Never"))

    sources_total = sum(sources)
    print("Sources Total:" + str(sources_total))

    weighted_total = sum(weighted_list)
    print("Weighted Total: " + str(weighted_total))

    expected_value = weighted_total / n
    print("Expected Value: " + str(expected_value))

    # Testing family of distributions
    index_k = []
    obs_k_values = sources
    exp_expected_value = []
    expected_value_to_k = []
    k_factorial = []
    p_k_values = []
    exp_k_values = []
    for k in range(len(sources)):
        # This is for poisson, find it for normal distribution
        part1 = np.exp(-expected_value)
        part2 = expected_value**k
        part3 = math.factorial(k)
        part4 = part1 * (part2 / part3)
        result = sources_total * part4

        index_k.append(k)
        exp_expected_value.append(part1)
        expected_value_to_k.append(part2)
        k_factorial.append(part3)
        p_k_values.append(part4)
        exp_k_values.append(result)

    family_of_dist_df = pd.DataFrame({"k": index_k, "n": n, "exp(-expected value)": exp_expected_value,
                                      "(expected value)^k": expected_value_to_k, "k!": k_factorial, "Pk":p_k_values, "Exp(K)": exp_k_values})
    print(tabulate(family_of_dist_df, headers='keys', tablefmt='psql', numalign="right", showindex= "Never"))

    # chi Square
    chi_square = 0
    obs_k_values = sources
    numerator = []
    denominator = []
    result = []
    for i in range(len(sources)):
        num = (obs_k_values[i] - exp_k_values[i]) ** 2
        den = exp_k_values[i]
        res = num / den

        numerator.append(num)
        denominator.append(den)
        result.append(res)

    family_of_dist_df = pd.DataFrame(
        {"k": index_k, "Obs(K)": obs_k_values, "Exp(k)": exp_k_values, "Num": numerator, "Den": denominator,
         "Num/Den": result})
    print(tabulate(family_of_dist_df, headers='keys', tablefmt='psql', numalign="right", showindex="Never"))

    chi_square = sum(result)
    print("Chi-Square: " + str(chi_square))


def problem2():
    print("\n----------------------- Problem 2. The Sign Test -----------------------")
    # Perform a Sign Test
    # Given:
    #   -File Raw.txt
    #   -Target median of 400

    # Read in the data from the file
    path = "/Users/eddiecarrizales/Library/CloudStorage/Raw.txt"
    raw_data = np.loadtxt(path, delimiter = ",")
    print("Raw Data:")
    print(raw_data)

    # ---------- Perform the Sign Test -----------

    # First 1:
    # Null Hypothesis (Ho): The close the data is to 400, the safer the system is.

    # Second 2:
    # Count the number of values above and below our target median 400.
    target_median = 400
    values_greater_than_median = []
    values_less_than_median = []

    for i in range(len(raw_data)):
        if raw_data[i] > target_median:
            values_greater_than_median.append(raw_data[i])
        elif raw_data[i] < target_median:
            values_less_than_median.append(raw_data[i])

    print("\nGreater than target median:")
    print(values_greater_than_median)
    count_greater_than_median = len(values_greater_than_median)
    print("Count: " + str(count_greater_than_median))

    print("\nLess than target median:")
    print(values_less_than_median)
    count_less_than_median = len(values_less_than_median)
    print("Count: " + str(count_less_than_median))

    print("\nWe can see that more than half of the population is above (greater than) the target median 400")

    number_of_datapoints = len(raw_data)
    print("\nNumber of Datapoints = " + str(number_of_datapoints))

    # Third 3: Calculating the Binomial Distributions of each side
    probability_less_than_median = stats.binom.cdf(count_less_than_median, number_of_datapoints, 0.5)
    print("\nBinomial distribution for values LESS THAN target median 400:")
    print(probability_less_than_median)

    probability_greater_than_median = stats.binom.cdf(count_greater_than_median, number_of_datapoints, 0.5)
    print("\nBinomial distribution for values GREATER THAN target median 400:")
    print(probability_greater_than_median)

    # Fourth 4: Calculating 2 * Minimum and checking P-Value Test
    print("\n2 * Minimum of our two binomial distribution values:")
    value = 2 * min(probability_less_than_median, probability_greater_than_median)
    print(value)

    print("\nFor a P-Value Test, the min value is greater than 0.01. Thus, this system is safe.")

def problem3():
    print("\n----------------------- Problem 3. Wilcoxian Signed Rank Test -----------------------\n")
    # Given:
    problem3_given_dataset = [37.1, 37.3, 37.4, 37.5, 36.2, 35.9, 38.1, 38.2, 38.3, 39.5]
    # significance is 1%
    # guess is 37

    # Find:
    # Perform the Wilcoxon Sign Test
    # Show the values of alpha
    # Show the value of the Sign Test for the right-hand side
    # Calculate the sum of the positive ranks
    # Construct a table

    # Value of alpha (since significance is 1%, alpha is .01)
    alpha = .01

    # Given guess
    guess = 37

    # Construct Wilcoxon Sign Test Table
    index_list = []
    sample_list = []
    difference_list = []
    distance_list = []
    sign_list = []

    for i in range(len(problem3_given_dataset)):
        sample = problem3_given_dataset[i]
        difference = sample - guess
        distance = abs(difference)
        sign = None

        # Calculate the sign
        if difference > 0:
            sign = "+"
        else:
            sign = "-"

        # append to our lists
        index_list.append(i + 1)
        sample_list.append(round(sample, 3))
        difference_list.append(round(difference, 3))
        distance_list.append(round(distance, 3))
        sign_list.append(sign)

    # --- Calculate the rank (I just did this manually) ---
    rank_list = [1, 2, 3, 4, "", "", 6.5, 8, 9, 10]

    wilcoxon_df = pd.DataFrame({"i": index_list, "Xi": sample_list, "Difference (Xi - Guess)": difference_list, "Distance": distance_list, "Rank": rank_list, "Sign": sign_list})
    print(tabulate(wilcoxon_df, headers='keys', tablefmt='psql', numalign="center", showindex= "Never"))

    sum_of_positive_ranks = 43.5
    print("\nThe value of alpha is: 0.01")
    print("\nSum of the positive ranks is: " + str(sum_of_positive_ranks))
    print("\nThe sum of positive ranks is 43.6 and our P-value min(positive ranks, negative ranks) / total ranks =  2/10 = 0.2 "
          "\n is greater than the significance level 1% (i.e., alpha = 0.01), thus the Wilcoxian Signed Rank Test fails.")

def problem4():
    print("\n----------------------- Problem 4. Bootstrap -----------------------")
    # We are asked to find the BootStrap Estimator for the Median.

    # Steps to Bootstrap Estimator for the Median:

    # 1. Collect the Data
    # In this case we are given a bootstrap_sample:
    bootstrap_sample = [5, 7, 11, 13]
    print("\nBootstrap sample: ")
    print(bootstrap_sample)

    # 2. Resampling with Replacement
    # Bootstrap involves selecting data points from the dataset with replacement to create a resampled dataset.
    # This resampling is usually done randomly and a large number of times.
    # However, what I will do is a "cartesian product" meaning I will generate all possible combinations
    #   with replacement and with different orders. So since our dataset has 4 values,
    #   we will have 4^4 = 256 different samples.

    samples_from_sample = []
    # Creating all bootstrap Samples from the sample (sampling with replacement)
    samples_from_sample = product(bootstrap_sample, repeat = len(bootstrap_sample))
    samples_from_sample = list(samples_from_sample)  # cast the object to a list
    print("\nAll Bootstrap samples from the sample (sampling with replacement):")
    # print(samples_from_sample)
    print("We will have 4^4 = 256 possible values.")

    # Show 20 values per row
    temp_values = []
    values_per_row = 20
    for i in range(len(samples_from_sample)):
        temp_values.append(samples_from_sample[i])
        if i % values_per_row == 0:
           print(temp_values)
           temp_values.clear()

    # 3. Calculate the medians:
    # For each bootstrap sample, we have to calculate the median.
    # The median is the middle value of data when sorted in ascending order.
    medians_from_all_samples = []
    for i in range(len(samples_from_sample)):
        sorted_samples_from_sample = sorted(samples_from_sample[i])
        sample_median = np.median(sorted_samples_from_sample)
        medians_from_all_samples.append(sample_median)
    print("\nMedians from all the samples:")
    print(medians_from_all_samples)
    # 4. Calculate BootStrap Estimator for the Median

    # 4a: Count the different medians
    medians_count_dict = {}
    for i in range(len(medians_from_all_samples)):
        if medians_from_all_samples[i] not in medians_count_dict:
            medians_count_dict[medians_from_all_samples[i]] = 1
        else:
            medians_count_dict[medians_from_all_samples[i]] += 1
    print("\nCount for each median in the samples:")
    print(medians_count_dict)

    # 4b: plug into formula
    equation_first_part = []
    equation_second_part = []
    for key in medians_count_dict:
        part1_result = (key**2) * (medians_count_dict[key] / len(medians_from_all_samples) )
        part2_result = key * (medians_count_dict[key] / len(medians_from_all_samples) )
        equation_first_part.append(part1_result)
        equation_second_part.append(part2_result)

    part1_sum = sum(equation_first_part)
    part2_sum = sum(equation_second_part)

    bootstrap_estimator_for_median = part1_sum - (part2_sum**2)
    print("\nBootstrap Estimator for median:")
    print(bootstrap_estimator_for_median)

def problem5():
    print("\n----------------------- Problem 5. Bayesian Inference -----------------------")
    # Bayesian Inference
    # Given:
    min = 40000
    max = 70000
    count_n = 100
    mean_X = 48000
    samples_std_observed_s = 12000
    # confidence interval = 95%
    z_alpha = 1.96
    prior_mean_mu = 55000
    prior_std_tao = 7653

    # --- Formulas for calculations ---

    # posterior mean
    mu_x = ( ((count_n * mean_X)/(samples_std_observed_s ** 2)) + (prior_mean_mu/(prior_std_tao ** 2)) ) / ( (count_n/(samples_std_observed_s ** 2)) + (1/(prior_std_tao ** 2)) )
    print("\nThe value of μ_x is correct:")
    print(mu_x)

    # posterior std
    tao_x = (1 / ( math.sqrt((count_n/(samples_std_observed_s ** 2)) + (1/(prior_std_tao ** 2))) ))
    print("\nThe value of 𝜏_x is not correct, the correct value is:")
    print(tao_x)

    # lower bound
    lower_bound = mu_x - (z_alpha * tao_x)
    print("\nThe Lower Bound of the credible set:")
    print(lower_bound)

    # upper bound
    upper_bound = mu_x + (z_alpha * tao_x)
    print("\nThe Upper Bound of the credible set:")
    print(upper_bound)

    print("\nThe critical set for a 95% confidence interval is:")
    print("[" + str(lower_bound) + ", " + str(upper_bound) + "]")

def problem6():
    print("\n----------------------- Problem 6. Bayesian Hypothesis -----------------------")
    # Bayesian Hypothesis

    # Given:
    # Dataset "observations.txt"
    # problem has a normal model and a normal distribution
    # Has a confidence interval of 95%, and it is a duality
    prior_mean_mu = 2500
    prior_std_tao = 250
    z_alpha = 1.96  # since alpha is 0.05, then 0.05/2 = 0.025 = then from z-table = 1.96
    target_values_Ho = [1000, 2400, 2500, 2600, 4000]

    # Find:
    # Posterior Mean: mu_x
    # Posterior Standard Deviation tao_x
    # Lower Bound of the credible set
    # Upper Bound of the credible set
    # Which target values support Ho, and which support HA

    # Read in Observations.txt
    path = "/Users/eddiecarrizales/Library/CloudStorage/Observations.txt"
    observations_data = np.loadtxt(path, delimiter = ",")
    print("\nObservations Data:")
    print(observations_data)

    count_n = len(observations_data)
    x_bar = np.mean(observations_data)
    samples_std_observed_s = np.std(observations_data, ddof=1)

    # --- Formulas for calculations ---

    # posterior mean
    numerator_1 = (count_n * x_bar) / (samples_std_observed_s ** 2)
    numerator_2 = prior_mean_mu / (prior_std_tao ** 2)
    numerator = numerator_1 + numerator_2
    denominator_1 = count_n / (samples_std_observed_s ** 2)
    denominator_2 = 1 / (prior_std_tao ** 2)
    denominator = denominator_1 + denominator_2
    mu_x = numerator/denominator
    print("\nPosterior Mean μ_x:")
    print(mu_x)

    # posterior std
    numerator = 1
    denominator_1 = count_n / (samples_std_observed_s ** 2)
    denominator_2 = 1 / (prior_std_tao ** 2)
    denominator = math.sqrt(denominator_1 + denominator_2)
    tau_x = numerator/denominator
    print("\nPosterior Standard Deviation 𝜏_x:")
    print(tau_x)

    # lower bound
    lower_bound = mu_x - (z_alpha * tau_x)
    print("\nLower Bound the of credible set:")
    print(lower_bound)

    # upper bound
    upper_bound = mu_x + (z_alpha * tau_x)
    print("\nUpper Bound the of credible set:")
    print(upper_bound)

    print("\nThe critical set for a 95% confidence interval is:")
    print("[" + str(lower_bound) + ", " + str(upper_bound) + "]")

    print("\nThus, values [2500, 2600], support the null hypothesis H_0.")
    print("And, values [1000, 2400, 4000], support the alternative hypothesis H_A.")

def main():
    problem1()
    problem2()
    problem3()
    problem4()
    problem5()
    problem6()

if __name__ == "__main__":
    main()