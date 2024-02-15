# Bayesian Inference and Hypothesis, Bootstrap, and Statistical-tests

## Chi-Square Tests
Given:
(IMAGE HERE)

• And has a known standard deviation of 2.0049
• And has a Normal Distribution

Find:
• What is the chi-square test for this dataset?
(IMAGE HERE)

##The Sign Test
Given:
• The file Raw.txt
• And a target median of 400
Find:
• Perform a sign test
(IMAGE HERE)

## Wilcoxian Signed Rank Test
Given:
An example of the Wilcoxon Sign test is as follows:
• Given the values of {7,5.5,9.5,6,3.5,9}
• And an esFmaFon of 5
• And n=size(set)=6
• And a signicance of 5%, which is alpha=0.05
• From the Wilcoxon Table, the sum of the posistive ranks must be less than 19 for a
right-side test.
• Construct the following table:
• Column i is the number of the sample.
• Xi is the value of the sample.
• (Xi-Guess) is the difference between the sample and the guess.
• The distance is the absolute value of the difference.
• Once the distance is found, assign a rank based on the size of the distance.
• The smaller the distance, the earlier the rank.
• Then, check to see if the distance is posiFve or negaFve.
• Summate only the posiFve ranks.
• If the positive ranks is less than 19, then the test passes.

(image here)

Find:
• Perform another Wilcoxon Sign Test on the dataset { 37.1, 37.3, 37.4, 37.5, 36.2, 35.9,
38.1, 38.2, 38.3, 39.5}
• The significance is 1%
• And the guess is 37.
• Show the values of alpha, the value of the Sign Test for the right-handed side, and
calculate the sum of the positive ranks.

(image here)

## Bootstrap

Given:
• A sample set of sample = (5, 7, 11, 13)

Find:
• Write a program in R, MATLAB Python, C++, Java or similar language .
• Creating all Bootstrap Samples from the sample. (sampling with replacement)
• Yes, you should show all the possible values.
• Find the bootstrap estimator for the Median.

(image here)

## Bayesian Inference

Given:
• Min=40000
• Max=70000
• Count n=100
• 𝑋̅=48000
• Samples standard deviation (Observed) s = 12000
• ConAidence Interval is 95%, Zalpha=1.96
• 𝜇=(40000+7000)2⁄ = 55000
• 𝜏=(55000−40000) / 1.96 = 7653

(image here)

Find:
• Are the values of 𝜇𝑥 _and 𝜏𝑥 _correct? What values do you get?
• What is the Creditable Set? Show both the lower and upper bounds.

(image here)

## Bayesian Hypothesis
Given:
• Prior Mean 𝜇=2500
• Prior Standard Deviation 𝜏=250
• Observations.txt
• The problem has a normal model and a normal distribution
• A confidence of 95%(𝛼=0.05), and is a duality.
• Target values: for 𝐻0: {1000, 2400, 2500, 2600, 4000}
• 𝑃{𝐻0|𝑋=𝑥} = 𝑃{𝜃 − 𝜇𝑥 * 𝜏𝑥 ≥ 𝑇𝑎𝑟𝑔𝑒𝑡 − 𝜇𝑥 * 𝜏𝑥}= 1−Φ (𝑇𝑎𝑟𝑔𝑒𝑡−𝜇𝑥 * 𝜏𝑥)
• And 𝐻𝐴 is going to be the compliment of 𝐻0.
• 𝑃{𝐻𝐴|𝑋=𝑥}= Φ( 𝑇𝑎𝑟𝑔𝑒𝑡 − 𝜇𝑥 * 𝜏𝑥)

Find:
• Posterior Mean 𝜇𝑥
• Posterior Standard Deviation 𝜏𝑥
• Lower bound of the credible set
• Upper bound of the credible set
• Which target values support 𝐻0_, and which support 𝐻𝐴?

(image here)
