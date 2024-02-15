# Bayesian Inference and Hypothesis, Bootstrap, and Statistical-tests

## Chi-Square Tests <br />
Given: <br />
![Image1]()


• And has a known standard deviation of 2.0049 <br />
• And has a Normal Distribution <br />

Find: <br />
• What is the chi-square test for this dataset? <br />
![Image1]()

##The Sign Test <br />
Given: <br />
• The file Raw.txt <br />
• And a target median of 400 <br />

Find: <br />
• Perform a sign test <br />
![Image1]()

## Wilcoxian Signed Rank Test <br />
Given: <br />
An example of the Wilcoxon Sign test is as follows: <br />
• Given the values of {7,5.5,9.5,6,3.5,9} <br />
• And an esFmaFon of 5 <br />
• And n=size(set)=6 <br />
• And a significance of 5%, which is alpha=0.05 <br /> 
• From the Wilcoxon Table, the sum of the positive ranks must be less than 19 for a right-side test. <br />

• Construct the following table: <br />
• Column i is the number of the sample. <br />
• Xi is the value of the sample. <br />
• (Xi-Guess) is the difference between the sample and the guess. <br />
• The distance is the absolute value of the difference. <br />
• Once the distance is found, assign a rank based on the size of the distance. <br />
• The smaller the distance, the earlier the rank. <br />
• Then, check to see if the distance is positive or negative. <br />
• Summate only the positive ranks. <br />
• If the positive ranks is less than 19, then the test passes. <br />

![Image1]()

Find: <br />
• Perform another Wilcoxon Sign Test on the dataset { 37.1, 37.3, 37.4, 37.5, 36.2, 35.9, 38.1, 38.2, 38.3, 39.5} <br />
• The significance is 1% <br />
• And the guess is 37. <br />
• Show the values of alpha, the value of the Sign Test for the right-handed side, and calculate the sum of the positive ranks. <br />

![Image1]()

## Bootstrap <br />

Given: <br />
• A sample set of sample = (5, 7, 11, 13) <br />

Find: <br />
• Write a program in R, MATLAB Python, C++, Java or similar language. <br />
• Creating all Bootstrap Samples from the sample. (sampling with replacement) <br />
• Yes, you should show all the possible values. <br />
• Find the bootstrap estimator for the Median. <br />

![Image1]()

## Bayesian Inference <br />

Given: <br />
• Min=40000 <br />
• Max=70000 <br />
• Count n=100 <br />
• 𝑋̅=48000 <br />
• Samples standard deviation (Observed) s = 12000 <br />
• ConAidence Interval is 95%, Zalpha=1.96 <br />
• 𝜇=(40000+7000)2⁄ = 55000 <br />
• 𝜏=(55000−40000) / 1.96 = 7653 <br />

![Image1]()

Find: <br />
• Are the values of 𝜇𝑥 _and 𝜏𝑥 _correct? What values do you get? <br />
• What is the Creditable Set? Show both the lower and upper bounds. <br />

![Image1]()

## Bayesian Hypothesis <br />
Given: <br />
• Prior Mean 𝜇=2500 <br />
• Prior Standard Deviation 𝜏=250 <br />
• Observations.txt <br />
• The problem has a normal model and a normal distribution <br />
• A confidence of 95%(𝛼=0.05), and is a duality. <br />
• Target values: for 𝐻0: {1000, 2400, 2500, 2600, 4000} <br />
• 𝑃{𝐻0|𝑋=𝑥} = 𝑃{𝜃 − 𝜇𝑥 * 𝜏𝑥 ≥ 𝑇𝑎𝑟𝑔𝑒𝑡 − 𝜇𝑥 * 𝜏𝑥}= 1−Φ (𝑇𝑎𝑟𝑔𝑒𝑡−𝜇𝑥 * 𝜏𝑥) <br />
• And 𝐻𝐴 is going to be the compliment of 𝐻0. <br />
• 𝑃{𝐻𝐴|𝑋=𝑥}= Φ( 𝑇𝑎𝑟𝑔𝑒𝑡 − 𝜇𝑥 * 𝜏𝑥) <br />

Find: <br />
• Posterior Mean 𝜇𝑥 <br />
• Posterior Standard Deviation 𝜏𝑥 <br />
• Lower bound of the credible set <br />
• Upper bound of the credible set <br />
• Which target values support 𝐻0_, and which support 𝐻𝐴? <br />

![Image1]()
