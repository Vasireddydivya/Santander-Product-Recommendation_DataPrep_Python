# Santander-Product-Recommendation
This is a group project. Ajay Raghunathan, Divya Vasireddy, Kavya Goli
•	The purpose of this project is to build a better recommendation system and presumably do a better job of advertising services to the people
•	Initially in Data Pre-processing step for imputing the missing values for each column I find the distribution/frequency of each variable and replaced appropriately. I have done the data preprocessing in both 'Python' and R. But for the next steps in project I used 'R'
•	In the Feature engineering step merged train and test data sets based on customer id and added few additional features for better understanding the trend of each product for every month
•	The scoring metric which we have used for building this recommendation model is MAP@7. The intuition behind this scoring metric is that it rewards solutions where the person actually added one of the items you recommended, and you get more points if the purchased item was earlier in your list of recommendations
•	We have used XGBoost algorithm for predictions
