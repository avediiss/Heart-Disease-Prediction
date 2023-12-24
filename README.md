
This project delves into the Heart Disease Dataset sourced from Kaggle, focusing on the crucial task of Heart Disease Prediction and Risk Assessment. By analyzing patient characteristics and medical measurements, our objective is to develop robust predictive models for accurate heart disease diagnosis. The dataset's selection is justified by its relevance to global cardiovascular health, its reliability from a reputable platform, and its comprehensive structure conducive to predictive modeling. Leveraging machine learning techniques on this dataset holds significant implications for healthcare stakeholders, aiding in early diagnosis and informed decision-making. Moreover, beyond practical applications, its educational value in data analysis and healthcare analytics is noteworthy. The report emphasizes the dataset's pivotal role in addressing heart disease prediction, advocating for improved patient outcomes and healthcare practices through enhanced detection and management strategies.
Keywords: Heart Disease, Predictive Modeling, Risk Assessment, Machine Learning, Data Analysis.


Focused on leveraging the Heart Disease Dataset from Kaggle, our report centers on developing precise predictive models for heart disease diagnosis. Through comprehensive analysis of patient attributes and medical measurements, our aim is to enhance early detection and risk assessment, addressing critical healthcare challenges in cardiovascular disease management. This research emphasizes the dataset's significance in advancing healthcare practices and decision-making through machine learning applications.

Data Preparation
There are 14 columns in the dataset starting 
1.	Age : displays the age of the individual.
2.	Sex : displays the gender of the individual using the following format : 
1 = male  0 = female.
3.	Chest-pain type : displays the type of chest-pain experienced by the individual using the following format :
           1 = typical angina
           2 = atypical angina
           3 = non - anginal pain
           4 = asymptotic
4.	Resting Blood Pressure : displays the resting blood pressure value of an individual in mmHg (unit)
5.	Serum Cholesterol : displays the serum cholesterol in mg/dl (unit)
6.	Fasting Blood Sugar : compares the fasting blood sugar value of an individual with 120 mg/dl. 
   If fasting blood sugar > 120 mg/dl 
then : 1  (true)
                                else : 0   (false)
7.	Resting ECG : 
              0 = normal
              1 = having ST-T wave abnormality
              2 = left ventricular hypertrophy
8.	Max heart rate achieved : displays the max heart rate achieved by an individual.
9.	Exercise induced angina : 
              1 = yes
              0 = no
10.	Oldpeak: ST depression induced by exercise relative to rest.
11.	Peak exercise ST segment : 
              1 = upsloping
              2 = flat
              3 = downsloping
12.	Number of major vessels (0-3) colored by fluoroscopy : displays the value as integer or float.
13.	Thal : displays the thalassemia : 
              3 = normal
              6 = fixed defect
              7 = reversible defect
14.	Diagnosis of heart disease : Displays whether the individual is suffering from heart disease or not : 
              0 = absence
              1,2,3,4 = present.

Why these parameters:
Age: Age is the most important risk factor in developing cardiovascular or heart diseases, with approximately a tripling of risk with each decade of life. Coronary fatty streaks can begin to form in adolescence. It is estimated that 82 percent of people who die of coronary heart disease are 65 and older. Simultaneously, the risk of stroke doubles every decade after age 55. 
Sex: Men are at greater risk of heart disease than premenopausal women. Once past menopause, it has been argued that a woman’s risk is similar to a man’s although more recent data from the WHO and UN disputes this. If a female has diabetes, she is more likely to develop heart disease than a male with diabetes.
Angina (Chest Pain): Angina is chest pain or discomfort caused when your heart muscle doesn’t get enough oxygen-rich blood. It may feel like pressure or squeezing in your chest. The discomfort also can occur in your shoulders, arms, neck, jaw, or back. Angina pain may even feel like indigestion.
Resting Blood Pressure: Over time, high blood pressure can damage arteries that feed your heart. High blood pressure that occurs with other conditions, such as obesity, high cholesterol or diabetes, increases your risk even more.
Serum Cholesterol: A high level of low-density lipoprotein (LDL) cholesterol (the “bad” cholesterol) is most likely to narrow arteries. A high level of triglycerides, a type of blood fat related to your diet, also ups your risk of a heart attack. However, a high level of high-density lipoprotein (HDL) cholesterol (the “good” cholesterol) lowers your risk of a heart attack.
Fasting Blood Sugar: Not producing enough of a hormone secreted by your pancreas (insulin) or not responding to insulin properly causes your body’s blood sugar levels to rise, increasing your risk of a heart attack.
Resting ECG: For people at low risk of cardiovascular disease, the USPSTF concludes with moderate certainty that the potential harms of screening with resting or exercise ECG equal or exceed the potential benefits. For people at intermediate to high risk, current evidence is insufficient to assess the balance of benefits and harms of screening.
Max heart rate achieved: The increase in cardiovascular risk, associated with the acceleration of heart rate, was comparable to the increase in risk observed with high blood pressure. It has been shown that an increase in heart rate by 10 beats per minute was associated with an increase in the risk of cardiac death by at least 20%, and this increase in the risk is similar to the one observed with an increase in systolic blood pressure by 10 mm Hg.
Exercise induced angina: The pain or discomfort associated with angina usually feels tight, gripping or squeezing, and can vary from mild to severe. Angina is usually felt in the center of your chest but may spread to either or both of your shoulders, or your back, neck, jaw or arm. It can even be felt in your hands. o Types of Angina a. Stable Angina / Angina Pectoris b. Unstable Angina c. Variant (Prinzmetal) Angina d. Microvascular Angina.
Peak exercise ST segment: A treadmill ECG stress test is considered abnormal when there is a horizontal or down-sloping ST-segment depression ≥ 1 mm at 60–80 ms after the J point. Exercise ECGs with up-sloping ST-segment depressions are typically reported as an ‘equivocal’ test. In general, the occurrence of horizontal or down-sloping ST-segment depression at a lower workload (calculated in METs) or heart rate indicates a worse prognosis and higher likelihood of multi-vessel disease. The duration of ST-segment depression is also important, as prolonged recovery after peak stress is consistent with a positive treadmill ECG stress test. Another finding that is highly indicative of significant CAD is the occurrence of ST-segment elevation > 1 mm (often suggesting transmural ischemia); these patients are frequently referred urgently for coronary angiography.

Analysis:
	To guarantee that the dataset was reliable for analysis, it went through multiple cleaning and exploratory steps during the first phase of data preparation. We started by checking every attribute for any missing values. Upon examination, we found that there were 2 null values in the ‘thal’ column and 4 null values in the ‘ca’ column. We rightly went on to eliminate these missing value cases from the dataset in order to preserve data integrity. Then, we used the ‘describe()’ method to get descriptive statistics that provided important information about the distribution properties, count, mean, standard deviation, minimum, and maximum values of the dataset. After other data types were examined, 3 features of type float64 and 11 features of type int64 were found. A phase that was also included was the search for duplicate entries, which upon examination confirmed that there were no duplicate items in the dataset. 
The dataset shows a distinct age distribution among individuals with heart disease, notably concentrated around ages 58 and 57, with a significant prevalence in those aged 50 and above. Moreover, a gender-based analysis reveals an interesting trend: females diagnosed with heart disease tend to be older than their male counterparts, indicating an age-related gender disparity within the dataset.
An in-depth correlation analysis, visualized through a comprehensive heatmap, uncovers significant associations between various features and heart disease ('target'). Positive correlations are identified with chest pain ('cp'), maximum heart rate achieved ('thalach'), and the slope of the peak exercise ST segment ('slope'). Conversely, features like exercise-induced angina ('exang'), number of major vessels ('ca'), ST depression induced by exercise relative to rest ('oldpeak'), and thalassemia ('thal') exhibit negative correlations with the 'target,' suggesting their relevance in predicting or assessing heart disease risk.
We used a variety of supervised learning models, including Decision Tree, Random Forest, Naive Bayes, Logistic Regression, SVM, and KNN, to address our classification problem. Our goal is to determine which model, out of the dataset, best predicts heart disease.
We divided the data into 80:20 segments for training (80%) and testing (20%). We then carefully evaluated the performance of each model using the four key assessment metrics: accuracy, precision, recall, and F1 score. Naïve Bayes was the ensemble's best performer overall, with exceptional recall rate, accuracy, and precision. This makes Naïve Bayes a solid and reliable method for predicting heart attacks.
Furthermore, employing an error rate plot function for 1/K in the KNN analysis revealed crucial insights. Our quest for the optimal 'K' value was guided by seeking the least error rates, particularly within the range of 0.2 to 0.0. Remarkably, our analysis pinpointed 'K' value 14 as the optimal choice, aligning with superior predictive accuracy and reinforcing its efficacy in heart disease prediction. After doing “Hyperparameter-tuning” the accuracy on the test set was 0.8778, but the accuracy on the training set was 0.8164. This difference suggests that the model may have overfitted to the complexities of the training set, which is cause for concern. Interestingly, just one potential model was taken into account during the hyperparameter tuning procedure, which might have limited the search for the best model for this dataset. Even though the obtained accuracy is really good, it's important to understand that it might not fully represent the total performance of the model. These results imply that evaluating the model's success purely on the basis of accuracy is not appropriate. But overall, the current results indicate promise in the model's ability to predict heart disease. 

The complexity of our model reduced and improved its generalizability. We maintained balance training accuracy and generalization ability, making it easier to make predictions on new unseen data. We built a robust prediction model that gives accurate predictions on heart disease of an individual.
