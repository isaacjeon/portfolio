*A collection of data analytics and machine learning projects focused on exploratory analysis, predictive modeling, and dashboard development.*

## Table of Contents

**[Machine Learning Projects](#machine-learning-projects)**
  - [Retail Product Sales Forecasting](#retail-product-sales-forecasting) - time series forecasting of retail product sales
  - [Depression Prediction](#depression-prediction) - identification of mental health risk with binary classification
  - [News Article Classification](#news-article-classification) - categorization of BBC news articles using NLP techniques
  - [SMS Spam Detection](#sms-spam-detection) - detection of spam messages through dimensionality reduction and clustering

**[Dashboards](#dashboards)**
  - [Brazilian E-Commerce](#brazilian-e-commerce) - analysis of Brazilian e-commerce orders across multiple dimensions, including location, payment types, and review scores
  - [Japan Tourism](#japan-tourism) - regional segmentation of tourism data in Japan across various categories
  - [Student Progress Report](#student-progress-report) - dashboard for tracking student performance and learning trends in a middle school math class

# Machine Learning Projects

The following machine learning projects follow the workflow of data cleaning and processing, exploratory data analysis (EDA), and model training and evaluation. The code is written in **Python** using **Jupyter Notebook**.

Core libraries used across all projects include:
- **pandas** and **NumPy** for data manipulation
- **matplotlib** and **seaborn** for visualization
- **scikit-learn** for modeling and evaluation

Tools, models, and techniques specific to each project are mentioned in their respective descriptions.

***

### Retail Product Sales Forecasting

This project focuses on forecasting daily sales of a specific style of work pants using real-world sales data from a small clothing retailer. The goal was to identify trends, seasonality, and other patterns in the data to forecast sales up to 28 days ahead.

Exploratory analysis included statistical methods such as correlation analysis and STL decomposition, along with visualization tools like ACF/PACF plots and KDEs. Forecasting models were developed using **Holt-Winters**, **SARIMAX**, Meta's **Prophet**, and **LightGBM**. Features were engineered based on calendar dates (e.g. day of week, month, holidays), price changes, lag values, and moving averages. Models were evaluated using expanding window cross-validation to simulate real-world forecasting conditions.

The data revealed a nonlinear trend: a significant increase in sales between 2020 and 2021, then a plateau and gradual decline. Seasonality was strongest around Christmas and back-to-school periods. Notably, increased volatility in residuals from 2020-2023 appeared to reflect post-COVID shifts in consumer behavior and fashion trends influenced by social media.

Among the models, Prophet achieved the lowest RMSE, while LightGBM offered competitive performance with faster training times. An ensemble of all four models further yielded improved and more stable results. Although daily-level forecasting proved challenging with a limited dataset, aggregating to weekly or bi-weekly levels produced more reliable results, demonstrating that forecasting remains viable even for small retailers with limited data resources.

Links: <a href="https://github.com/isaacjeon/product_sales_forecasting" target="_blank">Github repository</a>,
<a href="https://isaacjeon.github.io/portfolio/assets/sales_forecasting_report.pdf" target="_blank">Project report</a>,
<a href="https://isaacjeon.github.io/portfolio/assets/sales_forecasting_slides.pdf" target="_blank">PowerPoint slides</a>
<br>

<p align="center" >
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/product_sales_plot.png" width="100%"/>
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/stl_decomposition_plots.png" width="100%"/>
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/cv_fold_rmse_plot.png" width="100%"/>
</p>

***

### Depression Prediction
This project builds a classification model to predict whether an individual is at risk of depression based on factors such as age, gender, location, and work/study status. The workflow involved data cleaning, exploratory analysis, and training models using **Random Forest**, **XGBoost**, and **Support Vector Machine (SVM)**.

The best-performing model was an SVM with a linear kernel, achieving a recall of 98.9% i.e. correctly identifying nearly all depressed surveyees. This is especially valuable in real-world scenarios where minimizing false negatives is critical. Random Forest and XGBoost also provided feature importance values, helping to identify key contributors to model performance. Notably, age had the highest importance and showed a significant negative correlation with depression, indicating that younger individuals were more likely to be at risk.

Links: <a href="https://github.com/isaacjeon/depression-prediction" target="_blank">Github repository</a>, <a href="https://nbviewer.org/github/isaacjeon/depression-prediction/blob/main/depression-prediction.ipynb" target="_blank">Jupyter Notebook</a><br>

<p align="center" >
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/correlation_heatmap.png" style="height: 265px; width: auto;"/>
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/precision_recall_curve.png" style="height: 265px; width: auto;"/>
</p>

***

### News Article Classification

This project builds models to classify BBC news articles into their correct categories using natural language processing techniques. The text data was processed using **TF-IDF vectorization**, and two models were trained: **Non-negative Matrix Factorization (NMF)** for topic modeling and a **Linear Support Vector Classifier (SVC)** for supervised classification.

The LinearSVC model achieved an accuracy of 98.1%, outperforming the NMF model at 97.0%. This level of performance suggests that the model could be effectively applied to automate the classification of text data for tasks such as content organization, filtering, or topic-based recommendation systems.

Links: <a href="https://github.com/isaacjeon/news_classification" target="_blank">Github repository</a>, <a href="https://nbviewer.org/github/isaacjeon/news_classification/blob/main/bbc-news-classification-nmf-and-linearsvc.ipynb" target="_blank">Jupyter Notebook</a><br>

<p align="center" >
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/word_clouds.png" style="height: 350px; width: auto;"/>
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/news_confusion_matrix.png" style="height: 350px; width: auto;"/>
</p>

***

### SMS Spam Detection

This project involves the use of unsupervised learning techniques for dimensionality reduction in detecting SMS spam messages. The text data was first converted to a Bag-of-Words (BoW) representation, then transformed using three *dimensionality reduction* methods: **TruncatedSVD**, **Non-Negative Matrix Factorization (NMF)**, and **Uniform Manifold Approximation and Projection (UMAP)**. Each reduced dataset was evaluated using **Logistic Regression** for *classification* and **K-Means** for *clustering*.

Of these methods, NMF proved to be the most effective overall (precision = 98.3%, recall = 88.5%). It achieved classification performance comparable to the baseline logistic regression model (precision = 96.7%, recall = 89.3%), reduced the original 2562-dimensional feature space to just 3 dimensions, and resulted in clear clustering. NMF also produced interpretable latent topics, including one that clearly aligned with spam messages.

These results suggest that dimensionality reduction and clustering methods can support not just spam detection, but also tasks such as topic discovery, pattern identification, content grouping visualization, and outlier detection.

Links: <a href="https://github.com/isaacjeon/spam_detection" target="_blank">Github repository</a>, <a href="https://nbviewer.org/github/isaacjeon/spam_detection/blob/main/sms-spam-detection.ipynb" target="_blank">Jupyter Notebook</a><br>

<p align="center" >
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/nmf_clusters.png" width="99%"/>
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/spam_confusion_matrix.png" style="height: 245px; width: auto;"/>
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/cluster_texts.png" style="height: 245px; width: auto;"/>
</p>

***

# Dashboards

### Brazilian E-Commerce
This project explores Brazilian e-commerce order data across several dimensions including customer and seller demographics, product categories, payment types, delivery logistics, and review scores. It utilizes **SQL** for data structuring, cleaning, and exploratory data analysis (EDA), and **Tableau** for building interactive dashboards for visualizing monthly sales performance, regional trends, delivery times, and customer purchasing behavior and satisfaction to uncover operational and marketing insights.

Links: <a href="https://github.com/isaacjeon/brazilian_ecommerce" target="_blank">Github repository</a>, <a href="https://public.tableau.com/views/BrazilianE-Commerce_17547219521680/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link" target="_blank">Tableau Public dashboard</a><br>

<p align="center">
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/brazilian_ecommerce_dashboard.png" width="100%"/>
</p>

***

### Japan Tourism
This project builds a Tableau dashboard for visualization of Japan tourism data on a regional level collected across various categories (e.g., Accommodation Type, Nationality, Length of Stay. The original Excel file was cleaned and restructured, before importing the data into **MySQL**. The tables were then unified and prepared for exploration using **Tableau**. The interactive dashboard enables identification and filtering of characteristics of tourists that visit each region of Japan, which may provide some insight into which demographics each region may or may not appeal to and can allow for more informative tourism advertising such as in promotion of travel services and attractions.

Links: <a href="https://github.com/isaacjeon/japan-tourism-segmentation" target="_blank">Github repository</a>, <a href="https://public.tableau.com/views/JapanTourismbyPrefectureRegion/Dashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link" target="_blank">Tableau Public dashboard</a><br>

<p align="center">
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/japan_tourism_dashboard.png" width="100%"/>
</p>

***

### Student Progress Report
A **Google Sheets** dashboard built to track individual student as well as overall class performance in a middle school math class (Algebra 1). The tool includes anonymized data for four students and provides dynamic visualizations of homework and assessment scores across chapters and time. Features include automatic grade calculations, trend charts, and filters for analyzing individual or class-wide performance over time. Additional functions allow users to search the class schedule by date to look up assignments or filter by chapter to view chapter-specific scores and section topics.

This dashboard is a modified version that I made for personal use in classroom data analysis. It enabled me to easily identify dips in individual student performance by comparing their results to class averages across specific topics, allowing me to pinpoint concepts that a specific student or the overall class found particularly challenging or excelled in. Additionally, by observing performance trends over time I could detect periods of relatively poor results compared to their usual performance, possibly indicating an issue stemming from outside the classroom. The information gained from this analysis would then be used to adjust classroom instruction and provide targeted support accordingly.

Links: <a href="https://docs.google.com/spreadsheets/d/18JRLqKFpRUV40HY5y26woV0SE-YZLjCWwkF2F0uZEls/edit?usp=sharing" target="_blank">Google Sheets</a><br>

<p align="center">
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/student_progress_dashboard.png" width="100%"/>
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/student_progress_dashboard_2.png" width="45%"/>
  <img src="https://raw.githubusercontent.com/isaacjeon/portfolio/refs/heads/main/assets/student_progress_dashboard_3.png" width="45%"/>
</p>

***
