import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn import svm, linear_model, tree, neighbors
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Title and Introduction
st.title('Dự đoán rủi ro Sinh viên Bỏ học thông qua Máy học')
st.write("""
Việc dự đoán rủi ro sinh viên bỏ học là vô cùng quan trọng đối với các cơ sở giáo dục nhằm nâng cao tỷ lệ thành công để giữ chân sinh viên tiếp tục học tại trường.
Trang web được sinh ra để so sánh bốn kỹ thuật máy học— Supper vector machine, Logistic regression, Desicion Tree và K-Nearest Neighbor
để dự đoán tỷ lệ bỏ học của sinh viên, đồng thời cung cấp cái nhìn sâu sắc về hiệu quả và khả năng áp dụng của chúng.
""")

@st.cache_data
def prepare_data_and_evaluate_models():
    data = pd.read_csv('MYResult.csv')
    
    # Display the first few rows of the dataframe to understand its structure
    data.head()
    
    # Define categorical and numeric features
    categorical_features = data.select_dtypes(include=['object', 'bool']).drop(['Dropout'], axis=1).columns
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Create transformers for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
        ('scaler', StandardScaler())  # Scale data
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Impute missing values with 'missing'
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical data
    ])
    
    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Prepare the target variable
    y = data['Dropout']
    
    # Prepare feature matrix
    X = data.drop('Dropout', axis=1)
    
    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models with default settings
    models = {
        'SVM': SVC(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # Create a full pipeline with preprocessing and the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Predict on the validation set
        y_pred = pipeline.predict(X_valid)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_valid, y_pred)
        results[name] = accuracy
    
    # Identify the best model based on accuracy
    best_model_name, best_model_instance = max(models.items(), key=lambda x: results[x[0]])
    
    return results, preprocessor, best_model_instance


# Call the function and output the results, scaler, and best model instance
results, scaler, best_model_instance = prepare_data_and_evaluate_models()

st.header('Tiêu chí so sánh')
st.write("""
Hiệu quả của các kỹ thuật máy học trong việc dự đoán tỷ lệ sinh viên bỏ học có thể được đánh giá qua nhiều tiêu chí:
- **Độ chính xác**: Tỷ lệ của kết quả đúng trong tổng số trường hợp được xem xét.
- **Khả năng Mở rộng**: Khả năng duy trì hoặc cải thiện hiệu suất khi tăng kích thước của bộ dữ liệu.
- **Dễ hiểu**: Mức độ dễ dàng hiểu được kết quả bởi con người.
- **Yêu cầu về Tính toán**: Lượng tài nguyên tính toán cần thiết để huấn luyện và chạy mô hình.
- **Khả năng Áp dụng Thực tế**: Tính thực tiễn khi áp dụng mô hình trong môi trường giáo dục thực tế.
""")

criteria_expander = st.expander("**Tìm hiểu thêm về mỗi tiêu chí**")
criteria_expander.write("""
- **Độ Chính xác** đảm bảo rằng các dự đoán chính xác với kết quả thực tế, dựa vào đó làm cơ sở để đưa ra các quyết định và hành động dựa trên kết quả dự đoán.
- **Khả năng Mở rộng** đảm bảo khi có thêm dữ liệu sinh viên, mô hình có thể xử lý bộ dữ liệu lớn hơn mà không giảm đáng kể hiệu suất.
- **Dễ hiểu** quan trọng đối với các bên liên quan để hiểu các dự đoán và lý do của mô hình, tạo dựng sự tin tưởng và cho phép ra quyết định thông tin.
- **Yêu cầu về Tính toán** ảnh hưởng đến khả năng thực thi việc huấn luyện và triển khai mô hình, đặc biệt là trong các cơ sở có tài nguyên hạn chế.
- **Khả năng Áp dụng Thực tế** xem xét sự tích hợp của mô hình với các hệ thống và quy trình hiện có, đảm bảo rằng các dự đoán có thể được sử dụng hiệu quả để hỗ trợ sinh viên có nguy cơ.
""")

st.header('Tổng quan về Kỹ thuật Máy học')
techniques = {
    'SVM (Support Vector Machine)': 'Một mô hình học có giám sát có thể phân loại các trường hợp bằng cách tìm một bộ phân tách. SVM hoạt động tốt với biên độ phân tách rõ ràng và hiệu quả trong không gian nhiều chiều.',
    'Logistic Regression': 'Một mô hình thống kê sử dụng hàm logistic để mô hình hóa một biến phụ thuộc nhị phân ở dạng cơ bản nhất.',
    'Decision Tree': 'Một công cụ hỗ trợ quyết định sử dụng mô hình giống như cây của các quyết định. Nó đơn giản để hiểu và dễ để giải thích, làm cho nó phổ biến cho các nhiệm vụ phân loại.',
    'K-Nearest Neighbors (KNN)': 'Một phương pháp không tham số được sử dụng cho phân loại và hồi quy. Trong cả hai trường hợp, đầu vào bao gồm k ví dụ đào tạo gần nhất trong không gian đặc trưng. Kết quả phụ thuộc vào việc KNN được sử dụng cho phân loại hay hồi quy.'
}

for technique, description in techniques.items():
    st.subheader(technique)
    st.write(description)

# Assuming 'prepare_data_and_evaluate_models' function is defined and cached as in the previous code snippet

st.header('Detailed Comparison')

# Evaluate the models (reusing the previously defined function)
results, scaler, best_model = prepare_data_and_evaluate_models()

# Display the results in a table
st.subheader('Model Performance')
st.write(pd.DataFrame(results.items(), columns=['Model', 'Accuracy']).sort_values('Accuracy', ascending=False))

# Visualization of model performances
st.subheader('Accuracy Visualization')
fig, ax = plt.subplots()
pd.DataFrame(results.items(), columns=['Model', 'Accuracy']).set_index('Model').sort_values('Accuracy').plot(kind='barh', legend=None, ax=ax)
plt.title('Comparison of Model Accuracies')
plt.xlabel('Accuracy')
plt.ylabel('Model')
st.pyplot(fig)

st.subheader('Detailed Comparison Across Criteria')

additional_criteria = {
    'Scalability': {'SVM (Support Vector Machine)': 'Cao', 'Logistic Regression': 'Trung bình', 'Decision Tree': 'Cao', 'K-Nearest Neighbors (KNN)': 'Thấp'},
    'Ease of Interpretation': {'SVM (Support Vector Machine)': 'Thấp', 'Logistic Regression': 'Cao', 'Decision Tree': 'Cao', 'K-Nearest Neighbors (KNN)': 'Trung bình'},
    'Computational Requirements': {'SVM (Support Vector Machine)': 'Cao', 'Logistic Regression': 'Thấp', 'Decision Tree': 'Trung bình', 'K-Nearest Neighbors (KNN)': 'Trung bình'},
    'Real-world Applicability': {'SVM (Support Vector Machine)': 'Trung bình', 'Logistic Regression': 'Cao', 'Decision Tree': 'Cao', 'K-Nearest Neighbors (KNN)': 'Thấp'}
}

# For displaying purposes, we can format it into a DataFrame
df_criteria = pd.DataFrame(additional_criteria)
st.write(df_criteria)

# To provide context or further explanation, you might want to add descriptions for why a model received a particular score or status.
st.write("""
**Khả năng Mở rộng** phản ánh khả năng của mô hình để xử lý hiệu quả lượng dữ liệu tăng lên. Các mô hình như Cây quyết định thường được coi là có khả năng mở rộng do sự đơn giản và dễ dàng trong việc xử lý nhiều công việc song song cùng một lần của chúng.

**Dễ hiểu** nói về mức độ dễ dàng để hiểu các quyết định của mô hình. Hồi quy Logistic và Cây quyết định có điểm cao vì chúng tạo ra các mô hình dễ để hiểu hơn so với các mô hình phức tạp hơn như SVM.

**Yêu cầu về Tính toán** xem xét nguồn lực tính toán cần thiết cho việc đào tạo và dự đoán. SVM, đặc biệt với không gian đặc trưng lớn, thường yêu cầu nhiều nguồn lực hơn.

**Khả năng Áp dụng Thực tế** xem xét tính thực tiễn khi triển khai mô hình trong các cài đặt giáo dục thực tế, xem xét các yếu tố như khả năng giải thích, khả năng mở rộng, và yêu cầu về nguồn lực. Hồi quy Logistic và Cây quyết định thường có điểm cao hơn do sự cân bằng của chúng giữa độ chính xác, khả năng giải thích, và hiệu quả tính toán.
""")

st.header('Interactive Prediction Tool')

st.write("Công cụ tương tác này cho phép bạn nhập các đặc điểm của sinh viên dựa trên mô tả tập dữ liệu được cung cấp. Nó mô phỏng cách các mô hình máy học có thể dự đoán rủi ro bỏ học dựa trên các feature đặc trưng của chúng.")

# Dropdown and input fields for the dataset description
# major = st.selectbox('Bạn đang học Chuyên ngành gì?', range(1, 70))
# major_change = st.radio('Bạn có chuyển ngành khi nhập học không?', 
#                             ('Không, đây là chuyên ngành tôi đăng ký.',
#                              'Có, tôi đã chuyển ngành lúc nhập học.',
#                              'Có, tôi đã chuyển ngành khi học hết năm 1.'))
# age = st.slider('Bạn bao nhiêu tuổi?', 16,100,16)
# gender = st.radio('Giới tính ?', ('Nam', 'Nữ'))
# ethnicity = st.radio('Dân Tộc?', ('Kinh', 'Khác'))
# language = st.radio('Ngôn ngữ chính của Bạn là gì?', ('Tiếng Việt', 'Tiếng Anh', 'Khác'))
# financial_situation = st.radio('Tình hình tài chính gia đình Bạn thế nào?', ('Không tốt', 'Tốt', 'Khá tốt'))
# father_education = st.radio('Trình độ học vấn của Bố bạn là gì?', ('Sau Đại Học', 'Đại Học', 'Cao Đẳng', 'Thấp hơn'))
# mother_education = st.radio('Trình độ học vấn của Mẹ bạn là gì?', ('Sau Đại Học', 'Đại Học', 'Cao Đẳng', 'Thấp hơn'))
# mother_occupation = st.radio('Nghề nghiệp của Mẹ bạn là gì?', 
#                                   ('Nghỉ Hưu', 'Nội trợ', 'Công chức nhà nước', 'Kinh doanh', 'Lao động tự do', 'Khác'))
# father_occupation = st.radio('Nghề nghiệp của Bố bạn là gì?', 
#                                   ('Nghỉ Hưu', 'Nội trợ', 'Công chức nhà nước', 'Kinh doanh', 'Lao động tự do', 'Khác'))
# mental_health_status = st.radio('Tình trạng sức khỏe tâm thần của Bạn hiện giờ thế nào ?', ('Tốt', 'Không tốt'))
# physical_health_status = st.radio('Tình trạng sức khỏe thể chất của Bạn hiện giờ thế nào ?', ('Tốt', 'Không tốt'))
# family_support = st.radio('Gia đình có ủng hộ ngành Bạn đang học không?', ('Có', 'Không'))
# family_financial_support = st.radio('Gia đình có hỗ trợ cho Bạn không?', ('Có', 'Không'))
# peer_support = st.radio('Bạn có nhóm bạn cùng hỗ trợ nhau học tập hoặc hỗ trợ nhau trong cuộc sống không?', ('Có', 'Không'))
# academic_activities = st.radio('Bạn từng tham gia hoạt động học thuật nào chưa ?', ('Có', 'Chưa bao giờ.'))
# club_activities = st.radio('Bạn có tham gia sinh hoạt câu lạc bộ nào không ?', ('Có', 'Không'))
# part_time_job = st.radio('Bạn có đi làm thêm không ?', ('Có', 'Không'))
# # For the input text fields, we'll use placeholders as an example to simplify
# part_time_job_details = st.text_input('Công việc làm thêm của bạn là gì ?')
# part_time_hours = st.text_input('Tổng thời gian bạn dành cho công việc làm thêm là bao nhiêu giờ một ngày ?')
# part_time_income = st.text_input('Thu nhập từ công việc làm thêm là bao nhiêu ?')
# internet_usage_for_study = st.radio('Bạn có sử dụng máy tính và internet để phục vụ việc học của mình không?', ('Có', 'Không'))
# online_tools_usage = st.radio('Bạn có sử dụng các công cụ trực tuyến để phục vụ việc học tập của mình không?', ('Có', 'Không'))
# social_media_time = st.radio('Thời gian Bạn dành cho các mạng xã hội như Facebook, Tiktok... mỗi ngày là bao nhiêu?',
#                              ('Nhiều hơn 2 tiếng', 'Ít hơn 2 tiếng', 'Trung bình 10 - 30 phút tôi kiểm tra một lần.', 'Tôi không dùng thường xuyên, ít hơn 30\' mỗi ngày.'))
# facility_quality = st.radio('Bạn đánh giá thế nào về cơ sở vật chất của Trường?', ('Rất tốt', 'Tốt', 'Chưa tốt'))
# teaching_quality = st.radio('Bạn đánh giá thế nào về chất lượng giảng dạy của Giảng viên ?', ('Rất tốt', 'Tốt', 'Chưa hài lòng'))
# recent_life_event = st.radio('Trong thời gian gần đây Bạn có gặp biến cố gì lớn trong cuộc sống không ?', ('Không', 'Có'))
# local_economic_condition = st.radio('Điều kiện kinh tế nơi Bạn sống tốt không?', ('Tốt', 'Chưa tốt'))
# local_cultural_level = st.radio('Trình độ văn hóa, văn minh nơi bạn sống thế nào?', ('Cao', 'Trung Bình', 'Còn thấp'))
# continuing_education = st.radio('Bạn có tiếp tục học ở trường không ?', ('Có', 'Không', 'Đang cân nhắc thêm'))

# # Text input for multiple-choice question where users can specify reasons affecting their choices
# #survey_impact_reasons = st.text_area('Trong các câu khảo sát ở trước (Câu 1->30) thì đâu là những lý do ảnh hưởng tới lựa chọn của bạn ?')

# if st.button('Predict Dropout Risk'):
#     # Map the user inputs to the expected model input format
#     user_inputs = {
#         'col1': major,
#         'col2': {'Không, đây là chuyên ngành tôi đăng ký.': 1, 'Có, tôi đã chuyển ngành lúc nhập học.': 2, 'Có, tôi đã chuyển ngành khi học hết năm 1.': 3}[major_change],
#         'col3': age,
#         'col4': {'Nam': 1, 'Nữ': 2}[gender],
#         'col5': {'Kinh': 1, 'Khác': 2}[ethnicity],
#         'col6': {'Tiếng Việt': 1, 'Tiếng Anh': 2, 'Khác': 3}[language],
#         'col7': {'Không tốt': 1, 'Tốt': 2, 'Khá tốt': 3}[financial_situation],
#         'col8': {'Sau Đại Học': 4, 'Đại Học': 3, 'Cao Đẳng': 2, 'Thấp hơn': 1}[father_education],
#         'col9': {'Sau Đại Học': 4, 'Đại Học': 3, 'Cao Đẳng': 2, 'Thấp hơn': 1}[mother_education],
#         'col10': {'Nghỉ Hưu': 1, 'Nội trợ': 2, 'Công chức nhà nước': 3, 'Kinh doanh': 4, 'Lao động tự do': 5, 'Khác': 6}[mother_occupation],
#         'col11': {'Nghỉ Hưu': 1, 'Nội trợ': 2, 'Công chức nhà nước': 3, 'Kinh doanh': 4, 'Lao động tự do': 5, 'Khác': 6}[father_occupation],
#         'col12': {'Tốt': 1, 'Không tốt': 2}[mental_health_status],
#         'col13': {'Tốt': 1, 'Không tốt': 2}[physical_health_status],
#         'col14': {'Có': 1, 'Không': 2}[family_support],
#         'col15': {'Có': 1, 'Không': 2}[family_financial_support],
#         'col16': {'Có': 1, 'Không': 2}[peer_support],
#         'col17': {'Có': 1, 'Chưa bao giờ.': 2}[academic_activities],
#         'col18': {'Có': 1, 'Không': 2}[club_activities],
#         'col19': {'Có': 1, 'Không': 2}[part_time_job],
#         # Assuming textual inputs for 'col20', 'col21', 'col22' need specific handling based on your model
#         #'col20': part_time_job_details,  # Directly use the input string for now
#         #'col21': part_time_hours,
#         #'col22': part_time_income,
#         'col23': {'Có': 1, 'Không': 2}[internet_usage_for_study],
#         'col24': {'Có': 1, 'Không': 2}[online_tools_usage],
#         'col25': {'Nhiều hơn 2 tiếng': 1, 'Ít hơn 2 tiếng': 2, 'Trung bình 10 - 30 phút tôi kiểm tra một lần.': 3, 'Tôi không dùng thường xuyên, ít hơn 30\' mỗi ngày.': 4}[social_media_time],
#         'col26': {'Rất tốt': 3, 'Tốt': 2, 'Chưa tốt': 1}[facility_quality],
#         'col27': {'Rất tốt': 3, 'Tốt': 2, 'Chưa hài lòng': 1}[teaching_quality],
#         'col28': {'Không': 1, 'Có': 2}[recent_life_event],
#         'col29': {'Tốt': 2, 'Chưa tốt': 1}[local_economic_condition],
#         'col30': {'Cao': 3, 'Trung Bình': 2, 'Còn thấp': 1}[local_cultural_level],
#         'col31': {'Có': 3, 'Không': 1, 'Đang cân nhắc thêm': 2}[continuing_education],
#         #'col32': survey_impact_reasons,  # Assuming this is handled correctly in your model
#     }

#     # Convert this dictionary into a DataFrame
#     input_df = pd.DataFrame([user_inputs])
#     st.write(input_df)
#     scaled_inputs = scaler.transform(input_df)  # Placeholder for the actual scaler used in training

    # Load the best model
    
    # Make a prediction
    # prediction = best_model.predict(scaled_inputs)
    
    # Display the prediction result
    # st.write(f"Dựa vào dữ liệu được cung cấp ở trên, thì mức độ rủi ro mà sinh viên này bỏ học là: {'**Thấp**' if prediction[0] == 1 else '**Cao**'}")
    
# Title and Introduction
st.title('Dự đoán rủi ro Sinh viên Bỏ học thông qua Máy học')
st.write("Việc dự đoán rủi ro sinh viên bỏ học là vô cùng quan trọng đối với các cơ sở giáo dục nhằm nâng cao tỷ lệ giữ chân sinh viên.")

# Uploading the CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Here's the first five entries of your file:")
    st.write(data.head())

    # Assuming 'data' needs some preprocessing and model is already trained and saved (you can load model here)
    # For demo, using a placeholder for model prediction
    # You will need to replace this with actual model loading and prediction logic
    # Placeholder: randomly assign dropout prediction
    np.random.seed(42)
    data['prediction'] = np.random.choice(['dropout', 'continue'], size=len(data))

    # Display the predictions in the table
    st.write("Predictions:")
    st.dataframe(data[['StudentId', 'prediction']])  # Assuming 'student_id' column exists

    # Display summary statistics
    dropout_rate = np.mean(data['prediction'] == 'dropout')
    continue_rate = np.mean(data['prediction'] == 'continue')
    st.write(f"Percentage of students predicted to dropout: {dropout_rate * 100:.2f}%")
    st.write(f"Percentage of students predicted to continue: {continue_rate * 100:.2f}%")

st.header('Feedback')
st.write('Bạn thấy dự đoán bị sai? vui lòng phản hồi ở dưới :).')

# Feedback form (simplified version)
user_feedback = st.text_area("Nhập phản hồi của bạn ở đây")
if st.button('Gửi Phản hồi'):
    st.write('Cảm ơn bạn đã gửi phản hồi!')

st.header('Kết luận')
st.write("""
Trang web này cung cấp một cái nhìn so sánh về bốn kỹ thuật máy học để dự đoán tỷ lệ sinh viên bỏ học.
Mỗi phương pháp có điểm mạnh và điểm yếu riêng, và việc chọn kỹ thuật tốt nhất có thể phụ thuộc vào bối cảnh và yêu cầu cụ thể.
Chúng tôi khuyến khích các cơ sở giáo dục và nhà nghiên cứu khám phá thêm các kỹ thuật này để tìm ra giải pháp phù hợp nhất với nhu cầu của họ.
""")

st.subheader('Tài liệu tham khảo')
st.write("""
- [Scikit-Learn](https://scikit-learn.org/stable/documentation.html) để biết thông tin chi tiết về các mô hình máy học.
- [Streamlit](https://docs.streamlit.io) để học cách xây dựng ứng dụng web tương tác cho dự án dữ liệu của bạn.
- [Towards Data Science](https://towardsdatascience.com/) để đọc các bài báo sâu sắc về máy học và khoa học dữ liệu.
""")