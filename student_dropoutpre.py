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
    
        # Check for NaN values in each column
    nan_columns = data.isnull().sum()
    print("Number of NaN values in each column:\n", nan_columns)

    # Filter out columns with any NaN values
    columns_with_nan = nan_columns[nan_columns > 0].index.tolist()
    print("Columns with NaN values:", columns_with_nan)

    # Locate rows where any cell in each row has NaN
    nan_rows = data[data.isnull().any(axis=1)]
    print("Rows with NaN values:\n", nan_rows)

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
- **Khả năng mở rộng**: Khả năng duy trì hoặc cải thiện hiệu suất khi tăng kích thước của bộ dữ liệu.
- **Dễ hiểu**: Mức độ dễ dàng hiểu được kết quả bởi con người.
- **Yêu cầu về tính toán**: Lượng tài nguyên tính toán cần thiết để huấn luyện và chạy mô hình.
- **Khả năng á dụng thực tế**: Tính thực tiễn khi áp dụng mô hình trong môi trường giáo dục thực tế.
""")

criteria_expander = st.expander("**Tìm hiểu thêm về mỗi tiêu chí**")
criteria_expander.write("""
- **Độ chính xác** đảm bảo rằng các dự đoán chính xác với kết quả thực tế, dựa vào đó làm cơ sở để đưa ra các quyết định và hành động dựa trên kết quả dự đoán.
- **Khả năng mở rộng** đảm bảo khi có thêm dữ liệu sinh viên, mô hình có thể xử lý bộ dữ liệu lớn hơn mà không giảm đáng kể hiệu suất.
- **Dễ hiểu** quan trọng đối với các bên liên quan để hiểu các dự đoán và lý do của mô hình, tạo dựng sự tin tưởng và cho phép ra quyết định thông tin.
- **Yêu cầu về tính toán** ảnh hưởng đến khả năng thực thi việc huấn luyện và triển khai mô hình, đặc biệt là trong các cơ sở có tài nguyên hạn chế.
- **Khả năng áp dụng thực tế** xem xét sự tích hợp của mô hình với các hệ thống và quy trình hiện có, đảm bảo rằng các dự đoán có thể được sử dụng hiệu quả để hỗ trợ sinh viên có nguy cơ.
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
results, preprocessor, best_model = prepare_data_and_evaluate_models()

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

st.subheader('Phân tích các tiêu chí')

additional_criteria = {
    'Khả năng mở rộng': {'SVM (Support Vector Machine)': 'Cao', 'Logistic Regression': 'Trung bình', 'Decision Tree': 'Cao', 'K-Nearest Neighbors (KNN)': 'Thấp'},
    'Dễ hiểu': {'SVM (Support Vector Machine)': 'Thấp', 'Logistic Regression': 'Cao', 'Decision Tree': 'Cao', 'K-Nearest Neighbors (KNN)': 'Trung bình'},
    'Yêu cầu về tính toán': {'SVM (Support Vector Machine)': 'Cao', 'Logistic Regression': 'Thấp', 'Decision Tree': 'Trung bình', 'K-Nearest Neighbors (KNN)': 'Trung bình'},
    'Khả năng áp dụng thực tế': {'SVM (Support Vector Machine)': 'Trung bình', 'Logistic Regression': 'Cao', 'Decision Tree': 'Cao', 'K-Nearest Neighbors (KNN)': 'Thấp'}
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
    
# Title and Introduction
st.title('Dự đoán rủi ro Sinh viên Bỏ học thông qua Máy học')
st.write("Việc dự đoán rủi ro sinh viên bỏ học là vô cùng quan trọng đối với các cơ sở giáo dục nhằm nâng cao tỷ lệ giữ chân sinh viên.")

# Uploading the CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Đây là năm mẫu dữ liệu đầu tiên trong file bạn vừa tải lên:")
        st.write(data.head())

        # Apply preprocessing to the necessary columns
        # This is a simplified example; adjust according to your actual preprocessing needs
        scaled_inputs = preprocessor.transform(data)

        # Predict using the trained model
        data['Dropout?'] = best_model.predict(scaled_inputs)

        # Display the predictions
        st.write("Kết quả dự đoán:")
        st.dataframe(data[['StudentId', 'Dropout?']])

        # Display summary statistics
        dropout_rate = np.mean(data['Dropout?'] == True)
        continue_rate = np.mean(data['Dropout?'] == False)
        st.write(f"Dự đoán phần trăm học sinh bỏ học: {dropout_rate * 100:.2f}%")
        st.write(f"Dự đoán phần trăm học sinh tiếp tục học: {continue_rate * 100:.2f}%")

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