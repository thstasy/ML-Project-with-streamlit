# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st
# from PIL import Image
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical


# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train = x_train / 255.0
# x_test = x_test / 255.0

# y_train = np.squeeze(y_train)
# y_test = np.squeeze(y_test)

# num_classes = 10
# y_train_one_hot = to_categorical(y_train, num_classes)
# y_test_one_hot = to_categorical(y_test, num_classes)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train, y_train_one_hot, batch_size=64, epochs=10, validation_data=(x_test, y_test_one_hot))


# def main():
#     st.title('Cifar10 Web Classifier')
#     st.write('Upload any image that you think fits into one of the classes')
#     file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
#     if file:
#         image = Image.open(file)
#         st.image(image, use_column_width=True)
#         resized_image = image.resize((32, 32))
#         img_array = img_array.reshape((1, 32, 32, 3))
#         model = tf.keras.models.load_model('cifar10_model.h5')

#         predictions = model.predict(img_array)
#         cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck']

#         fig, ax = plt.subplots()
#         y_pos = np.arrange(len(cifar10_classes))
#         ax.barh(y_pos, predictions[0], align='center')
#         ax.set_yticks(y_pos)
#         ax.set_yticklabels(cifar10_classes)
#         ax.invert_yaxis()
#         ax.set_xlabel("Probability")
#         ax.set_title('CIFAR10 Predictions')
#         st.pyplot(fig)
#     else:
#         st.text('You have not uploaded an image yet.')

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to integers
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

# Convert integer labels to one-hot encoding
num_classes = 10
y_train_one_hot = to_categorical(y_train, num_classes)
y_test_one_hot = to_categorical(y_test, num_classes)

# Define the model
model = Sequential([
    Flatten(input_shape=(32, 32, 3)), 
    Dense(1000, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=10, validation_data=(x_test, y_test_one_hot))

def main():
    st.title('Cifar10 Web Classifier')
    st.write('Upload any image that you think fits into one of the classes')
    file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        resized_image = image.resize((32, 32))
        img_array = img_array.reshape((1, 32, 32, 3))
        model = tf.keras.models.load_model('cifar10_model.h5')

        predictions = model.predict(img_array)
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_classes))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_classes)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title('CIFAR10 Predictions')
        st.pyplot(fig)
    else:
        st.text('You have not uploaded an image yet.')

if __name__ == '__main__':
    main()
