import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

def welcome():
    return 'welcome all'

def prediction(MIP,	SDIP, KIP, SIP, MDM, SDDM, KDM, SDM):

    prediction = classifier.predict_proba(
        [[MIP, SDIP, KIP, SIP, MDM, SDDM, KDM, SDM]])
    return prediction




  # giving the webpage a title


# here we define some of the front end elements of the web page like
# the font and background color, the padding and the text to be displayed
html_temp = """

<div style ="background-color:#FBB62C;padding:13px">
<h1 style ="color:black;text-align:center;">Pulsar Detection Using Machine Learning </h1>
</div>
"""




# this line allows us to display the front end aspects we have
# defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)

html_temp = """
<br><br>
"""
# this line allows us to display the front end aspects we have
# defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)

st.write("""
Universe provides astronomers with huge amount of data regarding Pulsars. It is usually difficult for
astronomers to look for patterns in data and use their brains to find a promising candidate. So I have tried to implement
Machine Learning models to aid them in finding Pulsars. You can find more about the project here https://thevoxium.github.io/Pulsars-Detection/
""")
# the following lines create text boxes in which the user can enter
# the data required to make the prediction


html_temp = """
<br><br>
"""
# this line allows us to display the front end aspects we have
# defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)



MIP = st.slider("Mean of the integrated profile", min_value=-100.00, max_value=1000.00)
SDIP = st.slider("Standard deviation of the integrated profile", min_value=-100.00, max_value=1000.00)
KIP = st.slider("Excess kurtosis of the integrated profile", min_value=-100.00, max_value=1000.00)
SIP = st.slider("Skewness of the integrated profile", min_value=-100.00, max_value=1000.00)
MDM = st.slider("Mean of the DM-SNR curve", min_value=-100.00, max_value=1000.00)
SDDM = st.slider("Standard deviation of the DM-SNR curve", min_value=-100.00, max_value=1000.00)
KDM = st.slider("Excess kurtosis of the DM-SNR curve", min_value=-100.00, max_value=1000.00)
SDM = st.slider("Skewness of the DM-SNR curve", min_value=-100.00, max_value=1000.00)


# the below line ensures that when the button called 'Predict' is clicked,
# the prediction function defined above is called to make the prediction
# and store it in the variable result
if st.button("Predict"):
    global result
    result = prediction(MIP, SDIP, KIP, SIP, MDM, SDDM, KDM, SDM)
    p1 = result[0][0]*100
    p2 = result[0][1]*100
    st.success("The Star has "+str(p1)+"% chance of being a Pulsar and "+str(p2)+"% chance of not being a Pulsar.")
