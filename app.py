import numpy as np
import streamlit as st
import pandas as pd
import hashlib
import sqlite3 
import chatbot
import predict_cancer
import breast_cancer


def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def main():
    st.title("AI Cancer Shield")
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT_XzWVjHbcWLOcLVbnrdWhcRBEdIYlXP-qVQ&s")
    st.header("Welcome to Cancer Awareness and Detection Platform")
    st.write("""
    Cancer is one of the leading causes of death worldwide. Early detection can help save lives.
    This platform provides information about cancer, along with tools like a chatbot and an image-based brain cancer prediction model.
    Notable research by experts like Tata has played a significant role in cancer research and treatment.
    """)
    menu = ["Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Login":
        st.subheader("Login Section")
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:
                st.success("Logged In as {}".format(username))
                PAGES = {
                    "Brain cancer ": predict_cancer,
                    "Breast Cancer": breast_cancer,
                    "ChatBot": chatbot
                    }
                st.sidebar.title('AI Cancer Shield')
                selection = st.sidebar.radio("Go to", list(PAGES.keys()))
                page = PAGES[selection]
                page.app()
            else:
                st.warning("Incorrect Username/Password")

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')
        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")



if __name__ == '__main__':
    main()