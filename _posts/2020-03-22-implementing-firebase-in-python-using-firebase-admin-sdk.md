---
layout: writing
title:  implementing firebase in python using firebase admin sdk
date:   2020-03-22 20:50:50 +0530
category: Python
tags: beginner python firebase firebase-admin firebase-auth firebase-storage
comment: true
---
Firebase is a mobile and web application development platform by Google. Firebase platform has approx 18 products ready to be implemented in the project.

Features of firebase can be implemented in python using Firebase Admin SDK for Python. In this post, we will tackle Firebase Authentication and Storage which are essential for most of the projects.
<!-- more -->

![https://firebase.google.com/](https://miro.medium.com/max/1280/0*rgZc-kbOQV3WDKMC.jpg)

### Create a Firebase App

1. Open your [Firebase Console](https://console.firebase.google.com/) and create an app or find an existing app.

2. Download the Firebase config file which is in JSON format this file is needed to be mentioned inside your project. In the project‘s console, the config file can be downloaded from Settings > Project Settings >Service Accounts > Firebase Admin SDK > Generate New Private Key

### Install the Firebase SDK

Install the Firebase Admin SDK for python using pip.

{% highlight bash linenos %}
pip install firebase_admin
{% endhighlight %}

### Configure SDK in Python Project

1. Importing Firebase Admin to project

{% highlight python linenos %}
from firebase_admin import credentials
{% endhighlight %}

2. Creating a Firebase app instance

{% highlight python linenos %}
def create_firebase_app():
 cred = credentials.Certificate('PATH OF FIREBASE SERVICE FILE ')
 firebase=firebase_admin.initialize_app(cred)
 return firebase
{% endhighlight %}

{% highlight python linenos %}
firebase_instance=create_firebase_app()
{% endhighlight %}

### Implementing Firebase Auth

Firebase Auth is a service that can authenticate users using only client-side code.

1. Importing Firebase Auth to project

{% highlight python linenos %}
from firebase_admin import auth
{% endhighlight %}
2. Creating a User

{% highlight python linenos %}
def create_user(username,name,email,password):
 try:
  auth.create_user(email=email,email_verified=True,password=password,display_name=name)
  return f'{username} created successfully'
 except Exception as e:
  return f'ERROR: {e}'
 #Taking User Info Logic Here( GUI or Input)
 username="tarun_11"
 name="Tarun"
 password="PASSWORD"
 email="tarunbisht@company.com"
 #Create User
 status=create_user(username=username,name=name,email=email,password=password)
 print(status)
{% endhighlight %}

3. Retrieving User Data

Functions used to retrieve user data returns [UserRecord](https://firebase.google.com/docs/reference/admin/python/firebase_admin.auth?authuser=0#userrecord) object using which we can access user information.

* Using User Id

{% highlight python linenos %}
def get_user(id):
 user = auth.get_user(id)
 return user
{% endhighlight %}

* Using Email Id

{% highlight python linenos %}
def get_user(email):
 user = auth.get_user_by_email(email)
 return user
{% endhighlight %}

* All Users

{% highlight python linenos %}
for user in auth.list_users().iterate_all():
 print('User: ' + user.uid)
{% endhighlight %}

4. Updating User Data

{% highlight python linenos %}

# need to specify a uid

user = auth.update_user(
 uid,
 email='new_email@company.com',
 phone_number='+10000010200',
 email_verified=True,
 password='newPassword',
 display_name='New Name',
 photo_url='URL',
 disabled=True
 )
{% endhighlight %}

5. Deleting User Data

{% highlight python linenos %}
auth.delete_user(uid)
{% endhighlight %}

### Implementing Firebase Storage

Firebase Storage provides secure file uploads and downloads for Firebase apps. The developer can use it to store images, audio, video, or other user-generated content.

1. Importing Firebase Storage to project

{% highlight python linenos %}
from firebase_admin import storage
{% endhighlight %}

2. Configuring Firebase Storage bucket

{% highlight python linenos %}

# Edit Create App function to provide storage bucket address

def create_firebase_app():
 cred = credentials.Certificate('PATH OF FIREBASE SERVICE FILE ')
 firebase=firebase_admin.initialize_app(cred, {
 'storageBucket': 'BUCKET URL FROM FIREBASE STORAGE DASHBOARD'
 })
 return firebase
{% endhighlight %}

{% highlight python linenos %}
firebase_instance=create_firebase_app()
{% endhighlight %}

3. Uploading File to Firebase Storage

{% highlight python linenos %}
'''
Arguments:
file= File to upload
name=file name by which it is stored in bucket
format (optional)= Format of file (like jpg,png or pdf etc..)
'''
def upload_file(file,name,format="image/webp"):
 bucket = storage.bucket()
 blob = bucket.blob(name)
 blob.upload_from_filename(file,content_type=format)
{% endhighlight %}

{% highlight python linenos %}
# Uploading a file in root
upload_file(file='image.webp',name='cover.webp',format="image/webp")
{% endhighlight %}

{% highlight python linenos %}
# Uploading a file in a folder
upload_blob(file='image.webp',name=os.path.join(FOLDER_NAME, 'cover.webp'),format="image/webp")
{% endhighlight %}

4. Delete File from Firebase Storage

{% highlight python linenos %}
'''
Arguments:
filename= Name of file to delete
folder_name (optional)= Name of folder inside bucket where specified file resides
'''
def delete_file(filename,folder_name=None):
 bucket=storage.bucket()
 deleted=None
 if folder_name:
  blob=bucket.blob(os.path.join(folder_name,filename))
  deleted=blob.delete()
 else:
  blob=bucket.blob(filename)
  deleted=blob.delete()
  return deleted
{% endhighlight %}

{% highlight python linenos %}
# Deleting a file from root
deleted_file=delete_file('cover.webp')
{% endhighlight %}

{% highlight python linenos %}
# Deleting a file from a folder
deleted_file =delete_file(os.path.join(FOLDER_NAME, 'cover.webp'))
{% endhighlight %}

5. Making a File Public

{% highlight python linenos %}
'''
Arguments:
filename= Name of file to delete
folder_name (optional)= Name of folder inside bucket where specified file resides
'''
def make_file_public(filename,folder_name=None):
 bucket=storage.bucket()
 if folder_name:
  blob=bucket.blob(os.path.join(folder_name,filename))
  blob.make_public()
 else:
  blob=bucket.blob(filename)
  blob.make_public()
{% endhighlight %}

{% highlight python linenos %}
# Making file public from a folder
make_file_public('cover.webp')
{% endhighlight %}

{% highlight python linenos %}
# Making file public from root
make_file_public(os.path.join(FOLDER_NAME, 'cover.webp'))
{% endhighlight %}

6. Generating access URL for a file

The saved file inside bucket needs to be accessed in some manner most common way to access a file is through URL. Firebase provides a way to generate

* Signed URL- URL which is valid for a specific time after which they get expired.

* Public URL- URL which is valid till the file is present in storage. To get a public URL file should be public.

{% highlight python linenos %}
# Signed URL
'''
Arguments:
name=Name of file for which url is to generate
minutes (optional)=Expiry time of url
'''
def generate_signed_url (name,minutes=43800):
 bucket=storage.bucket()
 blob=bucket.blob(name)
 return blob.generate_signed_url(timedelta(minutes=minutes))
{% endhighlight %}

{% highlight python linenos %}
# generating signed url for a file in root of bucket
url= generate_signed_url ('cover.webp')
{% endhighlight %}

{% highlight python linenos %}
# generating signed url for file inside a folder
url= generate_signed_url (os.path.join(FOLDER_NAME, 'cover.webp'))
{% endhighlight %}

{% highlight python linenos %}
# Public URL
'''
Arguments:
Name=Name of file for which url is to generate
'''
def generate_public_url (name):
 bucket=storage.bucket()
 blob=bucket.blob(name)
 return blob.public_url
{% endhighlight %}

{% highlight python linenos %}
# generating public url for a file in root of bucket
url= generate_public_url ('cover.webp')
{% endhighlight %}

{% highlight python linenos %}
# generating public url for file inside a folder
url= generate_public_url (os.path.join(FOLDER_NAME, 'cover.webp'))
{% endhighlight %}

This article just provides a configuration of Firebase and basic implementation of some features of Firebase. For more references please refer to Firebase docs.

Thanks for being till last.😊😊

References:

[Firebase Docs](https://firebase.google.com/docs)

[Google Cloud Blob Docs](https://googleapis.dev/python/storage/latest/blobs.html)
